"""Code chat API routes (FastAPI).

Provides RAG-powered code chat with SSE streaming.
Uses the stateless retrieval/completion functions from core/stateless.
"""

import json
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import Settings

from ..deps import (
    get_current_user,
    get_db_manager,
    get_pipeline,
    get_project_manager,
    get_conversation_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["code-chat"])

# Code-specific system prompt addition
CODE_SYSTEM_PROMPT = """You are a code intelligence assistant for the CodeLoom platform.
When answering questions about code:
- Cite file paths with line numbers (e.g., src/auth/login.py:42)
- Show function signatures when referencing code
- Format code blocks with appropriate language tags
- Explain how different pieces of code connect to each other
- Be precise about which functions call which, and what dependencies exist
"""

CONCISE_INSTRUCTION = "\nBe concise. Give short, direct answers. Skip lengthy explanations unless asked."


RESPONSE_FORMAT_INSTRUCTIONS = {
    "analytical": "\nProvide a thorough analytical response. Structure your answer with clear sections, evidence, and reasoning.",
    "brief": "\nBe brief and direct. Answer in 2-3 sentences maximum unless more detail is essential.",
    "detailed": "\nProvide a comprehensive, detailed answer with examples and thorough explanations.",
}


def _apply_request_settings(data: "ChatRequest", system_prompt: str) -> str:
    """Apply per-request settings (temperature, response_type, response_format) and return adjusted prompt."""
    if data.temperature is not None:
        Settings.llm.temperature = data.temperature

    # response_format takes precedence over legacy response_type
    if data.response_format and data.response_format in RESPONSE_FORMAT_INSTRUCTIONS:
        return system_prompt + RESPONSE_FORMAT_INSTRUCTIONS[data.response_format]

    if data.response_type == "concise":
        return system_prompt + CONCISE_INSTRUCTION
    return system_prompt


def _effective_top_k(data: "ChatRequest") -> int:
    """Return the effective top_k: explicit top_k overrides max_sources."""
    if data.top_k is not None:
        return max(1, min(data.top_k, 20))  # clamp 1-20
    return data.max_sources


class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None
    session_id: str | None = None
    include_history: bool = True
    max_sources: int = 6
    temperature: float | None = None
    response_type: str = "detailed"  # "detailed" | "concise"
    mode: str = "chat"  # "chat" | "impact"
    # Query options (knowledge project + advanced code chat)
    model: str | None = None           # LLM model override
    reranker_enabled: bool | None = None  # toggle reranking
    response_format: str | None = None  # analytical|detailed|brief|default
    top_k: int | None = None            # retrieval result count (overrides max_sources)


class FeedbackRequest(BaseModel):
    trace_id: str
    helpful: bool
    feedback_category: str | None = None  # inaccurate|irrelevant|incomplete|helpful|other
    user_message: str | None = None


@router.post("/projects/{project_id}/chat/feedback")
async def submit_feedback(
    project_id: str,
    data: FeedbackRequest,
    user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
    db_manager=Depends(get_db_manager),
):
    """Submit thumbs up/down feedback on a chat response.

    Stores in rag_feedback table and forwards score to Langfuse.
    """
    from codeloom.core.services.feedback_service import FeedbackService
    from uuid import uuid4

    user_id = user["user_id"]

    try:
        svc = FeedbackService(pipeline=pipeline, db_manager=db_manager)
        feedback_id = svc.submit_feedback(
            trace_id=data.trace_id,
            query_id=str(uuid4()),
            user_id=user_id,
            project_id=project_id,
            helpful=data.helpful,
            feedback_category=data.feedback_category,
            user_message=data.user_message,
        )
        return {"success": True, "feedback_id": feedback_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        logger.error(f"Feedback submission failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Project-wide analysis types (no entity resolution needed) ─────────
_PROJECT_WIDE_TYPES = {"dead_code", "complexity", "module_deps"}


@router.post("/projects/{project_id}/chat")
async def code_chat(
    project_id: str,
    data: ChatRequest,
    user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
    pm=Depends(get_project_manager),
    conversation_store=Depends(get_conversation_store),
    db_manager=Depends(get_db_manager),
):
    """Non-streaming code chat endpoint.

    Retrieves relevant code chunks, builds context, and returns
    a complete LLM response with source citations.
    """
    from codeloom.core.stateless import (
        fast_retrieve,
        build_context_with_history,
        format_sources,
        execute_query,
        load_conversation_history,
        save_conversation_turn,
        generate_session_id,
    )

    # Validate project
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("ast_status") not in ("complete", "not_applicable"):
        raise HTTPException(
            status_code=400,
            detail=f"Project not ready for chat (status: {project.get('ast_status')})",
        )

    user_id = data.user_id or user["user_id"]
    session_id = data.session_id or generate_session_id()
    start_time = time.time()

    # Start Langfuse trace (no-op if disabled)
    from codeloom.core.observability import get_tracer
    tracer = get_tracer()
    trace_id = tracer.start_trace(
        name="code_chat",
        user_id=user_id,
        notebook_id=project_id,
        query=data.query,
        metadata={"session_id": session_id},
    )

    # Load conversation history
    conversation_history = []
    if data.include_history and conversation_store:
        conversation_history = load_conversation_history(
            conversation_store=conversation_store,
            project_id=project_id,
            user_id=user_id,
            max_history=10,
        )

    # Retrieve relevant code chunks
    nodes = pipeline._get_cached_nodes(project_id)
    retrieval_results = []

    if nodes and pipeline._engine and pipeline._engine._retriever:
        retrieval_results = fast_retrieve(
            nodes=nodes,
            query=data.query,
            project_id=project_id,
            vector_store=pipeline._vector_store,
            retriever_factory=pipeline._engine._retriever,
            llm=Settings.llm,
            top_k=_effective_top_k(data),
        )

    # ASG expansion: enrich results with graph neighbors before context building
    if retrieval_results and project.get("asg_status") == "complete":
        try:
            from codeloom.core.asg_builder.expander import ASGExpander
            expander = ASGExpander(db_manager)
            retrieval_results = expander.expand(
                results=retrieval_results,
                project_id=project_id,
                cached_nodes=nodes,
                max_expansion=data.max_sources * 2,
                score_decay=0.7,
            )
        except Exception as e:
            logger.warning(f"ASG expansion failed, using base results: {e}")

    # Relationship query detection -- runs before impact detection
    relationship_context = ""
    relationship_intent = None
    if project.get("asg_status") == "complete":
        relationship_intent = _detect_relationship_intent(data.query)
        if relationship_intent:
            try:
                rel_type = relationship_intent["relation_type"]
                if rel_type in _PROJECT_WIDE_TYPES:
                    # Project-wide queries skip entity resolution
                    relationship_context, _ = _build_relationship_context(
                        db_manager, project_id, relationship_intent, resolved_units=None
                    )
                    logger.info(
                        "Project-wide analysis query: type=%s",
                        rel_type,
                    )
                else:
                    resolved = _resolve_entity(db_manager, project_id, relationship_intent["entity_name"])
                    if resolved:
                        relationship_context, _ = _build_relationship_context(
                            db_manager, project_id, relationship_intent, resolved
                        )
                        logger.info(
                            "Relationship query: type=%s entity=%s resolved=%d",
                            relationship_intent["relation_type"],
                            relationship_intent["entity_name"],
                            len(resolved),
                        )
            except Exception as e:
                logger.warning(f"Relationship context building failed: {e}")

    # Blast radius detection -- skip if relationship intent already handled
    run_impact = (
        not relationship_intent
        and (data.mode == "impact" or _detect_impact_intent(data.query))
    )
    blast_radius_context = ""
    if run_impact and retrieval_results and project.get("asg_status") == "complete":
        try:
            blast_radius_context, _ = _build_blast_radius_context(
                db_manager, project_id, retrieval_results, depth=3
            )
        except Exception as e:
            logger.warning(f"Blast radius detection failed: {e}")

    # Deep analysis narrative enrichment
    deep_narrative = ""
    if retrieval_results and project.get("deep_analysis_status") == "completed":
        try:
            result_unit_ids = [
                nws.node.metadata.get("unit_id")
                for nws in retrieval_results
                if nws.node.metadata.get("unit_id")
            ]
            if result_unit_ids:
                narratives = _get_relevant_narratives(
                    db_manager, project_id, result_unit_ids
                )
                if narratives:
                    deep_narrative = "\n\n## FUNCTIONAL NARRATIVE\n" + "\n\n".join(narratives)
        except Exception as e:
            logger.warning(f"Deep analysis narrative lookup failed: {e}")

    # Build context with code-specific enrichment
    context = build_context_with_history(
        retrieval_results=retrieval_results,
        conversation_history=conversation_history,
        max_chunks=data.max_sources,
    )
    if relationship_context:
        context = relationship_context + "\n\n" + context
    if blast_radius_context:
        context = blast_radius_context + "\n\n" + context
    if deep_narrative:
        context = deep_narrative + "\n\n" + context

    # Log retrieval span to Langfuse
    retrieval_ms = (time.time() - start_time) * 1000
    tracer.log_span(
        trace_id=trace_id,
        name="retrieval",
        input_data={"query": data.query, "node_count": len(nodes)},
        output_data={
            "result_count": len(retrieval_results),
            "top_scores": [round(r.score or 0, 4) for r in retrieval_results[:5]],
        },
        timing_ms=retrieval_ms,
    )

    # Add code system prompt to context (with response_type / temperature)
    effective_prompt = _apply_request_settings(data, CODE_SYSTEM_PROMPT)
    full_context = f"{effective_prompt}\n\n{context}"

    # Execute LLM query
    llm_start = time.time()
    response_text = execute_query(
        query=data.query,
        context=full_context,
        llm=Settings.llm,
    )
    llm_ms = (time.time() - llm_start) * 1000

    # Log LLM generation to Langfuse
    tracer.log_generation(
        trace_id=trace_id,
        name="llm_response",
        model=getattr(Settings.llm, "model", "unknown"),
        prompt=data.query,
        completion=response_text[:500] if response_text else "",
        timing_ms=llm_ms,
    )

    # Format sources
    sources = format_sources(retrieval_results, max_sources=data.max_sources)
    # Enrich sources with code metadata
    for i, nws in enumerate(retrieval_results[: len(sources)]):
        meta = nws.node.metadata or {}
        sources[i]["unit_name"] = meta.get("unit_name")
        sources[i]["unit_type"] = meta.get("unit_type")
        sources[i]["start_line"] = meta.get("start_line")
        sources[i]["end_line"] = meta.get("end_line")
        sources[i]["language"] = meta.get("language")

    # Save conversation turn
    if conversation_store:
        save_conversation_turn(
            conversation_store=conversation_store,
            project_id=project_id,
            user_id=user_id,
            user_message=data.query,
            assistant_response=response_text,
        )

    elapsed = time.time() - start_time

    # End Langfuse trace
    tracer.end_trace(
        trace_id=trace_id,
        status="success",
        response=response_text[:1000] if response_text else "",
        metadata={"execution_time_ms": int(elapsed * 1000)},
    )

    return {
        "success": True,
        "response": response_text,
        "session_id": session_id,
        "trace_id": trace_id,
        "sources": sources,
        "metadata": {
            "execution_time_ms": int(elapsed * 1000),
            "node_count": len(nodes),
            "retrieval_count": len(retrieval_results),
        },
    }


@router.post("/projects/{project_id}/chat/stream")
async def code_chat_stream(
    project_id: str,
    data: ChatRequest,
    user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
    pm=Depends(get_project_manager),
    conversation_store=Depends(get_conversation_store),
    db_manager=Depends(get_db_manager),
):
    """SSE streaming code chat endpoint.

    Emits events in order:
      1. type=sources  -- list of source citations
      2. type=content  -- streaming LLM response chunks
      3. type=done     -- final metadata
    """
    from codeloom.core.stateless import (
        fast_retrieve,
        build_context_with_history,
        format_sources,
        execute_query_streaming,
        load_conversation_history,
        save_conversation_turn,
        generate_session_id,
    )

    # Validate project
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("ast_status") not in ("complete", "not_applicable"):
        raise HTTPException(
            status_code=400,
            detail=f"Project not ready for chat (status: {project.get('ast_status')})",
        )

    user_id = data.user_id or user["user_id"]
    session_id = data.session_id or generate_session_id()

    # Start Langfuse trace for streaming endpoint
    from codeloom.core.observability import get_tracer
    stream_tracer = get_tracer()
    stream_trace_id = stream_tracer.start_trace(
        name="code_chat_stream",
        user_id=user_id,
        notebook_id=project_id,
        query=data.query,
        metadata={"session_id": session_id, "streaming": True},
    )

    def event_generator():
        start_time = time.time()
        response_text = ""

        try:
            # Load conversation history
            conversation_history = []
            if data.include_history and conversation_store:
                conversation_history = load_conversation_history(
                    conversation_store=conversation_store,
                    project_id=project_id,
                    user_id=user_id,
                    max_history=10,
                )

            # Retrieve relevant code chunks
            nodes = pipeline._get_cached_nodes(project_id)
            retrieval_results = []

            if nodes and pipeline._engine and pipeline._engine._retriever:
                retrieval_results = fast_retrieve(
                    nodes=nodes,
                    query=data.query,
                    project_id=project_id,
                    vector_store=pipeline._vector_store,
                    retriever_factory=pipeline._engine._retriever,
                    llm=Settings.llm,
                    top_k=_effective_top_k(data),
                )

            # ASG expansion
            if retrieval_results and project.get("asg_status") == "complete":
                try:
                    from codeloom.core.asg_builder.expander import ASGExpander
                    expander = ASGExpander(db_manager)
                    retrieval_results = expander.expand(
                        results=retrieval_results,
                        project_id=project_id,
                        cached_nodes=nodes,
                        max_expansion=data.max_sources * 2,
                        score_decay=0.7,
                    )
                except Exception as e:
                    logger.warning(f"ASG expansion failed, using base results: {e}")

            # Relationship query detection -- runs before impact detection
            relationship_context = ""
            relationship_data = None
            relationship_intent = None
            if project.get("asg_status") == "complete":
                relationship_intent = _detect_relationship_intent(data.query)
                if relationship_intent:
                    try:
                        rel_type = relationship_intent["relation_type"]
                        if rel_type in _PROJECT_WIDE_TYPES:
                            # Project-wide queries skip entity resolution
                            relationship_context, relationship_data = _build_relationship_context(
                                db_manager, project_id, relationship_intent, resolved_units=None
                            )
                            logger.info(
                                "Project-wide analysis query: type=%s",
                                rel_type,
                            )
                        else:
                            resolved = _resolve_entity(db_manager, project_id, relationship_intent["entity_name"])
                            if resolved:
                                relationship_context, relationship_data = _build_relationship_context(
                                    db_manager, project_id, relationship_intent, resolved
                                )
                                logger.info(
                                    "Relationship query: type=%s entity=%s resolved=%d",
                                    relationship_intent["relation_type"],
                                    relationship_intent["entity_name"],
                                    len(resolved),
                                )
                    except Exception as e:
                        logger.warning(f"Relationship context building failed: {e}")

            # Emit relationship event if detected
            if relationship_data:
                yield f"data: {json.dumps({'type': 'relationships', 'relationships': relationship_data})}\n\n"

            # Blast radius detection -- skip if relationship intent already handled
            run_impact = (
                not relationship_intent
                and (data.mode == "impact" or _detect_impact_intent(data.query))
            )
            blast_radius_context = ""
            impact_data = None
            impact_status = None

            if run_impact:
                asg_status = project.get("asg_status")
                logger.info("Impact mode: asg_status=%s, retrieval_count=%d", asg_status, len(retrieval_results))
                if not retrieval_results:
                    impact_status = "No code chunks retrieved \u2014 try a more specific query."
                elif asg_status != "complete":
                    impact_status = "ASG (code graph) not built for this project. Re-upload and parse the codebase to enable impact analysis."
                else:
                    try:
                        blast_radius_context, impact_data = _build_blast_radius_context(
                            db_manager, project_id, retrieval_results, depth=3
                        )
                        if not impact_data:
                            impact_status = "No dependents found in the code graph for the retrieved units."
                    except Exception as e:
                        logger.warning(f"Blast radius detection failed: {e}")
                        impact_status = "Impact analysis encountered an error."

            # Emit impact event (structured blast radius data) or status
            if impact_data:
                yield f"data: {json.dumps({'type': 'impact', 'impact': impact_data})}\n\n"
            elif run_impact and impact_status:
                yield f"data: {json.dumps({'type': 'impact_status', 'status': 'unavailable', 'message': impact_status})}\n\n"

            # Emit sources event
            sources = format_sources(retrieval_results, max_sources=data.max_sources)
            for i, nws in enumerate(retrieval_results[: len(sources)]):
                meta = nws.node.metadata or {}
                sources[i]["unit_name"] = meta.get("unit_name")
                sources[i]["unit_type"] = meta.get("unit_type")
                sources[i]["start_line"] = meta.get("start_line")
                sources[i]["end_line"] = meta.get("end_line")
                sources[i]["language"] = meta.get("language")

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Build context (with response_type / temperature)
            context = build_context_with_history(
                retrieval_results=retrieval_results,
                conversation_history=conversation_history,
                max_chunks=data.max_sources,
            )
            if relationship_context:
                context = relationship_context + "\n\n" + context
            if blast_radius_context:
                context = blast_radius_context + "\n\n" + context
            effective_prompt = _apply_request_settings(data, CODE_SYSTEM_PROMPT)
            full_context = f"{effective_prompt}\n\n{context}"

            # Stream LLM response
            for chunk in execute_query_streaming(
                query=data.query,
                context=full_context,
                llm=Settings.llm,
            ):
                response_text += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            # Save conversation turn
            if conversation_store and response_text:
                save_conversation_turn(
                    conversation_store=conversation_store,
                    project_id=project_id,
                    user_id=user_id,
                    user_message=data.query,
                    assistant_response=response_text,
                )

            elapsed = time.time() - start_time
            # End Langfuse trace
            stream_tracer.end_trace(
                stream_trace_id, status="success",
                response=response_text[:1000] if response_text else "",
                metadata={"execution_time_ms": int(elapsed * 1000)},
            )

            yield f"data: {json.dumps({'type': 'done', 'trace_id': stream_trace_id, 'metadata': {'execution_time_ms': int(elapsed * 1000), 'session_id': session_id, 'node_count': len(nodes), 'retrieval_count': len(retrieval_results)}})}\n\n"

        except Exception as e:
            logger.error(f"Error in code chat stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_IMPACT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"impact",
        r"blast\s*radius",
        r"affected",
        r"what\s+happens\s+if",
        r"change.*affect",
        r"ripple",
        r"downstream",
        r"upstream",
        r"who\s+uses",
        r"breaking\s+change",
    ]
]


def _detect_impact_intent(query: str) -> bool:
    """Detect whether a chat query expresses impact-analysis intent."""
    return any(pattern.search(query) for pattern in _IMPACT_PATTERNS)


# ── Relationship intent detection ──────────────────────────────────────

# Entity name: backtick-wrapped, quoted, or a code-like identifier
# (camelCase, PascalCase, snake_case, dot.separated -- excludes common English words)
_RELATIONSHIP_INTENT_PROMPT = """\
You are a code relationship intent classifier. Given a user's chat query about a codebase, determine if they are asking about a specific code relationship or analysis.

Relation types:
- callers: who/what calls a function/method
- callees: what does a function/method call
- dependencies: what does a unit depend on (imports, calls, inherits)
- dependents: what depends on / uses a unit (reverse dependencies)
- inheritors: what classes inherit from / extend a class
- implementors: what classes implement an interface
- call_chain: path/chain from one function to another

Additional analysis types:
- all_callers: "show all callers of X transitively", "who calls X recursively", "full caller tree"
- all_callees: "what does X call transitively", "full call tree of X", "everything X calls"
- dead_code: "find dead code", "unused functions", "unreachable code", "functions with no callers"
- complexity: "most complex functions", "complexity report", "cyclomatic complexity", "longest functions"
- decorators: "functions with @X", "methods annotated with X", "find @Override", "uses @Transactional"
- module_deps: "module dependencies", "file dependencies", "which modules depend on X", "directory dependency graph"

For dead_code, complexity, and module_deps: entity_name can be "ALL" (project-wide scan).
For decorators: entity_name is the decorator name (e.g., "Override", "Transactional", "app.route").

Respond with EXACTLY one line in this format:
- If relationship/analysis query: RELATION|entity_name|relation_type|target_name_or_NONE
- If NOT a relationship query: NONE

Examples:
Query: "What does processOrder call?"
RELATION|processOrder|callees|NONE

Query: "Who calls validateInput?"
RELATION|validateInput|callers|NONE

Query: "What depends on BaseModel?"
RELATION|BaseModel|dependents|NONE

Query: "Show call path from main to saveRecord"
RELATION|main|call_chain|saveRecord

Query: "What uses AuthMiddleware?"
RELATION|AuthMiddleware|dependents|NONE

Query: "What inherits from BaseModel?"
RELATION|BaseModel|inheritors|NONE

Query: "Show all transitive callers of processOrder"
RELATION|processOrder|all_callers|NONE

Query: "What does main call recursively?"
RELATION|main|all_callees|NONE

Query: "Find dead code in the project"
RELATION|ALL|dead_code|NONE

Query: "Show me the most complex functions"
RELATION|ALL|complexity|NONE

Query: "Find all functions with @Override"
RELATION|Override|decorators|NONE

Query: "Show module dependencies"
RELATION|ALL|module_deps|NONE

Query: "Which functions are annotated with @Transactional?"
RELATION|Transactional|decorators|NONE

Query: "explain the auth flow"
NONE

Query: "how does the login work"
NONE

Query: "{query}"
"""


def _detect_relationship_intent(query: str) -> dict | None:
    """Detect whether a chat query asks about code relationships using LLM classification.

    Makes a lightweight LLM call to classify the query intent.
    Returns dict with entity_name, relation_type, depth, target_name (for call_chain)
    or None if no relationship intent detected.
    """
    from llama_index.core import Settings as LISettings

    prompt = _RELATIONSHIP_INTENT_PROMPT.format(query=query)

    try:
        response = LISettings.llm.complete(prompt)
        result = response.text.strip()
    except Exception as e:
        logger.warning("Relationship intent classification failed: %s", e)
        return None

    if not result or result == "NONE":
        return None

    if not result.startswith("RELATION|"):
        return None

    parts = result.split("|")
    if len(parts) < 4:
        return None

    _, entity_name, relation_type, target_raw = parts[0], parts[1], parts[2], parts[3]

    valid_types = {
        "callers", "callees", "dependencies", "dependents",
        "inheritors", "implementors", "call_chain",
        # Extended analysis types
        "all_callers", "all_callees",
        "dead_code", "complexity", "decorators", "module_deps",
    }
    if relation_type not in valid_types:
        return None

    entity_name = entity_name.strip().strip("`'\"")
    if not entity_name:
        return None

    # Project-wide queries don't need a specific entity name
    if relation_type in _PROJECT_WIDE_TYPES:
        entity_name = entity_name or "ALL"

    target_name = None
    if target_raw.strip() not in ("NONE", ""):
        target_name = target_raw.strip().strip("`'\"")

    depth_map = {
        "callers": 2, "callees": 2,
        "dependencies": 2, "dependents": 2,
        "inheritors": 3, "implementors": 1,
        "call_chain": 10,
        "all_callers": 5, "all_callees": 5,
        "dead_code": 1, "complexity": 1,
        "decorators": 1, "module_deps": 1,
    }

    return {
        "entity_name": entity_name,
        "relation_type": relation_type,
        "depth": depth_map.get(relation_type, 2),
        "target_name": target_name if relation_type == "call_chain" else None,
    }


def _resolve_entity(
    db_manager,
    project_id: str,
    name: str,
    max_results: int = 10,
) -> list[dict]:
    """Resolve a function/class name to code_unit records.

    Tries exact match first, then qualified_name suffix, then fuzzy ILIKE.
    Returns list of {unit_id, name, qualified_name, unit_type, language, file_path}.
    """
    from uuid import UUID as _UUID
    from codeloom.core.db.models import CodeUnit, CodeFile
    from sqlalchemy import func

    pid = _UUID(project_id)

    with db_manager.get_session() as session:
        # 1. Exact name match (case-insensitive)
        units = (
            session.query(CodeUnit, CodeFile.file_path)
            .outerjoin(CodeFile, CodeUnit.file_id == CodeFile.file_id)
            .filter(
                CodeUnit.project_id == pid,
                func.lower(CodeUnit.name) == name.lower(),
            )
            .limit(max_results)
            .all()
        )

        # 2. Qualified name suffix match
        if not units:
            units = (
                session.query(CodeUnit, CodeFile.file_path)
                .outerjoin(CodeFile, CodeUnit.file_id == CodeFile.file_id)
                .filter(
                    CodeUnit.project_id == pid,
                    CodeUnit.qualified_name.ilike(f"%.{name}"),
                )
                .limit(max_results)
                .all()
            )

        # 3. Fuzzy ILIKE match
        if not units:
            units = (
                session.query(CodeUnit, CodeFile.file_path)
                .outerjoin(CodeFile, CodeUnit.file_id == CodeFile.file_id)
                .filter(
                    CodeUnit.project_id == pid,
                    CodeUnit.name.ilike(f"%{name}%"),
                )
                .limit(max_results)
                .all()
            )

        return [
            {
                "unit_id": str(u.unit_id),
                "name": u.name,
                "qualified_name": u.qualified_name,
                "unit_type": u.unit_type,
                "language": u.language,
                "file_path": fp or "",
            }
            for u, fp in units
        ]


def _build_relationship_context(
    db_manager,
    project_id: str,
    intent: dict,
    resolved_units: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Build LLM context from ASG graph traversal for a relationship query.

    For project-wide queries (dead_code, complexity, module_deps), resolved_units
    can be None. For entity-specific queries, resolved_units must be provided.

    Returns (markdown_context, structured_data).
    """
    from codeloom.core.asg_builder.queries import (
        get_callers, get_callees, get_dependencies, get_dependents,
        find_call_path,
    )
    from collections import defaultdict

    rel_type = intent["relation_type"]
    depth = intent["depth"]
    sections: list[str] = []
    structured: list[dict] = []

    # ── Extended analysis types (new) ─────────────────────────────────
    if rel_type in ("all_callers", "all_callees", "dead_code", "complexity",
                     "decorators", "module_deps"):
        return _build_extended_analysis_context(
            db_manager, project_id, intent, resolved_units
        )

    # ── Original relationship types ───────────────────────────────────
    # Map relation_type to query function + direction label
    query_map = {
        "callers": (get_callers, "Callers of"),
        "callees": (get_callees, "Callees of"),
        "dependencies": (get_dependencies, "Dependencies of"),
        "dependents": (get_dependents, "Dependents of"),
        "inheritors": (get_dependents, "Classes that inherit from"),
        "implementors": (get_dependents, "Classes that implement"),
    }

    if rel_type == "call_chain":
        # Special: find path between two entities
        target_name = intent.get("target_name")
        if target_name:
            target_units = _resolve_entity(db_manager, project_id, target_name, max_results=5)
        else:
            target_units = []

        for src in (resolved_units or [])[:3]:
            for tgt in target_units[:3]:
                try:
                    path = find_call_path(
                        db_manager, project_id,
                        src["unit_id"], tgt["unit_id"],
                        max_depth=depth,
                    )
                except Exception as e:
                    logger.debug(f"Call path lookup failed: {e}")
                    path = None

                if path:
                    section = f"### Call Path: `{src['name']}` -> `{tgt['name']}`\n\n"
                    for i, step in enumerate(path, 1):
                        prefix = "   -> calls\n" if i > 1 else ""
                        loc = f", {step.get('language', '')}" if step.get('language') else ""
                        section += f"{prefix}{i}. `{step['qualified_name'] or step['name']}()` ({step['unit_type']}{loc})\n"
                    section += f"\nPath length: {len(path)} hops\n"
                    sections.append(section)
                    structured.append({
                        "type": "call_chain",
                        "source": src["name"],
                        "target": tgt["name"],
                        "path": path,
                        "hops": len(path),
                    })
                else:
                    sections.append(
                        f"### No call path found from `{src['name']}` to `{tgt['name']}` "
                        f"within {depth} hops.\n"
                    )

        if not target_units:
            sections.append(
                f"### Could not resolve target entity `{intent.get('target_name', '?')}`\n"
            )

    elif rel_type in query_map:
        query_fn, label = query_map[rel_type]

        # For inheritors/implementors, filter to specific edge types
        edge_filter = None
        if rel_type == "inheritors":
            edge_filter = "inherits"
        elif rel_type == "implementors":
            edge_filter = "implements"

        for unit in (resolved_units or [])[:5]:
            try:
                results = query_fn(db_manager, project_id, unit["unit_id"], depth=depth)
            except Exception as e:
                logger.debug(f"Relationship query failed for {unit['name']}: {e}")
                continue

            # Apply edge type filter if needed
            if edge_filter:
                results = [r for r in results if r.get("edge_type") == edge_filter]

            if not results:
                sections.append(
                    f"### {label} `{unit['name']}` ({unit['unit_type']} in {unit['file_path']})\n\n"
                    f"No results found.\n"
                )
                continue

            section = (
                f"### {label} `{unit['name']}` ({unit['unit_type']} in {unit['file_path']})\n\n"
            )

            # Group by depth
            by_depth: dict[int, list] = defaultdict(list)
            for r in results:
                by_depth[r.get("depth", 1)].append(r)

            for d in sorted(by_depth.keys()):
                items = by_depth[d]
                depth_label = "Direct" if d == 1 else f"Depth {d}"
                section += f"**{depth_label}:**\n"
                for r in items[:20]:
                    qual = r.get("qualified_name", r["name"])
                    edge = r.get("edge_type", "")
                    section += f"- `{qual}` ({r['unit_type']}) via {edge}\n"
                if len(items) > 20:
                    section += f"- ... and {len(items) - 20} more\n"
                section += "\n"

            sections.append(section)
            structured.append({
                "type": rel_type,
                "unit_name": unit["name"],
                "unit_type": unit["unit_type"],
                "file_path": unit["file_path"],
                "results": results[:30],
                "total": len(results),
            })

    if not sections:
        return "", []

    header = "## Code Relationship Analysis\n\n"
    return header + "\n".join(sections), structured


def _build_extended_analysis_context(
    db_manager,
    project_id: str,
    intent: dict,
    resolved_units: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Build LLM context for extended analysis types.

    Handles: all_callers, all_callees, dead_code, complexity, decorators, module_deps.

    Returns (markdown_context, structured_data).
    """
    # Import with fallback -- these functions are in the same queries module
    try:
        from codeloom.core.asg_builder.queries import (
            get_all_callers, get_all_callees, get_dead_code,
            get_complexity_report, find_units_by_decorator,
            get_module_dependency_graph,
        )
    except ImportError:
        logger.warning("Extended ASG query functions not available")
        return "Extended analysis queries are not available in this version.", []

    rel_type = intent["relation_type"]
    depth = intent["depth"]
    sections: list[str] = []
    structured: list[dict] = []

    if rel_type == "all_callers":
        if not resolved_units:
            return "Could not resolve entity for transitive caller analysis.", []

        for unit in resolved_units[:3]:
            try:
                data = get_all_callers(
                    db_manager, project_id, unit["unit_id"], max_depth=depth
                )
            except Exception as e:
                logger.debug(f"Transitive callers lookup failed for {unit['name']}: {e}")
                continue

            results = data.get("results", [])
            total = data.get("total", 0)
            max_depth_reached = data.get("max_depth_reached", 0)

            section = (
                f"### Transitive Callers of `{unit['name']}` "
                f"({unit['unit_type']} in {unit['file_path']})\n\n"
                f"Found {total} callers across {max_depth_reached} depth levels:\n\n"
            )
            for r in results[:30]:
                qual = r.get("qualified_name", r["name"])
                fp = r.get("file_path", "")
                section += f"  [{r.get('depth', '?')}] `{qual}` ({fp})\n"
            if total > 30:
                section += f"\n  ... and {total - 30} more\n"

            sections.append(section)
            structured.append({
                "type": "all_callers",
                "unit_name": unit["name"],
                "total": total,
                "max_depth_reached": max_depth_reached,
                "results": results[:30],
            })

    elif rel_type == "all_callees":
        if not resolved_units:
            return "Could not resolve entity for transitive callee analysis.", []

        for unit in resolved_units[:3]:
            try:
                data = get_all_callees(
                    db_manager, project_id, unit["unit_id"], max_depth=depth
                )
            except Exception as e:
                logger.debug(f"Transitive callees lookup failed for {unit['name']}: {e}")
                continue

            results = data.get("results", [])
            total = data.get("total", 0)
            max_depth_reached = data.get("max_depth_reached", 0)

            section = (
                f"### Transitive Callees of `{unit['name']}` "
                f"({unit['unit_type']} in {unit['file_path']})\n\n"
                f"Found {total} callees across {max_depth_reached} depth levels:\n\n"
            )
            for r in results[:30]:
                qual = r.get("qualified_name", r["name"])
                fp = r.get("file_path", "")
                section += f"  [{r.get('depth', '?')}] `{qual}` ({fp})\n"
            if total > 30:
                section += f"\n  ... and {total - 30} more\n"

            sections.append(section)
            structured.append({
                "type": "all_callees",
                "unit_name": unit["name"],
                "total": total,
                "max_depth_reached": max_depth_reached,
                "results": results[:30],
            })

    elif rel_type == "dead_code":
        try:
            data = get_dead_code(db_manager, project_id, limit=30)
        except Exception as e:
            logger.debug(f"Dead code scan failed: {e}")
            return "Dead code analysis encountered an error.", []

        section = f"### Potentially Dead Code ({len(data)} functions with no callers)\n\n"
        for r in data:
            qual = r.get("qualified_name", r["name"])
            fp = r.get("file_path", "")
            sl = r.get("start_line", "?")
            section += f"  - `{qual}` in {fp}:{sl}\n"

        if not data:
            section += "No potentially dead functions found.\n"

        sections.append(section)
        structured.append({
            "type": "dead_code",
            "total": len(data),
            "results": data,
        })

    elif rel_type == "complexity":
        try:
            data = get_complexity_report(db_manager, project_id, sort="desc", limit=20)
        except Exception as e:
            logger.debug(f"Complexity report failed: {e}")
            return "Complexity analysis encountered an error.", []

        section = "### Most Complex Functions\n\n"
        for r in data:
            name = r.get("name", "?")
            fp = r.get("file_path", "")
            complexity = r.get("complexity", 0)
            line_count = r.get("line_count", 0)
            branch_count = r.get("branch_count", 0)
            section += (
                f"  - `{name}` ({fp}) -- "
                f"complexity: {complexity} "
                f"(lines: {line_count}, branches: {branch_count})\n"
            )

        if not data:
            section += "No functions found for complexity analysis.\n"

        sections.append(section)
        structured.append({
            "type": "complexity",
            "total": len(data),
            "results": data,
        })

    elif rel_type == "decorators":
        decorator_name = intent.get("entity_name", "")
        if not decorator_name or decorator_name == "ALL":
            return "Please specify a decorator name (e.g., 'find functions with @Override').", []

        try:
            data = find_units_by_decorator(
                db_manager, project_id, decorator_name, limit=30
            )
        except Exception as e:
            logger.debug(f"Decorator search failed: {e}")
            return "Decorator search encountered an error.", []

        section = f"### Functions/Methods with @{decorator_name}\n\n"
        for r in data:
            qual = r.get("qualified_name", r["name"])
            fp = r.get("file_path", "")
            sl = r.get("start_line", "?")
            section += f"  - `{qual}` ({fp}:{sl})\n"

        if not data:
            section += f"No functions/methods found with @{decorator_name}.\n"

        sections.append(section)
        structured.append({
            "type": "decorators",
            "decorator": decorator_name,
            "total": len(data),
            "results": data,
        })

    elif rel_type == "module_deps":
        try:
            data = get_module_dependency_graph(
                db_manager, project_id, level="directory"
            )
        except Exception as e:
            logger.debug(f"Module dependency graph failed: {e}")
            return "Module dependency analysis encountered an error.", []

        links = data.get("links", [])
        nodes = data.get("nodes", [])

        section = f"### Module Dependencies (directory level, {len(nodes)} modules)\n\n"
        for link in links[:30]:
            section += (
                f"  {link['source']} -> {link['target']} "
                f"(weight: {link['weight']})\n"
            )
        if len(links) > 30:
            section += f"\n  ... and {len(links) - 30} more edges\n"

        if not links:
            section += "No inter-module dependencies found.\n"

        sections.append(section)
        structured.append({
            "type": "module_deps",
            "total_modules": len(nodes),
            "total_links": len(links),
            "links": links[:30],
        })

    if not sections:
        return "", []

    header = "## Code Analysis Results\n\n"
    return header + "\n".join(sections), structured


def _traverse_single(
    db_manager,
    project_id: str,
    unit_id: str,
    direction: str,
    edge_type: str,
) -> list[dict]:
    """Single-hop graph lookup (no recursion). Returns immediate neighbors."""
    from sqlalchemy import text
    from uuid import UUID

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uid = UUID(unit_id) if isinstance(unit_id, str) else unit_id

    if direction == "incoming":
        filter_col, select_col = "target_unit_id", "source_unit_id"
    else:
        filter_col, select_col = "source_unit_id", "target_unit_id"

    sql = text(f"""
        SELECT u.unit_id, u.name, u.qualified_name, u.unit_type, u.file_id
        FROM code_edges e
        JOIN code_units u ON e.{select_col} = u.unit_id
        WHERE e.{filter_col} = :uid
          AND e.project_id = :pid
          AND e.edge_type = :etype
    """)

    with db_manager.get_session() as session:
        result = session.execute(sql, {"pid": pid, "uid": uid, "etype": edge_type})
        return [
            {
                "unit_id": str(row.unit_id),
                "name": row.name,
                "qualified_name": row.qualified_name,
                "unit_type": row.unit_type,
                "file_id": str(row.file_id) if row.file_id else "",
            }
            for row in result.fetchall()
        ]


def _compute_impact_score(
    direct: int,
    indirect: int,
    files_affected: int,
    total_files: int,
    max_depth: int,
    edge_types: set,
    unit_type: str,
) -> tuple[float, str]:
    """Compute blast radius impact score (0.0 = isolated, 1.0 = critical).

    Five dimensions, weighted:
    - Reach (40%): How many units are affected
    - Spread (20%): How many files are touched
    - Depth (15%): Transitive dependency chain length
    - Coupling (15%): Diversity of edge types (inherits > calls > imports)
    - Criticality (10%): Unit type weight (interface > class > method)

    Returns (score, level) where level is "critical"/"high"/"moderate"/"low".
    """
    # 1. Reach: saturates at 50 total dependents, direct weighted 2x
    reach = min(1.0, (direct * 2 + indirect) / 50)

    # 2. Spread: normalized against 20% of project files
    denom = max(1, total_files * 0.2)
    spread = min(1.0, files_affected / denom)

    # 3. Depth: saturates at depth 5
    depth_score = min(1.0, max_depth / 5)

    # 4. Coupling: score based on edge type diversity
    coupling = 0.0
    if "inherits" in edge_types or "implements" in edge_types:
        coupling += 0.5
    if "calls" in edge_types or "calls_sp" in edge_types:
        coupling += 0.3
    if "type_dep" in edge_types:
        coupling += 0.15
    if "imports" in edge_types:
        coupling += 0.05
    coupling = min(1.0, coupling)

    # 5. Criticality: based on unit type
    criticality_map = {
        "interface": 1.0,
        "class": 0.8,
        "struct": 0.7,
        "module": 0.7,
        "method": 0.5,
        "function": 0.5,
        "constructor": 0.4,
        "property": 0.3,
    }
    criticality = criticality_map.get(unit_type, 0.3)

    # Weighted sum
    score = (
        reach * 0.40
        + spread * 0.20
        + depth_score * 0.15
        + coupling * 0.15
        + criticality * 0.10
    )
    score = round(min(1.0, score), 2)

    # Classification
    if score >= 0.8:
        level = "critical"
    elif score >= 0.5:
        level = "high"
    elif score >= 0.25:
        level = "moderate"
    else:
        level = "low"

    return score, level


def _build_blast_radius_context(
    db_manager,
    project_id: str,
    retrieval_results,
    depth: int = 3,
    max_units: int = 5,
) -> tuple[str, list[dict]]:
    """Build a structured impact report with multi-root expansion.

    Phase 1: Collect retrieval roots from retrieval results.
    Phase 2: Expand roots via implements/inherits edges (ASG-based).
    Phase 3: Multi-root traversal -- get_dependents for all roots.
    Phase 4: Build enriched LLM context with relationship graph.

    Returns (markdown_context, structured_impact_list).
    """
    from codeloom.core.asg_builder.queries import get_dependents, get_dependencies
    from collections import defaultdict

    # -- Phase 1: Collect retrieval roots --------------------------------
    roots: dict[str, dict] = {}  # unit_id -> {name, file_path, source}
    seen_names: set[str] = set()

    for nws in retrieval_results[:max_units]:
        meta = nws.node.metadata or {}
        unit_id = meta.get("unit_id")
        unit_name = meta.get("unit_name") or meta.get("name", "unknown")
        file_path = meta.get("file_path", "")

        if not unit_id or unit_name in seen_names:
            continue
        seen_names.add(unit_name)
        roots[str(unit_id)] = {
            "name": unit_name,
            "file_path": file_path,
            "source": "retrieval",
        }

    # -- Phase 1b: Resolve parent classes for method-level units ----------
    # Retrieval often returns methods; implements/inherits edges live on the class.
    # Walk up via incoming 'contains' edge to find the parent class.
    parent_roots: dict[str, dict] = {}
    for uid, info in list(roots.items()):
        try:
            # Incoming contains: class --contains--> this method
            incoming = _traverse_single(
                db_manager, project_id, uid, direction="incoming", edge_type="contains",
            )
            for parent in incoming:
                pid_str = str(parent["unit_id"])
                if parent["unit_type"] in ("class", "interface") and parent["name"] not in seen_names:
                    parent_roots[pid_str] = {
                        "name": parent["name"],
                        "file_path": "",
                        "source": "parent_class",
                    }
                    seen_names.add(parent["name"])
                    logger.info("Resolved parent class: %s -> %s", info["name"], parent["name"])
        except Exception:
            continue

    # Merge parent classes into roots (they get full traversal)
    roots.update(parent_roots)

    # -- Phase 2: ASG-based root expansion --------------------------------
    expanded: dict[str, dict] = {}
    relationship_lines: list[str] = []

    for uid, info in roots.items():
        try:
            outgoing = get_dependencies(db_manager, project_id, uid, depth=1)
        except Exception:
            continue

        for dep in outgoing:
            dep_id = str(dep["unit_id"])
            if dep["edge_type"] in ("implements", "inherits") and dep["name"] not in seen_names:
                expanded[dep_id] = {
                    "name": dep["name"],
                    "file_path": "",
                    "source": "expanded",
                    "unit_type": dep.get("unit_type", ""),
                }
                seen_names.add(dep["name"])
                relationship_lines.append(
                    f"- {info['name']} {dep['edge_type']} {dep['name']}"
                )
            elif dep["edge_type"] == "imports":
                relationship_lines.append(
                    f"- {info['name']} imports {dep['name']}"
                )

    # Cap expanded roots to avoid runaway expansion
    max_expanded = max_units * 2
    all_roots = {**roots}
    for k, v in list(expanded.items())[:max_expanded - len(roots)]:
        all_roots[k] = v

    logger.info(
        "Blast radius: %d retrieval roots + %d expanded roots",
        len(roots), len(all_roots) - len(roots),
    )

    # -- Phase 3: Multi-root traversal ------------------------------------
    # Get total file count for impact score normalization
    total_files = 1
    try:
        from codeloom.core.db.models import CodeFile
        from uuid import UUID as _UUID
        with db_manager.get_session() as session:
            total_files = max(1, session.query(CodeFile).filter(
                CodeFile.project_id == _UUID(project_id)
            ).count())
    except Exception:
        pass

    sections: list[str] = []
    impact_list: list[dict] = []
    topology_lines: list[str] = []

    for uid, info in all_roots.items():
        try:
            dependents = get_dependents(db_manager, project_id, uid, depth=depth)
        except Exception as e:
            logger.debug("Blast radius lookup failed for unit %s: %s", uid, e)
            continue

        unit_name = info["name"]
        file_path = info["file_path"]
        source = info.get("source", "retrieval")

        logger.info("Impact: unit=%s dependents=%d source=%s", unit_name, len(dependents), source)

        if not dependents:
            impact_list.append({
                "unit_name": unit_name,
                "file_path": file_path,
                "direct": 0,
                "indirect": 0,
                "files_affected": 0,
                "impact_score": 0.0,
                "impact_level": "low",
                "dependents": [],
                "source": source,
            })
            continue

        # Group by depth
        by_depth: dict[int, list] = defaultdict(list)
        for dep in dependents:
            by_depth[dep["depth"]].append(dep)

        source_tag = " *(expanded)*" if source == "expanded" else ""
        section = f"### Changes to `{unit_name}`{source_tag} affect:\n"

        for d in sorted(by_depth.keys()):
            deps = by_depth[d]
            label = "Direct dependents" if d == 1 else f"Indirect dependents (depth {d})"
            section += f"\n**{label}**:\n"
            for dep in deps[:10]:
                qual = dep.get("qualified_name", dep["name"])
                edge = dep.get("edge_type", "depends")
                section += f"- `{qual}` ({edge})\n"
            if len(deps) > 10:
                section += f"- ... and {len(deps) - 10} more\n"

        total_direct = len(by_depth.get(1, []))
        total_indirect = sum(len(v) for k, v in by_depth.items() if k > 1)
        unique_files = len({dep.get("file_id") for dep in dependents if dep.get("file_id")})
        max_dep_depth = max(by_depth.keys()) if by_depth else 0

        # Compute impact score
        edge_type_set = {dep.get("edge_type", "") for dep in dependents}
        unit_type = info.get("unit_type", "")
        impact_score, impact_level = _compute_impact_score(
            direct=total_direct,
            indirect=total_indirect,
            files_affected=unique_files,
            total_files=total_files,
            max_depth=max_dep_depth,
            edge_types=edge_type_set,
            unit_type=unit_type,
        )

        section += (
            f"\n**Impact Score: {impact_score} ({impact_level.upper()})** "
            f"-- {total_direct} direct + {total_indirect} indirect dependents "
            f"across {unique_files} files\n"
        )
        sections.append(section)

        # Topology line for enriched LLM context
        topology_lines.append(
            f"- {unit_name}: impact={impact_score} ({impact_level}), "
            f"{total_direct} direct dependents, "
            f"{total_indirect} indirect (depth {depth}) across {unique_files} files"
        )

        # Build structured data for SSE event
        impact_list.append({
            "unit_name": unit_name,
            "file_path": file_path,
            "direct": total_direct,
            "indirect": total_indirect,
            "files_affected": unique_files,
            "impact_score": impact_score,
            "impact_level": impact_level,
            "source": source,
            "dependents": [
                {
                    "name": dep.get("qualified_name", dep["name"]),
                    "edge_type": dep.get("edge_type", "depends"),
                    "depth": dep["depth"],
                }
                for dep in dependents[:30]
            ],
        })

    if not sections:
        return "", impact_list

    # -- Phase 4: Enriched LLM context ------------------------------------
    asg_context = ""
    if relationship_lines or topology_lines:
        asg_parts = ["## ASG Context for Impact Analysis\n"]
        if relationship_lines:
            asg_parts.append("### Relationship Graph")
            asg_parts.extend(relationship_lines)
            asg_parts.append("")
        if topology_lines:
            asg_parts.append("### Impact Topology")
            asg_parts.extend(topology_lines)
            asg_parts.append("")
        asg_context = "\n".join(asg_parts) + "\n"

    return asg_context + "## IMPACT ANALYSIS\n\n" + "\n".join(sections), impact_list


def _get_relevant_narratives(
    db_manager,
    project_id: str,
    unit_ids: list,
    max_narratives: int = 3,
) -> list:
    """Look up deep analysis narratives covering the given units."""
    from uuid import UUID
    from sqlalchemy import text

    pid = UUID(project_id)
    placeholders = ", ".join(f":uid{i}" for i in range(len(unit_ids)))
    params = {"pid": pid, "limit": max_narratives}
    for i, uid in enumerate(unit_ids):
        params[f"uid{i}"] = UUID(uid) if isinstance(uid, str) else uid

    sql = f"""
        SELECT a.narrative
        FROM deep_analyses a
        JOIN (
            SELECT au.analysis_id,
                   COUNT(*) AS overlap_units,
                   MIN(au.min_depth) AS best_depth,
                   SUM(au.path_count) AS overlap_paths
            FROM analysis_units au
            WHERE au.project_id = :pid
              AND au.unit_id IN ({placeholders})
            GROUP BY au.analysis_id
        ) ov ON ov.analysis_id = a.analysis_id
        WHERE a.narrative IS NOT NULL
          AND a.narrative != ''
        ORDER BY ov.overlap_units DESC, ov.best_depth ASC, ov.overlap_paths DESC
        LIMIT :limit
    """

    with db_manager.get_session() as session:
        result = session.execute(text(sql), params)
        return [row.narrative for row in result.fetchall()]
