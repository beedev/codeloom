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


def _apply_request_settings(data: "ChatRequest", system_prompt: str) -> str:
    """Apply per-request settings (temperature, response_type) and return adjusted prompt."""
    if data.temperature is not None:
        Settings.llm.temperature = data.temperature

    if data.response_type == "concise":
        return system_prompt + CONCISE_INSTRUCTION
    return system_prompt


class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None
    session_id: str | None = None
    include_history: bool = True
    max_sources: int = 6
    temperature: float | None = None
    response_type: str = "detailed"  # "detailed" | "concise"
    mode: str = "chat"  # "chat" | "impact"


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
    if project.get("ast_status") not in ("complete",):
        raise HTTPException(
            status_code=400,
            detail=f"Project not ready for chat (status: {project.get('ast_status')})",
        )

    user_id = data.user_id or user["user_id"]
    session_id = data.session_id or generate_session_id()
    start_time = time.time()

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
            top_k=data.max_sources,
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

    # Blast radius detection — explicit mode toggle OR auto-detected intent
    run_impact = data.mode == "impact" or _detect_impact_intent(data.query)
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
    if blast_radius_context:
        context = blast_radius_context + "\n\n" + context
    if deep_narrative:
        context = deep_narrative + "\n\n" + context

    # Add code system prompt to context (with response_type / temperature)
    effective_prompt = _apply_request_settings(data, CODE_SYSTEM_PROMPT)
    full_context = f"{effective_prompt}\n\n{context}"

    # Execute LLM query
    response_text = execute_query(
        query=data.query,
        context=full_context,
        llm=Settings.llm,
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

    return {
        "success": True,
        "response": response_text,
        "session_id": session_id,
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
      1. type=sources  — list of source citations
      2. type=content  — streaming LLM response chunks
      3. type=done     — final metadata
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

    user_id = data.user_id or user["user_id"]
    session_id = data.session_id or generate_session_id()

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
                    top_k=data.max_sources,
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

            # Blast radius detection — explicit mode toggle OR auto-detected intent
            run_impact = data.mode == "impact" or _detect_impact_intent(data.query)
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
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'execution_time_ms': int(elapsed * 1000), 'session_id': session_id, 'node_count': len(nodes), 'retrieval_count': len(retrieval_results)}})}\n\n"

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
        r"depends?\s+on",
        r"what\s+happens\s+if",
        r"change.*affect",
        r"ripple",
        r"downstream",
        r"upstream",
        r"who\s+calls",
        r"who\s+uses",
        r"breaking\s+change",
    ]
]


def _detect_impact_intent(query: str) -> bool:
    """Detect whether a chat query expresses impact-analysis intent."""
    return any(pattern.search(query) for pattern in _IMPACT_PATTERNS)


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
    Phase 3: Multi-root traversal — get_dependents for all roots.
    Phase 4: Build enriched LLM context with relationship graph.

    Returns (markdown_context, structured_impact_list).
    """
    from codeloom.core.asg_builder.queries import get_dependents, get_dependencies
    from collections import defaultdict

    # ── Phase 1: Collect retrieval roots ──────────────────────────────
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

    # ── Phase 1b: Resolve parent classes for method-level units ─────────
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

    # ── Phase 2: ASG-based root expansion ─────────────────────────────
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

    # ── Phase 3: Multi-root traversal ─────────────────────────────────
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

    # ── Phase 4: Enriched LLM context ─────────────────────────────────
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
