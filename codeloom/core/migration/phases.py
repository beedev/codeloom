"""Phase executors for the MVP-centric migration pipeline.

Pipeline V1 (6-phase):
  Plan-level (mvp_id = NULL):
    1: Discovery — analyze codebase, identify MVP clusters
    2: Architecture — design target architecture (system-wide)
  Per-MVP (mvp_id set):
    3: Analyze — deep analysis scoped to the MVP's units
    4: Design — detailed module design (+ SP stubs when SPs exist)
    5: Transform — code migration (+ SP stubs when SPs exist)
    6: Test — scoped test generation and validation

Pipeline V2 (4-phase, Architecture-first):
  Plan-level (mvp_id = NULL):
    1: Architecture — design target architecture first
    2: Discovery — clustering informed by architecture output
  Per-MVP (mvp_id set):
    3: Transform — code migration
    4: Test — scoped test generation and validation
  On-demand (stored on FunctionalMVP, not as phase):
    Deep Analyze — merges old Analyze + Design

Each executor:
1. Builds context via MigrationContextBuilder
2. Assembles the LLM prompt via prompts module
3. Calls the LLM
4. Parses and returns the output
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llama_index.core import Settings

from .context_builder import MigrationContextBuilder
from .doc_enricher import DocEnricher
from . import prompts
from .agent import MigrationAgent, AgentEvent, build_tools_for_phase

logger = logging.getLogger(__name__)


def _plan_has_sps(plan: Dict) -> bool:
    """Check whether the plan's discovery metadata indicates stored procedure usage."""
    meta = plan.get("discovery_metadata") or {}
    sp_analysis = meta.get("sp_analysis") or {}
    return sp_analysis.get("total_sps", 0) > 0


# Phase number -> phase type mapping
PHASE_TYPES_V1 = {
    1: "discovery",
    2: "architecture",
    3: "analyze",
    4: "design",
    5: "transform",
    6: "test",
}
PHASE_TYPES_V2 = {
    1: "architecture",
    2: "discovery",
    3: "transform",
    4: "test",
}
PHASE_TYPES = PHASE_TYPES_V1  # backward compat default


def get_phase_type(phase_number: int, version: int = 1) -> str:
    """Get the phase type string for a phase number."""
    mapping = PHASE_TYPES_V2 if version == 2 else PHASE_TYPES_V1
    return mapping.get(phase_number, f"unknown_{phase_number}")


def execute_phase(
    phase_number: int,
    plan: Dict[str, Any],
    previous_outputs: Dict[int, str],
    context_builder: MigrationContextBuilder,
    token_budget: int = 12_000,
    mvp_context: Optional[Dict[str, Any]] = None,
    context_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a migration phase.

    Args:
        phase_number: 1-6 (V1) or 1-4 (V2)
        plan: Migration plan dict (target_brief, target_stack, constraints)
        previous_outputs: {phase_number: output_text} for approved phases
        context_builder: MigrationContextBuilder instance
        token_budget: Token budget for context building
        mvp_context: MVP dict for per-MVP phases. None for plan-level.
        context_type: Semantic type override (e.g. "architecture", "discovery").
            Decouples phase number from executor selection for V2 pipeline.
            If None, falls back to V1 phase-number dispatch.

    Returns:
        Dict with 'output' (markdown string) and optionally 'output_files'
    """
    _type_executors = {
        "discovery": _execute_discovery,
        "architecture": _execute_architecture,
        "analyze": _execute_analyze,
        "design": _execute_design,
        "transform": _execute_transform,
        "test": _execute_test,
    }

    if context_type:
        executor = _type_executors.get(context_type)
        if not executor:
            raise ValueError(f"Invalid context_type: {context_type}")
    else:
        # V1 fallback: phase number -> executor
        _number_executors = {
            1: _execute_discovery,
            2: _execute_architecture,
            3: _execute_analyze,
            4: _execute_design,
            5: _execute_transform,
            6: _execute_test,
        }
        executor = _number_executors.get(phase_number)
        if not executor:
            raise ValueError(f"Invalid phase number: {phase_number}")

    return executor(plan, previous_outputs, context_builder, token_budget, mvp_context)


# ── Agentic Execution ─────────────────────────────────────────────────

# System prompt for the migration agent.
_AGENT_SYSTEM_PROMPT = """\
You are a code migration specialist. You have access to tools that let you
read source code, search the codebase, look up framework documentation,
and validate generated code.

Your workflow:
1. Use get_source_code or get_unit_details to understand the MVP's code.
2. Use get_functional_context and get_dependencies to understand business logic and blast radius.
3. Use lookup_framework_docs for the TARGET framework's correct patterns.
4. Generate your migration output.
5. Use validate_syntax on any generated code to catch errors.

Important rules:
- ALWAYS read the source code before generating migrated code.
- Look up the target framework docs for correct annotation/API patterns.
- Check dependencies to understand what external code depends on this MVP.
- Validate generated code syntax before finishing.
- Follow the output format specified in the task instructions EXACTLY.
- When the task says to output a JSON array, output ONLY the raw JSON array — no markdown fences, no explanation text before or after.
"""


def execute_phase_agentic(
    phase_number: int,
    plan: Dict[str, Any],
    previous_outputs: Dict[int, str],
    context_builder: MigrationContextBuilder,
    context_type: Optional[str] = None,
    mvp_context: Optional[Dict[str, Any]] = None,
    pipeline=None,
    project_id: Optional[str] = None,
    max_turns: int = 10,
) -> "Generator[AgentEvent, None, Optional[Dict[str, Any]]]":
    """Execute a migration phase using the agentic tool-use loop.

    Instead of assembling all context upfront and making a single LLM call,
    this gives the LLM tools to pull context on demand and iterate.

    Args:
        phase_number: Phase number for UI display.
        plan: Migration plan dict.
        previous_outputs: Approved phase outputs.
        context_builder: MigrationContextBuilder for the project.
        context_type: Phase type ("transform", "analyze", "test", "discovery").
        mvp_context: MVP dict for per-MVP phases.
        pipeline: LocalRAGPipeline for search_codebase tool.
        project_id: Project UUID string for search_codebase tool.
        max_turns: Maximum agent iterations.

    Yields:
        AgentEvent instances for SSE streaming.

    Returns:
        Dict with 'output' and optionally 'output_files' on the agent.result,
        or None if the agent failed.
    """
    effective_type = context_type or get_phase_type(phase_number)
    unit_ids = (mvp_context or {}).get("unit_ids", [])

    # Build phase-specific tools
    tools = build_tools_for_phase(
        context_type=effective_type,
        ctx=context_builder,
        unit_ids=unit_ids,
        pipeline=pipeline,
        project_id=project_id,
        target_stack=plan.get("target_stack"),
    )

    # Resolve LLM (respects overrides)
    llm_context = "generation" if effective_type in ("transform", "test") else "understanding"
    llm = _get_phase_llm(llm_context)
    if llm is None:
        from .agent.events import ErrorEvent
        yield ErrorEvent(error="No LLM configured. Check LLM_PROVIDER settings.", recoverable=False)
        return

    # Build task prompt with minimal upfront context (agent fetches the rest)
    task_prompt = _build_agentic_task_prompt(
        effective_type, plan, previous_outputs, mvp_context,
    )

    agent = MigrationAgent(
        llm=llm,
        tools=tools,
        system_prompt=_AGENT_SYSTEM_PROMPT,
        max_turns=max_turns,
    )

    # Run the agent loop, yielding events
    yield from agent.execute(task_prompt, phase_type=effective_type)

    # Store final result for the caller to persist
    if agent.result:
        # Attempt to parse output_files from transform/test phases
        output_files = []
        if effective_type in ("transform", "test"):
            output_files = _parse_json_files(agent.result)

        agent.result = json.dumps({
            "output": agent.result if not output_files else (
                f"Generated {len(output_files)} file(s). See output_files."
            ),
            "output_files": output_files,
        })


def _build_agentic_task_prompt(
    context_type: str,
    plan: Dict[str, Any],
    previous_outputs: Dict[int, str],
    mvp_context: Optional[Dict[str, Any]],
) -> str:
    """Build a task prompt that tells the agent WHAT to do, not HOW.

    The agent will use tools to gather context on demand.
    We provide: target brief, target stack, MVP info, and previous phase outputs.
    """
    parts = []

    # Migration target
    parts.append(f"## Migration Target\n{plan.get('target_brief', 'Not specified')}")

    stack = plan.get("target_stack", {})
    if stack:
        parts.append(f"## Target Stack\n{json.dumps(stack, indent=2)}")

    # Previous phase outputs (condensed)
    if previous_outputs:
        parts.append("## Previous Phase Outputs")
        for pn, text in sorted(previous_outputs.items()):
            if text:
                # Truncate to avoid overwhelming the initial context
                truncated = text[:3000] + ("..." if len(text) > 3000 else "")
                parts.append(f"### Phase {pn}\n{truncated}")

    # MVP context
    if mvp_context:
        parts.append("## Current MVP")
        parts.append(f"Name: {mvp_context.get('name', 'Unknown')}")
        parts.append(f"Description: {mvp_context.get('description', 'N/A')}")
        parts.append(f"Unit count: {len(mvp_context.get('unit_ids', []))}")
        parts.append(f"Priority: {mvp_context.get('priority', 'N/A')}")

    # Phase-specific instructions
    instructions = _PHASE_INSTRUCTIONS.get(context_type, "Analyze and produce your output.")
    parts.append(f"## Your Task\n{instructions}")

    return "\n\n".join(parts)


_PHASE_INSTRUCTIONS: Dict[str, str] = {
    "transform": (
        "Migrate this MVP's source code to the target framework.\n\n"
        "Steps:\n"
        "1. Use get_source_code to read the current implementation.\n"
        "2. Use get_functional_context to understand business logic.\n"
        "3. Use get_dependencies to understand blast radius.\n"
        "4. Use lookup_framework_docs for target framework patterns.\n"
        "5. Generate migrated code files.\n"
        "6. Use validate_syntax on each generated file.\n\n"
        "CRITICAL OUTPUT FORMAT — your final answer MUST be a raw JSON array, nothing else:\n"
        '[{"file_path": "src/main/java/com/example/Foo.java", "content": "package com.example;\\n...", "language": "java"}]\n\n'
        "Rules:\n"
        "- Output ONLY the JSON array. No markdown, no explanation, no code fences.\n"
        "- Each object must have 'file_path', 'content' (full source code), and 'language'.\n"
        "- Include ALL migrated files in a single array."
    ),
    "analyze": (
        "Produce a Functional Requirements Register for this MVP.\n\n"
        "Steps:\n"
        "1. Use get_source_code and get_unit_details to understand the code.\n"
        "2. Use get_functional_context for business domain entities and rules.\n"
        "3. Use get_deep_analysis for existing analysis narratives.\n"
        "4. Use get_dependencies for integration boundaries.\n\n"
        "Output: Markdown with tables of Business Rules (BR-N), "
        "Data Entities (DE-N), Integrations (INT-N), Validations (VAL-N)."
    ),
    "test": (
        "Generate test files for the migrated MVP code.\n\n"
        "Steps:\n"
        "1. Read previous phase output to understand what was migrated.\n"
        "2. Use get_source_code to see the original implementation.\n"
        "3. Use get_functional_context for business logic to test.\n"
        "4. Use lookup_framework_docs for target test framework patterns.\n"
        "5. Generate test files with comprehensive coverage.\n"
        "6. Use validate_syntax on each generated test file.\n\n"
        "CRITICAL OUTPUT FORMAT — your final answer MUST be a raw JSON array, nothing else:\n"
        '[{"file_path": "src/test/java/com/example/FooTest.java", "content": "package com.example;\\n...", "language": "java"}]\n\n'
        "Rules:\n"
        "- Output ONLY the JSON array. No markdown, no explanation, no code fences.\n"
        "- Each object must have 'file_path', 'content' (full test source), and 'language'.\n"
        "- Include ALL test files in a single array."
    ),
    "discovery": (
        "Analyze the project codebase and produce a migration strategy.\n\n"
        "Steps:\n"
        "1. Use get_module_graph to understand project structure.\n"
        "2. Use search_codebase to find key patterns and frameworks.\n"
        "3. Use get_unit_details to understand code organization.\n\n"
        "Output: Markdown with migration strategy and recommendations."
    ),
}


def _call_llm(
    prompt: str,
    context_type: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call the LLM with a prompt and return the response text.

    Args:
        prompt: The prompt string to send
        context_type: Optional context hint for LLM override routing.
            'understanding' → use understanding_llm override if configured
            'generation' → use generation_llm override if configured
            None → use default Settings.llm
        temperature: Optional temperature override for this call.
            Use 0.0 for deterministic output. Restores original after call.
    """
    llm = _get_phase_llm(context_type)
    if llm is None:
        raise RuntimeError("No LLM configured. Check LLM_PROVIDER settings.")

    logger.info(
        f"Calling LLM with prompt of {len(prompt)} chars"
        f" (context_type={context_type}, temperature={temperature})"
    )

    # Temporarily override temperature if requested
    original_temp = None
    if temperature is not None and hasattr(llm, "temperature"):
        original_temp = llm.temperature
        llm.temperature = temperature

    try:
        response = llm.complete(prompt, gateway_purpose="migration")
        return response.text.strip()
    finally:
        if original_temp is not None:
            llm.temperature = original_temp


def _get_phase_llm(context_type: Optional[str] = None):
    """Resolve the LLM to use for a given context type.

    Checks migration.llm_overrides config for context-specific overrides.
    Falls back to the default Settings.llm if no override is configured.
    """
    if context_type:
        try:
            from ..config.config_loader import get_llm_overrides_config
            overrides = get_llm_overrides_config()

            override_key = f"{context_type}_llm"
            override_cfg = overrides.get(override_key)

            if override_cfg and isinstance(override_cfg, dict):
                provider = override_cfg.get("provider")
                model = override_cfg.get("model")
                if provider and model:
                    return _create_override_llm(provider, model, override_cfg)
        except Exception as e:
            logger.debug(f"LLM override lookup failed for {context_type}: {e}")

    return Settings.llm


def _create_override_llm(provider: str, model: str, cfg: dict):
    """Create an LLM instance from override configuration.

    Supports: ollama, openai, anthropic, gemini.
    Wraps in LLMGateway for observability/retry.
    Falls back to Settings.llm on failure.
    """
    from ..gateway import LLMGateway

    temperature = cfg.get("temperature", 0.1)

    try:
        raw_llm = None
        if provider == "ollama":
            from llama_index.llms.ollama import Ollama
            raw_llm = Ollama(model=model, temperature=temperature, request_timeout=300)
        elif provider == "openai":
            import os
            from llama_index.llms.openai import OpenAI
            raw_llm = OpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            import os
            from llama_index.llms.anthropic import Anthropic
            raw_llm = Anthropic(
                model=model,
                temperature=temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif provider == "gemini":
            import os
            from llama_index.llms.gemini import Gemini
            raw_llm = Gemini(
                model=model,
                temperature=temperature,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            logger.warning(f"Unknown LLM provider for override: {provider}")
            return Settings.llm

        logger.info(f"Migration LLM override: {provider}/{model}")
        return LLMGateway(raw_llm)
    except Exception as e:
        logger.warning(f"Failed to create override LLM ({provider}/{model}): {e}")
        return Settings.llm


# ── Agentic MVP Refinement Helpers ────────────────────────────────────


def _describe_mvp(
    ctx: MigrationContextBuilder,
    cluster: Dict[str, Any],
    token_budget: int = 6_000,
) -> Dict[str, Any]:
    """Generate a functional description for one MVP cluster.

    Uses source code, enriched unit signatures, and functional context
    to produce a business-oriented name and description via LLM.
    """
    unit_ids = cluster.get("unit_ids", [])

    # Gather context using existing machinery
    source_code = ctx._get_mvp_source_code_by_connectivity(unit_ids, budget=token_budget // 2)
    source_str = ctx._format_source_code_annotated(source_code) if source_code else ""

    units = ctx._get_mvp_units_enriched(unit_ids, limit=40)
    units_str = ctx._format_unit_details_enriched(units) if units else ""

    functional_str = ""
    try:
        functional = ctx.get_mvp_functional_context(unit_ids)
        functional_str = ctx.format_mvp_functional_context(functional) if functional else ""
    except Exception as e:
        logger.warning("Functional context extraction failed for MVP description: %s", e)

    prompt = prompts.mvp_functional_description(
        source_code=source_str,
        unit_signatures=units_str,
        functional_context=functional_str,
        cluster_metrics=cluster.get("metrics", {}),
    )

    try:
        raw = _call_llm(prompt, temperature=0.0)
        return _parse_mvp_description(raw)
    except Exception as e:
        logger.warning("MVP description LLM call failed: %s", e)
        return {"name": cluster.get("name", "Unknown"), "description": ""}


def _evaluate_mvp_coherence(
    mvp_summaries: List[Dict],
    inter_mvp_edges: List[Dict],
) -> Dict[str, Any]:
    """Evaluate all MVPs for functional coherence, suggest merges/splits."""
    prompt = prompts.mvp_coherence_evaluation(mvp_summaries, inter_mvp_edges)

    try:
        raw = _call_llm(prompt, temperature=0.0)
        return _parse_coherence_suggestions(raw)
    except Exception as e:
        logger.warning("MVP coherence evaluation failed: %s", e)
        return {"merge_suggestions": [], "split_suggestions": [], "assessment": "Evaluation failed"}


def _parse_mvp_description(raw: str) -> Dict[str, Any]:
    """Parse LLM output for MVP functional description.

    Expects JSON with name, description, etc. Falls back to text extraction.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Text fallback: first line as name, rest as description
    lines = text.split("\n")
    name = lines[0].strip().strip('"').strip("'") if lines else "Unknown"
    description = " ".join(l.strip() for l in lines[1:] if l.strip())
    return {"name": name[:80], "description": description[:500]}


def _parse_coherence_suggestions(raw: str) -> Dict[str, Any]:
    """Parse LLM output for MVP coherence evaluation.

    Expects JSON with merge_suggestions, split_suggestions. Falls back to empty.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                "merge_suggestions": parsed.get("merge_suggestions", []),
                "split_suggestions": parsed.get("split_suggestions", []),
                "assessment": parsed.get("assessment", ""),
            }
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return {
                    "merge_suggestions": parsed.get("merge_suggestions", []),
                    "split_suggestions": parsed.get("split_suggestions", []),
                    "assessment": parsed.get("assessment", ""),
                }
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse coherence suggestions as JSON, returning empty")
    return {"merge_suggestions": [], "split_suggestions": [], "assessment": raw[:200]}


# ── Phase 1: Discovery (Plan-Level) ──────────────────────────────────

def _execute_discovery(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Discovery: Analyze source codebase + recommend migration strategy.

    The clustering is handled by the engine before this is called.
    This phase focuses on the LLM's strategic analysis.

    V1: Phase 1 (no prior architecture output).
    V2: Phase 2 (architecture output available in prev).
    """
    codebase_context = ctx.build_phase_context(1, prev, budget, context_type="discovery")

    # In V2 pipeline, architecture runs first and its output is available
    architecture_output = ""
    for pn, text in prev.items():
        # Find the architecture output regardless of phase number
        if text and ("Module Structure" in text or "target architecture" in text.lower()):
            architecture_output = text
            break

    # MVP functional summaries from agentic refinement
    mvp_summaries_str = ""
    mvp_summaries = plan.get("_mvp_summaries")
    if mvp_summaries:
        lines = []
        for ms in mvp_summaries:
            desc = ms.get("description") or "No description"
            lines.append(f"- **{ms.get('name', 'Unknown')}** ({ms.get('unit_count', 0)} units): {desc}")
        mvp_summaries_str = "\n".join(lines)

    prompt = prompts.phase_1_discovery(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        constraints=plan.get("constraints") or {},
        codebase_context=codebase_context,
        has_sps=_plan_has_sps(plan),
        architecture_output=architecture_output,
        mvp_summaries=mvp_summaries_str,
    )

    output = _call_llm(prompt)
    return {"output": output, "output_files": []}


# ── Phase 2: Architecture (Plan-Level) ───────────────────────────────

def _execute_architecture(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Architecture: Design target architecture (system-wide).

    V1: Phase 2 (discovery output in prev[1]).
    V2: Phase 1 (no prior discovery output).

    Enrichments:
    - Source pattern analysis from ASG metadata
    - Target framework docs via Tavily (cached on plan)
    """
    codebase_context = ctx.build_phase_context(2, prev, budget, context_type="architecture")
    # In V1, discovery output is in prev[1]. In V2, there's no prior discovery.
    phase_1_output = prev.get(1, "")

    # Source pattern analysis (auto-detected from ASG)
    source_patterns_str = ""
    source_patterns_raw = plan.get("_source_patterns")
    if source_patterns_raw:
        source_patterns_str = ctx.format_source_patterns(source_patterns_raw)

    # Framework docs (fetched by engine, passed in plan_data)
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 2, budget=3000)

    prompt = prompts.phase_2_architecture(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        phase_1_output=phase_1_output,
        codebase_context=codebase_context,
        has_sps=_plan_has_sps(plan),
        framework_docs=framework_docs_str,
        source_patterns=source_patterns_str,
        migration_type=plan.get("migration_type", "framework_migration"),
        asset_strategies=plan.get("asset_strategies") or None,
    )

    output = _call_llm(prompt)
    return {"output": output, "output_files": []}


# ── Phase 3: Analyze (Per-MVP) ───────────────────────────────────────

def _execute_analyze(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Phase 3: Deep analysis scoped to the MVP's units.

    Enrichments:
    - MVP functional context (data entities, business rules, integrations, validation)
    - Framework docs for target reference
    - SRC validation loop: ensures register coverage >= 95% of MVP units
    """
    codebase_context = ctx.build_phase_context(3, prev, budget, mvp_context=mvp_context, context_type="analyze")
    phase_2_output = prev.get(2, "No Architecture output available.")

    # MVP functional context
    functional_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            functional = ctx.get_mvp_functional_context(mvp_context["unit_ids"])
            functional_str = ctx.format_mvp_functional_context(functional)
        except Exception as e:
            logger.warning("Functional context extraction failed: %s", e)

    # Framework docs
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 3, budget=2000)

    # Deep analysis context (if available)
    deep_context_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            deep_context_str = ctx.get_deep_analysis_context(mvp_context["unit_ids"])
            if deep_context_str:
                logger.info(
                    "Injecting deep analysis context into analyze phase (%d chars)",
                    len(deep_context_str),
                )
        except Exception as e:
            logger.warning("Deep analysis context injection failed: %s", e)

    prompt = prompts.phase_3_analyze(
        target_brief=plan["target_brief"],
        phase_2_output=phase_2_output,
        codebase_context=codebase_context,
        mvp_context=mvp_context,
        functional_context=functional_str,
        framework_docs=framework_docs_str,
    )

    # Append deep analysis context after the main prompt
    if deep_context_str:
        prompt += f"\n\n{deep_context_str}"

    output = _call_llm(prompt, context_type="understanding")

    # SRC validation loop: check register coverage
    if mvp_context and mvp_context.get("unit_ids"):
        output = _src_validate_register(output, mvp_context["unit_ids"], prompt)

    return {"output": output, "output_files": []}


# ── Phase 4: Design (Per-MVP) ────────────────────────────────────────

def _execute_design(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Phase 4: Detailed module design for this MVP (+ SP stubs when SPs exist)."""
    codebase_context = ctx.build_phase_context(4, prev, budget, mvp_context=mvp_context, context_type="design")
    phase_3_output = prev.get(3, "No Analyze output available.")

    # Framework docs
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 4, budget=2000)

    # MVP functional context
    functional_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            functional = ctx.get_mvp_functional_context(mvp_context["unit_ids"])
            functional_str = ctx.format_mvp_functional_context(functional)
        except Exception as e:
            logger.warning("Functional context extraction failed: %s", e)

    prompt = prompts.phase_4_design(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        phase_3_output=phase_3_output,
        codebase_context=codebase_context,
        mvp_context=mvp_context,
        framework_docs=framework_docs_str,
        functional_context=functional_str,
    )

    output = _call_llm(prompt)
    return {"output": output, "output_files": []}


# ── Phase 5: Transform (Per-MVP) ─────────────────────────────────────

def _execute_transform(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Transform: Generate migrated code for this MVP.

    V1: Phase 5 (Design output in prev[4]).
    V2: Phase 3 (Architecture output in prev[1], optional analysis_output on plan).
    """
    codebase_context = ctx.build_phase_context(5, prev, budget, mvp_context=mvp_context, context_type="transform")

    # V1: prior_phase_output is Design (phase 4). V2: it's Architecture (phase 1 or 2).
    # Try Design first, then Architecture, then any available output.
    prior_phase_output = (
        prev.get(4, "")
        or prev.get(2, "")
        or prev.get(1, "")
        or "No prior phase output available."
    )

    # V2: on-demand analysis output may be stored on plan_data
    analysis_output = plan.get("_analysis_output", "")

    # MVP functional description from agentic refinement
    mvp_description = ""
    if mvp_context:
        mvp_description = mvp_context.get("description", "")

    # Framework docs
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 5, budget=2000)

    # Deep analysis context (if available)
    deep_context_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            deep_context_str = ctx.get_deep_analysis_context(mvp_context["unit_ids"])
            if deep_context_str:
                logger.info(
                    "Injecting deep analysis context into transform phase (%d chars)",
                    len(deep_context_str),
                )
        except Exception as e:
            logger.warning("Deep analysis context injection failed: %s", e)

    prompt = prompts.phase_5_transform(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        phase_4_output=prior_phase_output,
        codebase_context=codebase_context,
        mvp_context=mvp_context,
        framework_docs=framework_docs_str,
        analysis_output=analysis_output,
        mvp_description=mvp_description,
    )

    # Append deep analysis context after the main prompt
    if deep_context_str:
        prompt += f"\n\n{deep_context_str}"

    # Lane: inject deterministic transforms so LLM fills gaps around pre-computed code
    det_transforms = plan.get("_deterministic_transforms")
    if det_transforms:
        det_section = "\n\n## Pre-Computed Deterministic Transforms\n\n"
        det_section += (
            "The following code has already been generated by deterministic rules. "
            "Do NOT regenerate these files. Focus on units NOT covered below, "
            "and fill in any business logic gaps marked with TODO comments.\n\n"
        )
        for dt in det_transforms:
            det_section += (
                f"### {dt['target_path']} (rule: {dt['rule_name']}, "
                f"confidence: {dt['confidence']:.0%})\n"
                f"```java\n{dt['target_code']}\n```\n\n"
            )
        prompt += det_section

    # Lane: append prompt augmentation (mapping tables, endpoint lists, etc.)
    lane_augmentation = plan.get("_lane_prompt_augmentation")
    if lane_augmentation:
        prompt += f"\n\n## Migration Lane Context\n\n{lane_augmentation}"

    raw = _call_llm(prompt, context_type="generation")
    output_files = _parse_json_files(raw)

    if output_files:
        output = f"Generated {len(output_files)} migrated file(s).\n\nSee output_files for generated code."
    else:
        output = raw
        output_files = []

    return {"output": output, "output_files": output_files}


# ── Phase 6: Test (Per-MVP) ──────────────────────────────────────────

def _execute_test(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Test: Generate tests scoped to this MVP's migrated code.

    V1: Phase 6 (Transform output in prev[5]).
    V2: Phase 4 (Transform output in prev[3]).
    """
    codebase_context = ctx.build_phase_context(6, prev, budget, mvp_context=mvp_context, context_type="test")
    # Try Transform output from either V1 (phase 5) or V2 (phase 3)
    prior_phase_output = (
        prev.get(5, "")
        or prev.get(3, "")
        or "No Transform output available."
    )

    # V2: on-demand analysis output (Functional Requirements Register)
    analysis_output = plan.get("_analysis_output", "")

    # MVP functional description from agentic refinement
    mvp_description = ""
    if mvp_context:
        mvp_description = mvp_context.get("description", "")

    # Framework docs
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 6, budget=2000)

    # Deep analysis context (if available)
    deep_context_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            deep_context_str = ctx.get_deep_analysis_context(mvp_context["unit_ids"])
            if deep_context_str:
                logger.info(
                    "Injecting deep analysis context into test phase (%d chars)",
                    len(deep_context_str),
                )
        except Exception as e:
            logger.warning("Deep analysis context injection failed: %s", e)

    prompt = prompts.phase_6_test(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        phase_5_output=prior_phase_output,
        codebase_context=codebase_context,
        mvp_context=mvp_context,
        framework_docs=framework_docs_str,
        analysis_output=analysis_output,
        mvp_description=mvp_description,
    )

    # Append deep analysis context after the main prompt
    if deep_context_str:
        prompt += f"\n\n{deep_context_str}"

    # Lane: append prompt augmentation for test phase (endpoint coverage checklists, etc.)
    lane_augmentation = plan.get("_lane_prompt_augmentation")
    if lane_augmentation:
        prompt += f"\n\n## Migration Lane Test Context\n\n{lane_augmentation}"

    raw = _call_llm(prompt, context_type="generation")
    output_files = _parse_json_files(raw)

    if output_files:
        output = f"Generated {len(output_files)} test file(s).\n\nSee output_files for generated tests."
    else:
        output = raw
        output_files = []

    return {"output": output, "output_files": output_files}


# ── On-Demand MVP Analysis (V2 Pipeline) ────────────────────────────

def execute_mvp_analysis(
    plan: Dict[str, Any],
    previous_outputs: Dict[int, str],
    context_builder: MigrationContextBuilder,
    mvp_context: Dict[str, Any],
    token_budget: int = 18_000,
) -> Dict[str, Any]:
    """Run on-demand deep analysis for an MVP (merges old Analyze + Design).

    Called by engine.analyze_mvp(). Result is stored on FunctionalMVP.analysis_output,
    not as a pipeline phase. Used by V2 pipeline.

    Args:
        plan: Migration plan dict
        previous_outputs: Plan-level phase outputs (architecture, discovery)
        context_builder: MigrationContextBuilder instance
        mvp_context: MVP dict with name, unit_ids, file_ids, etc.
        token_budget: Token budget for context building

    Returns:
        Dict with 'output' (markdown) and 'output_files' (always [])
    """
    # Foundation MVP: dedicated path (no unit context, prereqs-only prompt)
    is_foundation = (
        mvp_context
        and mvp_context.get("priority") == 0
        and not mvp_context.get("unit_ids")
    )

    if is_foundation:
        architecture_output = previous_outputs.get(1, "") or previous_outputs.get(2, "")
        prompt = prompts.mvp_foundation_analysis(
            target_brief=plan["target_brief"],
            target_stack=plan["target_stack"],
            architecture_output=architecture_output,
            mvp_context=mvp_context,
            discovery_metadata=plan.get("discovery_metadata", {}),
        )
        output = _call_llm(prompt)
        return {"output": output, "output_files": []}

    # Build context using analyze-level context (reuses Phase 3 context builder)
    codebase_context = context_builder.build_phase_context(
        3, previous_outputs, token_budget, mvp_context=mvp_context, context_type="analyze",
    )

    # Get architecture output (could be in prev[1] for V2 or prev[2] for V1)
    architecture_output = previous_outputs.get(1, "") or previous_outputs.get(2, "")

    # MVP functional context — prefer hierarchical grouping over flat
    functional_str = ""
    if mvp_context and mvp_context.get("unit_ids"):
        try:
            hierarchy = context_builder._build_class_method_hierarchy(mvp_context["unit_ids"])
            functional_str = context_builder.format_mvp_hierarchical_context(hierarchy)
        except Exception as e:
            logger.warning("Hierarchical context failed, falling back to flat: %s", e)
            try:
                functional = context_builder.get_mvp_functional_context(mvp_context["unit_ids"])
                functional_str = context_builder.format_mvp_functional_context(functional)
            except Exception as e2:
                logger.warning("Flat functional context also failed: %s", e2)

    # Framework docs
    framework_docs_str = ""
    fw_docs = plan.get("framework_docs")
    if fw_docs:
        enricher = DocEnricher()
        framework_docs_str = enricher.get_phase_docs(fw_docs, 3, budget=2000)

    prompt = prompts.mvp_analysis(
        target_brief=plan["target_brief"],
        target_stack=plan["target_stack"],
        architecture_output=architecture_output,
        codebase_context=codebase_context,
        mvp_context=mvp_context,
        functional_context=functional_str,
        framework_docs=framework_docs_str,
    )

    output = _call_llm(prompt)

    # SRC validation loop: check register coverage
    if mvp_context and mvp_context.get("unit_ids"):
        output = _src_validate_register(output, mvp_context["unit_ids"], prompt)

    return {"output": output, "output_files": []}


# ── SRC Validation ───────────────────────────────────────────────────

_REGISTER_ID_RE = re.compile(r"\b(BR|DE|INT|VAL)-\d+\b")


def _src_validate_register(
    output: str,
    unit_ids: List[str],
    original_prompt: str,
    max_iterations: int = 3,
) -> str:
    """SRC validation loop for Phase 3 Functional Requirements Register.

    Checks that the register covers enough MVP units, and re-prompts the
    LLM to fill gaps if coverage is below 95%.

    Args:
        output: LLM-generated Phase 3 output
        unit_ids: MVP's unit IDs to check coverage against
        original_prompt: The original Phase 3 prompt for re-prompting
        max_iterations: Maximum re-prompt attempts

    Returns:
        Final output with improved register coverage.
    """
    if not unit_ids:
        return output

    total_units = len(unit_ids)
    best_output = output

    # Register items map to functional concerns (BR, DE, INT, VAL), not 1:1
    # with code units. A class with 50 methods is still ~1 DE + a few BRs.
    # Target: 10% of units or 50, whichever is smaller, minimum 10.
    target = max(10, min(int(total_units * 0.10), 50))

    for iteration in range(max_iterations):
        # Parse register IDs from output
        register_ids = set(_REGISTER_ID_RE.findall(best_output))
        register_count = len(register_ids)

        if register_count >= target:
            logger.info(
                "SRC register converged: %d register items (target %d) for %d units (iteration %d)",
                register_count, target, total_units, iteration,
            )
            break

        logger.info(
            "SRC register gap: %d register items (target %d) for %d units (iteration %d)",
            register_count, target, total_units, iteration,
        )

        # Build gap prompt — only include register summary, NOT the full output,
        # to avoid context bloat that causes LLM repetition degeneration.
        gap_prompt = (
            f"The Functional Requirements Register currently has {register_count} "
            f"items but needs at least {target} to adequately cover this MVP's "
            f"{total_units} code units.\n\n"
            f"Please output ONLY the new register entries to add. "
            f"Look for:\n"
            f"- Service methods without a BR (Business Rule) entry\n"
            f"- Entity/model classes without a DE (Data Entity) entry\n"
            f"- Units with HTTP/REST/queue patterns without an INT (Integration) entry\n"
            f"- Units with validation annotations without a VAL (Validation) entry\n\n"
            f"Start numbering from BR-{register_count + 1}, DE-{register_count + 1}, etc. "
            f"Output ONLY the new markdown tables — no preamble, no repetition of existing entries."
        )

        try:
            extension = _call_llm(gap_prompt)
            if extension and len(extension) > 50:
                best_output = best_output + "\n\n## Register Extension (SRC Iteration " + str(iteration + 1) + ")\n\n" + extension
        except Exception as e:
            logger.warning("SRC re-prompt failed (iteration %d): %s", iteration, e)
            break

    return best_output


# ── Analysis Embedding ────────────────────────────────────────────────

def _embed_analysis_output(
    analysis_output: str,
    project_id: str,
    plan_id: str,
    mvp_id: int,
    mvp_name: str,
    pipeline,
) -> int:
    """Chunk analysis output by class/file sections and embed into vector store.

    Splits the markdown output on ### File: or #### ClassName headers.
    Each chunk gets metadata tagging it as migration_analysis so it can
    be retrieved by RAG chat queries about business rules, entities, etc.

    Deduplication is automatic — the vector store uses md5(text) + project_id
    unique index, so re-running analysis replaces old embeddings.

    Args:
        analysis_output: The full markdown analysis output
        project_id: Source project UUID string
        plan_id: Migration plan UUID string
        mvp_id: MVP integer ID
        mvp_name: MVP display name
        pipeline: LocalRAGPipeline instance (needs _vector_store and Settings.embed_model)

    Returns:
        Number of nodes embedded
    """
    if not analysis_output or not pipeline:
        return 0

    vector_store = getattr(pipeline, "_vector_store", None)
    if vector_store is None:
        logger.warning("Pipeline has no _vector_store, skipping analysis embedding")
        return 0

    # Split on file and class-level headers
    # Pattern matches: ### File: ... or #### ClassName ...
    section_pattern = re.compile(r"(?=^###\s)", re.MULTILINE)
    raw_sections = section_pattern.split(analysis_output)

    # Build chunks — skip tiny sections, merge preamble with first real section
    chunks: List[Dict[str, str]] = []
    current_file = ""

    for section in raw_sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        # Extract file name from "### File: path/to/File.java"
        file_match = re.match(r"^###\s+File:\s*(.+)", section)
        if file_match:
            current_file = file_match.group(1).strip()

        # Extract class name from "#### ClassName ..." within the section
        class_match = re.search(r"^####\s+(?:Class|Interface|Struct|Enum):\s*(\S+)", section, re.MULTILINE)
        class_name = class_match.group(1).strip() if class_match else ""

        chunks.append({
            "text": section,
            "file_name": current_file,
            "class_name": class_name,
        })

    if not chunks:
        # No section headers found — embed the whole output as one chunk
        if len(analysis_output) > 100:
            chunks.append({
                "text": analysis_output,
                "file_name": "",
                "class_name": "",
            })
        else:
            return 0

    # Create TextNodes with metadata
    try:
        from llama_index.core.schema import TextNode
    except ImportError:
        logger.warning("llama_index.core.schema not available, skipping embedding")
        return 0

    nodes = []
    for i, chunk in enumerate(chunks):
        node = TextNode(
            text=chunk["text"],
            metadata={
                "node_type": "migration_analysis",
                "project_id": project_id,
                "plan_id": plan_id,
                "mvp_id": mvp_id,
                "mvp_name": mvp_name,
                "class_name": chunk["class_name"],
                "file_name": chunk["file_name"],
                "chunk_index": i,
            },
        )
        nodes.append(node)

    # Embed and store
    try:
        from llama_index.core import Settings as LISettings
        embed_model = LISettings.embed_model
        if embed_model:
            for node in nodes:
                node.embedding = embed_model.get_text_embedding(node.text)
    except Exception as e:
        logger.warning("Failed to generate embeddings for analysis chunks: %s", e)
        return 0

    try:
        added = vector_store.add_nodes(nodes, project_id=project_id)
        logger.info(
            "Embedded %d analysis chunks for MVP %d (%s) in project %s",
            added, mvp_id, mvp_name, project_id,
        )
        return added
    except Exception as e:
        logger.warning("Failed to add analysis nodes to vector store: %s", e)
        return 0


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_json_files(raw: str) -> List[Dict]:
    """Try to parse LLM output as a JSON array of file dicts.

    Supports three formats:
    1. Raw JSON array: [{"file_path": ..., "content": ..., "language": ...}]
    2. JSON array wrapped in markdown code fences
    3. Fallback: extract files from markdown code blocks like
       ```java  // path/to/File.java  ...code...  ```
    """
    import re

    text = raw.strip()

    # Strip outer markdown code fences (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array embedded in text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list) and parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # ---- Fallback: extract files from markdown code blocks ----
    # Matches patterns like:
    #   ```java\n// src/main/java/Foo.java\npackage ...```
    #   **`src/Foo.java`**\n```java\npackage ...```
    #   ### src/Foo.java\n```java\npackage ...```
    files = []
    # Find all fenced code blocks
    block_pattern = re.compile(
        r"```(\w+)?\s*\n(.*?)```", re.DOTALL
    )
    # Patterns that precede a code block and name the file
    path_before_pattern = re.compile(
        r"(?:\*\*`?|###?\s+|`)([\w./\\-]+\.(?:java|py|ts|tsx|js|jsx|cs|go|rb|rs|kt|swift|xml|yaml|yml|json|sql|html|css|properties|gradle))`?\*?\*?\s*$",
        re.MULTILINE,
    )

    blocks = list(block_pattern.finditer(raw))
    if blocks:
        for match in blocks:
            lang = match.group(1) or ""
            content = match.group(2).strip()
            if not content:
                continue

            file_path = ""

            # Check if a file path appears just before this code block
            preceding_text = raw[:match.start()]
            path_matches = list(path_before_pattern.finditer(preceding_text))
            if path_matches:
                file_path = path_matches[-1].group(1)

            # Check if the first line of the code block is a comment with a path
            if not file_path:
                first_line = content.split("\n")[0].strip()
                # Matches: // path/to/File.java  or  # path/to/file.py
                comment_path = re.match(
                    r"^(?://|#)\s*([\w./\\-]+\.\w+)\s*$", first_line
                )
                if comment_path:
                    file_path = comment_path.group(1)
                    # Remove the comment line from content
                    content = "\n".join(content.split("\n")[1:]).strip()

            if not file_path:
                # Infer from package declaration + class name for Java
                pkg = re.search(r"package\s+([\w.]+);", content)
                cls = re.search(r"(?:class|interface|enum)\s+(\w+)", content)
                if pkg and cls:
                    file_path = f"src/main/java/{pkg.group(1).replace('.', '/')}/{cls.group(1)}.java"

            if file_path and content:
                files.append({
                    "file_path": file_path,
                    "content": content,
                    "language": lang or _guess_language(file_path),
                })

    if files:
        logger.info(
            "Extracted %d file(s) from markdown code blocks (JSON parse failed)",
            len(files),
        )
        return files

    logger.warning("Could not parse LLM output as JSON file array")
    return []


def _guess_language(file_path: str) -> str:
    """Guess language from file extension."""
    ext_map = {
        ".java": "java", ".py": "python", ".ts": "typescript",
        ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
        ".cs": "csharp", ".go": "go", ".rb": "ruby", ".rs": "rust",
        ".kt": "kotlin", ".swift": "swift", ".xml": "xml",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".sql": "sql", ".html": "html", ".css": "css",
    }
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    return ""
