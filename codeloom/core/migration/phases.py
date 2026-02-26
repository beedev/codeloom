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
from typing import Any, Dict, List, Optional, Tuple

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
        "- Include ALL migrated files in a single array.\n\n"
        "Quality Rules (MANDATORY):\n"
        "- NO STUBS: Every method must have real migrated logic, not placeholders.\n"
        "- COMPLETE BODIES: 50-line source = ~50-line migrated output.\n"
        "- Use // MIGRATION-TODO only inside otherwise complete method bodies.\n"
        "- SINGLE ENTRY POINT: Max one main()/app.listen() per MVP.\n"
        "- Include package.json/pom.xml/.csproj with ALL referenced packages.\n"
        "- DI registrations must use class-based tokens, not string literals.\n"
        "- Every import must reference a file in your output or a declared dependency."
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
    max_tokens: Optional[int] = None,
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
        max_tokens: Optional max output tokens. Passed to the provider to
            prevent truncation for phases that need large outputs (transform).
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
        kwargs: Dict[str, Any] = {"gateway_purpose": "migration"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = llm.complete(prompt, **kwargs)
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

    # Ground truth constraints (layer summary + language guidance)
    ground_truth = plan.get("_ground_truth")
    layer_summary_str = ""
    if ground_truth:
        # Prepend verified language distribution so LLM sees correct stats first
        lang_summary = ground_truth.format_language_summary()
        if lang_summary:
            codebase_context = lang_summary + "\n\n" + codebase_context

        layer_summary_str = ground_truth.format_layer_summary()
        lang_guidance = ground_truth.format_language_guidance(plan.get("target_stack", {}))
        if lang_guidance:
            layer_summary_str = layer_summary_str + "\n\n" + lang_guidance if layer_summary_str else lang_guidance

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
        layer_summary=layer_summary_str,
    )

    output = _call_llm(prompt)

    # Spot-check output against ground truth
    if ground_truth:
        issues = ground_truth.validate_phase_output("architecture", output)
        if issues:
            logger.warning(
                "Phase 2 (architecture) output has %d grounding issues: %s",
                len(issues), "; ".join(i.message for i in issues[:5]),
            )

    return {"output": output, "output_files": []}


def _build_language_override(plan: Dict, mvp_context: Optional[Dict]) -> str:
    """Build a language incompatibility override for per-MVP prompts.

    If the MVP's dominant language is incompatible with the plan's target
    framework, returns a CRITICAL instruction to keep-as-is. Otherwise empty.
    """
    ground_truth = plan.get("_ground_truth")
    if not ground_truth or not mvp_context or not mvp_context.get("unit_ids"):
        return ""
    mvp_lang = ground_truth.get_mvp_dominant_language(mvp_context["unit_ids"])
    if not mvp_lang:
        return ""
    if ground_truth.is_language_compatible(mvp_lang, plan.get("target_stack", {})):
        return ""
    return (
        f"\n\nCRITICAL: This MVP contains **{mvp_lang}** code which is NOT compatible "
        f"with the target framework ({plan.get('target_brief', '')}).\n"
        f"Do NOT migrate this code to the target framework. Instead:\n"
        f"- Analyze and document it as-is\n"
        f"- Flag it for separate handling or exclusion from migration\n"
        f"- Do NOT generate target-framework equivalents for {mvp_lang} code\n"
    )


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

    # Language compatibility override for incompatible MVPs
    prompt += _build_language_override(plan, mvp_context)

    # Append deep analysis context after the main prompt
    if deep_context_str:
        prompt += f"\n\n{deep_context_str}"

    output = _call_llm(prompt, context_type="understanding")

    # SRC validation loop: check register coverage
    ground_truth = plan.get("_ground_truth")
    if mvp_context and mvp_context.get("unit_ids"):
        output = _src_validate_register(
            output, mvp_context["unit_ids"], prompt, ground_truth=ground_truth,
        )

    # Spot-check output against ground truth
    if ground_truth:
        issues = ground_truth.validate_phase_output("analyze", output)
        if issues:
            logger.warning(
                "Phase 3 (analyze) output has %d grounding issues: %s",
                len(issues), "; ".join(i.message for i in issues[:5]),
            )

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

    prompt += _build_language_override(plan, mvp_context)

    output = _call_llm(prompt)
    return {"output": output, "output_files": []}


# ── Phase 5: Transform (Per-MVP) ─────────────────────────────────────

def _execute_transform(
    plan: Dict, prev: Dict[int, str],
    ctx: MigrationContextBuilder, budget: int,
    mvp_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Transform: Generate migrated code for this MVP.

    For MVP 99 (integration_mvp=True), delegates to the three-pass
    integration transform instead of the standard code migration."""

    # Integration MVP: branch to three-pass architecture
    if mvp_context and (mvp_context.get("metrics") or {}).get("integration_mvp"):
        return _execute_integration_transform(
            plan, prev, ctx, budget, mvp_context,
            all_mvp_transform_outputs=plan.get("_all_mvp_transform_outputs"),
            all_mvp_analysis_outputs=plan.get("_all_mvp_analysis_outputs"),
        )

    # V1: Phase 5 (Design output in prev[4]).
    # V2: Phase 3 (Architecture output in prev[1], optional analysis_output on plan).
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

    # Fix 1: Cross-MVP file manifest — tells this MVP what prior MVPs already generated
    prior_manifest_str = ""
    prior_manifest = plan.get("_prior_mvp_file_manifest", {})
    if prior_manifest:
        lines: List[str] = []
        for mvp_name, paths in prior_manifest.items():
            for p in paths:
                lines.append(f"  {p}  [{mvp_name}]")
        prior_manifest_str = "\n".join(lines)
        logger.info(
            "Transform: injecting prior MVP manifest (%d files from %d MVPs)",
            len(lines), len(prior_manifest),
        )

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
        prior_mvp_manifest=prior_manifest_str,
    )

    prompt += _build_language_override(plan, mvp_context)

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

    raw = _call_llm(prompt, context_type="generation", max_tokens=16_384)
    output_files = _parse_json_files(raw)

    # Retry once if parse failed but raw text contains code indicators
    if not output_files and raw and ("```" in raw or "import " in raw):
        logger.warning(
            "Transform parse failed (%d chars raw output). "
            "Retrying with explicit JSON instruction.",
            len(raw),
        )
        retry_prompt = (
            "Your previous response contained code but was not in the required JSON format.\n\n"
            "Reformat your ENTIRE response as a JSON array. Each element must have:\n"
            '{"file_path": "src/...", "content": "...", "language": "typescript"}\n\n'
            "Previous response to reformat:\n\n" + raw[:30_000]
        )
        retry_raw = _call_llm(retry_prompt, context_type="generation", max_tokens=16_384)
        output_files = _parse_json_files(retry_raw)
        if output_files:
            logger.info("Retry succeeded: parsed %d files", len(output_files))

    if output_files:
        total_content_len = sum(len(f.get("content", "")) for f in output_files)
        output = f"Generated {len(output_files)} migrated file(s).\n\nSee output_files for generated code."
        logger.info(
            "Transform output: %d files, %d chars total content",
            len(output_files), total_content_len,
        )
    else:
        output = raw
        output_files = []
        logger.warning(
            "Transform produced no parseable files (%d chars raw output)",
            len(raw) if raw else 0,
        )

    # Stub quality check (soft gate — log and report, never blocks)
    stub_quality = _check_stub_quality(output_files) if output_files else {}
    if stub_quality.get("stub_count", 0) > 0:
        logger.warning(
            "Transform stub quality: %d/%d files contain stubs — %s",
            stub_quality["stub_count"],
            stub_quality["total_files"],
            ", ".join(f["file_path"] for f in stub_quality.get("stub_files", [])[:5]),
        )

    return {"output": output, "output_files": output_files, "stub_quality": stub_quality}


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
    ground_truth: Optional[Any] = None,
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

    prompt += _build_language_override(plan, mvp_context)

    output = _call_llm(prompt)

    # SRC validation loop: check register coverage
    if mvp_context and mvp_context.get("unit_ids"):
        output = _src_validate_register(
            output, mvp_context["unit_ids"], prompt, ground_truth=ground_truth,
        )

    # Spot-check output against ground truth
    if ground_truth:
        issues = ground_truth.validate_phase_output("analyze", output)
        if issues:
            logger.warning(
                "MVP analysis output has %d grounding issues: %s",
                len(issues), "; ".join(i.message for i in issues[:5]),
            )

    return {"output": output, "output_files": []}


# ── Integration MVP Analysis ────────────────────────────────────────

# Regex patterns to extract register rows from MVP analysis markdown tables.
# Each register type has its own prefix (BR-, DE-, INT-, VAL-).
_REGISTER_ROW_RE = re.compile(
    r"^\|\s*((?:BR|DE|INT|VAL)-\d+)\s*\|(.+)$", re.MULTILINE,
)


def _parse_mvp_registers(analysis_text: str) -> Dict[str, List[str]]:
    """Extract functional requirements register rows from an MVP's analysis output.

    Returns dict with keys 'BR', 'DE', 'INT', 'VAL', each containing
    a list of full table-row strings (with ID prefix).
    """
    registers: Dict[str, List[str]] = {"BR": [], "DE": [], "INT": [], "VAL": []}
    for match in _REGISTER_ROW_RE.finditer(analysis_text):
        reg_id = match.group(1)          # e.g. "BR-3"
        full_row = match.group(0).strip()  # full "| BR-3 | ... |" line
        prefix = reg_id.split("-")[0]      # "BR"
        if prefix in registers:
            registers[prefix].append(full_row)
    return registers


def _merge_registers(
    all_mvp_analyses: Dict[str, Dict],
) -> Tuple[str, Dict[str, List[Tuple[str, str]]]]:
    """Aggregate functional requirements from all MVPs into a single register.

    Returns:
        (merged_markdown, per_type_rows) where per_type_rows maps
        register type → list of (mvp_name, row_text) tuples.
    """
    per_type: Dict[str, List[Tuple[str, str]]] = {
        "BR": [], "DE": [], "INT": [], "VAL": [],
    }

    for mvp_name, mvp_info in sorted(all_mvp_analyses.items()):
        analysis_text = mvp_info.get("output", "")
        regs = _parse_mvp_registers(analysis_text)
        for reg_type, rows in regs.items():
            for row in rows:
                per_type[reg_type].append((mvp_name, row))

    # Build markdown
    lines: List[str] = []
    type_labels = {
        "BR": "Business Rules",
        "DE": "Data Entities",
        "INT": "External Integrations",
        "VAL": "Validation Rules",
    }
    for reg_type, label in type_labels.items():
        entries = per_type[reg_type]
        lines.append(f"### {label} ({len(entries)} total)")
        if entries:
            for mvp_name, row in entries:
                lines.append(f"- **[{mvp_name}]** {row}")
        else:
            lines.append("_(none found)_")
        lines.append("")

    return "\n".join(lines), per_type


def _detect_cross_mvp_dependencies(
    per_type_rows: Dict[str, List[Tuple[str, str]]],
) -> str:
    """Detect shared entities and integrations across MVPs.

    Returns a markdown summary of cross-MVP dependencies.
    """
    # Track which MVPs reference each entity/integration name
    entity_mvps: Dict[str, set] = {}
    integration_mvps: Dict[str, set] = {}

    for mvp_name, row in per_type_rows.get("DE", []):
        # Extract entity name from table row: "| DE-1 | EntityName | ..."
        parts = [p.strip() for p in row.split("|") if p.strip()]
        if len(parts) >= 2:
            entity_name = parts[1].strip()
            entity_mvps.setdefault(entity_name, set()).add(mvp_name)

    for mvp_name, row in per_type_rows.get("INT", []):
        parts = [p.strip() for p in row.split("|") if p.strip()]
        if len(parts) >= 2:
            integration_name = parts[1].strip()
            integration_mvps.setdefault(integration_name, set()).add(mvp_name)

    lines: List[str] = []

    # Shared entities (referenced by 2+ MVPs)
    shared_entities = {k: v for k, v in entity_mvps.items() if len(v) >= 2}
    lines.append(f"### Shared Data Entities ({len(shared_entities)})")
    if shared_entities:
        for entity, mvps in sorted(shared_entities.items()):
            lines.append(f"- **{entity}**: {', '.join(sorted(mvps))}")
    else:
        lines.append("_(no shared entities detected)_")
    lines.append("")

    # Shared integrations (referenced by 2+ MVPs)
    shared_integrations = {k: v for k, v in integration_mvps.items() if len(v) >= 2}
    lines.append(f"### Shared External Integrations ({len(shared_integrations)})")
    if shared_integrations:
        for integ, mvps in sorted(shared_integrations.items()):
            lines.append(f"- **{integ}**: {', '.join(sorted(mvps))}")
    else:
        lines.append("_(no shared integrations detected)_")
    lines.append("")

    # Summary stats
    total_entities = len(entity_mvps)
    total_integrations = len(integration_mvps)
    lines.append(f"### Summary")
    lines.append(f"- Total unique entities: {total_entities} ({len(shared_entities)} shared)")
    lines.append(f"- Total unique integrations: {total_integrations} ({len(shared_integrations)} shared)")

    return "\n".join(lines)


def execute_integration_analysis(
    plan: Dict[str, Any],
    previous_outputs: Dict[int, str],
    mvp_context: Dict[str, Any],
    all_mvp_analyses: Dict[str, Dict],
) -> Dict[str, Any]:
    """Run integration analysis for MVP 99 — aggregates all other MVPs.

    Instead of analyzing source code (MVP 99 has none), this:
    1. Parses each MVP's functional requirements register (BR, DE, INT, VAL)
    2. Merges into a unified cross-project register
    3. Detects cross-MVP dependencies (shared entities, overlapping integrations)
    4. Generates integration-specific analysis via LLM

    Args:
        plan: Migration plan dict (target_brief, target_stack, etc.)
        previous_outputs: Plan-level phase outputs (architecture, discovery)
        mvp_context: MVP 99 dict (for metadata only — has no units)
        all_mvp_analyses: Dict[mvp_name → {mvp_id, priority, unit_count, output}]

    Returns:
        Dict with 'output' (markdown) and 'output_files' (always [])
    """
    if not all_mvp_analyses:
        logger.warning("Integration analysis: no MVP analyses available — skipping LLM call")
        return {
            "output": "## Integration Analysis\n\n"
                      "No completed MVP analyses found. Run analysis for individual MVPs first.",
            "output_files": [],
        }

    # ── Step 1: Parse + merge registers (deterministic) ──
    merged_register, per_type_rows = _merge_registers(all_mvp_analyses)

    total_items = sum(len(v) for v in per_type_rows.values())
    logger.info(
        "Integration analysis: merged %d register items from %d MVPs "
        "(BR=%d, DE=%d, INT=%d, VAL=%d)",
        total_items, len(all_mvp_analyses),
        len(per_type_rows["BR"]), len(per_type_rows["DE"]),
        len(per_type_rows["INT"]), len(per_type_rows["VAL"]),
    )

    # ── Step 2: Cross-MVP dependency detection (deterministic) ──
    cross_deps = _detect_cross_mvp_dependencies(per_type_rows)

    # ── Step 3: Build MVP summary table ──
    mvp_lines = ["| MVP | Priority | Units | Status |", "|-----|----------|-------|--------|"]
    for mvp_name, info in sorted(all_mvp_analyses.items(), key=lambda x: x[1].get("priority", 0)):
        mvp_lines.append(
            f"| {mvp_name} | {info.get('priority', '?')} | "
            f"{info.get('unit_count', 0)} | analyzed |"
        )
    mvp_summary_table = "\n".join(mvp_lines)

    # ── Step 4: LLM integration analysis ──
    architecture_output = previous_outputs.get(1, "") or previous_outputs.get(2, "")

    prompt = prompts.integration_analysis_prompt(
        architecture_output=architecture_output,
        target_stack=plan.get("target_stack", {}),
        merged_register=merged_register,
        cross_mvp_dependencies=cross_deps,
        mvp_summary_table=mvp_summary_table,
    )

    output = _call_llm(prompt)

    logger.info(
        "Integration analysis complete: %d chars output, %d register items aggregated",
        len(output), total_items,
    )

    return {"output": output, "output_files": []}


# ── SRC Validation ───────────────────────────────────────────────────

_REGISTER_ID_RE = re.compile(r"\b(?:BR|DE|INT|VAL)-\d+\b")

_REGISTER_ENTRY_RE = re.compile(
    r"\|\s*((?:BR|DE|INT|VAL)-\d+)\s*\|([^|]+)\|",
)


def _extract_register_summary(output: str, max_entries: int = 30) -> str:
    """Extract register entries with descriptions for gap-prompt context.

    Returns a concise summary of already-covered entries so the LLM
    knows what's been documented and can generate genuinely new items.
    """
    entries = _REGISTER_ENTRY_RE.findall(output)
    if not entries:
        return ""
    lines = ["Already covered entries:"]
    for reg_id, desc in entries[:max_entries]:
        lines.append(f"- {reg_id}: {desc.strip()}")
    return "\n".join(lines)


def _strip_duplicate_rows(extension: str, existing_ids: set) -> str:
    """Remove table rows from extension whose register ID already exists."""
    if not existing_ids:
        return extension
    lines = extension.split("\n")
    filtered = []
    stripped = 0
    for line in lines:
        match = _REGISTER_ENTRY_RE.search(line)
        if match and match.group(1) in existing_ids:
            stripped += 1
            continue
        filtered.append(line)
    if stripped:
        logger.info("SRC dedup: stripped %d duplicate register rows", stripped)
    return "\n".join(filtered)


def _src_validate_register(
    output: str,
    unit_ids: List[str],
    original_prompt: str,
    max_iterations: int = 3,
    ground_truth: Optional[Any] = None,
) -> str:
    """SRC validation loop for Phase 3 Functional Requirements Register.

    Checks that the register covers enough MVP units, and re-prompts the
    LLM to fill gaps if coverage is below target. When ground_truth is
    provided, the gap prompt is anchored to verified codebase facts to
    prevent hallucination. Without ground_truth, falls back to extracting
    context from original_prompt.

    Args:
        output: LLM-generated Phase 3 output
        unit_ids: MVP's unit IDs to check coverage against
        original_prompt: The original Phase 3 prompt (used as fallback
            context if ground_truth is not available)
        max_iterations: Maximum re-prompt attempts
        ground_truth: Optional CodebaseGroundTruth instance for grounded
            re-prompting with verified unit names from DB.

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

    # Extract codebase anchor once (before loop).
    # Prefer ground_truth (verified DB facts) over prompt parsing (fragile).
    codebase_anchor = ""
    if ground_truth is not None:
        codebase_anchor = ground_truth.build_src_gap_context(unit_ids, set())
    elif original_prompt:
        # Fallback: extract the MVP Functional Context section from prompt
        codebase_anchor = _extract_prompt_section(
            original_prompt, "MVP Functional Context", max_chars=3000,
        ) or _extract_prompt_section(
            original_prompt, "Current MVP", max_chars=2000,
        )

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

        # Build entry-aware summary so LLM sees what's already covered
        existing_ids_str = ", ".join(sorted(register_ids)) if register_ids else "none yet"
        existing_summary = _extract_register_summary(best_output)

        # Build gap prompt with codebase context to prevent hallucination.
        gap_prompt = (
            f"You are extending a Functional Requirements Register for a codebase migration.\n\n"
            f"The register currently has {register_count} items "
            f"but needs at least {target} to cover this MVP's {total_units} code units.\n\n"
        )

        if existing_summary:
            gap_prompt += f"{existing_summary}\n\n"
        else:
            gap_prompt += f"Already covered IDs: {existing_ids_str}\n\n"

        if codebase_anchor:
            gap_prompt += (
                f"Here are the ACTUAL code units in this MVP. Use ONLY these real names "
                f"and qualified paths. Do NOT invent classes, methods, or entities that "
                f"are not listed here:\n\n"
                f"{codebase_anchor}\n\n"
            )

        gap_prompt += (
            f"Generate ONLY new register entries for units NOT already covered. "
            f"Do NOT repeat any of the entries listed above.\n\n"
            f"Categories:\n"
            f"- BR-N: Business rules (service methods, handlers, use cases)\n"
            f"- DE-N: Data entities (model/entity classes with fields)\n"
            f"- INT-N: External integrations (HTTP clients, queues, adapters)\n"
            f"- VAL-N: Validation rules (validators, constraints)\n\n"
            f"Start numbering from BR-{register_count + 1}, DE-{register_count + 1}, etc.\n"
            f"Output ONLY the new markdown tables — no preamble, no repetition of existing entries."
        )

        try:
            extension = _call_llm(gap_prompt)
            if extension and len(extension) > 50:
                # Strip rows whose IDs already exist (dedup)
                extension = _strip_duplicate_rows(extension, register_ids)

                # Spot-check for hallucinated entities if ground truth available
                if ground_truth is not None:
                    issues = ground_truth._check_hallucinated_entities(extension)
                    if issues:
                        logger.warning(
                            "SRC iteration %d: %d potentially hallucinated entities detected",
                            iteration + 1, len(issues),
                        )
                best_output = best_output + "\n\n## Register Extension (SRC Iteration " + str(iteration + 1) + ")\n\n" + extension
        except Exception as e:
            logger.warning("SRC re-prompt failed (iteration %d): %s", iteration, e)
            break

    return best_output


def _extract_prompt_section(prompt: str, header_prefix: str, max_chars: int = 4000) -> str:
    """Extract a markdown section from prompt by header prefix.

    Scans for a line starting with ## or ### followed by header_prefix,
    returns everything up to the next same-level header or EOF.
    Truncates to max_chars to keep token budget bounded.

    Used as fallback context extraction when ground_truth is not available.
    """
    pattern = re.compile(
        rf"^(#{2,3}\s+{re.escape(header_prefix)}.*?)(?=\n#{2,3}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(prompt)
    if not match:
        return ""
    section = match.group(1).strip()
    if len(section) > max_chars:
        section = section[:max_chars] + "\n[... truncated]"
    return section


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


# ── Stub Detection ────────────────────────────────────────────────────

# Compiled once at module level for performance
_STUB_PATTERNS = [
    re.compile(r"(?i)not\s+(?:yet\s+)?implemented"),
    re.compile(r"(?i)\bplaceholder\b"),
    re.compile(r"(?i)\bdummy\b"),
    # "stub" but not the JSON field "is_sp_stub"
    re.compile(r"(?i)(?<!is_sp_)\bstub\b"),
    re.compile(r"throw\s+new\s+NotImplementedError"),
    re.compile(r"raise\s+NotImplementedError"),
    re.compile(r"(?i)(?://|#)\s*TODO:\s*implement"),
    # Python pass-only body
    re.compile(r"^\s*pass\s*$", re.MULTILINE),
    # Empty return as sole body: return null/None/undefined/0/""/[]/{}
    re.compile(r"^\s*return\s+(?:null|None|undefined|0|\"\"|\[\]|\{\})\s*;?\s*(?://.*)?$", re.MULTILINE),
]


def _check_stub_quality(output_files: List[Dict]) -> Dict:
    """Scan transform output files for stub/placeholder patterns.

    Uses a heuristic: files with <5 non-blank non-comment lines that also
    match a stub pattern are flagged.  Large files with an occasional TODO
    inside real logic are acceptable and NOT flagged.

    Returns:
        {stub_count, stub_files: [{file_path, patterns_found}],
         total_files, stub_ratio}
    """
    if not output_files:
        return {}

    stub_files: List[Dict] = []

    for fd in output_files:
        fp = fd.get("file_path", "unknown")
        content = fd.get("content", "")
        if not content:
            # Completely empty file is a stub
            stub_files.append({"file_path": fp, "patterns_found": ["empty_file"]})
            continue

        # Count substantive lines (non-blank, non-comment-only)
        lines = content.splitlines()
        substantive = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("//", "#", "/*", "*", "*/", "<!--")):
                continue
            # import/using/package lines are boilerplate, not logic
            if stripped.startswith(("import ", "from ", "using ", "package ")):
                continue
            substantive += 1

        # Check each pattern
        found_patterns: List[str] = []
        for pat in _STUB_PATTERNS:
            if pat.search(content):
                found_patterns.append(pat.pattern[:60])

        if not found_patterns:
            continue

        # Heuristic: small file with stub pattern = stub.
        # Large file with stub pattern = acceptable (TODO in real code).
        if substantive < 5:
            stub_files.append({"file_path": fp, "patterns_found": found_patterns})
        elif len(found_patterns) >= 3:
            # Multiple stub patterns even in a larger file = likely hollow
            stub_files.append({"file_path": fp, "patterns_found": found_patterns})

    total = len(output_files)
    stub_count = len(stub_files)

    return {
        "stub_count": stub_count,
        "stub_files": stub_files,
        "total_files": total,
        "stub_ratio": round(stub_count / total, 3) if total else 0.0,
    }


# ── Contract Card Extraction (Deterministic) ──────────────────────────

_CONTRACT_PATTERNS: Dict[str, Dict[str, List[re.Pattern]]] = {
    "python": {
        "imports": [
            re.compile(r"^import\s+(\S+)", re.MULTILINE),
            re.compile(r"^from\s+(\S+)\s+import", re.MULTILINE),
        ],
        "provides_class": [re.compile(r"^class\s+(\w+)", re.MULTILINE)],
        "provides_func": [re.compile(r"^def\s+(\w+)", re.MULTILINE)],
        "di": [re.compile(r"@inject", re.IGNORECASE)],
        "entry_points": [re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')],
    },
    "java": {
        "imports": [re.compile(r"^import\s+(.+?);", re.MULTILINE)],
        "provides_class": [
            re.compile(r"public\s+(?:class|interface|enum)\s+(\w+)"),
        ],
        "di": [
            re.compile(r"@(?:Inject|Autowired|Component|Service|Repository|Controller|Bean)\b"),
        ],
        "entry_points": [re.compile(r"public\s+static\s+void\s+main\s*\(")],
        "routes": [
            re.compile(r'@(?:Request|Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']'),
        ],
    },
    "csharp": {
        "imports": [re.compile(r"^using\s+(.+?);", re.MULTILINE)],
        "provides_class": [
            re.compile(r"public\s+(?:class|interface|struct|record)\s+(\w+)"),
        ],
        "di": [re.compile(r"\[Inject\]"), re.compile(r"services\.Add(?:Scoped|Transient|Singleton)")],
        "entry_points": [re.compile(r"static\s+(?:async\s+)?(?:Task\s+)?(?:void\s+)?Main\s*\(")],
        "routes": [
            re.compile(r'\[(?:Http(?:Get|Post|Put|Delete|Patch)|Route)\s*\(\s*["\']([^"\']+)["\']'),
        ],
    },
    "typescript": {
        "imports": [
            re.compile(r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"),
        ],
        "provides_class": [
            re.compile(r"export\s+(?:class|function|interface|type|const|enum)\s+(\w+)"),
        ],
        "di": [re.compile(r"@Injectable\s*\("), re.compile(r"container\.bind\(")],
        "entry_points": [
            re.compile(r"app\.listen\s*\("),
            re.compile(r"createServer\s*\("),
            re.compile(r"bootstrap\s*\(\s*\)"),
        ],
        "routes": [
            re.compile(r"@(?:Get|Post|Put|Delete|Patch)\s*\(\s*['\"]([^'\"]+)['\"]"),
            re.compile(r"(?:app|router)\.(?:get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]"),
        ],
    },
    "javascript": {},  # Populated below
}
# JS shares TS patterns
_CONTRACT_PATTERNS["javascript"] = _CONTRACT_PATTERNS["typescript"]


def _extract_contract_cards(
    all_mvp_files: Dict[str, List[Dict]],
) -> List[Dict]:
    """Extract contract cards from all MVP transform outputs.

    Language-agnostic regex extraction of provides/consumes/routes/DI
    from each MVP's generated files.

    Args:
        all_mvp_files: {mvp_name: [output_file_dict, ...]}

    Returns:
        List of per-file contract cards.
    """
    cards: List[Dict] = []

    for mvp_name, files in all_mvp_files.items():
        for fd in files:
            fp = fd.get("file_path", "unknown")
            content = fd.get("content", "")
            lang = (fd.get("language", "") or _guess_language(fp)).lower()
            if not content:
                continue

            # Resolve pattern set — fall back to all patterns combined
            patterns = _CONTRACT_PATTERNS.get(lang, {})
            if not patterns:
                # Try to match by extension patterns
                for plang, ppats in _CONTRACT_PATTERNS.items():
                    if plang in lang:
                        patterns = ppats
                        break

            provides: List[str] = []
            consumes: List[str] = []
            routes: List[str] = []
            di_registrations: List[str] = []
            entry_points: List[str] = []

            # Provides (classes, functions, interfaces)
            for pat in patterns.get("provides_class", []):
                provides.extend(pat.findall(content))
            for pat in patterns.get("provides_func", []):
                # Exclude __dunder__ methods and test_ functions
                for m in pat.findall(content):
                    if not m.startswith("_") and not m.startswith("test_"):
                        provides.append(m)

            # Consumes (imports)
            for pat in patterns.get("imports", []):
                matches = pat.findall(content)
                for m in matches:
                    # findall may return tuples for multi-group patterns
                    if isinstance(m, tuple):
                        consumes.extend(part.strip() for part in m if part.strip())
                    else:
                        consumes.append(m.strip())

            # Routes
            for pat in patterns.get("routes", []):
                routes.extend(pat.findall(content))

            # DI registrations
            for pat in patterns.get("di", []):
                di_registrations.extend(pat.findall(content) or [pat.pattern[:30]])

            # Entry points
            for pat in patterns.get("entry_points", []):
                if pat.search(content):
                    entry_points.append(pat.pattern[:50])

            cards.append({
                "mvp_name": mvp_name,
                "file_path": fp,
                "language": lang,
                "provides": list(set(provides)),
                "consumes": list(set(consumes)),
                "routes": list(set(routes)),
                "di_registrations": di_registrations,
                "entry_points": entry_points,
            })

    return cards


def _build_cross_reference_matrix(cards: List[Dict]) -> Dict:
    """Build cross-reference matrix from contract cards using pure set operations.

    Zero LLM involvement — deterministic gap detection.
    """
    # Index by MVP
    provides_by_mvp: Dict[str, set] = {}
    consumes_by_mvp: Dict[str, set] = {}
    all_provides: set = set()
    entry_points_all: List[Dict] = []
    has_manifest = False
    injectable_classes: set = set()
    di_registered: set = set()

    for card in cards:
        mvp = card["mvp_name"]

        # Provides
        pvd = set(card.get("provides", []))
        provides_by_mvp.setdefault(mvp, set()).update(pvd)
        all_provides.update(pvd)

        # Consumes — extract just the symbol name (last segment of import path)
        raw_consumes = card.get("consumes", [])
        cleaned = set()
        for c in raw_consumes:
            # "com.example.UserService" → "UserService"
            # "./services/UserService" → "UserService"
            parts = re.split(r"[./\\]", c)
            last = parts[-1].strip() if parts else c.strip()
            if last and not last.startswith(("@", "{", "}")):
                cleaned.add(last)
        consumes_by_mvp.setdefault(mvp, set()).update(cleaned)

        # Entry points
        if card.get("entry_points"):
            entry_points_all.append({
                "mvp_name": mvp,
                "file_path": card["file_path"],
                "patterns": card["entry_points"],
            })

        # Manifest detection
        fp_lower = card["file_path"].lower()
        if fp_lower in ("package.json", "pom.xml", "requirements.txt") or fp_lower.endswith(".csproj"):
            has_manifest = True

        # DI tracking
        if card.get("di_registrations"):
            injectable_classes.update(card.get("provides", []))
        # Look for container.bind() or services.Add() patterns
        for reg in card.get("di_registrations", []):
            if isinstance(reg, str):
                di_registered.add(reg)

    # Unresolved imports: what each MVP consumes that no other MVP provides
    unresolved: List[Dict] = []
    for mvp, consumes in consumes_by_mvp.items():
        own_provides = provides_by_mvp.get(mvp, set())
        other_provides = all_provides - own_provides
        # Also exclude well-known stdlib/framework modules
        for symbol in consumes - other_provides - own_provides:
            # Skip single-char or all-lowercase likely stdlib names
            if len(symbol) <= 2:
                continue
            unresolved.append({
                "consumer_mvp": mvp,
                "import_ref": symbol,
            })

    # Duplicate exports: symbols provided by 2+ MVPs
    duplicates: List[Dict] = []
    symbol_to_mvps: Dict[str, List[str]] = {}
    for mvp, pvd in provides_by_mvp.items():
        for s in pvd:
            symbol_to_mvps.setdefault(s, []).append(mvp)
    for s, mvps in symbol_to_mvps.items():
        if len(mvps) > 1:
            duplicates.append({"symbol": s, "mvps": mvps})

    # Partial DI
    partial_di: List[str] = sorted(injectable_classes - di_registered)

    return {
        "unresolved_imports": unresolved,
        "duplicate_exports": duplicates,
        "entry_point_count": len(entry_points_all),
        "entry_points": entry_points_all,
        "missing_manifest": not has_manifest,
        "partial_di": partial_di,
        "provides_by_mvp": {k: sorted(v) for k, v in provides_by_mvp.items()},
        "total_provides": len(all_provides),
        "total_consumes": sum(len(v) for v in consumes_by_mvp.values()),
    }


def _check_requirements_coverage(
    all_mvp_analysis: Dict[str, Any],
    all_provides: set,
) -> Dict:
    """Check functional requirements coverage against generated code.

    Extracts key entity nouns from each MVP's analysis_output and checks
    if they appear in the provides set from contract cards.

    Args:
        all_mvp_analysis: {mvp_name: analysis_output_dict_or_str}
        all_provides: union of all provides symbols across all MVPs

    Returns:
        {uncovered_requirements: [{text, source_mvp}], coverage_ratio}
    """
    import json as _json

    provides_lower = {p.lower() for p in all_provides}
    uncovered: List[Dict] = []
    total_reqs = 0
    covered_count = 0

    for mvp_name, analysis in all_mvp_analysis.items():
        # Extract requirements text from analysis_output
        reqs_text = ""
        if isinstance(analysis, dict):
            # Look for functional requirements register in various keys
            for key in ("functional_requirements", "requirements", "register",
                        "business_rules", "output"):
                if key in analysis:
                    val = analysis[key]
                    reqs_text += _json.dumps(val) if isinstance(val, (dict, list)) else str(val)
                    reqs_text += "\n"
            if not reqs_text and analysis:
                reqs_text = _json.dumps(analysis)
        elif isinstance(analysis, str):
            reqs_text = analysis

        if not reqs_text:
            continue

        # Extract key entities (capitalised words, likely class/service names)
        # Pattern: 2+ word sequences starting with uppercase
        entity_pattern = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:Service|Controller|Repository|Manager|Handler|Factory|Gateway|Provider|Adapter|Client|Engine|Processor|Orchestrator|Builder)?)\b")
        entities = set(entity_pattern.findall(reqs_text))

        for entity in entities:
            total_reqs += 1
            if entity.lower() in provides_lower:
                covered_count += 1
            else:
                # Only report if it looks like a service/class name (not generic words)
                if any(suffix in entity for suffix in (
                    "Service", "Controller", "Repository", "Manager",
                    "Handler", "Factory", "Gateway", "Provider",
                    "Adapter", "Client", "Engine", "Processor",
                    "Orchestrator", "Builder",
                )):
                    uncovered.append({
                        "text": entity,
                        "source_mvp": mvp_name,
                    })

    coverage_ratio = round(covered_count / total_reqs, 3) if total_reqs else 1.0

    return {
        "uncovered_requirements": uncovered,
        "coverage_ratio": coverage_ratio,
        "total_entities_checked": total_reqs,
        "covered_count": covered_count,
    }


# ── Integration Transform (MVP 99) ────────────────────────────────────

def _execute_integration_transform(
    plan: Dict,
    prev: Dict[int, str],
    ctx: "MigrationContextBuilder",
    budget: int,
    mvp_context: Optional[Dict] = None,
    all_mvp_transform_outputs: Optional[Dict[str, List[Dict]]] = None,
    all_mvp_analysis_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Integration Transform for MVP 99 — three-pass architecture.

    Pass 1: Deterministic contract card extraction + cross-reference matrix
    Pass 2: Requirements coverage check
    Pass 3: LLM resolves pre-computed gaps
    Verify: Re-run Pass 1 on combined output → gap report
    """
    if not all_mvp_transform_outputs:
        return {
            "output": "Integration skipped: no MVP transform outputs available.",
            "output_files": [],
        }

    # ── Pass 1: Contract cards + cross-reference matrix ──
    cards = _extract_contract_cards(all_mvp_transform_outputs)
    cross_ref = _build_cross_reference_matrix(cards)

    logger.info(
        "Integration Pass 1: %d contract cards, %d unresolved imports, "
        "%d duplicate exports, %d entry points, manifest=%s",
        len(cards),
        len(cross_ref["unresolved_imports"]),
        len(cross_ref["duplicate_exports"]),
        cross_ref["entry_point_count"],
        "present" if not cross_ref["missing_manifest"] else "MISSING",
    )

    # Fix 4: Collect all external packages across all MVPs for package.json generation
    import re as _re_pkg
    _imp_re = _re_pkg.compile(r'''import\s+.*?\s+from\s+['"]([^'"./][^'"]*?)['"]''')
    _builtins = {'path', 'fs', 'os', 'util', 'crypto', 'http', 'https', 'stream', 'events', 'url', 'buffer'}
    _all_pkgs: set = set()
    for _mvp_files in all_mvp_transform_outputs.values():
        for _f in _mvp_files:
            if _f.get("language") in ("typescript", "javascript"):
                for _m in _imp_re.finditer(_f.get("content", "")):
                    _raw = _m.group(1)
                    _pkg = '/'.join(_raw.split('/')[:2]) if _raw.startswith('@') else _raw.split('/')[0]
                    if _pkg not in _builtins:
                        _all_pkgs.add(_pkg)
    cross_ref["all_external_packages"] = sorted(_all_pkgs)
    logger.info("Integration Pass 1: detected %d unique external packages", len(_all_pkgs))

    # ── Pass 2: Requirements coverage ──
    all_provides = set()
    for mvp_provides in cross_ref.get("provides_by_mvp", {}).values():
        all_provides.update(mvp_provides)

    req_coverage = _check_requirements_coverage(
        all_mvp_analysis_outputs or {}, all_provides,
    )

    logger.info(
        "Integration Pass 2: requirements coverage=%.1f%%, "
        "%d uncovered requirements",
        req_coverage["coverage_ratio"] * 100,
        len(req_coverage["uncovered_requirements"]),
    )

    # ── Pass 3: LLM integration review ──
    # Build architecture context from previous outputs
    architecture_output = prev.get(1, "") or prev.get(2, "") or ""
    if len(architecture_output) > 8000:
        architecture_output = architecture_output[:8000] + "\n... [truncated]"

    # Fix 4: Build enriched file listing with provides from contract cards
    card_by_file: Dict[str, Dict] = {c.get("file_path", ""): c for c in cards}
    mvp_listing_parts: List[str] = []
    for mvp_name, files in all_mvp_transform_outputs.items():
        file_summaries = []
        for fd in files[:30]:
            fp = fd.get("file_path", "unknown")
            card = card_by_file.get(fp, {})
            provides = card.get("provides", [])
            provides_str = f"  → {', '.join(provides[:4])}" if provides else ""
            file_summaries.append(f"    - {fp}{provides_str}")
        mvp_listing_parts.append(
            f"  **{mvp_name}** ({len(files)} files):\n" + "\n".join(file_summaries)
        )
    mvp_file_listing = "\n".join(mvp_listing_parts)

    prompt = prompts.integration_review_prompt(
        architecture_output=architecture_output,
        cross_ref_matrix=cross_ref,
        requirements_coverage=req_coverage,
        target_stack=plan.get("target_stack", {}),
        mvp_file_listing=mvp_file_listing,
    )

    raw = _call_llm(prompt, context_type="generation", max_tokens=16_384)
    output_files = _parse_json_files(raw)

    if not output_files and raw:
        logger.warning(
            "Integration review parse failed (%d chars). Retrying with JSON instruction.",
            len(raw),
        )
        retry_prompt = (
            "Your previous response contained code but was not in the required JSON format.\n\n"
            "Reformat your ENTIRE response as a JSON array. Each element must have:\n"
            '{"file_path": "src/...", "content": "...", "language": "typescript"}\n\n'
            "Previous response to reformat:\n\n" + raw[:30_000]
        )
        retry_raw = _call_llm(retry_prompt, context_type="generation", max_tokens=16_384)
        output_files = _parse_json_files(retry_raw)

    # ── Verification pass: re-check combined output ──
    gap_report: Dict = {}
    if output_files:
        # Combine all MVP files + integration files for re-check
        combined: Dict[str, List[Dict]] = dict(all_mvp_transform_outputs)
        combined["MVP 99 \u2014 Integration"] = output_files

        verify_cards = _extract_contract_cards(combined)
        verify_matrix = _build_cross_reference_matrix(verify_cards)

        resolved_count = (
            len(cross_ref["unresolved_imports"])
            - len(verify_matrix["unresolved_imports"])
        )

        gap_report = {
            "remaining_unresolved": verify_matrix["unresolved_imports"],
            "remaining_duplicates": verify_matrix["duplicate_exports"],
            "entry_point_count_after": verify_matrix["entry_point_count"],
            "manifest_present_after": not verify_matrix["missing_manifest"],
            "resolved_count": resolved_count,
            "total_gaps_before": len(cross_ref["unresolved_imports"]),
        }

        logger.info(
            "Integration verification: resolved %d/%d unresolved imports, "
            "%d remaining, entry_points=%d",
            resolved_count,
            len(cross_ref["unresolved_imports"]),
            len(verify_matrix["unresolved_imports"]),
            verify_matrix["entry_point_count"],
        )

    output_text = (
        f"Integration Transform: generated {len(output_files)} integration file(s)."
    )
    if gap_report:
        output_text += (
            f"\nResolved {gap_report.get('resolved_count', 0)}/"
            f"{gap_report.get('total_gaps_before', 0)} gaps."
        )
        remaining = len(gap_report.get("remaining_unresolved", []))
        if remaining:
            output_text += f"\n{remaining} unresolved imports remain (see gap_report)."

    # Run stub check on integration output too
    stub_quality = _check_stub_quality(output_files) if output_files else {}

    # Round 9: Lane-agnostic DI generation + package.json patch
    if output_files:
        output_files = _post_transform_consolidate(
            all_mvp_transform_outputs,
            output_files,
            plan.get("target_stack", {}),
            plan=plan,
        )

    return {
        "output": output_text,
        "output_files": output_files,
        "stub_quality": stub_quality,
        "integration_metadata": {
            "cross_ref_matrix": cross_ref,
            "requirements_coverage": req_coverage,
            "gap_report": gap_report,
        },
    }


# ── DI Framework Detection + Lane-Agnostic Generators ─────────────────────


def _detect_di_framework(
    all_files: List[Dict],
    plan: Optional[Dict] = None,
) -> str:
    """Detect DI framework from generated code signatures.

    Priority order:
    1. Code signatures in generated files (most reliable — what did the LLM produce?)
    2. BaseLane.di_framework override (optional per-lane declaration)
    3. target_stack.frameworks / .languages keywords (secondary signal)

    Returns one of: "tsyringe" | "spring" | "microsoft_di" | "nestjs" | "inversify" | "none"
    """
    import re as _re

    # Aggregate content by language for fast scanning
    java_content = ""
    csharp_content = ""
    ts_content = ""
    for f in all_files:
        lang = (f.get("language") or "").lower()
        c = f.get("content", "")
        if lang == "java":
            java_content += c
        elif lang in ("csharp", "c#"):
            csharp_content += c
        elif lang in ("typescript", "javascript"):
            ts_content += c

    # ── 1. Code-signature detection (highest priority) ──
    # NestJS: @Module decorator (check before tsyringe — both use @Injectable)
    if _re.search(r'@Module\s*\(', ts_content):
        return "nestjs"
    # InversifyJS: container.bind() is distinctive
    if _re.search(r'container\.bind\s*\(', ts_content):
        return "inversify"
    # tsyringe: @injectable()/@singleton() decorators
    if _re.search(r'@(?:injectable|singleton)\s*\(\s*\)', ts_content, _re.IGNORECASE):
        return "tsyringe"
    # Angular: @NgModule providers (no separate container needed)
    if _re.search(r'providers\s*:\s*\[', ts_content) and _re.search(r'@NgModule', ts_content):
        return "none"
    # Spring Boot (Java)
    if _re.search(r'@(?:Service|Component|Repository|Controller|Bean|Autowired)\b', java_content):
        return "spring"
    # .NET Core / ASP.NET Core
    if _re.search(r'services\.Add(?:Scoped|Transient|Singleton)', csharp_content):
        return "microsoft_di"
    if _re.search(r'public\s+class\s+\w+\s*:\s*I\w+', csharp_content):
        return "microsoft_di"

    # ── 2. BaseLane.di_framework property (optional override) ──
    if plan:
        try:
            from .lanes import LaneRegistry
            asset_strategies = plan.get("asset_strategies") or {}
            lane_ids: set = set()
            for spec in asset_strategies.values():
                if isinstance(spec, dict):
                    if spec.get("lane_id"):
                        lane_ids.add(spec["lane_id"])
                    for st in (spec.get("sub_types") or {}).values():
                        if isinstance(st, dict) and st.get("lane_id"):
                            lane_ids.add(st["lane_id"])
            if not lane_ids and plan.get("migration_lane_id"):
                lane_ids.add(plan["migration_lane_id"])
            for lid in lane_ids:
                lane = LaneRegistry.get_lane(lid)
                if lane and getattr(lane, "di_framework", None):
                    return lane.di_framework  # type: ignore[return-value]
        except Exception:
            pass

    # ── 3. target_stack hints (fallback) ──
    if plan:
        ts_data = plan.get("target_stack") or {}
        fw = [f.lower() for f in ts_data.get("frameworks", [])]
        lang = [ll.lower() for ll in ts_data.get("languages", [])]
        if any("spring" in f or "springboot" in f for f in fw):
            return "spring"
        if any(k in f for f in fw for k in ("dotnet", ".net", "aspnet")):
            return "microsoft_di"
        if "nestjs" in fw or "nest" in fw:
            return "nestjs"
        if any(k in f for f in fw for k in ("express", "koa", "fastify", "node")):
            return "tsyringe"
        if "java" in lang:
            return "spring"
        if "csharp" in lang or "c#" in lang:
            return "microsoft_di"
        if "typescript" in lang or "javascript" in lang:
            return "tsyringe"

    return "none"


def _gen_tsyringe_container(all_files: List[Dict]) -> Optional[Dict]:
    """Generate a complete tsyringe container.ts from scratch.

    Uses line-by-line lookahead to match each @injectable()/@singleton()
    decorator to the class declared in the next 1-5 lines.  This avoids
    the "collect all classes in decorated file" bug from the old patch approach.
    """
    import re as _re
    _dec_re = _re.compile(r'@(?:injectable|singleton)\s*\(\s*\)', _re.IGNORECASE)
    _cls_re = _re.compile(r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)')

    injectable: Dict[str, str] = {}   # {ClassName: import_path}
    for f in all_files:
        if (f.get("language") or "").lower() not in ("typescript", "javascript"):
            continue
        fp = f.get("file_path", "")
        lines = f.get("content", "").splitlines()
        for i, line in enumerate(lines):
            if _dec_re.search(line):
                # Look ahead up to 5 lines for the decorated class name
                for j in range(i + 1, min(i + 6, len(lines))):
                    m = _cls_re.search(lines[j])
                    if m:
                        cls = m.group(1)
                        # Build relative import path without extension
                        imp = fp
                        if imp.startswith("src/"):
                            imp = "./" + imp[4:]
                        for ext in (".ts", ".js"):
                            if imp.endswith(ext):
                                imp = imp[:-len(ext)]
                                break
                        injectable[cls] = imp
                        break   # one class per decorator occurrence

    if not injectable:
        return None

    out: List[str] = [
        "import 'reflect-metadata';",
        "import { container } from 'tsyringe';",
        "",
    ]
    for cls, path in sorted(injectable.items(), key=lambda x: x[1]):
        out.append(f"import {{ {cls} }} from '{path}';")
    out += [
        "",
        "// DI registrations — auto-generated by CodeLoom post-transform pass",
    ]
    for cls in sorted(injectable):
        out.append(f"container.registerSingleton({cls}, {cls});")
    out += ["", "export { container };", ""]

    logger.info(
        "Post-transform [tsyringe]: generated container.ts with %d bindings: %s",
        len(injectable),
        ", ".join(sorted(injectable)[:10]),
    )
    return {"file_path": "src/container.ts", "language": "typescript",
            "content": "\n".join(out)}


def _gen_spring_bootstrap(all_files: List[Dict]) -> Optional[Dict]:
    """Verify @SpringBootApplication exists; generate Application.java if missing.

    Spring Boot uses classpath component scanning — no explicit container file
    is needed.  We just ensure the entry-point annotation is present.
    """
    import re as _re
    for f in all_files:
        if (f.get("language") or "").lower() == "java":
            if _re.search(r'@SpringBootApplication', f.get("content", "")):
                return None  # Already present — nothing to do

    # Not found — generate a minimal Application.java
    pkg = "com.example.app"
    for f in all_files:
        if (f.get("language") or "").lower() == "java":
            m = _re.search(r'^package\s+([\w.]+)\s*;', f.get("content", ""), _re.MULTILINE)
            if m:
                parts = m.group(1).split(".")
                pkg = ".".join(parts[:3]) if len(parts) >= 3 else m.group(1)
                break

    content = (
        f"package {pkg};\n\n"
        "import org.springframework.boot.SpringApplication;\n"
        "import org.springframework.boot.autoconfigure.SpringBootApplication;\n\n"
        "@SpringBootApplication\n"
        "public class Application {\n"
        "    public static void main(String[] args) {\n"
        "        SpringApplication.run(Application.class, args);\n"
        "    }\n"
        "}\n"
    )
    pkg_path = pkg.replace(".", "/")
    logger.info("Post-transform [spring]: generated Application.java (package=%s)", pkg)
    return {"file_path": f"src/main/java/{pkg_path}/Application.java",
            "language": "java", "content": content}


def _gen_dotnet_program(all_files: List[Dict]) -> Optional[Dict]:
    """Generate Program.cs with services.AddScoped<IFoo, Foo> for all class:IFoo pairs.

    Scans C# files for public class Foo : IFoo declarations.  Skips classes
    that are already registered in an existing Startup/Program file.
    """
    import re as _re
    _pair_re = _re.compile(r'public\s+class\s+(\w+)\s*:\s*(I\w+)')
    _existing_re = _re.compile(
        r'services\.Add(?:Scoped|Transient|Singleton)\s*<\s*\w+\s*,\s*(\w+)\s*>'
    )
    registered: set = set()
    pairs: list = []
    for f in all_files:
        if (f.get("language") or "").lower() not in ("csharp", "c#"):
            continue
        c = f.get("content", "")
        if "ConfigureServices" in c or "builder.Services" in c:
            for m in _existing_re.finditer(c):
                registered.add(m.group(1))
        for m in _pair_re.finditer(c):
            cls, iface = m.group(1), m.group(2)
            if iface.startswith("I") and len(iface) > 1:
                pairs.append((iface, cls))

    new_pairs = list({(i, c) for i, c in pairs if c not in registered})
    if not new_pairs:
        return None

    reg = "\n".join(
        f"builder.Services.AddScoped<{i}, {c}>();"
        for i, c in sorted(new_pairs, key=lambda x: x[1])
    )
    content = (
        "// Program.cs — auto-generated by CodeLoom post-transform pass\n"
        "var builder = WebApplication.CreateBuilder(args);\n\n"
        f"// Service registrations\n{reg}\n\n"
        "builder.Services.AddControllers();\n"
        "builder.Services.AddEndpointsApiExplorer();\n"
        "builder.Services.AddSwaggerGen();\n\n"
        "var app = builder.Build();\n"
        "app.UseHttpsRedirection();\n"
        "app.UseAuthorization();\n"
        "app.MapControllers();\n"
        "app.Run();\n"
    )
    logger.info("Post-transform [microsoft_di]: generated Program.cs with %d registrations",
                len(new_pairs))
    return {"file_path": "src/Program.cs", "language": "csharp", "content": content}


def _gen_nestjs_module(all_files: List[Dict]) -> Optional[Dict]:
    """Generate AppModule.ts importing all detected @Module()-decorated modules.

    Used for NestJS migrations — NestJS uses module-based DI, so a root
    AppModule that imports all feature modules is the equivalent of a container.
    """
    import re as _re
    _mod_re = _re.compile(r'@Module\s*\(')
    _cls_re = _re.compile(r'(?:export\s+)?class\s+(\w+Module)\b')
    modules: Dict[str, str] = {}   # {ModuleName: import_path}
    for f in all_files:
        if (f.get("language") or "").lower() not in ("typescript", "javascript"):
            continue
        c = f.get("content", "")
        if not _mod_re.search(c):
            continue
        fp = f.get("file_path", "")
        imp = fp
        if imp.startswith("src/"):
            imp = "./" + imp[4:]
        for ext in (".ts", ".js"):
            if imp.endswith(ext):
                imp = imp[:-len(ext)]
                break
        for m in _cls_re.finditer(c):
            modules[m.group(1)] = imp

    # Return None if an AppModule already exists (nothing to do)
    if any("app" in name.lower() for name in modules):
        return None
    if not modules:
        return None

    imports_str = "\n".join(
        f"import {{ {n} }} from '{p}';" for n, p in sorted(modules.items(), key=lambda x: x[1])
    )
    mod_list = ", ".join(sorted(modules.keys()))
    content = (
        "import { Module } from '@nestjs/common';\n"
        f"{imports_str}\n\n"
        "@Module({\n"
        f"  imports: [{mod_list}],\n"
        "})\n"
        "export class AppModule {}\n"
    )
    logger.info("Post-transform [nestjs]: generated AppModule.ts importing %d modules",
                len(modules))
    return {"file_path": "src/app.module.ts", "language": "typescript", "content": content}


# DI generator dispatch table — add new frameworks here without touching any other code
_DI_GENERATORS: Dict[str, Any] = {
    "tsyringe":     _gen_tsyringe_container,
    "spring":       _gen_spring_bootstrap,
    "microsoft_di": _gen_dotnet_program,
    "nestjs":       _gen_nestjs_module,
    # "inversify": _gen_inversify_container,  # Add when InversifyJS lane is implemented
}


def _post_transform_consolidate(
    all_mvp_outputs: Dict[str, List[Dict]],
    integration_files: List[Dict],
    target_stack: Dict,
    plan: Optional[Dict] = None,
) -> List[Dict]:
    """Deterministic post-pass: generate DI container + patch package.json.

    DI generation is framework-agnostic — detects the target DI framework
    from code signatures in generated files, then dispatches to the correct
    generator.  New frameworks: add a generator function + entry to _DI_GENERATORS.

    package.json patching is unchanged — scans all TS/JS files for external
    imports and adds any missing packages to dependencies.
    """
    import re as _re
    import json as _json

    # Flatten all files for detection and generation
    all_files: List[Dict] = []
    for files in all_mvp_outputs.values():
        all_files.extend(files)
    all_files.extend(integration_files)

    amended = list(integration_files)

    # ── DI container generation (lane-agnostic dispatch) ──
    effective_plan = plan or {"target_stack": target_stack}
    di_fw = _detect_di_framework(all_files, effective_plan)
    logger.info("Post-transform: detected DI framework = %s", di_fw)

    generator = _DI_GENERATORS.get(di_fw)
    if generator:
        new_file = generator(all_files)
        if new_file:
            fp = new_file["file_path"]
            # Replace existing file with the same path, or append
            idx = next((i for i, f in enumerate(amended)
                        if f.get("file_path") == fp), None)
            if idx is not None:
                amended[idx] = new_file
                logger.info("Post-transform [%s]: replaced %s", di_fw, fp)
            else:
                amended.append(new_file)
                logger.info("Post-transform [%s]: appended new %s", di_fw, fp)
        else:
            logger.info("Post-transform [%s]: no container action needed (already complete)", di_fw)
    else:
        logger.info("Post-transform: no DI generator registered for '%s' — skipping", di_fw)

    # ── package.json: add missing external dependencies (unchanged) ──
    import_re = _re.compile(r'''import\s+.*?\s+from\s+['"]([^'"./][^'"]*?)['"]''')
    known_builtins = {
        'path', 'fs', 'os', 'util', 'crypto', 'http', 'https',
        'stream', 'events', 'url', 'buffer', 'assert', 'child_process',
    }
    external_pkgs: set = set()
    for f in all_files:
        if (f.get("language") or "").lower() in ("typescript", "javascript"):
            for m in import_re.finditer(f.get("content", "")):
                raw = m.group(1)
                pkg = '/'.join(raw.split('/')[:2]) if raw.startswith('@') else raw.split('/')[0]
                if pkg not in known_builtins:
                    external_pkgs.add(pkg)

    pkg_idx = next(
        (i for i, f in enumerate(amended) if f.get('file_path', '').endswith('package.json')),
        None,
    )
    if pkg_idx is not None and external_pkgs:
        try:
            pkg_data = _json.loads(amended[pkg_idx]['content'])
            have = (
                set(pkg_data.get('dependencies', {}).keys())
                | set(pkg_data.get('devDependencies', {}).keys())
            )
            missing = external_pkgs - have
            if missing:
                pkg_data.setdefault('dependencies', {})
                for p in sorted(missing):
                    pkg_data['dependencies'][p] = '*'
                amended[pkg_idx] = dict(amended[pkg_idx])
                amended[pkg_idx]['content'] = _json.dumps(pkg_data, indent=2)
                logger.info(
                    "Post-transform: added %d missing packages to package.json: %s",
                    len(missing),
                    ', '.join(sorted(missing)[:15]),
                )
        except Exception as exc:
            logger.warning("Post-transform: could not patch package.json: %s", exc)

    return amended
