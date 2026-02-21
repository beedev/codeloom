"""LLM-generated behavioral diagram generation.

Uses the existing LLM infrastructure (Settings.llm) to generate PlantUML
for Sequence, Use Case, Activity, and Deployment diagrams.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text

from . import prompts

logger = logging.getLogger(__name__)

# Diagram types handled by this module
BEHAVIORAL_TYPES = {"sequence", "usecase", "activity", "deployment"}


def generate_behavioral_diagram(
    diagram_type: str,
    mvp_context: Dict[str, Any],
    db,
    project_id: str,
) -> Dict[str, str]:
    """Generate a behavioral PlantUML diagram via LLM.

    Args:
        diagram_type: One of 'sequence', 'usecase', 'activity', 'deployment'
        mvp_context: MVP dict with name, unit_ids, file_ids, analysis_output, etc.
        db: DatabaseManager instance
        project_id: Project UUID string

    Returns:
        {"puml": "...", "title": "...", "description": "..."}
    """
    if diagram_type not in BEHAVIORAL_TYPES:
        raise ValueError(f"Unknown behavioral diagram type: {diagram_type}")

    # Gather context data
    unit_ids = mvp_context.get("unit_ids", [])
    signatures = _get_unit_signatures(db, project_id, unit_ids)
    call_paths = _get_call_paths(db, project_id, unit_ids)
    source_snippets = _get_source_snippets(db, project_id, unit_ids, budget=4000)

    mvp_name = mvp_context.get("name", "Unnamed MVP")

    # Build the appropriate prompt
    if diagram_type == "sequence":
        prompt = prompts.sequence_diagram_prompt(
            mvp_name, signatures, call_paths, source_snippets,
        )
    elif diagram_type == "usecase":
        functional_ctx = _build_functional_context(mvp_context)
        prompt = prompts.usecase_diagram_prompt(
            mvp_name, signatures, functional_ctx,
        )
    elif diagram_type == "activity":
        # Enrich with RAPTOR L1 summaries for semantic understanding
        raptor_ctx = _get_raptor_summaries(db, project_id, unit_ids)
        prompt = prompts.activity_diagram_prompt(
            mvp_name, source_snippets, call_paths, raptor_summaries=raptor_ctx,
        )
    elif diagram_type == "deployment":
        target_stack = _format_target_stack(mvp_context)
        arch_ctx = _build_architecture_context(mvp_context, signatures, call_paths)
        detected_infra = mvp_context.get("detected_infra", "")
        prompt = prompts.deployment_diagram_prompt(
            mvp_name, target_stack, arch_ctx, detected_infra,
        )
    else:
        raise ValueError(f"Unhandled diagram type: {diagram_type}")

    # Call LLM
    from ..migration.phases import _call_llm
    response_text = _call_llm(prompt, context_type="generation")

    # Extract PlantUML from response
    puml = _extract_plantuml(response_text)
    if not puml:
        logger.warning("LLM response did not contain valid PlantUML for %s diagram", diagram_type)
        puml = f"@startuml\nnote \"Generation failed — no valid PlantUML in LLM response\" as N\n@enduml"

    # Strip swim lane syntax (|Name|) from activity diagrams — causes syntax errors
    if diagram_type == "activity":
        puml = _strip_swimlanes(puml)

    # Inject clean theme into LLM-generated diagram
    puml = _inject_theme(puml)

    title = f"{diagram_type.replace('usecase', 'Use Case').title()} Diagram: {mvp_name}"

    return {
        "puml": puml,
        "title": title,
        "description": f"LLM-generated {diagram_type} diagram for {mvp_name}",
    }


_BEHAVIORAL_THEME = """
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontSize 12
skinparam roundCorner 8
skinparam sequence {
  ArrowColor #495057
  LifeLineBorderColor #6C757D
  ParticipantBackgroundColor #F8F9FA
  ParticipantBorderColor #495057
  ParticipantFontColor #212529
  BoxBackgroundColor #F1F3F5
  BoxBorderColor #ADB5BD
  GroupBackgroundColor #E9ECEF
  DividerBackgroundColor #DEE2E6
}
skinparam activity {
  BackgroundColor #F8F9FA
  BorderColor #495057
  FontColor #212529
  ArrowColor #495057
  DiamondBackgroundColor #E9ECEF
  DiamondBorderColor #495057
}
skinparam usecase {
  BackgroundColor #F8F9FA
  BorderColor #495057
  ActorBorderColor #495057
  ArrowColor #6C757D
}
skinparam node {
  BackgroundColor #F8F9FA
  BorderColor #495057
  FontColor #212529
}
""".strip()


def _strip_swimlanes(puml: str) -> str:
    """Remove PlantUML swim lane lines (|Name|) from activity diagrams.

    Swim lanes cause syntax errors when mixed with fork/join, if/endif,
    or when placed after ``start``.  Replace them with a comment so the
    diagram still renders.
    """
    lines = puml.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\|[^|]+\|$", stripped):
            # Convert swimlane to a note-like label or just skip
            lane_name = stripped.strip("|")
            result.append(f"' -- {lane_name} --")
        else:
            result.append(line)
    return "\n".join(result)


def _inject_theme(puml: str) -> str:
    """Inject a clean theme into PlantUML generated by LLM.

    Replaces any existing skinparam blocks with our standard theme,
    inserted right after @startuml.
    """
    # Remove existing skinparam lines the LLM may have added
    lines = puml.split("\n")
    filtered = []
    skip_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("skinparam ") and "{" in stripped:
            skip_block = True
            continue
        if skip_block:
            if stripped == "}":
                skip_block = False
            continue
        if stripped.startswith("skinparam ") and "{" not in stripped:
            continue  # single-line skinparam
        filtered.append(line)

    # Inject our theme right after @startuml
    result = []
    for line in filtered:
        result.append(line)
        if line.strip() == "@startuml":
            result.append(_BEHAVIORAL_THEME)
            result.append("")
    return "\n".join(result)


def _extract_plantuml(text: str) -> Optional[str]:
    """Extract PlantUML code from LLM response text.

    Handles responses that wrap in markdown code blocks or include extra text.
    """
    # Try direct @startuml...@enduml extraction
    match = re.search(r"@startuml.*?@enduml", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Try extracting from markdown code blocks
    match = re.search(r"```(?:plantuml|puml|uml)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        if inner.startswith("@startuml"):
            return inner
        return f"@startuml\n{inner}\n@enduml"

    return None


def _get_unit_signatures(db, project_id: str, unit_ids: List[str], limit: int = 60) -> str:
    """Get formatted unit signatures for prompt context."""
    if not unit_ids:
        return "(no units)"
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids[:limit]]

    with db.get_session() as session:
        rows = session.execute(text("""
            SELECT cu.qualified_name, cu.unit_type, cu.signature, cf.file_path
            FROM code_units cu
            JOIN code_files cf ON cu.file_id = cf.file_id
            WHERE cu.unit_id = ANY(:uids)
            ORDER BY cu.qualified_name
        """), {"uids": uids})
        units = [dict(r._mapping) for r in rows.fetchall()]

    lines = []
    for u in units:
        sig = u.get("signature") or u["qualified_name"]
        lines.append(f"- [{u['unit_type']}] {sig}  ({u.get('file_path', '')})")

    return "\n".join(lines) if lines else "(no signatures)"


def _get_call_paths(db, project_id: str, unit_ids: List[str], limit: int = 100) -> str:
    """Get formatted call paths for prompt context."""
    if not unit_ids:
        return "(no call paths)"
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    with db.get_session() as session:
        rows = session.execute(text("""
            SELECT
                su.qualified_name AS caller,
                tu.qualified_name AS callee,
                ce.edge_type
            FROM code_edges ce
            JOIN code_units su ON ce.source_unit_id = su.unit_id
            JOIN code_units tu ON ce.target_unit_id = tu.unit_id
            WHERE ce.project_id = :pid
              AND ce.edge_type IN ('calls', 'imports')
              AND ce.source_unit_id = ANY(:uids)
            ORDER BY ce.edge_type, su.qualified_name
            LIMIT :lim
        """), {"pid": pid, "uids": uids, "lim": limit})
        edges = [dict(r._mapping) for r in rows.fetchall()]

    lines = []
    for e in edges:
        arrow = "→" if e["edge_type"] == "calls" else "⇢"
        lines.append(f"- {e['caller']} {arrow} {e['callee']}")

    return "\n".join(lines) if lines else "(no call paths)"


def _get_source_snippets(db, project_id: str, unit_ids: List[str], budget: int = 4000) -> str:
    """Get source code snippets within character budget."""
    if not unit_ids:
        return "(no source code)"
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    with db.get_session() as session:
        rows = session.execute(text("""
            SELECT cu.qualified_name, cu.source, cu.language
            FROM code_units cu
            WHERE cu.unit_id = ANY(:uids) AND cu.source IS NOT NULL
            ORDER BY cu.qualified_name
        """), {"uids": uids})
        units = [dict(r._mapping) for r in rows.fetchall()]

    char_budget = budget * 4  # ~4 chars per token
    used = 0
    snippets = []

    for u in units:
        source = u.get("source", "")
        if not source:
            continue
        entry = f"### {u['qualified_name']} ({u.get('language', '')})\n```\n{source}\n```\n"
        if used + len(entry) > char_budget:
            break
        snippets.append(entry)
        used += len(entry)

    return "\n".join(snippets) if snippets else "(no source code available)"


def _build_functional_context(mvp_context: Dict) -> str:
    """Build functional context from MVP analysis output and metadata."""
    parts = []

    desc = mvp_context.get("description")
    if desc:
        parts.append(f"**Description**: {desc}")

    analysis = mvp_context.get("analysis_output")
    if analysis and isinstance(analysis, dict):
        output = analysis.get("output", "")
        if output:
            # Truncate to keep prompt manageable
            parts.append(f"**Deep Analysis**:\n{output[:3000]}")

    metrics = mvp_context.get("metrics", {})
    if metrics:
        parts.append(
            f"**Metrics**: {metrics.get('size', '?')} units, "
            f"cohesion={metrics.get('cohesion', '?')}, "
            f"coupling={metrics.get('coupling', '?')}"
        )

    return "\n\n".join(parts) if parts else "(no functional context available)"


def _format_target_stack(mvp_context: Dict) -> str:
    """Format target stack info from plan context."""
    stack = mvp_context.get("target_stack")
    if not stack:
        return "(target stack not specified)"

    parts = []
    if isinstance(stack, dict):
        langs = stack.get("languages", [])
        frameworks = stack.get("frameworks", [])
        versions = stack.get("versions", {})

        if langs:
            parts.append(f"**Languages**: {', '.join(langs)}")
        if frameworks:
            parts.append(f"**Frameworks**: {', '.join(frameworks)}")
        if versions:
            ver_strs = [f"{k}: {v}" for k, v in versions.items()]
            parts.append(f"**Versions**: {', '.join(ver_strs)}")
    else:
        parts.append(str(stack))

    return "\n".join(parts)


def _get_raptor_summaries(db, project_id: str, unit_ids: List[str], budget: int = 6000) -> str:
    """Load RAPTOR L1 summaries for files that contain the MVP's units.

    L1 summaries are per-file semantic descriptions generated by RAPTOR
    during ingestion.  They give the LLM real behavioral understanding
    of what each file does — critical for declarative MVPs that have
    no call-tree data.

    Args:
        db: DatabaseManager instance
        project_id: Project UUID string
        unit_ids: Code unit IDs belonging to the MVP
        budget: Character budget for the combined summaries

    Returns:
        Formatted string of RAPTOR summaries, or empty string.
    """
    if not unit_ids:
        return ""

    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uids = [UUID(u) if isinstance(u, str) else u for u in unit_ids]

    try:
        with db.get_session() as session:
            # Step 1: Get distinct file_ids for this MVP's units
            file_rows = session.execute(text("""
                SELECT DISTINCT cu.file_id::text, cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
            """), {"uids": uids}).fetchall()

            if not file_rows:
                return ""

            file_ids = [r[0] for r in file_rows]
            file_paths = {r[0]: r[1] for r in file_rows}

            # Step 2: Fetch L1 summaries whose source_id matches these file_ids
            summary_rows = session.execute(text("""
                SELECT "text",
                       metadata_::jsonb->>'source_id' AS source_id
                FROM data_embeddings
                WHERE metadata_::jsonb->>'project_id' = :pid
                  AND metadata_::jsonb->>'node_type' = 'raptor_summary'
                  AND (metadata_::jsonb->>'tree_level')::int = 1
                  AND metadata_::jsonb->>'source_id' = ANY(:fids)
                ORDER BY metadata_::jsonb->>'source_id'
            """), {"pid": str(pid), "fids": file_ids}).fetchall()

            if not summary_rows:
                return ""

            # Step 3: Format within budget
            char_budget = budget * 4  # ~4 chars per token
            used = 0
            parts = []
            for row in summary_rows:
                fpath = file_paths.get(row[1], row[1] or "unknown")
                # Use just the filename for brevity
                short_path = fpath.rsplit("/", 1)[-1] if fpath else "unknown"
                entry = f"### {short_path}\n{row[0]}\n"
                if used + len(entry) > char_budget:
                    break
                parts.append(entry)
                used += len(entry)

            return "\n".join(parts)

    except Exception as e:
        logger.warning("RAPTOR summary fetch failed: %s", e)
        return ""


def _build_architecture_context(mvp_context: Dict, signatures: str, call_paths: str) -> str:
    """Build architecture context for deployment diagram."""
    parts = [f"**MVP Units**:\n{signatures}", f"**Call Paths**:\n{call_paths}"]

    desc = mvp_context.get("description")
    if desc:
        parts.insert(0, f"**Description**: {desc}")

    return "\n\n".join(parts)
