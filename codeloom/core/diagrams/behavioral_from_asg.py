"""Deterministic behavioral diagram generation from ASG call tree data.

Generates Sequence, Activity, and Use Case diagrams directly from
ChainTracer's call tree — no LLM calls needed. Every arrow, participant,
and activity corresponds to a real code path in the ASG.

Deployment diagrams remain LLM-assisted (see behavioral.py) since
infrastructure topology can't be deterministically inferred from code alone.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..understanding.chain_tracer import ChainTracer
from ..understanding.models import CallTreeNode, EntryPoint, EntryPointType

logger = logging.getLogger(__name__)

# Reuse the behavioral theme from behavioral.py
_THEME = """
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

# ── Sequence Diagram ─────────────────────────────────────────────────

_MAX_SEQUENCE_INTERACTIONS = 25
_MAX_SEQUENCE_DEPTH = 8


def generate_sequence_diagram(
    call_trees: List[CallTreeNode],
    mvp_name: str,
) -> Dict[str, str]:
    """Generate a PlantUML sequence diagram from call tree data.

    Walks the call tree depth-first, emitting participants and call arrows
    that correspond to real code paths.

    Args:
        call_trees: List of CallTreeNode roots (one per entry point)
        mvp_name: Name of the MVP for the title

    Returns:
        {"puml": "...", "title": "...", "description": "..."}
    """
    if not call_trees:
        return _empty_result("Sequence", mvp_name)

    # Collect unique participants and interactions
    participants: Dict[str, str] = {}  # qualified_name -> short alias
    interactions: List[Tuple[str, str, str, str]] = []  # (from_qn, to_qn, method_name, edge_type)

    for tree in call_trees:
        _walk_sequence(tree, participants, interactions, depth=0, visited=set())

    if not interactions:
        return _empty_result("Sequence", mvp_name)

    # Build PlantUML
    lines = ["@startuml", _THEME, ""]
    lines.append(f"title {_safe(mvp_name)} — Request Flow")
    lines.append("")

    # Emit participants in discovery order
    for qn, alias in participants.items():
        short_name = _class_name(qn)
        ptype = "participant"
        lines.append(f'{ptype} "{short_name}" as {alias}')
    lines.append("")

    # Emit interactions with activate/deactivate
    active_stack: List[str] = []
    for i, (src_qn, tgt_qn, method, edge_type) in enumerate(interactions[:_MAX_SEQUENCE_INTERACTIONS]):
        src_alias = participants.get(src_qn)
        tgt_alias = participants.get(tgt_qn)
        if not src_alias or not tgt_alias:
            continue

        arrow = "->" if edge_type == "calls" else "->>"
        label = _short_method(method)
        lines.append(f"{src_alias} {arrow} {tgt_alias}: {label}")
        lines.append(f"activate {tgt_alias}")
        active_stack.append(tgt_alias)

    # Deactivate in reverse order
    for alias in reversed(active_stack):
        lines.append(f"deactivate {alias}")

    lines.append("")
    lines.append("@enduml")

    return {
        "puml": "\n".join(lines),
        "title": f"Sequence Diagram: {mvp_name}",
        "description": f"Call flow for {mvp_name} ({len(interactions)} interactions from ASG)",
    }


def _walk_sequence(
    node: CallTreeNode,
    participants: Dict[str, str],
    interactions: List[Tuple[str, str, str, str]],
    depth: int,
    visited: Optional[Set[str]] = None,
) -> None:
    """Recursively walk the call tree, collecting participants and interactions."""
    if depth > _MAX_SEQUENCE_DEPTH:
        return
    if len(interactions) >= _MAX_SEQUENCE_INTERACTIONS:
        return
    if visited is None:
        visited = set()
    if node.unit_id in visited:
        return
    visited.add(node.unit_id)

    # Resolve to class-level participant
    parent_qn = _to_class_qn(node)
    if parent_qn not in participants:
        participants[parent_qn] = f"P{len(participants)}"

    for child in node.children:
        child_class_qn = _to_class_qn(child)
        if child_class_qn not in participants:
            participants[child_class_qn] = f"P{len(participants)}"

        interactions.append((
            parent_qn,
            child_class_qn,
            child.name,
            child.edge_type,
        ))

        _walk_sequence(child, participants, interactions, depth + 1, visited)


# ── Activity Diagram ─────────────────────────────────────────────────

_MAX_ACTIVITIES = 25


def generate_activity_diagram(
    call_trees: List[CallTreeNode],
    mvp_name: str,
) -> Dict[str, str]:
    """Generate a PlantUML activity diagram from call tree data.

    Maps the call tree to an activity flow:
    - Sequential calls become sequential activities
    - Multiple children at same level become fork/join
    - Leaf nodes are terminal activities

    Args:
        call_trees: List of CallTreeNode roots
        mvp_name: Name of the MVP

    Returns:
        {"puml": "...", "title": "...", "description": "..."}
    """
    if not call_trees:
        return _empty_result("Activity", mvp_name)

    lines = ["@startuml", _THEME, ""]
    lines.append(f"title {_safe(mvp_name)} — Activity Flow")
    lines.append("")
    lines.append("start")
    lines.append("")

    activity_count = [0]  # mutable counter
    seen_classes: Set[str] = set()

    for tree in call_trees:
        _walk_activity(tree, lines, activity_count, seen_classes, depth=0, visited=set())

    lines.append("")
    lines.append("stop")
    lines.append("@enduml")

    return {
        "puml": "\n".join(lines),
        "title": f"Activity Diagram: {mvp_name}",
        "description": f"Control flow for {mvp_name} ({activity_count[0]} activities from ASG)",
    }


def _walk_activity(
    node: CallTreeNode,
    lines: List[str],
    count: List[int],
    seen_classes: Set[str],
    depth: int,
    visited: Optional[Set[str]] = None,
) -> None:
    """Recursively emit activity nodes from call tree.

    NOTE: We intentionally avoid PlantUML swim lane syntax (``|name|``)
    because swim lane changes are illegal inside fork/join and if/endif
    blocks, and inconsistent lane usage (some activities in lanes, some
    not) also triggers syntax errors.  Instead, the first activity from
    each class/module gets a ``ClassName.`` prefix in its label.
    """
    if count[0] >= _MAX_ACTIVITIES:
        return
    if depth > 8:
        return
    if visited is None:
        visited = set()
    if node.unit_id in visited:
        return
    visited.add(node.unit_id)

    class_name = _class_name(_to_class_qn(node))

    # Emit the activity — prefix with class name on first occurrence
    activity_label = _safe(node.name)
    if node.unit_type in ("method", "function"):
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            lines.append(f":{_safe(class_name)}.{activity_label};")
        else:
            lines.append(f":{activity_label};")
        count[0] += 1

    children = node.children
    if not children:
        return

    if len(children) == 1:
        # Sequential flow
        _walk_activity(children[0], lines, count, seen_classes, depth + 1, visited)
    else:
        # Multiple children — use fork/join for parallel or if/else for sequential
        if _looks_like_branching(node):
            # Decision point
            first_child = children[0]
            lines.append(f"if ({_safe_condition(first_child.name)}) then (yes)")
            before = count[0]
            _walk_activity(first_child, lines, count, seen_classes, depth + 1, visited)
            if count[0] == before:  # empty branch — add placeholder
                lines.append(f":{_safe(first_child.name)};")
                count[0] += 1
            for alt in children[1:]:
                lines.append(f"elseif ({_safe_condition(alt.name)}) then")
                before = count[0]
                _walk_activity(alt, lines, count, seen_classes, depth + 1, visited)
                if count[0] == before:
                    lines.append(f":{_safe(alt.name)};")
                    count[0] += 1
            lines.append("endif")
        else:
            # Sequential calls — show all
            for child in children:
                _walk_activity(child, lines, count, seen_classes, depth + 1, visited)


def _looks_like_branching(node: CallTreeNode) -> bool:
    """Heuristic: if children have different class origins, likely branching."""
    if len(node.children) < 2:
        return False
    classes = {_to_class_qn(c) for c in node.children}
    # If all children go to the same class, it's likely sequential calls, not branching
    return len(classes) > 1


# ── Use Case Diagram ─────────────────────────────────────────────────

_MAX_USECASES = 15

# Map entry point types to actor names
_ACTOR_MAP = {
    EntryPointType.HTTP_ENDPOINT: "User",
    EntryPointType.MESSAGE_HANDLER: "Message Queue",
    EntryPointType.SCHEDULED_TASK: "Scheduler",
    EntryPointType.CLI_COMMAND: "CLI User",
    EntryPointType.EVENT_LISTENER: "Event Bus",
    EntryPointType.STARTUP_HOOK: "System",
    EntryPointType.PUBLIC_API: "Client",
    EntryPointType.UNKNOWN: "External",
}


def generate_usecase_diagram(
    entry_points: List[EntryPoint],
    mvp_name: str,
    mvp_unit_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate a PlantUML use case diagram from detected entry points.

    Groups entry points by actor type, creating meaningful use case
    groupings instead of dumping every method.

    Args:
        entry_points: Detected entry points from ChainTracer
        mvp_name: Name of the MVP
        mvp_unit_ids: If provided, filter to only entry points within the MVP

    Returns:
        {"puml": "...", "title": "...", "description": "..."}
    """
    if not entry_points:
        return _empty_result("Use Case", mvp_name)

    # Filter to MVP scope if unit_ids provided
    if mvp_unit_ids:
        mvp_set = set(mvp_unit_ids)
        entry_points = [ep for ep in entry_points if ep.unit_id in mvp_set]

    if not entry_points:
        return _empty_result("Use Case", mvp_name)

    lines = ["@startuml", _THEME, ""]
    lines.append("left to right direction")
    lines.append("")

    # Group by actor type
    by_actor: Dict[str, List[EntryPoint]] = defaultdict(list)
    for ep in entry_points:
        actor_name = _ACTOR_MAP.get(ep.entry_type, "External")
        by_actor[actor_name].append(ep)

    # Emit actors
    actor_aliases: Dict[str, str] = {}
    for i, actor_name in enumerate(by_actor.keys()):
        alias = f"A{i}"
        actor_aliases[actor_name] = alias
        lines.append(f'actor "{actor_name}" as {alias}')
    lines.append("")

    # System boundary
    lines.append(f'rectangle "{_safe(mvp_name)}" {{')

    uc_aliases: Dict[str, str] = {}
    uc_idx = 0
    for actor_name, eps in by_actor.items():
        for ep in eps[:_MAX_USECASES]:
            alias = f"UC{uc_idx}"
            uc_aliases[ep.unit_id] = alias
            # Clean up method name into a verb phrase
            uc_label = _to_use_case_label(ep.name)
            lines.append(f'  usecase "{uc_label}" as {alias}')
            uc_idx += 1

    lines.append("}")
    lines.append("")

    # Connect actors to their use cases
    for actor_name, eps in by_actor.items():
        actor_alias = actor_aliases[actor_name]
        for ep in eps[:_MAX_USECASES]:
            uc_alias = uc_aliases.get(ep.unit_id)
            if uc_alias:
                lines.append(f"{actor_alias} --> {uc_alias}")

    lines.append("")
    lines.append("@enduml")

    return {
        "puml": "\n".join(lines),
        "title": f"Use Case Diagram: {mvp_name}",
        "description": f"Entry points for {mvp_name} ({len(entry_points)} use cases from ASG)",
    }


# ── Shared Helpers ───────────────────────────────────────────────────

def _to_class_qn(node: CallTreeNode) -> str:
    """Resolve a node to its class-level qualified name.

    For methods like 'com.example.UserService.findUser', returns 'com.example.UserService'.
    For top-level functions, returns the module name from file_path.
    """
    qn = node.qualified_name
    if node.unit_type in ("method", "constructor"):
        parts = qn.rsplit(".", 1)
        if len(parts) > 1:
            return parts[0]
    if node.unit_type == "function":
        # Use file path as the class-level grouping
        fp = node.file_path
        if fp:
            return fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return qn


def _class_name(qualified_name: str) -> str:
    """Extract short class name from qualified name."""
    return qualified_name.rsplit(".", 1)[-1] if "." in qualified_name else qualified_name


def _short_method(name: str) -> str:
    """Clean up a method name for diagram labels."""
    # Remove common prefixes
    clean = name.replace("__", "").strip("_")
    # Truncate long names
    if len(clean) > 30:
        clean = clean[:27] + "..."
    return _safe(clean) if clean else "call"


def _to_use_case_label(method_name: str) -> str:
    """Convert a method name into a use-case-style label.

    'getUserById' -> 'Get User By Id'
    'handle_payment' -> 'Handle Payment'
    """
    # Split camelCase
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", method_name)
    # Split snake_case
    spaced = spaced.replace("_", " ")
    # Title case, limit length
    label = spaced.strip().title()
    if len(label) > 35:
        label = label[:32] + "..."
    return label or "Action"


def _safe_condition(text: str) -> str:
    """Sanitize a method name for use inside PlantUML ``if (...)`` conditions.

    Strips parentheses and other characters that would create unbalanced
    delimiters inside the ``if (condition?)`` syntax.
    """
    cleaned = re.sub(r'[()<>{}\[\]@#$%^&*;\'"/\\|~`]', "", text)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned[:30] if cleaned else "condition"


def _safe(text: str) -> str:
    """Sanitize text for PlantUML — remove characters that break syntax."""
    cleaned = re.sub(r'[<>{}@#$%^&*;\'"/\\|~`]', "", text)
    cleaned = cleaned.replace("\n", " ").replace("\r", "").strip()
    return re.sub(r"\s+", " ", cleaned) or "unnamed"


def _empty_result(diagram_type: str, mvp_name: str) -> Dict[str, str]:
    """Return an empty diagram result."""
    return {
        "puml": f"""@startuml
{_THEME}

note "No call tree data available for {_safe(mvp_name)}.\\nRun Deep Understanding analysis first." as N
@enduml""",
        "title": f"{diagram_type} Diagram: {mvp_name}",
        "description": f"No call tree data available for {diagram_type} diagram",
    }
