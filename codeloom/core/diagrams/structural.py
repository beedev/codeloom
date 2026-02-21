"""Deterministic PlantUML generators for structural diagrams.

Takes pre-queried ASG data and produces PlantUML syntax.
No LLM calls — purely data-driven.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Clean, readable PlantUML theme — light elements on neutral background.
# Uses only standard fonts and colors the PlantUML public server supports.
_SKINPARAM = """
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontSize 12
skinparam roundCorner 8
skinparam linetype ortho
skinparam nodesep 60
skinparam ranksep 40

skinparam class {
  BackgroundColor #F8F9FA
  BorderColor #495057
  ArrowColor #6C757D
  FontColor #212529
  AttributeFontColor #495057
  StereotypeFontColor #6C757D
  HeaderBackgroundColor #E9ECEF
}

skinparam package {
  BackgroundColor #F1F3F5
  BorderColor #ADB5BD
  FontColor #343A40
  FontStyle bold
}

skinparam component {
  BackgroundColor #F8F9FA
  BorderColor #495057
  FontColor #212529
  StereotypeFontColor #6C757D
}

skinparam arrow {
  Color #495057
  FontColor #6C757D
  FontSize 10
}

skinparam note {
  BackgroundColor #FFF3CD
  BorderColor #FFCA2C
  FontColor #664D03
}
""".strip()

# Max dependency edges per source class to avoid arrow spaghetti
_MAX_DEPS_PER_CLASS = 6


def _sanitize(name: str) -> str:
    """Sanitize a name for safe use as a PlantUML identifier.

    Removes characters that break PlantUML syntax, keeping it readable.
    """
    # Replace problematic characters with underscores
    cleaned = re.sub(r'[<>{}()\[\]@#$%^&*;:\'"/\\|~`!]', "", name)
    cleaned = cleaned.replace("\n", " ").replace("\r", "").strip()
    # Collapse multiple spaces/underscores
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "unnamed"


def _safe_alias(name: str, idx: int) -> str:
    """Create a safe PlantUML alias from a name."""
    # Use only alphanumeric + underscore for aliases
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name)[:30]
    return f"{safe}_{idx}"


def generate_class_diagram(class_data: Dict) -> str:
    """Generate a PlantUML class diagram from ASG class/interface data."""
    classes: List[Dict] = class_data.get("classes", [])
    edges: List[Dict] = class_data.get("edges", [])

    if not classes:
        return _empty_diagram("Class Diagram", "No classes found in this MVP")

    lines = ["@startuml", _SKINPARAM, ""]

    # Left-to-right layout reduces crossing arrows for wide hierarchies
    if len(classes) > 4:
        lines.append("left to right direction")
        lines.append("")

    # Map qualified names to safe aliases
    alias_map: Dict[str, str] = {}

    # Group classes by package (from qualified_name prefix)
    packages: Dict[str, List[Dict]] = {}
    for cls in classes:
        qn = cls.get("qualified_name", cls["name"])
        parts = qn.rsplit(".", 1)
        pkg = parts[0] if len(parts) > 1 else ""
        packages.setdefault(pkg, []).append(cls)

    alias_idx = 0
    for pkg_name, pkg_classes in sorted(packages.items()):
        if pkg_name:
            lines.append(f"package {_sanitize(_short_name(pkg_name))} {{")

        for cls in pkg_classes:
            rendered, aliases = _render_class(cls, alias_idx)
            alias_map.update(aliases)
            alias_idx += len(aliases)
            lines.extend(rendered)
            lines.append("")

        if pkg_name:
            lines.append("}")
            lines.append("")

    # Render edges using aliases — structural edges first, then capped depends
    structural_edges = [e for e in edges if e["edge_type"] != "depends"]
    depends_edges = [e for e in edges if e["edge_type"] == "depends"]

    for edge in structural_edges:
        src_alias = alias_map.get(edge["source"])
        tgt_alias = alias_map.get(edge["target"])
        if not src_alias or not tgt_alias:
            continue
        if edge["edge_type"] == "inherits":
            lines.append(f"{src_alias} --|> {tgt_alias}")
        elif edge["edge_type"] == "implements":
            lines.append(f"{src_alias} ..|> {tgt_alias}")
        elif edge["edge_type"] == "overrides":
            lines.append(f"{src_alias} ..> {tgt_alias} : overrides")

    # Cap dependency edges per source to avoid arrow spaghetti
    deps_per_source: Dict[str, int] = {}
    for edge in depends_edges:
        src_alias = alias_map.get(edge["source"])
        tgt_alias = alias_map.get(edge["target"])
        if not src_alias or not tgt_alias:
            continue
        count = deps_per_source.get(src_alias, 0)
        if count >= _MAX_DEPS_PER_CLASS:
            continue
        deps_per_source[src_alias] = count + 1
        lines.append(f"{src_alias} --> {tgt_alias}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def _render_class(cls: Dict, start_idx: int) -> tuple[List[str], Dict[str, str]]:
    """Render a single class/interface with its members.

    Returns (lines, {qualified_name: alias}) mapping.
    """
    unit_type = cls.get("unit_type", "class")
    qn = cls.get("qualified_name", cls["name"])
    display_name = _sanitize(_short_name(qn))
    alias = _safe_alias(display_name, start_idx)
    metadata = cls.get("metadata") or {}
    modifiers = metadata.get("modifiers", [])

    # Determine class keyword
    if unit_type == "interface":
        keyword = "interface"
    elif "abstract" in [m.lower() for m in modifiers]:
        keyword = "abstract class"
    else:
        keyword = "class"

    lines = [f'{keyword} "{display_name}" as {alias} {{']

    # Render members
    members: List[Dict] = cls.get("members", [])
    fields = [m for m in members if m.get("unit_type") in ("field", "property", "attribute")]
    methods = [m for m in members if m.get("unit_type") in ("method", "function", "constructor")]

    for field in fields[:15]:
        vis = _visibility_symbol(field)
        fname = _sanitize(field.get("name", "?"))
        ftype = _sanitize(_extract_type(field))
        type_str = f" : {ftype}" if ftype else ""
        lines.append(f"  {vis}{fname}{type_str}")

    if len(fields) > 15:
        lines.append(f"  .. +{len(fields) - 15} more ..")

    if fields and methods:
        lines.append("  --")

    for method in methods[:20]:
        vis = _visibility_symbol(method)
        sig = _method_signature(method)
        lines.append(f"  {vis}{sig}")

    if len(methods) > 20:
        lines.append(f"  .. +{len(methods) - 20} more ..")

    lines.append("}")
    return lines, {qn: alias}


def _visibility_symbol(unit: Dict) -> str:
    """Map modifiers to UML visibility symbols."""
    metadata = unit.get("metadata") or {}
    modifiers = [m.lower() for m in metadata.get("modifiers", [])]

    if "private" in modifiers:
        return "-"
    if "protected" in modifiers:
        return "#"
    return "+"


def _method_signature(unit: Dict) -> str:
    """Build a concise method signature from metadata."""
    name = _sanitize(unit.get("name", "?"))
    metadata = unit.get("metadata") or {}
    params = metadata.get("parsed_params", [])
    ret = metadata.get("return_type") or ""

    if params:
        param_strs = []
        for p in params[:4]:
            if isinstance(p, dict):
                pname = _sanitize(p.get("name", "?"))
                ptype = _sanitize(p.get("type", ""))
                param_strs.append(f"{pname}: {ptype}" if ptype else pname)
            else:
                param_strs.append(_sanitize(str(p)))
        if len(params) > 4:
            param_strs.append("..")
        param_str = ", ".join(param_strs)
    else:
        param_str = ""

    ret_str = f" : {_sanitize(ret)}" if ret else ""
    return f"{name}({param_str}){ret_str}"


def _extract_type(unit: Dict) -> str:
    """Extract type annotation from metadata."""
    metadata = unit.get("metadata") or {}
    return metadata.get("return_type") or metadata.get("type", "")


def generate_package_diagram(package_data: Dict) -> str:
    """Generate a PlantUML package diagram from file/directory groupings."""
    packages: List[Dict] = package_data.get("packages", [])
    imports: List[Dict] = package_data.get("imports", [])

    if not packages:
        return _empty_diagram("Package Diagram", "No packages found in this MVP")

    lines = ["@startuml", _SKINPARAM, ""]

    if len(packages) > 3:
        lines.append("left to right direction")
        lines.append("")

    aliases = {}
    for i, pkg in enumerate(packages):
        alias = f"P{i}"
        dir_name = pkg["directory"]
        aliases[dir_name] = alias
        short = _sanitize(dir_name.split("/")[-1] if "/" in dir_name else dir_name)
        file_count = len(pkg["files"])
        lines.append(f'package "{short} ({file_count} files)" as {alias} {{')
        for f in pkg["files"][:8]:
            fname = _sanitize(f["name"])
            lines.append(f'  file "{fname}"')
        if file_count > 8:
            lines.append(f'  file ".. +{file_count - 8} more"')
        lines.append("}")
        lines.append("")

    # Show top import relationships, capped and deduplicated
    seen_edges: set = set()
    edge_count = 0
    for imp in imports[:30]:
        src_alias = aliases.get(imp["source_dir"])
        tgt_alias = aliases.get(imp["target_dir"])
        if src_alias and tgt_alias and src_alias != tgt_alias:
            key = (src_alias, tgt_alias)
            if key not in seen_edges:
                seen_edges.add(key)
                count = imp.get("count", 1)
                label = f" : {count}" if count > 1 else ""
                lines.append(f"{src_alias} --> {tgt_alias}{label}")
                edge_count += 1
                if edge_count >= 20:
                    break

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def generate_component_diagram(component_data: Dict) -> str:
    """Generate a PlantUML component diagram with architectural stereotypes."""
    components: List[Dict] = component_data.get("components", [])
    connectors: List[Dict] = component_data.get("connectors", [])

    if not components:
        return _empty_diagram("Component Diagram", "No components found in this MVP")

    lines = ["@startuml", _SKINPARAM, ""]
    lines.append("top to bottom direction")
    lines.append("")

    layers = {
        "controller": [],
        "service": [],
        "repository": [],
        "entity": [],
        "middleware": [],
        "utility": [],
        "config": [],
        "component": [],
        "test": [],
    }
    aliases: Dict[str, str] = {}

    for comp in components:
        st = comp.get("stereotype", "component")
        layers.setdefault(st, []).append(comp)

    layer_order = ["controller", "middleware", "service", "repository", "entity", "utility", "config", "component"]
    layer_labels = {
        "controller": "Presentation",
        "middleware": "Middleware",
        "service": "Business Logic",
        "repository": "Data Access",
        "entity": "Domain Model",
        "utility": "Utilities",
        "config": "Configuration",
        "component": "Components",
    }

    idx = 0
    for layer in layer_order:
        layer_comps = layers.get(layer, [])
        if not layer_comps:
            continue

        label = layer_labels.get(layer, layer.title())
        lines.append(f'package "{label}" {{')

        for comp in layer_comps:
            alias = f"C{idx}"
            aliases[comp["qualified_name"]] = alias
            short = _sanitize(comp["name"])
            stereo = f" <<{comp['stereotype']}>>" if comp["stereotype"] != "component" else ""
            lines.append(f'  [{short}]{stereo} as {alias}')
            idx += 1

        lines.append("}")
        lines.append("")

    # Cap connectors per source to avoid arrow spaghetti
    conns_per_source: Dict[str, int] = {}
    for conn in connectors[:60]:
        src_alias = aliases.get(conn["source"])
        tgt_alias = aliases.get(conn["target"])
        if src_alias and tgt_alias and src_alias != tgt_alias:
            count = conns_per_source.get(src_alias, 0)
            if count >= _MAX_DEPS_PER_CLASS:
                continue
            conns_per_source[src_alias] = count + 1
            arrow = "-->" if conn["edge_type"] == "calls" else "..>"
            lines.append(f"{src_alias} {arrow} {tgt_alias}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def _short_name(qualified_name: str) -> str:
    """Extract the short class/module name from a qualified name."""
    return qualified_name.rsplit(".", 1)[-1] if "." in qualified_name else qualified_name


def _empty_diagram(title: str, message: str) -> str:
    """Generate a minimal diagram with a note for empty data."""
    return f"""@startuml
{_SKINPARAM}

note "{message}" as N
@enduml"""
