"""OOP edge detectors — inherits, implements, overrides, type_dep.

All functions take an EdgeContext and return List[dict] of edge records
ready for bulk insertion.
"""

import logging
from typing import List, Set, Tuple
from uuid import UUID

from .context import EdgeContext, resolve_unit, extract_type_identifiers, extract_base_classes

logger = logging.getLogger(__name__)


# ── Inherits ─────────────────────────────────────────────────────────


def detect_inherits(ctx: EdgeContext) -> List[dict]:
    """Detect inheritance edges from metadata or class signatures.

    Prefers unit_metadata["extends"] (Java/C#) when available,
    falls back to regex parsing of signature (Python/JS/TS).
    """
    edges = []
    inheritable_types = ("class", "struct", "record")

    for u in ctx.units:
        if u.unit_type not in inheritable_types:
            continue

        meta = u.unit_metadata or {}
        base_names = []

        # Prefer metadata (Java/C# parsers set this explicitly)
        extends_meta = meta.get("extends")
        if extends_meta:
            if isinstance(extends_meta, str):
                base_names.append(extends_meta)
            elif isinstance(extends_meta, list):
                base_names.extend(extends_meta)
        elif u.signature:
            # Fallback: regex on signature (Python/JS/TS)
            base_names = extract_base_classes(u.signature, u.language)

        for base_name in base_names:
            target = resolve_unit(
                base_name, ctx,
                preferred_types=("class", "struct", "record"),
            )
            if target and target.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "inherits",
                    "edge_metadata": {"base_class": base_name},
                })
    return edges


# ── Implements ───────────────────────────────────────────────────────


def detect_implements(ctx: EdgeContext) -> List[dict]:
    """Detect implements edges from unit_metadata["implements"].

    Source: class/struct/record unit with metadata["implements"] list.
    Target: interface unit resolved by name.
    """
    edges = []
    for u in ctx.units:
        if u.unit_type not in ("class", "struct", "record"):
            continue

        meta = u.unit_metadata or {}
        implements_list = meta.get("implements")
        if not implements_list:
            continue

        if isinstance(implements_list, str):
            implements_list = [implements_list]

        for iface_name in implements_list:
            target = resolve_unit(
                iface_name, ctx,
                preferred_types=("interface",),
            )
            if target and target.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "implements",
                    "edge_metadata": {"interface_name": iface_name},
                })
    return edges


# ── Overrides ────────────────────────────────────────────────────────


def detect_overrides(ctx: EdgeContext) -> List[dict]:
    """Detect override edges for methods annotated with @Override or override modifier.

    Walks the inheritance chain to find the parent method being overridden.
    """
    edges = []
    for u in ctx.units:
        if u.unit_type != "method":
            continue

        meta = u.unit_metadata or {}

        # Check for override indicator
        is_override = meta.get("is_override", False)
        if not is_override:
            annotations = meta.get("annotations", [])
            if "@Override" not in annotations:
                modifiers = meta.get("modifiers", [])
                if "override" not in modifiers:
                    continue

        # Find the parent class
        parent_class_name = meta.get("parent_name")
        if not parent_class_name and u.qualified_name:
            parts = u.qualified_name.rsplit(".", 2)
            if len(parts) >= 2:
                parent_class_name = parts[-2]

        if not parent_class_name:
            continue

        # Find the parent class unit
        parent_class = resolve_unit(
            parent_class_name, ctx,
            preferred_types=("class", "struct", "record"),
        )
        if not parent_class:
            continue

        # Walk the inheritance chain to find the overridden method
        parent_class_meta = parent_class.unit_metadata or {}
        base_name = parent_class_meta.get("extends")
        if isinstance(base_name, list):
            base_name = base_name[0] if base_name else None

        # Also check implements for interface default methods
        search_bases = []
        if base_name:
            search_bases.append(base_name)
        impl_list = parent_class_meta.get("implements", [])
        if isinstance(impl_list, str):
            impl_list = [impl_list]
        search_bases.extend(impl_list)

        for base in search_bases:
            base_unit = resolve_unit(
                base, ctx,
                preferred_types=("class", "interface", "struct", "record"),
            )
            if not base_unit:
                continue

            # Look for a method with the same name in the base
            target_qn = (
                f"{base_unit.qualified_name}.{u.name}"
                if base_unit.qualified_name
                else f"{base_unit.name}.{u.name}"
            )
            target = ctx.unit_by_qualified.get(target_qn)

            if not target:
                # Broader search: any method with same name that belongs to base
                for qu_qn, qu in ctx.unit_by_qualified.items():
                    if qu.name == u.name and qu.unit_type == "method":
                        qu_meta = qu.unit_metadata or {}
                        qu_parent = qu_meta.get("parent_name", "")
                        if qu_parent == base_unit.name:
                            target = qu
                            break

            if target and target.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "overrides",
                    "edge_metadata": {
                        "overriding_class": parent_class_name,
                        "parent_class": base_unit.name,
                    },
                })
                break  # Found the overridden method, stop searching

    return edges


# ── Type Dependencies ────────────────────────────────────────────────


def detect_type_deps(ctx: EdgeContext) -> List[dict]:
    """Detect type dependency edges from structured metadata.

    Reads pre-extracted metadata (no regex) to find type references:
    - Strategy A: metadata["parsed_params"][*].type -> method/constructor depends on param type
    - Strategy B: metadata["return_type"] -> method depends on return type
    - Strategy C: metadata["fields"][*].type -> class depends on field type

    Direction: consumer --type_dep--> referenced_type
    """
    edges = []
    seen_pairs: Set[Tuple[UUID, UUID, str]] = set()

    for u in ctx.units:
        meta = u.unit_metadata or {}

        type_refs: List[Tuple[str, str]] = []  # (type_string, kind)

        # Strategy A: param types (methods/constructors)
        if u.unit_type in ("method", "constructor", "function"):
            parsed_params = meta.get("parsed_params", [])
            for param in parsed_params:
                ptype = param.get("type")
                if ptype:
                    type_refs.append((ptype, "param"))

        # Strategy B: return types (methods/functions)
        if u.unit_type in ("method", "function"):
            ret_type = meta.get("return_type")
            if ret_type:
                type_refs.append((ret_type, "return"))

        # Strategy C: field types (classes/interfaces/structs)
        if u.unit_type in ("class", "interface", "struct", "record"):
            fields = meta.get("fields", [])
            for field in fields:
                ftype = field.get("type")
                if ftype:
                    type_refs.append((ftype, "field"))

        # Resolve each type reference to project units
        for type_str, kind in type_refs:
            identifiers = extract_type_identifiers(type_str)
            for ident in identifiers:
                target = resolve_unit(
                    ident, ctx,
                    preferred_types=("class", "interface", "struct", "record", "enum"),
                )
                if not target or target.unit_id == u.unit_id:
                    continue

                pair_key = (u.unit_id, target.unit_id, kind)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "type_dep",
                    "edge_metadata": {"kind": kind},
                })

    logger.debug(f"type_dep detection: {len(edges)} edges")
    return edges
