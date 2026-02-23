"""Struts edge detectors — framework-specific cross-cutting edges.

Detects relationships between Struts XML config units, Java classes,
and JSP view pages. All functions take an EdgeContext and return
List[dict] of edge records ready for bulk insertion.

Edge types produced:
- struts_action_class:    struts_action / struts2_action -> Java class
- struts_action_form:     struts_action -> struts_form_bean (or Java class)
- struts_action_forward:  struts_action -> jsp_page
- struts_action_input:    struts_action -> jsp_page
- struts2_action_class:   struts2_action -> Java class
- struts2_result_view:    struts2_action -> jsp_page
- struts_tile_include:    struts_tile_def -> jsp_page
- struts_validation_form: struts_validation_rule -> struts_form_bean
- jsp_includes:           jsp_page -> jsp_page
"""

import logging
from typing import Dict, List, Optional

from ..db.models import CodeUnit
from .context import EdgeContext, resolve_unit

logger = logging.getLogger(__name__)


# ── JSP resolution helper ────────────────────────────────────────────


def _resolve_jsp(path: str, ctx: EdgeContext) -> Optional[CodeUnit]:
    """Resolve a forward/result path to a jsp_page unit.

    JSP units store their relative file path as ``name``. Forward paths
    like ``/WEB-INF/jsp/login.jsp`` or ``/login.jsp`` may not exactly
    match the stored name, so we use suffix matching after normalising
    leading slashes.
    """
    if not path:
        return None

    # Strip leading slash for suffix comparison
    normalised = path.lstrip("/")

    for u in ctx.units:
        if u.unit_type != "jsp_page":
            continue
        unit_name = u.name.lstrip("/")
        if unit_name == normalised or unit_name.endswith("/" + normalised):
            return u
    return None


# ── Action edges (Struts 1.x + 2.x) ─────────────────────────────────


def _detect_action_edges(ctx: EdgeContext) -> List[dict]:
    """Detect edges originating from struts_action and struts2_action units."""
    edges: List[dict] = []

    for u in ctx.units:
        meta = u.unit_metadata or {}

        if u.unit_type == "struts_action":
            # struts_action_class: action -> Java class via metadata["type"]
            action_type = meta.get("type", "")
            if action_type:
                target = resolve_unit(
                    action_type, ctx,
                    preferred_types=("class",),
                )
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "struts_action_class",
                        "edge_metadata": {"class": action_type},
                    })

            # struts_action_form: action -> form-bean via metadata["name"]
            form_name = meta.get("name", "")
            if form_name:
                target = resolve_unit(
                    form_name, ctx,
                    preferred_types=("struts_form_bean", "class"),
                )
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "struts_action_form",
                        "edge_metadata": {"form_name": form_name},
                    })

            # struts_action_forward: action -> jsp_page via forwards[].path
            forwards = meta.get("forwards", [])
            for fwd in forwards:
                fwd_path = fwd.get("path", "")
                jsp = _resolve_jsp(fwd_path, ctx)
                if jsp and jsp.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": jsp.unit_id,
                        "edge_type": "struts_action_forward",
                        "edge_metadata": {
                            "forward_name": fwd.get("name", ""),
                            "path": fwd_path,
                        },
                    })

            # struts_action_input: action -> jsp_page via metadata["input"]
            input_path = meta.get("input", "")
            if input_path:
                jsp = _resolve_jsp(input_path, ctx)
                if jsp and jsp.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": jsp.unit_id,
                        "edge_type": "struts_action_input",
                        "edge_metadata": {"path": input_path},
                    })

        elif u.unit_type == "struts2_action":
            # struts2_action_class: action -> Java class via metadata["class"]
            action_class = meta.get("class", "")
            if action_class:
                target = resolve_unit(
                    action_class, ctx,
                    preferred_types=("class",),
                )
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "struts2_action_class",
                        "edge_metadata": {"class": action_class},
                    })

            # struts2_result_view: action -> jsp_page via results[].value
            results = meta.get("results", [])
            for res in results:
                res_value = res.get("value", "")
                jsp = _resolve_jsp(res_value, ctx)
                if jsp and jsp.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": jsp.unit_id,
                        "edge_type": "struts2_result_view",
                        "edge_metadata": {
                            "result_name": res.get("name", ""),
                            "path": res_value,
                        },
                    })

    return edges


# ── JSP include edges ────────────────────────────────────────────────


def _detect_jsp_edges(ctx: EdgeContext) -> List[dict]:
    """Detect jsp_includes edges from JSP include directives."""
    edges: List[dict] = []

    for u in ctx.units:
        if u.unit_type != "jsp_page":
            continue

        meta = u.unit_metadata or {}
        includes = meta.get("includes", [])

        for inc_path in includes:
            target = _resolve_jsp(inc_path, ctx)
            if target and target.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "jsp_includes",
                    "edge_metadata": {"path": inc_path},
                })

    return edges


# ── Tile definition edges ────────────────────────────────────────────


def _detect_tile_edges(ctx: EdgeContext) -> List[dict]:
    """Detect struts_tile_include edges from tile definition paths and attributes."""
    edges: List[dict] = []

    for u in ctx.units:
        if u.unit_type != "struts_tile_def":
            continue

        meta = u.unit_metadata or {}

        # Main tile path -> JSP
        tile_path = meta.get("path", "")
        if tile_path:
            jsp = _resolve_jsp(tile_path, ctx)
            if jsp and jsp.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": jsp.unit_id,
                    "edge_type": "struts_tile_include",
                    "edge_metadata": {"path": tile_path},
                })

        # Attribute values that reference JSPs
        attributes = meta.get("attributes", [])
        for attr in attributes:
            attr_value = attr.get("value", "")
            if attr_value and attr_value.endswith(".jsp"):
                jsp = _resolve_jsp(attr_value, ctx)
                if jsp and jsp.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": jsp.unit_id,
                        "edge_type": "struts_tile_include",
                        "edge_metadata": {
                            "attribute_name": attr.get("name", ""),
                            "path": attr_value,
                        },
                    })

    return edges


# ── Validation edges ─────────────────────────────────────────────────


def _detect_validation_edges(ctx: EdgeContext) -> List[dict]:
    """Detect struts_validation_form edges linking validation rules to form beans."""
    edges: List[dict] = []

    for u in ctx.units:
        if u.unit_type != "struts_validation_rule":
            continue

        meta = u.unit_metadata or {}
        form_name = meta.get("form_name", "")
        if not form_name:
            continue

        target = resolve_unit(
            form_name, ctx,
            preferred_types=("struts_form_bean",),
        )
        if target and target.unit_id != u.unit_id:
            edges.append({
                "project_id": ctx.project_id,
                "source_unit_id": u.unit_id,
                "target_unit_id": target.unit_id,
                "edge_type": "struts_validation_form",
                "edge_metadata": {"form_name": form_name},
            })

    return edges


# ── Public entry point ───────────────────────────────────────────────


def detect_struts_edges(ctx: EdgeContext) -> List[dict]:
    """Detect all Struts-specific edges. Entry point called from builder."""
    edges: List[dict] = []
    edges.extend(_detect_action_edges(ctx))
    edges.extend(_detect_jsp_edges(ctx))
    edges.extend(_detect_tile_edges(ctx))
    edges.extend(_detect_validation_edges(ctx))
    logger.debug("Struts edge detection: %d edges", len(edges))
    return edges
