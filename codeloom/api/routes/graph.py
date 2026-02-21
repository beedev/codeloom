"""Graph API routes — ASG query endpoints.

Exposes the code relationship graph for frontend visualization
and programmatic access to callers, callees, and dependency chains.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_current_user, get_db_manager, get_project_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/graph", tags=["graph"])


def _require_asg(pm, project_id: str) -> dict:
    """Validate project exists and ASG is built."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("asg_status") != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"ASG not built for this project (status: {project.get('asg_status')}). "
            "Re-ingest the project to build the graph.",
        )
    return project


@router.get("/callers/{unit_id}")
async def get_callers(
    project_id: str,
    unit_id: str,
    depth: int = Query(default=1, ge=1, le=5),
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get units that call the given unit."""
    _require_asg(pm, project_id)

    from codeloom.core.asg_builder import get_callers as _get_callers
    results = _get_callers(db_manager, project_id, unit_id, depth=depth)
    return {"unit_id": unit_id, "direction": "callers", "depth": depth, "results": results}


@router.get("/callees/{unit_id}")
async def get_callees(
    project_id: str,
    unit_id: str,
    depth: int = Query(default=1, ge=1, le=5),
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get units that the given unit calls."""
    _require_asg(pm, project_id)

    from codeloom.core.asg_builder import get_callees as _get_callees
    results = _get_callees(db_manager, project_id, unit_id, depth=depth)
    return {"unit_id": unit_id, "direction": "callees", "depth": depth, "results": results}


@router.get("/dependencies/{unit_id}")
async def get_dependencies(
    project_id: str,
    unit_id: str,
    depth: int = Query(default=2, ge=1, le=5),
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get all units this unit depends on (calls + imports + inherits)."""
    _require_asg(pm, project_id)

    from codeloom.core.asg_builder import get_dependencies as _get_deps
    results = _get_deps(db_manager, project_id, unit_id, depth=depth)
    return {"unit_id": unit_id, "direction": "dependencies", "depth": depth, "results": results}


@router.get("/dependents/{unit_id}")
async def get_dependents(
    project_id: str,
    unit_id: str,
    depth: int = Query(default=2, ge=1, le=5),
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get all units that depend on this unit (blast radius)."""
    _require_asg(pm, project_id)

    from codeloom.core.asg_builder import get_dependents as _get_dependents
    results = _get_dependents(db_manager, project_id, unit_id, depth=depth)
    return {"unit_id": unit_id, "direction": "dependents", "depth": depth, "results": results}


@router.get("/full")
async def full_graph(
    project_id: str,
    edge_types: str = Query(default="calls,contains,inherits,imports,implements,overrides,type_dep"),
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get full graph data (nodes + edges) for visualization.

    Returns all code units as nodes and all edges, filtered by edge_type.
    Designed for force-directed graph rendering.
    """
    _require_asg(pm, project_id)

    from uuid import UUID
    from codeloom.core.db.models import CodeUnit, CodeEdge

    types = [t.strip() for t in edge_types.split(",") if t.strip()]

    with db_manager.get_session() as session:
        units = session.query(CodeUnit).filter(
            CodeUnit.project_id == UUID(project_id)
        ).all()

        edges = session.query(CodeEdge).filter(
            CodeEdge.project_id == UUID(project_id),
            CodeEdge.edge_type.in_(types),
        ).all()

        # Only include nodes that have at least one edge
        node_ids_in_edges = set()
        for e in edges:
            node_ids_in_edges.add(str(e.source_unit_id))
            node_ids_in_edges.add(str(e.target_unit_id))

        nodes = [
            {
                "id": str(u.unit_id),
                "name": u.name,
                "qualified_name": u.qualified_name,
                "unit_type": u.unit_type,
                "language": u.language,
                "file_id": str(u.file_id),
            }
            for u in units
            if str(u.unit_id) in node_ids_in_edges
        ]

        links = [
            {
                "source": str(e.source_unit_id),
                "target": str(e.target_unit_id),
                "edge_type": e.edge_type,
            }
            for e in edges
        ]

    return {
        "nodes": nodes,
        "links": links,
        "node_count": len(nodes),
        "edge_count": len(links),
    }


@router.get("/unit/{unit_id}")
async def get_unit_detail(
    project_id: str,
    unit_id: str,
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get full details of a code unit including source, edges, and file path."""
    _require_asg(pm, project_id)

    from uuid import UUID
    from codeloom.core.db.models import CodeUnit, CodeFile, CodeEdge

    with db_manager.get_session() as session:
        unit = session.query(CodeUnit).filter(
            CodeUnit.unit_id == UUID(unit_id),
            CodeUnit.project_id == UUID(project_id),
        ).first()

        if not unit:
            raise HTTPException(status_code=404, detail="Unit not found")

        # Get file path
        code_file = session.query(CodeFile).filter(
            CodeFile.file_id == unit.file_id,
        ).first()

        # Get all outgoing edges (this unit -> others)
        outgoing_rows = (
            session.query(CodeEdge, CodeUnit)
            .join(CodeUnit, CodeEdge.target_unit_id == CodeUnit.unit_id)
            .filter(
                CodeEdge.source_unit_id == UUID(unit_id),
                CodeEdge.project_id == UUID(project_id),
            )
            .all()
        )

        # Get all incoming edges (others -> this unit)
        incoming_rows = (
            session.query(CodeEdge, CodeUnit)
            .join(CodeUnit, CodeEdge.source_unit_id == CodeUnit.unit_id)
            .filter(
                CodeEdge.target_unit_id == UUID(unit_id),
                CodeEdge.project_id == UUID(project_id),
            )
            .all()
        )

        def _edge_unit(edge, u):
            return {
                "unit_id": str(u.unit_id),
                "name": u.name,
                "qualified_name": u.qualified_name,
                "unit_type": u.unit_type,
                "language": u.language,
                "edge_type": edge.edge_type,
            }

        # Group edges by type + direction
        outgoing = {}
        for edge, u in outgoing_rows:
            outgoing.setdefault(edge.edge_type, []).append(_edge_unit(edge, u))

        incoming = {}
        for edge, u in incoming_rows:
            incoming.setdefault(edge.edge_type, []).append(_edge_unit(edge, u))

    return {
        "unit_id": str(unit.unit_id),
        "name": unit.name,
        "qualified_name": unit.qualified_name,
        "unit_type": unit.unit_type,
        "language": unit.language,
        "file_id": str(unit.file_id),
        "file_path": code_file.file_path if code_file else None,
        "start_line": unit.start_line,
        "end_line": unit.end_line,
        "line_count": (unit.end_line - unit.start_line + 1) if unit.start_line and unit.end_line else None,
        "signature": unit.signature,
        "docstring": unit.docstring,
        "source": unit.source,
        "edges": {
            "outgoing": outgoing,
            "incoming": incoming,
        },
    }


@router.get("/overview")
async def graph_overview(
    project_id: str,
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pm=Depends(get_project_manager),
):
    """Get graph overview: edge type counts and summary stats."""
    _require_asg(pm, project_id)

    from codeloom.core.asg_builder import get_edge_stats
    stats = get_edge_stats(db_manager, project_id)
    total = sum(stats.values())

    return {
        "project_id": project_id,
        "total_edges": total,
        "edge_types": stats,
    }
