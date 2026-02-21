"""ASG graph traversal queries.

Provides functions to query the code_edges table for
callers, callees, dependencies, dependents, and structural views.
Uses recursive CTEs for transitive (depth > 1) queries.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)


def get_callers(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    depth: int = 1,
) -> List[Dict[str, Any]]:
    """Get units that call the given unit.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the target unit
        depth: Maximum traversal depth (1 = direct callers only)

    Returns:
        List of dicts with unit_id, name, qualified_name, unit_type, edge_type, depth
    """
    return _traverse(
        db, project_id, unit_id, depth,
        direction="incoming", edge_types=("calls",),
    )


def get_callees(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    depth: int = 1,
) -> List[Dict[str, Any]]:
    """Get units that the given unit calls.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the source unit
        depth: Maximum traversal depth

    Returns:
        List of dicts with unit info
    """
    return _traverse(
        db, project_id, unit_id, depth,
        direction="outgoing", edge_types=("calls",),
    )


def get_dependencies(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    depth: int = 2,
) -> List[Dict[str, Any]]:
    """Get all units this unit depends on (calls + imports + inherits, outgoing).

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the source unit
        depth: Maximum traversal depth

    Returns:
        List of dicts with unit info
    """
    return _traverse(
        db, project_id, unit_id, depth,
        direction="outgoing", edge_types=("calls", "calls_sp", "imports", "inherits", "implements", "overrides", "type_dep"),
    )


def get_dependents(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    depth: int = 2,
) -> List[Dict[str, Any]]:
    """Get all units that depend on this unit (blast radius).

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the target unit
        depth: Maximum traversal depth

    Returns:
        List of dicts with unit info
    """
    return _traverse(
        db, project_id, unit_id, depth,
        direction="incoming", edge_types=("calls", "calls_sp", "imports", "inherits", "implements", "overrides", "type_dep"),
    )


def get_import_graph(
    db: DatabaseManager,
    project_id: str,
) -> Dict[str, Any]:
    """Get the full import graph for a project.

    Returns:
        Dict with 'nodes' (list of units) and 'edges' (list of import edges)
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    e.source_unit_id,
                    e.target_unit_id,
                    su.name AS source_name,
                    su.qualified_name AS source_qualified,
                    tu.name AS target_name,
                    tu.qualified_name AS target_qualified
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'imports'
                ORDER BY su.name, tu.name
            """),
            {"pid": pid},
        )
        rows = result.fetchall()

    nodes_seen = set()
    nodes = []
    edges = []

    for row in rows:
        src_id = str(row.source_unit_id)
        tgt_id = str(row.target_unit_id)

        if src_id not in nodes_seen:
            nodes_seen.add(src_id)
            nodes.append({"unit_id": src_id, "name": row.source_name, "qualified_name": row.source_qualified})
        if tgt_id not in nodes_seen:
            nodes_seen.add(tgt_id)
            nodes.append({"unit_id": tgt_id, "name": row.target_name, "qualified_name": row.target_qualified})

        edges.append({"source": src_id, "target": tgt_id})

    return {"nodes": nodes, "edges": edges}


def get_class_hierarchy(
    db: DatabaseManager,
    project_id: str,
) -> Dict[str, Any]:
    """Get the class hierarchy (inherits + implements edges) for a project.

    Returns:
        Dict with 'nodes' (classes/interfaces) and 'edges' (inheritance/implements relationships)
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    e.source_unit_id,
                    e.target_unit_id,
                    e.edge_type,
                    su.name AS child_name,
                    su.qualified_name AS child_qualified,
                    tu.name AS parent_name,
                    tu.qualified_name AS parent_qualified
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type IN ('inherits', 'implements')
                ORDER BY e.edge_type, tu.name, su.name
            """),
            {"pid": pid},
        )
        rows = result.fetchall()

    nodes_seen = set()
    nodes = []
    edges = []

    for row in rows:
        child_id = str(row.source_unit_id)
        parent_id = str(row.target_unit_id)

        if child_id not in nodes_seen:
            nodes_seen.add(child_id)
            nodes.append({"unit_id": child_id, "name": row.child_name, "qualified_name": row.child_qualified})
        if parent_id not in nodes_seen:
            nodes_seen.add(parent_id)
            nodes.append({"unit_id": parent_id, "name": row.parent_name, "qualified_name": row.parent_qualified})

        edges.append({"child": child_id, "parent": parent_id, "edge_type": row.edge_type})

    return {"nodes": nodes, "edges": edges}


def get_interface_implementations(
    db: DatabaseManager,
    project_id: str,
) -> Dict[str, Any]:
    """Get the interface → implementor graph for a project.

    Returns:
        Dict with 'nodes' (interfaces and implementors) and
        'edges' (implements relationships)
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    e.source_unit_id,
                    e.target_unit_id,
                    su.name AS implementor_name,
                    su.qualified_name AS implementor_qualified,
                    su.unit_type AS implementor_type,
                    tu.name AS interface_name,
                    tu.qualified_name AS interface_qualified
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'implements'
                ORDER BY tu.name, su.name
            """),
            {"pid": pid},
        )
        rows = result.fetchall()

    nodes_seen = set()
    nodes = []
    edges = []

    for row in rows:
        impl_id = str(row.source_unit_id)
        iface_id = str(row.target_unit_id)

        if impl_id not in nodes_seen:
            nodes_seen.add(impl_id)
            nodes.append({
                "unit_id": impl_id,
                "name": row.implementor_name,
                "qualified_name": row.implementor_qualified,
                "unit_type": row.implementor_type,
            })
        if iface_id not in nodes_seen:
            nodes_seen.add(iface_id)
            nodes.append({
                "unit_id": iface_id,
                "name": row.interface_name,
                "qualified_name": row.interface_qualified,
                "unit_type": "interface",
            })

        edges.append({"implementor": impl_id, "interface": iface_id})

    return {"nodes": nodes, "edges": edges}


def get_sp_dependencies(
    db: DatabaseManager,
    project_id: str,
) -> List[Dict[str, Any]]:
    """Get all stored procedure dependencies: which app units call which SPs.

    Returns:
        List of dicts with sp_name, sp_unit_id, caller_name, caller_unit_id, call_pattern
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    e.source_unit_id AS caller_id,
                    e.target_unit_id AS sp_id,
                    e.metadata->>'sp_name' AS sp_name,
                    e.metadata->>'call_pattern' AS call_pattern,
                    su.name AS caller_name,
                    su.qualified_name AS caller_qualified,
                    su.unit_type AS caller_type,
                    su.language AS caller_language,
                    su.file_id AS caller_file_id,
                    tu.name AS sp_unit_name,
                    tu.qualified_name AS sp_qualified
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'calls_sp'
                ORDER BY tu.name, su.name
            """),
            {"pid": pid},
        )
        return [
            {
                "sp_name": row.sp_name or row.sp_unit_name,
                "sp_unit_id": str(row.sp_id),
                "sp_qualified_name": row.sp_qualified,
                "caller_name": row.caller_name,
                "caller_unit_id": str(row.caller_id),
                "caller_qualified_name": row.caller_qualified,
                "caller_type": row.caller_type,
                "caller_language": row.caller_language,
                "caller_file_id": str(row.caller_file_id),
                "call_pattern": row.call_pattern,
            }
            for row in result.fetchall()
        ]


def get_sp_impact_graph(
    db: DatabaseManager,
    project_id: str,
    sp_name: str,
) -> Dict[str, Any]:
    """Get the impact graph for a specific stored procedure.

    Shows all application code units that call this SP (blast radius).

    Args:
        db: DatabaseManager instance
        project_id: Project UUID string
        sp_name: Name of the stored procedure

    Returns:
        Dict with 'sp' (SP details), 'callers' (list of calling units),
        and 'caller_count'
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        # Find the SP unit
        sp_result = session.execute(
            text("""
                SELECT unit_id, name, qualified_name, metadata, file_id,
                       start_line, end_line
                FROM code_units
                WHERE project_id = :pid
                  AND unit_type IN ('stored_procedure', 'sql_function')
                  AND LOWER(name) = LOWER(:sp_name)
                LIMIT 1
            """),
            {"pid": pid, "sp_name": sp_name},
        )
        sp_row = sp_result.fetchone()
        if not sp_row:
            return {"sp": None, "callers": [], "caller_count": 0}

        # Find all callers
        caller_result = session.execute(
            text("""
                SELECT
                    su.unit_id, su.name, su.qualified_name,
                    su.unit_type, su.language, su.file_id,
                    e.metadata->>'call_pattern' AS call_pattern
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                WHERE e.target_unit_id = :sp_id
                  AND e.edge_type = 'calls_sp'
                  AND e.project_id = :pid
                ORDER BY su.language, su.name
            """),
            {"pid": pid, "sp_id": sp_row.unit_id},
        )
        callers = [
            {
                "unit_id": str(r.unit_id),
                "name": r.name,
                "qualified_name": r.qualified_name,
                "unit_type": r.unit_type,
                "language": r.language,
                "file_id": str(r.file_id),
                "call_pattern": r.call_pattern,
            }
            for r in caller_result.fetchall()
        ]

    return {
        "sp": {
            "unit_id": str(sp_row.unit_id),
            "name": sp_row.name,
            "qualified_name": sp_row.qualified_name,
            "file_id": str(sp_row.file_id),
            "start_line": sp_row.start_line,
            "end_line": sp_row.end_line,
        },
        "callers": callers,
        "caller_count": len(callers),
    }


def get_edge_stats(
    db: DatabaseManager,
    project_id: str,
) -> Dict[str, int]:
    """Get edge type counts for a project.

    Returns:
        Dict mapping edge_type to count
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT edge_type, COUNT(*) AS cnt
                FROM code_edges
                WHERE project_id = :pid
                GROUP BY edge_type
                ORDER BY cnt DESC
            """),
            {"pid": pid},
        )
        return {row.edge_type: row.cnt for row in result.fetchall()}


# ── Private helpers ──────────────────────────────────────────────────

def _traverse(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int,
    direction: str,
    edge_types: tuple,
) -> List[Dict[str, Any]]:
    """Generic graph traversal using recursive CTE.

    Args:
        db: DatabaseManager
        project_id: Project UUID string
        unit_id: Starting unit UUID string
        max_depth: Maximum recursion depth
        direction: "outgoing" (source -> target) or "incoming" (target -> source)
        edge_types: Tuple of edge type strings to follow

    Returns:
        List of neighbor dicts
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    uid = UUID(unit_id) if isinstance(unit_id, str) else unit_id

    if direction == "outgoing":
        seed_filter = "source_unit_id = :uid"
        seed_select = "target_unit_id"
        recurse_join = "e.source_unit_id = d.neighbor_id"
        recurse_select = "e.target_unit_id"
    else:
        seed_filter = "target_unit_id = :uid"
        seed_select = "source_unit_id"
        recurse_join = "e.target_unit_id = d.neighbor_id"
        recurse_select = "e.source_unit_id"

    # Build edge type filter
    type_placeholders = ", ".join(f":et{i}" for i in range(len(edge_types)))
    params = {"pid": pid, "uid": uid, "max_depth": max_depth}
    for i, et in enumerate(edge_types):
        params[f"et{i}"] = et

    sql = f"""
        WITH RECURSIVE deps AS (
            SELECT {seed_select} AS neighbor_id, edge_type, 1 AS depth
            FROM code_edges
            WHERE {seed_filter}
              AND project_id = :pid
              AND edge_type IN ({type_placeholders})

            UNION ALL

            SELECT {recurse_select}, e.edge_type, d.depth + 1
            FROM code_edges e
            JOIN deps d ON {recurse_join}
            WHERE d.depth < :max_depth
              AND e.project_id = :pid
              AND e.edge_type IN ({type_placeholders})
        )
        SELECT DISTINCT
            u.unit_id,
            u.name,
            u.qualified_name,
            u.unit_type,
            u.language,
            u.file_id,
            d.edge_type,
            MIN(d.depth) AS depth
        FROM deps d
        JOIN code_units u ON d.neighbor_id = u.unit_id
        WHERE u.unit_id != :uid
        GROUP BY u.unit_id, u.name, u.qualified_name, u.unit_type, u.language, u.file_id, d.edge_type
        ORDER BY depth, u.name
    """

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        return [
            {
                "unit_id": str(row.unit_id),
                "name": row.name,
                "qualified_name": row.qualified_name,
                "unit_type": row.unit_type,
                "language": row.language,
                "file_id": str(row.file_id),
                "edge_type": row.edge_type,
                "depth": row.depth,
            }
            for row in result.fetchall()
        ]
