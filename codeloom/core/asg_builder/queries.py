"""ASG graph traversal queries.

Provides functions to query the code_edges table for
callers, callees, dependencies, dependents, and structural views.

Also provides advanced analysis queries:
- Cyclomatic complexity reporting (from enricher metadata)
- Decorator/annotation and return-type search
- Dead code detection (uncalled functions with smart exclusions)
- Transitive call chains (full caller/callee closure)
- Module-level dependency graphs (file and directory aggregation)

Traversal queries try Apache AGE Cypher first (if graph is synced),
falling back to recursive CTEs on the relational code_edges table.
"""

import fnmatch
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)


# =========================================================================
# Core traversal queries
# =========================================================================


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
    """Get the interface -> implementor graph for a project.

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


def find_call_path(
    db: DatabaseManager,
    project_id: str,
    source_unit_id: str,
    target_unit_id: str,
    max_depth: int = 10,
    edge_types: tuple = ("calls", "calls_sp"),
) -> Optional[List[Dict[str, Any]]]:
    """Find the shortest call path between two units.

    Uses a recursive CTE with ARRAY path accumulator for cycle prevention.
    Returns the shortest path as a list of unit dicts ordered from source to
    target, or None if no path exists within max_depth hops.
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    src = UUID(source_unit_id) if isinstance(source_unit_id, str) else source_unit_id
    tgt = UUID(target_unit_id) if isinstance(target_unit_id, str) else target_unit_id

    type_placeholders = ", ".join(f":et{i}" for i in range(len(edge_types)))
    params: Dict[str, Any] = {
        "pid": pid, "src": src, "tgt": tgt, "max_depth": max_depth,
    }
    for i, et in enumerate(edge_types):
        params[f"et{i}"] = et

    sql = f"""
        WITH RECURSIVE paths AS (
            SELECT
                unit_id,
                ARRAY[unit_id] AS path,
                0 AS depth
            FROM code_units
            WHERE unit_id = :src AND project_id = :pid

            UNION ALL

            SELECT
                u.unit_id,
                p.path || u.unit_id,
                p.depth + 1
            FROM paths p
            JOIN code_edges e ON e.source_unit_id = p.unit_id
            JOIN code_units u ON u.unit_id = e.target_unit_id
            WHERE e.project_id = :pid
              AND e.edge_type IN ({type_placeholders})
              AND NOT (u.unit_id = ANY(p.path))
              AND p.depth < :max_depth
        )
        SELECT path
        FROM paths
        WHERE unit_id = :tgt
        ORDER BY array_length(path, 1) ASC
        LIMIT 1
    """

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        row = result.fetchone()
        if not row:
            return None

        # Resolve full unit info for each step in the path
        path_ids = [UUID(str(uid)) for uid in row.path]
        units = session.execute(
            text("""
                SELECT unit_id, name, qualified_name, unit_type, language, file_id
                FROM code_units
                WHERE unit_id = ANY(:ids) AND project_id = :pid
            """),
            {"ids": path_ids, "pid": pid},
        )
        unit_map = {
            row.unit_id: {
                "unit_id": str(row.unit_id),
                "name": row.name,
                "qualified_name": row.qualified_name,
                "unit_type": row.unit_type,
                "language": row.language,
                "file_id": str(row.file_id) if row.file_id else None,
            }
            for row in units.fetchall()
        }

        # Return path in order, preserving the traversal sequence
        return [unit_map[uid] for uid in path_ids if uid in unit_map]


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


# =========================================================================
# Feature 1: Cyclomatic complexity report
# =========================================================================


def get_complexity_report(
    db: DatabaseManager,
    project_id: str,
    sort: str = "desc",
    limit: int = 50,
    min_complexity: int = 1,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get a complexity report for callable units in a project.

    Uses McCabe cyclomatic complexity from enricher metadata when available,
    falling back to a heuristic estimate from source text otherwise.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        sort: "desc" (highest first) or "asc" (lowest first)
        limit: Maximum number of results
        min_complexity: Minimum complexity threshold to include
        language: Optional language filter

    Returns:
        List of dicts with unit info and complexity score
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    order = "DESC" if sort == "desc" else "ASC"
    lang_filter = "AND u.language = :lang" if language else ""

    # Primary: use enricher metadata when available
    sql = f"""
        SELECT u.unit_id, u.name, u.qualified_name, u.unit_type, u.language,
               cf.file_path, u.start_line, u.end_line,
               u.metadata->>'cyclomatic_complexity' AS cc_raw,
               u.source,
               u.metadata AS metadata
        FROM code_units u
        JOIN code_files cf ON u.file_id = cf.file_id
        WHERE u.project_id = :pid
          AND u.unit_type IN ('function', 'method', 'constructor', 'paragraph')
          {lang_filter}
        ORDER BY u.start_line
    """

    params: Dict[str, Any] = {"pid": pid}
    if language:
        params["lang"] = language

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        rows = result.fetchall()

    scored = []
    for row in rows:
        # Prefer enricher-computed McCabe complexity
        if row.cc_raw is not None:
            complexity = int(row.cc_raw)
        else:
            # Fallback: heuristic estimate from source text
            complexity = _estimate_branch_complexity(row.source or "")

        if complexity < min_complexity:
            continue

        line_count = None
        if row.start_line is not None and row.end_line is not None:
            line_count = row.end_line - row.start_line + 1

        scored.append({
            "unit_id": str(row.unit_id),
            "name": row.name,
            "qualified_name": row.qualified_name,
            "unit_type": row.unit_type,
            "language": row.language,
            "file_path": row.file_path,
            "start_line": row.start_line,
            "end_line": row.end_line,
            "line_count": line_count,
            "complexity": complexity,
        })

    reverse = sort != "asc"
    scored.sort(key=lambda x: x["complexity"], reverse=reverse)

    return scored[:limit]


# =========================================================================
# Feature 2: Decorator / annotation search and return type search
# =========================================================================


def find_units_by_decorator(
    db: DatabaseManager,
    project_id: str,
    decorator_name: str,
    unit_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Search units by annotation/decorator name.

    Searches across metadata annotations array, metadata modifiers array,
    and the unit's signature text for decorator matches.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        decorator_name: Decorator/annotation name (partial match, case-insensitive)
        unit_type: Optional filter by unit_type (e.g. "method", "class")
        limit: Maximum results

    Returns:
        List of dicts with unit info and matched decorator context
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    # Normalize: strip leading @ if present
    clean_name = decorator_name.lstrip("@").strip()
    pattern = f"%{clean_name}%"
    sig_pattern = f"%@{clean_name}%"
    unit_type_filter = "AND u.unit_type = :utype" if unit_type else ""

    sql = f"""
        SELECT u.unit_id, u.name, u.qualified_name, u.unit_type, u.language,
               cf.file_path, u.start_line, u.end_line,
               u.metadata->'modifiers' AS modifiers,
               u.metadata->'annotations' AS annotations,
               u.signature
        FROM code_units u
        JOIN code_files cf ON u.file_id = cf.file_id
        WHERE u.project_id = :pid
          AND (
            EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(
                    COALESCE(u.metadata->'annotations', '[]'::jsonb)
                ) a WHERE a ILIKE :pat
            )
            OR EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(
                    COALESCE(u.metadata->'modifiers', '[]'::jsonb)
                ) m WHERE m ILIKE :pat
            )
            OR u.signature ILIKE :sig_pat
          )
          {unit_type_filter}
        ORDER BY cf.file_path, u.name
        LIMIT :lim
    """

    params: Dict[str, Any] = {
        "pid": pid, "pat": pattern, "sig_pat": sig_pattern, "lim": limit,
    }
    if unit_type:
        params["utype"] = unit_type

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        return [
            {
                "unit_id": str(row.unit_id),
                "name": row.name,
                "qualified_name": row.qualified_name,
                "unit_type": row.unit_type,
                "language": row.language,
                "file_path": row.file_path,
                "start_line": row.start_line,
                "end_line": row.end_line,
                "modifiers": row.modifiers,
                "annotations": row.annotations,
                "signature": row.signature,
            }
            for row in result.fetchall()
        ]


def find_units_by_return_type(
    db: DatabaseManager,
    project_id: str,
    return_type: str,
    unit_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Search callable units by return type.

    Searches the metadata return_type field for a case-insensitive match.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        return_type: Return type to search for (partial match, case-insensitive)
        unit_type: Optional filter by unit_type
        limit: Maximum results

    Returns:
        List of dicts with unit info and return type
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    pattern = f"%{return_type}%"
    unit_type_filter = "AND u.unit_type = :utype" if unit_type else ""

    sql = f"""
        SELECT u.unit_id, u.name, u.qualified_name, u.unit_type, u.language,
               cf.file_path, u.start_line, u.end_line,
               u.metadata->>'return_type' AS return_type
        FROM code_units u
        JOIN code_files cf ON u.file_id = cf.file_id
        WHERE u.project_id = :pid
          AND u.metadata->>'return_type' ILIKE :pat
          {unit_type_filter}
        ORDER BY cf.file_path, u.name
        LIMIT :lim
    """

    params: Dict[str, Any] = {"pid": pid, "pat": pattern, "lim": limit}
    if unit_type:
        params["utype"] = unit_type

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        return [
            {
                "unit_id": str(row.unit_id),
                "name": row.name,
                "qualified_name": row.qualified_name,
                "unit_type": row.unit_type,
                "language": row.language,
                "file_path": row.file_path,
                "start_line": row.start_line,
                "end_line": row.end_line,
                "return_type": row.return_type,
            }
            for row in result.fetchall()
        ]


# =========================================================================
# Feature 3: Dead code detection
# =========================================================================

# Names that are typically entry points or framework-invoked, not dead code
_ENTRY_POINT_NAMES = frozenset({
    "main", "__init__", "__main__", "setUp", "tearDown", "setUpClass",
    "tearDownClass", "configure", "setup", "run", "execute", "handle",
    "init", "destroy", "close", "start", "stop",
})


def get_dead_code(
    db: DatabaseManager,
    project_id: str,
    exclude_patterns: Optional[List[str]] = None,
    exclude_decorators: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Detect potentially dead (uncalled) functions and methods.

    A unit is considered dead if no incoming 'calls' edge exists. Results
    are post-filtered in Python to exclude common false positives: entry
    points, overrides, test methods, and user-specified patterns/decorators.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        exclude_patterns: Optional list of fnmatch patterns on file_path to exclude
        exclude_decorators: Optional list of decorator names to exclude
        limit: Maximum results

    Returns:
        List of dicts with unit info for potentially dead functions
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    sql = """
        SELECT u.unit_id, u.name, u.qualified_name, u.unit_type, u.language,
               cf.file_path, u.start_line, u.end_line,
               u.metadata
        FROM code_units u
        JOIN code_files cf ON u.file_id = cf.file_id
        LEFT JOIN code_edges e ON e.target_unit_id = u.unit_id
            AND e.edge_type = 'calls' AND e.project_id = :pid
        WHERE u.project_id = :pid
          AND u.unit_type IN ('function', 'method')
          AND e.id IS NULL
        ORDER BY cf.file_path, u.name
    """

    with db.get_session() as session:
        result = session.execute(text(sql), {"pid": pid})
        raw_rows = result.fetchall()

    # Post-filter in Python for nuanced exclusion rules
    exclude_patterns = exclude_patterns or []
    exclude_decorators_lower = set(d.lower() for d in (exclude_decorators or []))
    filtered = []

    for row in raw_rows:
        name_lower = (row.name or "").lower()

        # 1. Exclude common entry-point names
        if name_lower in _ENTRY_POINT_NAMES:
            continue

        # 2. Exclude test methods (test_ prefix or Test class methods)
        if name_lower.startswith("test_") or name_lower.startswith("test"):
            # Be more specific: only skip test_ prefix (PEP) or Test* (JUnit)
            if row.name and (row.name.startswith("test_") or row.name.startswith("Test")):
                continue

        # 3. Exclude file path patterns
        if any(fnmatch.fnmatch(row.file_path or "", pat) for pat in exclude_patterns):
            continue

        # 4. Exclude overrides (from metadata)
        meta = row.metadata or {} if hasattr(row, 'metadata') else {}
        if meta.get("is_override"):
            continue

        # 5. Exclude units with specified decorators
        if exclude_decorators_lower:
            annotations = [a.lower() for a in (meta.get("annotations") or [])]
            modifiers = [m.lower() for m in (meta.get("modifiers") or [])]
            all_markers = set(annotations + modifiers)
            if any(d in marker for d in exclude_decorators_lower for marker in all_markers):
                continue

        filtered.append({
            "unit_id": str(row.unit_id),
            "name": row.name,
            "qualified_name": row.qualified_name,
            "unit_type": row.unit_type,
            "language": row.language,
            "file_path": row.file_path,
            "start_line": row.start_line,
            "end_line": row.end_line,
        })

        if len(filtered) >= limit:
            break

    return filtered


# =========================================================================
# Feature 4: Transitive call chains
# =========================================================================


def get_all_callers(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int = 10,
) -> Dict[str, Any]:
    """Get the full transitive closure of callers for a unit.

    Traverses the call graph in the incoming direction up to max_depth,
    returning all direct and indirect callers grouped by depth.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the target unit
        max_depth: Maximum traversal depth

    Returns:
        Dict with unit_id, total_count, by_depth (dict of depth -> units),
        max_depth_reached, and flat results list
    """
    results = _traverse(
        db, project_id, unit_id, max_depth,
        direction="incoming", edge_types=("calls",),
    )

    # Enrich with file_path
    results = _enrich_file_paths(db, project_id, results)

    by_depth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        d = r.get("depth", 1)
        by_depth[d].append(r)

    return {
        "unit_id": unit_id,
        "total_count": len(results),
        "max_depth_reached": max(by_depth.keys()) if by_depth else 0,
        "by_depth": dict(by_depth),
        "results": results,
    }


def get_all_callees(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int = 10,
) -> Dict[str, Any]:
    """Get the full transitive closure of callees for a unit.

    Traverses the call graph in the outgoing direction up to max_depth,
    returning all direct and indirect callees grouped by depth.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        unit_id: UUID string of the source unit
        max_depth: Maximum traversal depth

    Returns:
        Dict with unit_id, total_count, by_depth, max_depth_reached,
        and flat results list
    """
    results = _traverse(
        db, project_id, unit_id, max_depth,
        direction="outgoing", edge_types=("calls",),
    )

    # Enrich with file_path
    results = _enrich_file_paths(db, project_id, results)

    by_depth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        d = r.get("depth", 1)
        by_depth[d].append(r)

    return {
        "unit_id": unit_id,
        "total_count": len(results),
        "max_depth_reached": max(by_depth.keys()) if by_depth else 0,
        "by_depth": dict(by_depth),
        "results": results,
    }


# =========================================================================
# Feature 5: Module dependency graph
# =========================================================================


def get_module_dependency_graph(
    db: DatabaseManager,
    project_id: str,
    level: str = "file",
    prefix: Optional[str] = None,
    min_weight: int = 1,
    dir_depth: int = 2,
) -> Dict[str, Any]:
    """Get the module-level dependency graph for a project.

    Aggregates import edges between files or directories into a weighted
    dependency graph suitable for visualization.

    Args:
        db: DatabaseManager instance
        project_id: UUID string of the project
        level: "file" for file-level or "directory" for directory-level aggregation
        prefix: Optional path prefix filter (e.g. "src/main/java")
        min_weight: Minimum number of imports to include an edge
        dir_depth: Number of directory segments to use for directory-level grouping

    Returns:
        Dict with 'nodes' and 'links' for graph rendering
    """
    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    prefix_filter = "AND sf.file_path LIKE :prefix AND tf.file_path LIKE :prefix" if prefix else ""

    sql = f"""
        SELECT sf.file_id AS source_id, sf.file_path AS source_path,
               tf.file_id AS target_id, tf.file_path AS target_path,
               COUNT(*) AS weight
        FROM code_edges e
        JOIN code_units su ON e.source_unit_id = su.unit_id
        JOIN code_units tu ON e.target_unit_id = tu.unit_id
        JOIN code_files sf ON su.file_id = sf.file_id
        JOIN code_files tf ON tu.file_id = tf.file_id
        WHERE e.project_id = :pid AND e.edge_type = 'imports'
          AND sf.file_id != tf.file_id
          {prefix_filter}
        GROUP BY sf.file_id, sf.file_path, tf.file_id, tf.file_path
        HAVING COUNT(*) >= :min_w
        ORDER BY weight DESC
    """

    params: Dict[str, Any] = {"pid": pid, "min_w": min_weight}
    if prefix:
        params["prefix"] = f"{prefix}%"

    with db.get_session() as session:
        result = session.execute(text(sql), params)
        rows = result.fetchall()

    if level == "directory":
        return _aggregate_to_directory(rows, dir_depth)

    # File-level graph
    nodes_seen: Dict[str, Dict[str, Any]] = {}
    links = []

    for row in rows:
        src_id = str(row.source_id)
        tgt_id = str(row.target_id)

        if src_id not in nodes_seen:
            nodes_seen[src_id] = {
                "id": src_id,
                "file_path": row.source_path,
                "label": row.source_path.rsplit("/", 1)[-1] if "/" in row.source_path else row.source_path,
            }
        if tgt_id not in nodes_seen:
            nodes_seen[tgt_id] = {
                "id": tgt_id,
                "file_path": row.target_path,
                "label": row.target_path.rsplit("/", 1)[-1] if "/" in row.target_path else row.target_path,
            }

        links.append({
            "source": src_id,
            "target": tgt_id,
            "weight": row.weight,
        })

    return {
        "nodes": list(nodes_seen.values()),
        "links": links,
        "node_count": len(nodes_seen),
        "link_count": len(links),
        "level": level,
    }


def _aggregate_to_directory(rows: list, dir_depth: int) -> Dict[str, Any]:
    """Aggregate file-level import edges to directory-level.

    Groups files by their first dir_depth path segments and sums weights.
    """
    def _dir_key(path: str) -> str:
        parts = path.split("/")
        if len(parts) > dir_depth:
            return "/".join(parts[:dir_depth])
        return os.path.dirname(path) if "/" in path else "."

    dir_edges: Dict[tuple, int] = defaultdict(int)
    dir_set: set = set()

    for row in rows:
        src_dir = _dir_key(row.source_path)
        tgt_dir = _dir_key(row.target_path)
        if src_dir == tgt_dir:
            continue  # Skip internal dependencies
        dir_set.add(src_dir)
        dir_set.add(tgt_dir)
        dir_edges[(src_dir, tgt_dir)] += row.weight

    nodes = [{"id": d, "label": d} for d in sorted(dir_set)]
    links = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in sorted(dir_edges.items(), key=lambda x: -x[1])
    ]

    return {
        "nodes": nodes,
        "links": links,
        "node_count": len(nodes),
        "link_count": len(links),
        "level": "directory",
    }


# -- Private helpers ----------------------------------------------------------

# AGE edge type label map (lowercase -> uppercase)
_AGE_EDGE_MAP = {
    "calls": "CALLS", "contains": "CONTAINS", "imports": "IMPORTS",
    "inherits": "INHERITS", "implements": "IMPLEMENTS", "overrides": "OVERRIDES",
    "type_dep": "TYPE_DEP", "calls_sp": "CALLS_SP", "data_flow": "DATA_FLOW",
}

# Regex patterns for estimating branching complexity (fallback)
_BRANCH_PATTERNS = [
    re.compile(r'\b(if|elif|else if)\b'),
    re.compile(r'\b(for|foreach|while|do)\b'),
    re.compile(r'\b(switch|case)\b'),
    re.compile(r'\b(try|catch|except|finally)\b'),
    re.compile(r'\b(&&|\|\|)\b'),
    re.compile(r'\?.*:'),  # ternary operator
]


def _estimate_branch_complexity(source: str) -> int:
    """Estimate branching complexity from source text.

    Counts occurrences of branching keywords and logical operators.
    This is a rough heuristic fallback when enricher metadata is unavailable.
    Returns 1 + branch_count (McCabe convention).
    """
    if not source:
        return 1

    count = 1  # Base complexity
    for pattern in _BRANCH_PATTERNS:
        count += len(pattern.findall(source))
    return count


def _enrich_file_paths(
    db: DatabaseManager,
    project_id: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Batch-enrich traversal results with file_path from code_files table.

    The _traverse functions return file_id but not file_path. This helper
    resolves file_ids to file_paths in a single query.
    """
    if not results:
        return results

    file_ids = list({r.get("file_id") for r in results if r.get("file_id")})
    if not file_ids:
        return results

    pid = UUID(project_id) if isinstance(project_id, str) else project_id

    try:
        file_uuids = [UUID(fid) if isinstance(fid, str) else fid for fid in file_ids]
        with db.get_session() as session:
            rows = session.execute(
                text("SELECT file_id, file_path FROM code_files WHERE file_id = ANY(:ids) AND project_id = :pid"),
                {"ids": file_uuids, "pid": pid},
            )
            path_map = {str(r.file_id): r.file_path for r in rows.fetchall()}
    except Exception:
        path_map = {}

    for r in results:
        r["file_path"] = path_map.get(r.get("file_id", ""), "")

    return results


def _age_available(db: DatabaseManager, project_id: str) -> bool:
    """Check if AGE graph is synced for this project."""
    pid = UUID(project_id) if isinstance(project_id, str) else project_id
    try:
        with db.get_session() as session:
            result = session.execute(
                text("SELECT age_graph_status FROM projects WHERE project_id = :pid"),
                {"pid": pid},
            )
            row = result.fetchone()
            return row is not None and row.age_graph_status == "synced"
    except Exception:
        return False


def _traverse(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int,
    direction: str,
    edge_types: tuple,
) -> List[Dict[str, Any]]:
    """Generic graph traversal -- tries Cypher for single-type queries, SQL for multi-type.

    AGE doesn't support multi-label variable-length paths, so multi-edge-type
    traversals stay on the SQL recursive CTE for correctness.
    """
    if len(edge_types) == 1 and _age_available(db, project_id):
        try:
            return _traverse_cypher(db, project_id, unit_id, max_depth, direction, edge_types)
        except Exception as e:
            logger.debug(f"AGE traversal failed, falling back to SQL: {e}")

    return _traverse_sql(db, project_id, unit_id, max_depth, direction, edge_types)


def _traverse_cypher(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int,
    direction: str,
    edge_types: tuple,
) -> List[Dict[str, Any]]:
    """Graph traversal using Apache AGE Cypher (single edge type only).

    Called only when len(edge_types) == 1. Multi-type traversals stay on SQL
    because AGE doesn't support multi-label variable-length paths.
    """
    from .age_client import AGEClient

    age = AGEClient(db)
    uid = str(unit_id)
    age_label = _AGE_EDGE_MAP.get(edge_types[0], edge_types[0].upper())

    if direction == "outgoing":
        cypher = (
            f"MATCH path = (start {{unit_id: '{uid}'}})-[:{age_label}*1..{max_depth}]->(target) "
            f"WHERE start <> target "
            f"RETURN DISTINCT target.unit_id AS uid, target.name AS name, "
            f"target.qualified_name AS qname, target.unit_type AS utype, "
            f"target.language AS lang, target.file_id AS fid, "
            f"length(path) AS depth "
            f"ORDER BY depth, target.name"
        )
    else:
        cypher = (
            f"MATCH path = (source)-[:{age_label}*1..{max_depth}]->(end_node {{unit_id: '{uid}'}}) "
            f"WHERE source <> end_node "
            f"RETURN DISTINCT source.unit_id AS uid, source.name AS name, "
            f"source.qualified_name AS qname, source.unit_type AS utype, "
            f"source.language AS lang, source.file_id AS fid, "
            f"length(path) AS depth "
            f"ORDER BY depth, source.name"
        )

    rows = age.cypher(
        project_id, cypher,
        columns=["uid", "name", "qname", "utype", "lang", "fid", "depth"],
    )

    return [
        {
            "unit_id": str(r["uid"]),
            "name": r["name"],
            "qualified_name": r["qname"],
            "unit_type": r["utype"],
            "language": r["lang"],
            "file_id": str(r["fid"]) if r["fid"] else None,
            "edge_type": edge_types[0],
            "depth": int(r["depth"]),
        }
        for r in rows
    ]


def _traverse_sql(
    db: DatabaseManager,
    project_id: str,
    unit_id: str,
    max_depth: int,
    direction: str,
    edge_types: tuple,
) -> List[Dict[str, Any]]:
    """Graph traversal using recursive CTE (SQL fallback)."""
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
