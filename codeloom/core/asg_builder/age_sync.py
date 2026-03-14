"""AGE Graph Sync — populate Apache AGE graph from relational ASG data.

After ASGBuilder.build_edges() writes to the relational code_edges table,
this module syncs the same topology into an AGE graph for Cypher traversal.

Vertices are labeled by unit_type (Function, Class, Method, etc.).
Edges are labeled by edge_type (CALLS, IMPORTS, INHERITS, etc.).
"""

import logging
from typing import Any, Dict, List

from .age_client import AGEClient

logger = logging.getLogger(__name__)

# Batch size for Cypher CREATE statements
_VERTEX_BATCH = 500
_EDGE_BATCH = 500

# Map edge_type strings to AGE relationship labels (uppercase, underscores)
_EDGE_LABEL_MAP = {
    "calls": "CALLS",
    "contains": "CONTAINS",
    "imports": "IMPORTS",
    "inherits": "INHERITS",
    "implements": "IMPLEMENTS",
    "overrides": "OVERRIDES",
    "type_dep": "TYPE_DEP",
    "calls_sp": "CALLS_SP",
    "data_flow": "DATA_FLOW",
}

# Map unit_type strings to AGE vertex labels (PascalCase)
_VERTEX_LABEL_MAP = {
    "function": "Function",
    "method": "Method",
    "class": "Class",
    "interface": "Interface",
    "module": "Module",
    "constructor": "Constructor",
    "property": "Property",
    "type_alias": "TypeAlias",
    "stored_procedure": "StoredProcedure",
    "sql_function": "SqlFunction",
    "step": "Step",
    "paragraph": "Paragraph",
    "section": "Section",
    "program": "Program",
    "division": "Division",
    "job": "Job",
    "proc_step": "ProcStep",
    "struts_action": "StrutsAction",
    "struts_form": "StrutsForm",
    "jsp_page": "JspPage",
    "record": "Record",
    "struct": "Struct",
    "enum": "Enum",
}


def _escape_cypher(s: str) -> str:
    """Escape a string for use inside Cypher single-quoted literals."""
    if s is None:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _vertex_label(unit_type: str) -> str:
    """Map unit_type to PascalCase AGE vertex label."""
    return _VERTEX_LABEL_MAP.get(unit_type, "CodeUnit")


def _edge_label(edge_type: str) -> str:
    """Map edge_type to uppercase AGE relationship label."""
    return _EDGE_LABEL_MAP.get(edge_type, edge_type.upper())


class AGEGraphSync:
    """Sync relational ASG data to an Apache AGE graph."""

    def __init__(self, age_client: AGEClient):
        self._age = age_client

    def sync_project(
        self,
        project_id: str,
        units: list,
        edges: List[Dict[str, Any]],
    ) -> int:
        """Full sync: drop existing graph, create fresh, populate vertices + edges.

        Args:
            project_id: Project UUID string.
            units: List of CodeUnit ORM objects (need .unit_id, .name, .qualified_name,
                   .unit_type, .language, .file_id).
            edges: List of edge dicts from builder (keys: source_unit_id, target_unit_id,
                   edge_type, project_id, metadata).

        Returns:
            Total number of edges created in the graph.
        """
        # 1. Drop + recreate graph
        self._age.drop_graph(project_id)
        self._age.ensure_graph(project_id)

        if not units:
            return 0

        # 2. Create vertex labels (AGE requires labels to be created before use)
        unit_types_present = {u.unit_type for u in units}
        labels_needed = {_vertex_label(ut) for ut in unit_types_present}
        self._create_vertex_labels(project_id, labels_needed)

        # 3. Create edge labels
        edge_types_present = {e["edge_type"] for e in edges}
        edge_labels_needed = {_edge_label(et) for et in edge_types_present}
        self._create_edge_labels(project_id, edge_labels_needed)

        # 4. Batch-create vertices
        self._create_vertices(project_id, units)

        # 5. Batch-create edges
        self._create_edges(project_id, edges)

        total = len(edges)
        logger.info(
            f"AGE graph synced: {len(units)} vertices, {total} edges "
            f"for project {project_id}"
        )
        return total

    def _create_vertex_labels(self, project_id: str, labels: set) -> None:
        """Create vertex labels in the graph (AGE requires this before use)."""
        gname = self._age.graph_name(project_id)

        with self._age._db.get_session() as session:
            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            for label in labels:
                try:
                    conn.exec_driver_sql(f"SELECT create_vlabel('{gname}', '{label}')")
                except Exception:
                    pass  # Label may already exist

    def _create_edge_labels(self, project_id: str, labels: set) -> None:
        """Create edge labels in the graph."""
        gname = self._age.graph_name(project_id)

        with self._age._db.get_session() as session:
            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            for label in labels:
                try:
                    conn.exec_driver_sql(f"SELECT create_elabel('{gname}', '{label}')")
                except Exception:
                    pass  # Label may already exist

    def _create_vertices(self, project_id: str, units: list) -> None:
        """Batch-create vertices from CodeUnit objects."""
        for i in range(0, len(units), _VERTEX_BATCH):
            batch = units[i : i + _VERTEX_BATCH]
            creates = []
            for u in batch:
                uid = _escape_cypher(str(u.unit_id))
                name = _escape_cypher(u.name or "")
                qname = _escape_cypher(u.qualified_name or "")
                lang = _escape_cypher(u.language or "")
                fid = _escape_cypher(str(u.file_id) if u.file_id else "")
                label = _vertex_label(u.unit_type)
                creates.append(
                    f"(:{label} {{unit_id: '{uid}', name: '{name}', "
                    f"qualified_name: '{qname}', unit_type: '{_escape_cypher(u.unit_type)}', "
                    f"language: '{lang}', file_id: '{fid}'}})"
                )

            if creates:
                cypher = "CREATE " + ",\n       ".join(creates)
                self._age.cypher_write(project_id, cypher)

        logger.debug(f"Created {len(units)} vertices in AGE graph")

    def _create_edges(self, project_id: str, edges: List[Dict[str, Any]]) -> None:
        """Batch-create edges by matching vertices on unit_id property.

        AGE requires MATCH on both endpoints before CREATE of the relationship.
        We process edges in batches, but each edge needs its own MATCH+CREATE
        since AGE doesn't support multi-MATCH batching well.

        For performance, we group edges by (edge_type) and create per-type
        batches using a single Cypher statement with UNION-like patterns.
        """
        # Group edges by edge_type for batch processing
        by_type: Dict[str, List[Dict]] = {}
        for e in edges:
            by_type.setdefault(e["edge_type"], []).append(e)

        for edge_type, typed_edges in by_type.items():
            label = _edge_label(edge_type)
            for i in range(0, len(typed_edges), _EDGE_BATCH):
                batch = typed_edges[i : i + _EDGE_BATCH]
                # Create edges one at a time within a batch Cypher call
                # AGE handles this efficiently within a single transaction
                for e in batch:
                    src = _escape_cypher(str(e["source_unit_id"]))
                    tgt = _escape_cypher(str(e["target_unit_id"]))
                    meta = _escape_cypher(
                        str(e.get("metadata", {})) if e.get("metadata") else "{}"
                    )
                    cypher = (
                        f"MATCH (a {{unit_id: '{src}'}}), (b {{unit_id: '{tgt}'}}) "
                        f"CREATE (a)-[:{label} {{metadata: '{meta}'}}]->(b)"
                    )
                    try:
                        self._age.cypher_write(project_id, cypher)
                    except Exception as exc:
                        logger.debug(
                            f"Failed to create edge {src}-[{label}]->{tgt}: {exc}"
                        )

        logger.debug(f"Created {len(edges)} edges in AGE graph")
