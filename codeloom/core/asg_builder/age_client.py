"""Apache AGE client — thin wrapper for Cypher queries via SQLAlchemy.

Uses ag_catalog.cypher() SQL function. No external AGE Python driver needed.
Cypher queries use exec_driver_sql() to avoid SQLAlchemy interpreting
Cypher labels (e.g. :Function) as bind parameters.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)

# Regex to strip AGE type annotations from agtype values
# e.g. '{"id": 1, "name": "foo"}::vertex' -> '{"id": 1, "name": "foo"}'
_AGTYPE_SUFFIX_RE = re.compile(r"::(?:vertex|edge|path|numeric|integer|float|string|boolean|list|map)\s*$")


def _parse_agtype(value: Any) -> Any:
    """Parse an agtype value returned by AGE into a Python object.

    AGE returns results as the custom 'agtype' type which psycopg2
    delivers as strings with type suffixes (e.g. '42::integer',
    '"hello"::string', '{"id": 1}::vertex').

    Strategy: strip the type suffix, then json.loads the remainder.
    """
    if value is None:
        return None

    s = str(value).strip()

    # Strip type annotation suffix
    s = _AGTYPE_SUFFIX_RE.sub("", s).strip()

    # Try JSON parse
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        # Return as-is for simple scalars
        return s


class AGEClient:
    """Thin wrapper for Apache AGE Cypher queries.

    One graph per project: ``asg_<project_id_hex>``.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    # ── Graph lifecycle ──────────────────────────────────────────────

    @staticmethod
    def graph_name(project_id: str) -> str:
        """Derive AGE graph name from project UUID.

        AGE graph names must be valid SQL identifiers (no hyphens).
        """
        return f"asg_{project_id.replace('-', '')}"

    def ensure_graph(self, project_id: str) -> None:
        """Create the project's graph if it doesn't already exist."""
        gname = self.graph_name(project_id)
        with self._db.get_session() as session:
            # Check if graph already exists
            result = session.execute(
                text("SELECT 1 FROM ag_catalog.ag_graph WHERE name = :name"),
                {"name": gname},
            )
            if result.fetchone():
                return

            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            conn.exec_driver_sql(f"SELECT * FROM ag_catalog.create_graph('{gname}')")
            logger.info(f"Created AGE graph: {gname}")

    def drop_graph(self, project_id: str) -> None:
        """Drop the project's graph if it exists."""
        gname = self.graph_name(project_id)
        with self._db.get_session() as session:
            # Check if graph exists first
            result = session.execute(
                text("SELECT 1 FROM ag_catalog.ag_graph WHERE name = :name"),
                {"name": gname},
            )
            if not result.fetchone():
                return

            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            conn.exec_driver_sql(f"SELECT * FROM ag_catalog.drop_graph('{gname}', true)")
            logger.info(f"Dropped AGE graph: {gname}")

    def graph_exists(self, project_id: str) -> bool:
        """Check whether the project's AGE graph exists."""
        gname = self.graph_name(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("SELECT 1 FROM ag_catalog.ag_graph WHERE name = :name"),
                {"name": gname},
            )
            return result.fetchone() is not None

    # ── Cypher execution ─────────────────────────────────────────────

    def cypher(
        self,
        project_id: str,
        query: str,
        columns: List[str],
        column_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a read Cypher query against a project's graph.

        Args:
            project_id: Project UUID string.
            query: Cypher query string (without $$-quoting).
            columns: Column names for the result set.
            column_types: Optional SQL types for columns (default: 'agtype' for all).

        Returns:
            List of dicts mapping column names to parsed Python values.
        """
        gname = self.graph_name(project_id)

        # Build the column type list for the AS clause
        if column_types is None:
            column_types = ["agtype"] * len(columns)
        as_clause = ", ".join(
            f"{col} {ctype}" for col, ctype in zip(columns, column_types)
        )

        # Escape single quotes in Cypher query for the SQL wrapper
        safe_query = query.replace("'", "''")

        sql = f"""
            SELECT * FROM ag_catalog.cypher('{gname}', $$
                {query}
            $$) AS ({as_clause})
        """

        with self._db.get_session() as session:
            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            result = conn.exec_driver_sql(sql)
            rows = result.fetchall()

        return [
            {col: _parse_agtype(row[i]) for i, col in enumerate(columns)}
            for row in rows
        ]

    def cypher_write(
        self,
        project_id: str,
        query: str,
    ) -> None:
        """Execute a write Cypher query (CREATE, SET, DELETE, etc.).

        No result set expected — used for graph mutations.
        Uses exec_driver_sql to bypass SQLAlchemy's bind-parameter parsing,
        since Cypher labels like :Function would be misinterpreted.
        """
        gname = self.graph_name(project_id)

        sql = f"""
            SELECT * FROM ag_catalog.cypher('{gname}', $$
                {query}
            $$) AS (result agtype)
        """

        with self._db.get_session() as session:
            conn = session.connection()
            conn.exec_driver_sql("LOAD 'age'")
            conn.exec_driver_sql("SET search_path = ag_catalog, \"$user\", public")
            conn.exec_driver_sql(sql)

    # ── Utility ──────────────────────────────────────────────────────

    def vertex_count(self, project_id: str) -> int:
        """Count all vertices in the project's graph."""
        rows = self.cypher(
            project_id,
            "MATCH (n) RETURN count(n) AS cnt",
            columns=["cnt"],
        )
        return int(rows[0]["cnt"]) if rows else 0

    def edge_count(self, project_id: str) -> int:
        """Count all edges in the project's graph."""
        rows = self.cypher(
            project_id,
            "MATCH ()-[r]->() RETURN count(r) AS cnt",
            columns=["cnt"],
        )
        return int(rows[0]["cnt"]) if rows else 0
