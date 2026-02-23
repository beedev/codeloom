"""ASG Builder — extracts semantic edges from parsed code units.

Runs as a post-processing step after AST parsing. Reads all code units
for a project from the database, detects relationships (contains, imports,
calls, inherits, implements, overrides, type_dep, calls_sp), and stores
them as CodeEdge records.

Edge detection logic is delegated to domain-specific modules:
- structural.py: contains, imports, calls
- oop.py: inherits, implements, overrides, type_dep
- stored_proc.py: calls_sp

Each module receives an EdgeContext (shared lookup structures) and returns
List[dict] of edge records ready for bulk insertion.
"""

import logging
from typing import List, Set, Tuple
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..db import DatabaseManager
from ..db.models import CodeEdge, CodeUnit

from .context import EdgeContext
from . import structural, oop, stored_proc, struts as struts_edges

logger = logging.getLogger(__name__)


class ASGBuilder:
    """Build the Abstract Semantic Graph for a project.

    Edge types produced:
    - contains:   class -> method/property/constructor (from parent_name field)
    - imports:    file unit -> imported unit (resolved within project)
    - calls:      function/method -> function/method (identifier matching)
    - inherits:   class -> base class (from signature parsing or metadata)
    - implements: class/struct/record -> interface (from metadata["implements"])
    - overrides:  method -> parent method (from @Override / override modifier)
    - calls_sp:   app code unit -> stored procedure (SP invocation patterns)
    - type_dep:   consumer -> referenced type (field types, param types, return types)
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def build_edges(self, project_id: str) -> int:
        """Build all ASG edges for a project.

        Loads all code units, detects relationships, and bulk-inserts
        CodeEdge records. Uses ON CONFLICT DO NOTHING to be idempotent.

        Args:
            project_id: UUID string of the project

        Returns:
            Number of edges created
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        with self._db.get_session() as session:
            # Load all code units for this project
            units: List[CodeUnit] = (
                session.query(CodeUnit)
                .filter(CodeUnit.project_id == pid)
                .all()
            )

            if not units:
                logger.info(f"No code units found for project {project_id}")
                return 0

            logger.info(f"Building ASG for {len(units)} code units")

            # Build shared lookup context
            ctx = EdgeContext.from_units(units, pid)

            # Collect all edges from domain-specific detectors
            edges: List[dict] = []

            # Structural edges
            edges.extend(structural.detect_contains(ctx))
            edges.extend(structural.detect_imports(ctx))
            edges.extend(structural.detect_calls(ctx))

            # OOP edges
            edges.extend(oop.detect_inherits(ctx))
            edges.extend(oop.detect_implements(ctx))
            edges.extend(oop.detect_overrides(ctx))
            edges.extend(oop.detect_type_deps(ctx))

            # Domain-gated detectors
            if any(u.unit_type in ("stored_procedure", "sql_function") for u in units):
                edges.extend(stored_proc.detect_sp_calls(ctx))

            if any(u.unit_type.startswith("struts") or u.unit_type == "jsp_page" for u in units):
                edges.extend(struts_edges.detect_struts_edges(ctx))

            # Deduplicate
            seen: Set[Tuple] = set()
            unique_edges = []
            for e in edges:
                key = (e["source_unit_id"], e["target_unit_id"], e["edge_type"])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)

            # Bulk insert with ON CONFLICT DO NOTHING
            if unique_edges:
                stmt = pg_insert(CodeEdge).values(unique_edges)
                stmt = stmt.on_conflict_do_nothing(
                    constraint="uq_code_edge"
                )
                session.execute(stmt)

            logger.info(f"ASG built: {len(unique_edges)} edges for project {project_id}")
            return len(unique_edges)

    # ── Re-enrichment for existing projects ──────────────────────────

    @staticmethod
    def enrich_class_fields_from_db(db_manager: "DatabaseManager", project_id: str) -> int:
        """Re-enrich class units with field metadata for existing projects.

        For projects already ingested (class units in DB but no metadata["fields"]),
        this reads each class unit's stored source, parses it with tree-sitter,
        and updates the JSONB metadata with extracted fields.

        Called by the build-asg endpoint before edge building so that
        type_dep detection has field data to work with.

        Returns:
            Number of class units updated with field metadata.
        """
        from ..ast_parser.enricher import SemanticEnricher

        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        enricher = SemanticEnricher()
        updated = 0

        with db_manager.get_session() as session:
            class_units = (
                session.query(CodeUnit)
                .filter(
                    CodeUnit.project_id == pid,
                    CodeUnit.unit_type.in_(("class", "interface", "struct", "record")),
                )
                .all()
            )

            for cu in class_units:
                if not cu.source or not cu.language:
                    continue

                # Skip if fields already extracted
                meta = cu.unit_metadata or {}
                if meta.get("fields"):
                    continue

                fields = enricher.enrich_class_source(cu.source, cu.language)
                if fields:
                    new_meta = dict(meta)
                    new_meta["fields"] = fields
                    cu.unit_metadata = new_meta
                    updated += 1

        logger.info(f"Field re-enrichment: {updated}/{len(class_units)} class units updated")
        return updated
