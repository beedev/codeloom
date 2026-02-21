"""Entry point detection and call tree tracing.

Two-pass entry point detection:
  Pass 1 (heuristic): Units with zero incoming 'calls' edges
  Pass 2 (annotation): Units matching framework annotation patterns

Call tree tracing via recursive CTE with ARRAY path accumulator
for cycle prevention, extending the pattern from
core/asg_builder/queries.py:_traverse().
"""

import logging
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager
from .models import CallTreeNode, EntryPoint, EntryPointType

logger = logging.getLogger(__name__)


# Annotation patterns by language for Pass 2 detection
_ANNOTATION_PATTERNS = {
    "java": {
        EntryPointType.HTTP_ENDPOINT: [
            r"@(Get|Post|Put|Delete|Patch)Mapping",
            r"@RequestMapping",
            r"@RestController",
        ],
        EntryPointType.MESSAGE_HANDLER: [
            r"@KafkaListener",
            r"@JmsListener",
            r"@RabbitListener",
            r"@SqsListener",
        ],
        EntryPointType.SCHEDULED_TASK: [
            r"@Scheduled",
        ],
        EntryPointType.EVENT_LISTENER: [
            r"@EventListener",
            r"@TransactionalEventListener",
        ],
        EntryPointType.STARTUP_HOOK: [
            r"@PostConstruct",
            r"CommandLineRunner",
            r"ApplicationRunner",
        ],
    },
    "csharp": {
        EntryPointType.HTTP_ENDPOINT: [
            r"\[Http(Get|Post|Put|Delete|Patch)\]",
            r"\[Route\(",
            r": ControllerBase",
            r": Controller",
        ],
        EntryPointType.MESSAGE_HANDLER: [
            r"IConsumer<",
            r"IMessageHandler",
        ],
        EntryPointType.STARTUP_HOOK: [
            r"IHostedService",
            r"BackgroundService",
        ],
    },
    "python": {
        EntryPointType.HTTP_ENDPOINT: [
            r"@app\.(get|post|put|delete|patch|route)\(",
            r"@router\.(get|post|put|delete|patch)\(",
            r"class.*View\b.*:",
        ],
        EntryPointType.CLI_COMMAND: [
            r"@click\.command",
            r"def main\(",
            r'if __name__.*==.*"__main__"',
        ],
        EntryPointType.SCHEDULED_TASK: [
            r"@celery_app\.task",
            r"@shared_task",
        ],
    },
    "javascript": {
        EntryPointType.HTTP_ENDPOINT: [
            r"router\.(get|post|put|delete|patch)\(",
            r"app\.(get|post|put|delete|patch)\(",
            r"export\s+(default\s+)?async\s+function\s+(GET|POST|PUT|DELETE)",
        ],
        EntryPointType.MESSAGE_HANDLER: [
            r"\.on\(['\"]message['\"]",
            r"consumer\.subscribe",
        ],
    },
    "typescript": {
        EntryPointType.HTTP_ENDPOINT: [
            r"@(Get|Post|Put|Delete|Patch)\(",
            r"router\.(get|post|put|delete|patch)\(",
            r"export\s+(default\s+)?async\s+function\s+(GET|POST|PUT|DELETE)",
        ],
        EntryPointType.MESSAGE_HANDLER: [
            r"\.on\(['\"]message['\"]",
            r"@MessagePattern",
        ],
    },
}


class ChainTracer:
    """Detect entry points and trace call trees through the ASG.

    Args:
        db: DatabaseManager instance
        max_entry_points: Optional cap on detected entry points (from config)
    """

    def __init__(self, db: DatabaseManager, max_entry_points: Optional[int] = None):
        self._db = db
        self._max_entry_points = max_entry_points

    def detect_entry_points(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Detect all entry points for a project using dual-pass strategy.

        Pass 1 -- Heuristic: functions/methods with zero incoming 'calls'
            edges (conservative heuristic).

        Pass 2 -- Annotation: units whose source/metadata matches known
            framework annotation patterns (e.g., @GetMapping, @app.route).

        Results are merged (union by unit_id), with Pass 2 classifications
        taking precedence over Pass 1's UNKNOWN type.

        Args:
            project_id: UUID string of the project

        Returns:
            Deduplicated list of EntryPoint objects, sorted by file path
        """
        pass1 = self._detect_pass1_heuristic(project_id)
        pass2 = self._detect_pass2_annotations(project_id)
        merged = self._merge_entry_points(pass1, pass2)

        if self._max_entry_points and len(merged) > self._max_entry_points:
            logger.info(
                f"Capping entry points from {len(merged)} to {self._max_entry_points}"
            )
            merged = merged[:self._max_entry_points]

        return merged

    def trace_call_tree(
        self,
        project_id: str,
        entry_unit_id: str,
        max_depth: int = 10,
    ) -> CallTreeNode:
        """Trace the complete call tree from an entry point.

        Uses a recursive CTE with ARRAY path accumulator to:
        - Prevent cycles (unit_id already in path array)
        - Reconstruct the full tree structure from flat rows
        - Respect max_depth limit

        Args:
            project_id: UUID string
            entry_unit_id: UUID string of the entry point unit
            max_depth: Maximum call depth (default 10)

        Returns:
            CallTreeNode tree rooted at the entry point
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        uid = UUID(entry_unit_id) if isinstance(entry_unit_id, str) else entry_unit_id

        sql = self._build_call_tree_cte()

        with self._db.get_session() as session:
            result = session.execute(
                text(sql),
                {"pid": pid, "uid": uid, "max_depth": max_depth},
            )
            rows = result.fetchall()

        if not rows:
            return self._make_leaf_node(project_id, entry_unit_id)

        return self._build_tree_from_paths(rows, entry_unit_id)

    def get_flat_unit_membership(
        self,
        tree: CallTreeNode,
    ) -> List[Dict[str, Any]]:
        """Flatten a call tree into a list of unit memberships.

        Used to populate the analysis_units junction table.

        Args:
            tree: Root CallTreeNode

        Returns:
            List of dicts with unit_id, min_depth, path_count (deduplicated)
        """
        acc: Dict[str, Dict[str, int]] = {}

        def _walk(node: CallTreeNode, path_seen: Optional[Set[str]] = None):
            local_seen = set(path_seen or set())
            if node.unit_id in local_seen:
                return
            local_seen.add(node.unit_id)

            slot = acc.setdefault(node.unit_id, {"min_depth": node.depth, "path_count": 0})
            slot["min_depth"] = min(slot["min_depth"], node.depth)
            slot["path_count"] += 1

            for child in node.children:
                _walk(child, local_seen)

        _walk(tree)
        return [
            {
                "unit_id": uid,
                "min_depth": vals["min_depth"],
                "path_count": vals["path_count"],
            }
            for uid, vals in acc.items()
        ]

    # -- Private: Pass 1 (Heuristic) ----------------------------------------

    def _detect_pass1_heuristic(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Find units with zero incoming 'calls' edges.

        SQL: Select code_units that have no rows in code_edges where
        target_unit_id = unit_id AND edge_type = 'calls'.
        Filter to callable types (function, method) only.
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        sql = """
            SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
                   u.language, u.start_line, u.end_line,
                   f.file_path, u.metadata
            FROM code_units u
            JOIN code_files f ON u.file_id = f.file_id
            WHERE u.project_id = :pid
              AND u.unit_type IN ('function', 'method')
              AND NOT EXISTS (
                  SELECT 1 FROM code_edges e
                  WHERE e.target_unit_id = u.unit_id
                    AND e.edge_type = 'calls'
                    AND e.project_id = :pid
              )
            ORDER BY f.file_path, u.start_line
        """

        with self._db.get_session() as session:
            result = session.execute(text(sql), {"pid": pid})
            rows = result.fetchall()

        return [
            EntryPoint(
                unit_id=str(row.unit_id),
                name=row.name,
                qualified_name=row.qualified_name or row.name,
                file_path=row.file_path,
                entry_type=self._classify_entry_type(row),
                language=row.language or "unknown",
                metadata=row.metadata or {},
                detected_by="heuristic",
            )
            for row in rows
        ]

    # -- Private: Pass 2 (Annotations) --------------------------------------

    def _detect_pass2_annotations(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Find units matching framework annotation patterns.

        Scans code_units.source and code_units.metadata for patterns
        defined in _ANNOTATION_PATTERNS, grouped by language.
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        with self._db.get_session() as session:
            lang_result = session.execute(
                text("""
                    SELECT DISTINCT language
                    FROM code_units
                    WHERE project_id = :pid AND language IS NOT NULL
                """),
                {"pid": pid},
            )
            languages = [row.language for row in lang_result.fetchall()]

        entry_points: List[EntryPoint] = []

        for lang in languages:
            patterns = _ANNOTATION_PATTERNS.get(lang, {})
            if not patterns:
                continue

            for entry_type, regexes in patterns.items():
                combined = "|".join(f"({r})" for r in regexes)

                sql = """
                    SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
                           u.language, u.start_line, u.end_line,
                           f.file_path, u.metadata, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.language = :lang
                      AND (
                          u.source ~ :pattern
                          OR u.metadata::text ~ :pattern
                      )
                """

                with self._db.get_session() as session:
                    result = session.execute(
                        text(sql),
                        {"pid": pid, "lang": lang, "pattern": combined},
                    )
                    for row in result.fetchall():
                        entry_points.append(EntryPoint(
                            unit_id=str(row.unit_id),
                            name=row.name,
                            qualified_name=row.qualified_name or row.name,
                            file_path=row.file_path,
                            entry_type=entry_type,
                            language=row.language or lang,
                            metadata=row.metadata or {},
                            detected_by="annotation",
                        ))

        return entry_points

    # -- Private: Merge & Classify -------------------------------------------

    def _merge_entry_points(
        self,
        pass1: List[EntryPoint],
        pass2: List[EntryPoint],
    ) -> List[EntryPoint]:
        """Merge Pass 1 and Pass 2 results, deduplicating by unit_id.

        Pass 2 classifications take precedence (they have explicit type info).
        """
        by_id: Dict[str, EntryPoint] = {}

        for ep in pass1:
            by_id[ep.unit_id] = ep

        for ep in pass2:
            existing = by_id.get(ep.unit_id)
            if existing:
                existing.entry_type = ep.entry_type
                existing.detected_by = "both"
            else:
                by_id[ep.unit_id] = ep

        return sorted(by_id.values(), key=lambda e: (e.file_path, e.name))

    def _classify_entry_type(self, row) -> EntryPointType:
        """Classify entry type from heuristic detection (Pass 1)."""
        name = (row.name or "").lower()
        meta = row.metadata or {}
        modifiers = meta.get("modifiers", [])

        if name == "main" and "static" in modifiers:
            return EntryPointType.CLI_COMMAND
        if name.startswith("test_") or name.startswith("test"):
            return EntryPointType.UNKNOWN
        if meta.get("is_endpoint"):
            return EntryPointType.HTTP_ENDPOINT

        return EntryPointType.PUBLIC_API

    def _build_call_tree_cte(self) -> str:
        """Build the recursive CTE SQL for call tree tracing."""
        return """
            WITH RECURSIVE call_tree AS (
                -- Seed: the entry point itself
                SELECT
                    u.unit_id,
                    u.name,
                    u.qualified_name,
                    u.unit_type,
                    u.language,
                    u.start_line,
                    u.end_line,
                    u.source,
                    f.file_path,
                    0 AS depth,
                    'root'::text AS edge_type,
                    ARRAY[u.unit_id] AS path
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.unit_id = :uid

                UNION ALL

                -- Recurse: follow outgoing calls/calls_sp edges
                SELECT
                    tu.unit_id,
                    tu.name,
                    tu.qualified_name,
                    tu.unit_type,
                    tu.language,
                    tu.start_line,
                    tu.end_line,
                    tu.source,
                    tf.file_path,
                    ct.depth + 1,
                    e.edge_type,
                    ct.path || tu.unit_id
                FROM call_tree ct
                JOIN code_edges e ON e.source_unit_id = ct.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                JOIN code_files tf ON tu.file_id = tf.file_id
                WHERE e.project_id = :pid
                  AND e.edge_type IN ('calls', 'calls_sp')
                  AND ct.depth < :max_depth
                  AND NOT (tu.unit_id = ANY(ct.path))  -- cycle prevention
            )
            SELECT unit_id, name, qualified_name, unit_type, language,
                   start_line, end_line, source, file_path,
                   depth, edge_type, path
            FROM call_tree
            ORDER BY depth, name
        """

    def _build_tree_from_paths(
        self,
        rows: list,
        root_unit_id: str,
    ) -> CallTreeNode:
        """Reconstruct nested CallTreeNode from flat CTE rows.

        Algorithm: trie-like insertion using the path arrays.
        """
        row_data = {}
        for row in rows:
            uid = str(row.unit_id)
            if uid not in row_data or row.depth < row_data[uid]["depth"]:
                row_data[uid] = {
                    "unit_id": uid,
                    "name": row.name,
                    "qualified_name": row.qualified_name or row.name,
                    "unit_type": row.unit_type,
                    "language": row.language or "unknown",
                    "file_path": row.file_path,
                    "start_line": row.start_line or 0,
                    "end_line": row.end_line or 0,
                    "source": row.source,
                    "depth": row.depth,
                    "edge_type": row.edge_type,
                }

        # Build parent->children map from path arrays
        children_map: Dict[str, List[str]] = {}
        for row in rows:
            path = [str(p) for p in row.path]
            if len(path) >= 2:
                parent_id = path[-2]
                child_id = path[-1]
                if parent_id not in children_map:
                    children_map[parent_id] = []
                if child_id not in children_map[parent_id]:
                    children_map[parent_id].append(child_id)

        def _build(uid: str, ancestors: frozenset = frozenset()) -> CallTreeNode:
            data = row_data.get(uid, {})
            node = CallTreeNode(
                unit_id=uid,
                name=data.get("name", ""),
                qualified_name=data.get("qualified_name", ""),
                unit_type=data.get("unit_type", "unknown"),
                language=data.get("language", "unknown"),
                file_path=data.get("file_path", ""),
                start_line=data.get("start_line", 0),
                end_line=data.get("end_line", 0),
                source=data.get("source"),
                depth=data.get("depth", 0),
                edge_type=data.get("edge_type", "calls"),
            )
            new_ancestors = ancestors | {uid}
            for child_id in children_map.get(uid, []):
                if child_id not in new_ancestors:
                    node.children.append(_build(child_id, new_ancestors))
            return node

        return _build(str(root_unit_id))

    def _make_leaf_node(
        self,
        project_id: str,
        unit_id: str,
    ) -> CallTreeNode:
        """Create a leaf CallTreeNode for a unit with no outgoing calls."""
        pid = UUID(project_id)
        uid = UUID(unit_id)

        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
                           u.language, u.start_line, u.end_line, u.source,
                           f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.unit_id = :uid AND u.project_id = :pid
                """),
                {"pid": pid, "uid": uid},
            )
            row = result.fetchone()

        if not row:
            return CallTreeNode(
                unit_id=str(unit_id), name="unknown", qualified_name="unknown",
                unit_type="unknown", language="unknown", file_path="",
                start_line=0, end_line=0, source=None, depth=0,
            )

        return CallTreeNode(
            unit_id=str(row.unit_id),
            name=row.name,
            qualified_name=row.qualified_name or row.name,
            unit_type=row.unit_type,
            language=row.language or "unknown",
            file_path=row.file_path,
            start_line=row.start_line or 0,
            end_line=row.end_line or 0,
            source=row.source,
            depth=0,
        )
