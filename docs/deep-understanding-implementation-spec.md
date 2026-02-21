# Deep Understanding Engine — Implementation Specification

> **Status**: Draft
> **Version**: 1.0
> **Last Updated**: 2026-02-19
> **Scope**: `codeloom/core/understanding/` — new module
> **Dependencies**: Existing ASG, DB, RAPTOR, migration, and retrieval subsystems

---

## Table of Contents

1. [Module-Level Design](#1-module-level-design)
2. [Database Design](#2-database-design)
3. [Algorithm Specifications](#3-algorithm-specifications)
4. [Integration Points](#4-integration-points)
5. [API Design](#5-api-design)
6. [Prompt Engineering](#6-prompt-engineering)
7. [Worker State Machine](#7-worker-state-machine)
8. [Configuration Schema](#8-configuration-schema)
9. [Testing Strategy](#9-testing-strategy)
10. [Deployment & Rollout](#10-deployment--rollout)

---

## 1. Module-Level Design

### 1.1 Package Structure

```
codeloom/core/understanding/
├── __init__.py          # Public API exports
├── models.py            # Data contracts (dataclasses)
├── chain_tracer.py      # Entry point detection + call tree tracing
├── analyzer.py          # Tiered LLM chain analysis
├── worker.py            # Background job processing
├── engine.py            # Orchestrator (public API)
├── prompts.py           # Prompt templates
└── frameworks/
    ├── __init__.py      # detect_and_analyze() registry
    ├── base.py          # FrameworkAnalyzer ABC + FrameworkContext
    ├── spring.py        # Spring Boot / Spring MVC analyzer
    └── aspnet.py        # ASP.NET Core analyzer
```

### 1.2 `models.py` — Data Contracts

```python
"""Data contracts for the Deep Understanding Engine.

All structured types used across the understanding subsystem.
Kept as dataclasses (not ORM models) for transport between layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class EntryPointType(Enum):
    """Classification of detected entry points."""
    HTTP_ENDPOINT = "http_endpoint"       # REST/GraphQL controller action
    MESSAGE_HANDLER = "message_handler"   # Queue/event consumer
    SCHEDULED_TASK = "scheduled_task"     # Cron / timer-triggered
    CLI_COMMAND = "cli_command"           # Console / CLI entry
    EVENT_LISTENER = "event_listener"    # Domain event subscriber
    STARTUP_HOOK = "startup_hook"        # App initialization
    PUBLIC_API = "public_api"            # Exported library method
    UNKNOWN = "unknown"


class AnalysisTier(Enum):
    """Token budget tier for chain analysis."""
    TIER_1 = "tier_1"   # Full source, ≤100K tokens
    TIER_2 = "tier_2"   # Depth-prioritized truncation, ≤200K tokens
    TIER_3 = "tier_3"   # Summarization fallback, >200K tokens


@dataclass
class EvidenceRef:
    """Links an extracted fact back to source code.

    Every business rule, data entity, or integration discovered by the
    LLM must carry at least one EvidenceRef so reviewers can verify.
    """
    unit_id: str                  # UUID of the CodeUnit
    qualified_name: str           # e.g. "com.acme.OrderService.placeOrder"
    file_path: str                # Relative file path
    start_line: int
    end_line: int
    snippet: Optional[str] = None # Relevant code excerpt (≤10 lines)


@dataclass
class EntryPoint:
    """A detected entry point into the application."""
    unit_id: str
    name: str
    qualified_name: str
    file_path: str
    entry_type: EntryPointType
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Detection provenance
    detected_by: str = "heuristic"  # "heuristic" | "annotation" | "both"


@dataclass
class CallTreeNode:
    """A node in the traced call tree.

    Recursive structure: each node holds its children.
    The root node is the entry point itself.
    """
    unit_id: str
    name: str
    qualified_name: str
    unit_type: str           # function, method, class, stored_procedure
    language: str
    file_path: str
    start_line: int
    end_line: int
    source: Optional[str]    # Full source text (None if truncated)
    depth: int               # 0 = entry point
    edge_type: str = "calls" # Edge type that led here
    children: List["CallTreeNode"] = field(default_factory=list)
    token_count: int = 0     # Tokens in source (computed by analyzer)


@dataclass
class DeepContextBundle:
    """Complete analysis output for one entry point.

    The full bundle shape is persisted in deep_analyses.result_json
    for forward-compatible reprocessing and auditability.
    """
    entry_point: EntryPoint
    tier: AnalysisTier
    total_units: int
    total_tokens: int

    # Extracted knowledge — each item carries EvidenceRefs
    business_rules: List[Dict[str, Any]] = field(default_factory=list)
    data_entities: List[Dict[str, Any]] = field(default_factory=list)
    integrations: List[Dict[str, Any]] = field(default_factory=list)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    cross_cutting_concerns: List[str] = field(default_factory=list)

    # Narrative summary (for chat injection)
    narrative: str = ""
    confidence: float = 0.0        # 0..1 aggregate extraction confidence
    coverage: float = 0.0          # 0..1 chain coverage score
    chain_truncated: bool = False  # True when tiers 2/3 truncated or summarized deep code

    # Schema version for forward compatibility
    schema_version: int = 1
    prompt_version: str = "v1.0"
    analyzed_at: Optional[str] = None  # ISO timestamp


@dataclass
class AnalysisError:
    """Structured error from analysis pipeline."""
    phase: str          # "detection" | "tracing" | "analysis" | "storage"
    message: str
    unit_id: Optional[str] = None
    recoverable: bool = True
```

### 1.3 `chain_tracer.py` — ChainTracer

```python
"""Entry point detection and call tree tracing.

Two-pass entry point detection:
  Pass 1 (heuristic): Units with zero incoming 'calls' edges
  Pass 2 (annotation): Units matching framework annotation patterns

Call tree tracing via recursive CTE with ARRAY path accumulator
for cycle prevention, extending the pattern from
core/asg_builder/queries.py:_traverse().
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager
from ..db.models import CodeUnit
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
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    def detect_entry_points(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Detect all entry points for a project using dual-pass strategy.

        Pass 1 — Heuristic: functions/methods with zero incoming 'calls'
            edges (conservative heuristic).

        Pass 2 — Annotation: units whose source/metadata matches known
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
        return self._merge_entry_points(pass1, pass2)

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

        The CTE extends the pattern from core/asg_builder/queries.py:_traverse()
        (lines 436-524) with two additions:
        1. ARRAY path accumulator column for cycle detection
        2. Full path array returned for tree reconstruction

        Args:
            project_id: UUID string
            entry_unit_id: UUID string of the entry point unit
            max_depth: Maximum call depth (default 10)

        Returns:
            CallTreeNode tree rooted at the entry point
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        uid = UUID(entry_unit_id) if isinstance(entry_unit_id, str) else entry_unit_id

        # See Section 3.2 for the full CTE SQL
        sql = self._build_call_tree_cte()

        with self._db.get_session() as session:
            result = session.execute(
                text(sql),
                {"pid": pid, "uid": uid, "max_depth": max_depth},
            )
            rows = result.fetchall()

        if not rows:
            # Return a leaf node for the entry point itself
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

    # ── Private: Pass 1 (Heuristic) ────────────────────────────────────

    def _detect_pass1_heuristic(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Find units with zero incoming 'calls' edges.

        SQL: Select code_units that have no rows in code_edges where
        target_unit_id = unit_id AND edge_type = 'calls'.
        Filter to callable types (function, method) only.
        Note: nested-function exclusion is parser/language dependent and
        is intentionally not enforced in SQL in v1.
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

    # ── Private: Pass 2 (Annotations) ──────────────────────────────────

    def _detect_pass2_annotations(
        self,
        project_id: str,
    ) -> List[EntryPoint]:
        """Find units matching framework annotation patterns.

        Scans code_units.source and code_units.metadata for patterns
        defined in _ANNOTATION_PATTERNS, grouped by language.
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        # Get project languages
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

            # Build combined regex per entry type
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

    # ── Private: Merge & Classify ───────────────────────────────────────

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
                # Pass 2 wins on type; merge detected_by
                existing.entry_type = ep.entry_type
                existing.detected_by = "both"
            else:
                by_id[ep.unit_id] = ep

        return sorted(by_id.values(), key=lambda e: (e.file_path, e.name))

    def _classify_entry_type(self, row) -> EntryPointType:
        """Classify entry type from heuristic detection (Pass 1).

        Uses metadata hints: if the unit has modifiers like 'static',
        a name like 'main', or is a test function, classify accordingly.
        """
        name = (row.name or "").lower()
        meta = row.metadata or {}
        modifiers = meta.get("modifiers", [])

        if name == "main" and "static" in modifiers:
            return EntryPointType.CLI_COMMAND
        if name.startswith("test_") or name.startswith("test"):
            return EntryPointType.UNKNOWN  # Skip test functions
        if meta.get("is_endpoint"):
            return EntryPointType.HTTP_ENDPOINT

        return EntryPointType.PUBLIC_API

    def _build_call_tree_cte(self) -> str:
        """Build the recursive CTE SQL for call tree tracing.

        See Section 3.2 for detailed algorithm specification.
        """
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
        See Section 3.3 for detailed specification.
        """
        # Build lookup of unit_id -> row data
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

        # Recursive tree construction
        def _build(uid: str) -> CallTreeNode:
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
            for child_id in children_map.get(uid, []):
                node.children.append(_build(child_id))
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
```

### 1.4 `analyzer.py` — ChainAnalyzer

```python
"""Tiered LLM analysis of traced call chains.

Token budget algorithm:
  Tier 1: Full source fits in ≤100K tokens → send everything
  Tier 2: 100K < total ≤ 200K → depth-prioritized truncation
  Tier 3: total > 200K → summarize deep branches, send summaries + shallow source

Reuses TokenCounter from core/code_chunker/token_counter.py.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core import Settings

from ..code_chunker.token_counter import TokenCounter
from .models import (
    AnalysisTier, CallTreeNode, DeepContextBundle,
    EntryPoint, EvidenceRef,
)
from . import prompts

logger = logging.getLogger(__name__)

# Tier thresholds (tokens)
TIER_1_MAX = 100_000
TIER_2_MAX = 200_000


class ChainAnalyzer:
    """Analyze a traced call chain to extract structured understanding.

    Args:
        token_counter: Optional TokenCounter instance (created if None)
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._tc = token_counter or TokenCounter()

    def analyze_chain(
        self,
        entry_point: EntryPoint,
        call_tree: CallTreeNode,
        framework_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> DeepContextBundle:
        """Run tiered analysis on a call chain.

        Steps:
        1. Count total tokens across all units in the tree
        2. Select tier based on total tokens
        3. Prepare source payload (full, truncated, or summarized)
        4. Build prompt with framework hints
        5. Call LLM and parse structured JSON output
        6. Validate evidence references

        Args:
            entry_point: The entry point being analyzed
            call_tree: Traced call tree from ChainTracer
            framework_contexts: Optional framework analysis results

        Returns:
            DeepContextBundle with extracted knowledge
        """
        # Step 1: Count tokens
        total_tokens = self._count_tree_tokens(call_tree)

        # Step 2: Select tier
        tier = self._select_tier(total_tokens)
        logger.info(
            f"Analyzing {entry_point.qualified_name}: "
            f"{total_tokens} tokens → {tier.value}"
        )

        # Step 3: Prepare source payload
        source_payload = self._prepare_source(call_tree, tier, total_tokens)

        # Step 4: Build prompt
        prompt = prompts.build_chain_analysis_prompt(
            entry_point=entry_point,
            source_payload=source_payload,
            tier=tier,
            framework_contexts=framework_contexts or [],
        )

        # Step 5: Call LLM
        llm = Settings.llm
        if llm is None:
            raise RuntimeError("No LLM configured")

        response = llm.complete(prompt)
        raw_output = response.text.strip()

        # Step 6: Parse and validate
        parsed = self._parse_json_output(raw_output)
        bundle = self._build_bundle(
            entry_point=entry_point,
            tier=tier,
            total_tokens=total_tokens,
            call_tree=call_tree,
            parsed=parsed,
        )

        return bundle

    # ── Token Counting ──────────────────────────────────────────────────

    def _count_tree_tokens(self, node: CallTreeNode) -> int:
        """Recursively count tokens across all source in the tree."""
        count = 0
        if node.source:
            node.token_count = self._tc.count(node.source)
            count += node.token_count
        for child in node.children:
            count += self._count_tree_tokens(child)
        return count

    def _select_tier(self, total_tokens: int) -> AnalysisTier:
        """Select analysis tier based on total source tokens."""
        if total_tokens <= TIER_1_MAX:
            return AnalysisTier.TIER_1
        elif total_tokens <= TIER_2_MAX:
            return AnalysisTier.TIER_2
        else:
            return AnalysisTier.TIER_3

    # ── Source Preparation ──────────────────────────────────────────────

    def _prepare_source(
        self,
        tree: CallTreeNode,
        tier: AnalysisTier,
        total_tokens: int,
    ) -> str:
        """Prepare the source code payload for the LLM prompt.

        Tier 1: Full source from all units, formatted with file paths
        Tier 2: Depth-prioritized — full source for depth 0-2,
                 signatures only for depth 3+, fill remaining budget
                 with highest-connectivity deeper units
        Tier 3: Summarize branches at depth 3+ via separate LLM call,
                 include depth 0-1 full source + summaries
        """
        if tier == AnalysisTier.TIER_1:
            return self._format_full_source(tree)
        elif tier == AnalysisTier.TIER_2:
            return self._format_depth_prioritized(tree, budget=TIER_2_MAX)
        else:
            return self._format_with_summaries(tree, budget=TIER_2_MAX)

    def _format_full_source(self, node: CallTreeNode, indent: int = 0) -> str:
        """Format full source tree with file path headers."""
        parts = []
        prefix = "  " * indent

        header = f"{prefix}## {node.qualified_name}"
        header += f" [{node.file_path}:{node.start_line}-{node.end_line}]"
        header += f" (depth={node.depth}, type={node.unit_type})"
        parts.append(header)

        if node.source:
            parts.append(f"{prefix}```{node.language}")
            parts.append(node.source)
            parts.append(f"{prefix}```")

        for child in node.children:
            parts.append(self._format_full_source(child, indent + 1))

        return "\n".join(parts)

    def _format_depth_prioritized(
        self,
        tree: CallTreeNode,
        budget: int,
    ) -> str:
        """Depth-prioritized truncation for Tier 2.

        Algorithm:
        1. Include full source for depth 0-2
        2. For depth 3+, include only signatures
        3. If budget remains, fill with highest-connectivity deep units
        """
        parts = []
        remaining_budget = budget
        deep_candidates = []

        def _walk(node: CallTreeNode):
            nonlocal remaining_budget

            if node.depth <= 2 and node.source:
                source_tokens = node.token_count or self._tc.count(node.source)
                if source_tokens <= remaining_budget:
                    parts.append(self._format_unit_full(node))
                    remaining_budget -= source_tokens
                else:
                    parts.append(self._format_unit_signature(node))
                    remaining_budget -= 50  # Estimate for signature
            elif node.depth > 2:
                deep_candidates.append(node)
                parts.append(self._format_unit_signature(node))
                remaining_budget -= 50
            else:
                parts.append(self._format_unit_signature(node))
                remaining_budget -= 50

            for child in node.children:
                _walk(child)

        _walk(tree)

        # Fill remaining budget with deep units by connectivity
        deep_candidates.sort(
            key=lambda n: len(n.children), reverse=True
        )
        for node in deep_candidates:
            if remaining_budget <= 0:
                break
            if node.source and node.token_count <= remaining_budget:
                parts.append(f"\n### [Deep unit - full source] {node.qualified_name}")
                parts.append(f"```{node.language}\n{node.source}\n```")
                remaining_budget -= node.token_count

        return "\n".join(parts)

    def _format_with_summaries(
        self,
        tree: CallTreeNode,
        budget: int,
    ) -> str:
        """Tier 3: Summarize deep branches, include shallow full source.

        Algorithm:
        1. Full source for depth 0-1
        2. Group depth 2+ branches by their depth-1 parent
        3. Summarize each branch group via LLM
        4. Include summaries in payload
        """
        parts = []
        branch_sources = {}

        def _collect_branches(node: CallTreeNode, branch_key: str = "root"):
            if node.depth <= 1:
                if node.source:
                    parts.append(self._format_unit_full(node))
                # Each depth-1 child starts a new branch
                for child in node.children:
                    _collect_branches(child, branch_key=child.qualified_name)
            else:
                if branch_key not in branch_sources:
                    branch_sources[branch_key] = []
                if node.source:
                    branch_sources[branch_key].append(
                        f"// {node.qualified_name} [{node.file_path}:{node.start_line}]\n{node.source}"
                    )
                for child in node.children:
                    _collect_branches(child, branch_key)

        _collect_branches(tree)

        # Summarize each branch
        for branch_name, sources in branch_sources.items():
            if not sources:
                continue
            combined = "\n\n".join(sources)
            summary = self._summarize_branch(branch_name, combined)
            parts.append(f"\n### [Branch summary] {branch_name}\n{summary}")

        return "\n".join(parts)

    def _summarize_branch(self, branch_name: str, source: str) -> str:
        """Summarize a deep branch via LLM call."""
        prompt = prompts.build_branch_summary_prompt(branch_name, source)
        llm = Settings.llm
        if not llm:
            return f"[Summary unavailable for {branch_name}]"
        response = llm.complete(prompt)
        return response.text.strip()

    # ── Output Parsing ──────────────────────────────────────────────────

    def _parse_json_output(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences."""
        # Strip markdown code fences
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Attempting repair.")
            # Try to find the outermost { }
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"parse_error": str(e), "raw_output": raw[:2000]}

    def _build_bundle(
        self,
        entry_point: EntryPoint,
        tier: AnalysisTier,
        total_tokens: int,
        call_tree: CallTreeNode,
        parsed: Dict[str, Any],
    ) -> DeepContextBundle:
        """Build DeepContextBundle from parsed LLM output."""
        from datetime import datetime

        total_units = len(self._flatten_units(call_tree))

        return DeepContextBundle(
            entry_point=entry_point,
            tier=tier,
            total_units=total_units,
            total_tokens=total_tokens,
            business_rules=parsed.get("business_rules", []),
            data_entities=parsed.get("data_entities", []),
            integrations=parsed.get("integrations", []),
            side_effects=parsed.get("side_effects", []),
            cross_cutting_concerns=parsed.get("cross_cutting_concerns", []),
            narrative=parsed.get("narrative", ""),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            coverage=float(parsed.get("coverage", 0.0) or 0.0),
            chain_truncated=bool(parsed.get("chain_truncated", tier != AnalysisTier.TIER_1)),
            schema_version=1,
            prompt_version="v1.0",
            analyzed_at=datetime.utcnow().isoformat(),
        )

    def _flatten_units(self, node: CallTreeNode) -> List[str]:
        """Collect all unit_ids in the tree."""
        ids = [node.unit_id]
        for child in node.children:
            ids.extend(self._flatten_units(child))
        return ids

    # ── Formatting Helpers ──────────────────────────────────────────────

    def _format_unit_full(self, node: CallTreeNode) -> str:
        header = f"## {node.qualified_name} [{node.file_path}:{node.start_line}-{node.end_line}]"
        return f"{header}\n```{node.language}\n{node.source}\n```"

    def _format_unit_signature(self, node: CallTreeNode) -> str:
        sig_line = node.source.split("\n")[0] if node.source else node.name
        return f"- `{node.qualified_name}` ({node.unit_type}, depth={node.depth}): `{sig_line}`"
```

### 1.5 `worker.py` — UnderstandingWorker

```python
"""Background worker for deep understanding analysis.

Mirrors the RAPTORWorker pattern from core/raptor/worker.py:
- Daemon thread with its own asyncio event loop
- Queue-based job processing with Semaphore concurrency control
- Polls database for pending jobs

Adds distributed lease protocol:
- Claim jobs via FOR UPDATE SKIP LOCKED
- Heartbeat every 30s
- Stale reclaim at 120s
- Retry policy: configurable max retries with exponential backoff
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4

from llama_index.core import Settings
from sqlalchemy import text

from ..db import DatabaseManager
from .chain_tracer import ChainTracer
from .analyzer import ChainAnalyzer
from .models import DeepContextBundle

if TYPE_CHECKING:
    from .frameworks.base import FrameworkContext

logger = logging.getLogger(__name__)


@dataclass
class UnderstandingJob:
    """A job to analyze a project's entry points."""
    job_id: str          # UUID of the DeepAnalysisJob row
    project_id: str
    worker_id: str


class UnderstandingWorker:
    """Background worker for deep understanding analysis.

    Lifecycle:
    1. start() spawns daemon thread with asyncio loop
    2. _poll_pending() checks for claimable jobs every poll_interval
    3. _process_job() runs the full pipeline per job:
       a. Detect framework context
       b. Detect entry points
       c. Trace call trees
       d. Analyze each chain
       e. Store results + populate analysis_units
       f. Embed narratives for retrieval
    4. stop() signals shutdown
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        poll_interval: float = 15.0,
        max_concurrent: int = 2,
        heartbeat_interval: float = 30.0,
        stale_threshold: float = 120.0,
        max_retries: int = 2,
    ):
        self._db = db_manager
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.heartbeat_interval = heartbeat_interval
        self.stale_threshold = stale_threshold
        self.max_retries = max_retries
        self.worker_id = f"understanding-{uuid4()}"

        self._queue: Optional[asyncio.Queue] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Lazily initialized
        self._tracer: Optional[ChainTracer] = None
        self._analyzer: Optional[ChainAnalyzer] = None

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("Understanding worker already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="understanding-worker"
        )
        self._thread.start()
        logger.info("Understanding worker started")

    def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Understanding worker stopped")

    def _run_loop(self):
        """Run the async event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"Understanding worker loop error: {e}")
        finally:
            self._loop.close()

    async def _main_loop(self):
        """Main processing loop — mirrors RAPTORWorker._main_loop()."""
        poll_task = asyncio.create_task(self._poll_pending())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        while self._running:
            try:
                job = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.poll_interval,
                )
                asyncio.create_task(self._process_with_semaphore(job))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in understanding main loop: {e}")

        poll_task.cancel()
        heartbeat_task.cancel()

    async def _poll_pending(self):
        """Poll for claimable jobs using FOR UPDATE SKIP LOCKED."""
        while self._running:
            try:
                jobs = self._claim_pending_jobs()
                for job in jobs:
                    await self._queue.put(job)
            except Exception as e:
                logger.error(f"Error polling understanding jobs: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _heartbeat_loop(self):
        """Update heartbeat_at for all running jobs owned by this worker."""
        while self._running:
            try:
                self._update_heartbeats()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    async def _process_with_semaphore(self, job: UnderstandingJob):
        """Process job with semaphore for concurrency control."""
        async with self._semaphore:
            await asyncio.to_thread(self._process_job_sync, job)

    def _process_job_sync(self, job: UnderstandingJob):
        """Process a single understanding job (runs in thread pool).

        Pipeline:
        1. Initialize tracer/analyzer if needed
        2. Run framework detection
        3. Detect entry points
        4. For each entry point:
           a. Trace call tree
           b. Analyze chain
           c. Store DeepAnalysis row + analysis_units
        5. Mark job completed
        """
        logger.info(f"Processing understanding job {job.job_id} for project {job.project_id}")

        try:
            # Initialize
            if not self._tracer:
                self._tracer = ChainTracer(self._db)
            if not self._analyzer:
                self._analyzer = ChainAnalyzer()

            # Framework detection
            from .frameworks import detect_and_analyze
            framework_contexts = detect_and_analyze(self._db, job.project_id)

            # Detect entry points
            entry_points = self._tracer.detect_entry_points(job.project_id)
            logger.info(f"Found {len(entry_points)} entry points for project {job.project_id}")

            # Update job progress
            self._update_job_progress(job.job_id, len(entry_points), 0)

            # Process each entry point
            completed = 0
            errors = []

            for ep in entry_points:
                try:
                    # Trace
                    tree = self._tracer.trace_call_tree(
                        job.project_id, ep.unit_id, max_depth=10
                    )

                    # Analyze
                    bundle = self._analyzer.analyze_chain(
                        entry_point=ep,
                        call_tree=tree,
                        framework_contexts=framework_contexts,
                    )

                    # Store
                    self._store_analysis(job.job_id, job.project_id, bundle, tree)

                    completed += 1
                    self._update_job_progress(job.job_id, len(entry_points), completed)

                except Exception as e:
                    logger.error(
                        f"Error analyzing entry point {ep.qualified_name}: {e}",
                        exc_info=True,
                    )
                    errors.append({"entry_point": ep.qualified_name, "error": str(e)})

            # Mark completed
            status = "completed" if not errors or completed > 0 else "failed"
            self._complete_job(job.job_id, status, errors)

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            self._handle_job_failure(job.job_id, str(e))

    # ── DB Operations (see Section 7 for SQL) ───────────────────────────

    def _claim_pending_jobs(self, limit: int = 2) -> List[UnderstandingJob]:
        """Claim pending jobs using FOR UPDATE SKIP LOCKED.

        Also reclaims stale running jobs (heartbeat older than threshold).
        See Section 7.3 for exact SQL.
        """
        now = datetime.utcnow()
        stale_cutoff = now - timedelta(seconds=self.stale_threshold)

        with self._db.get_session() as session:
            # Reclaim stale jobs first
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = 'pending',
                        worker_id = NULL
                    WHERE status = 'running'
                      AND heartbeat_at < :stale_cutoff
                      AND retry_count < :max_retries
                """),
                {"stale_cutoff": stale_cutoff, "max_retries": self.max_retries},
            )

            # Claim pending jobs
            result = session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = 'running',
                        worker_id = :worker_id,
                        started_at = :now,
                        heartbeat_at = :now,
                        retry_count = retry_count + 1,
                        next_attempt_at = NULL
                    WHERE job_id IN (
                        SELECT job_id FROM deep_analysis_jobs
                        WHERE status = 'pending'
                          AND (next_attempt_at IS NULL OR next_attempt_at <= :now)
                        ORDER BY created_at
                        LIMIT :limit
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING job_id, project_id, worker_id
                """),
                {"now": now, "limit": limit, "worker_id": self.worker_id},
            )
            rows = result.fetchall()

        return [
            UnderstandingJob(
                job_id=str(row.job_id),
                project_id=str(row.project_id),
                worker_id=str(row.worker_id),
            )
            for row in rows
        ]

    def _update_heartbeats(self):
        """Update heartbeat_at for all jobs this worker is processing."""
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET heartbeat_at = :now
                    WHERE status = 'running'
                      AND worker_id = :worker_id
                """),
                {"now": datetime.utcnow(), "worker_id": self.worker_id},
            )

    def _update_job_progress(self, job_id: str, total: int, completed: int):
        """Update job progress counters."""
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET total_entry_points = :total,
                        completed_entry_points = :completed,
                        heartbeat_at = :now
                    WHERE job_id = :job_id
                      AND worker_id = :worker_id
                """),
                {
                    "job_id": UUID(job_id),
                    "total": total,
                    "completed": completed,
                    "now": datetime.utcnow(),
                    "worker_id": self.worker_id,
                },
            )

    def _store_analysis(
        self,
        job_id: str,
        project_id: str,
        bundle: DeepContextBundle,
        tree,
    ):
        """Store a DeepAnalysis row and populate analysis_units.

        Uses upsert (ON CONFLICT on project_id + entry_unit_id + schema_version).
        See Section 2 for table definitions.
        """
        import json as json_mod
        from .chain_tracer import ChainTracer  # for get_flat_unit_membership

        pid = UUID(project_id)
        entry_uid = UUID(bundle.entry_point.unit_id)

        # Serialize full bundle to JSON for replay/audit compatibility
        result_json = json_mod.dumps({
            "entry_point": {
                "unit_id": bundle.entry_point.unit_id,
                "name": bundle.entry_point.name,
                "qualified_name": bundle.entry_point.qualified_name,
                "file_path": bundle.entry_point.file_path,
                "entry_type": bundle.entry_point.entry_type.value,
                "language": bundle.entry_point.language,
                "metadata": bundle.entry_point.metadata,
                "detected_by": bundle.entry_point.detected_by,
            },
            "tier": bundle.tier.value,
            "total_units": bundle.total_units,
            "total_tokens": bundle.total_tokens,
            "business_rules": bundle.business_rules,
            "data_entities": bundle.data_entities,
            "integrations": bundle.integrations,
            "side_effects": bundle.side_effects,
            "cross_cutting_concerns": bundle.cross_cutting_concerns,
            "narrative": bundle.narrative,
            "confidence": bundle.confidence,
            "coverage": bundle.coverage,
            "chain_truncated": bundle.chain_truncated,
            "schema_version": bundle.schema_version,
            "prompt_version": bundle.prompt_version,
            "analyzed_at": bundle.analyzed_at,
        })

        with self._db.get_session() as session:
            # Upsert deep_analyses
            result = session.execute(
                text("""
                    INSERT INTO deep_analyses (
                        analysis_id, job_id, project_id, entry_unit_id,
                        entry_type, tier, total_units, total_tokens,
                        confidence_score, coverage_pct,
                        result_json, narrative, schema_version, prompt_version
                    ) VALUES (
                        gen_random_uuid(), :job_id, :pid, :entry_uid,
                        :entry_type, :tier, :total_units, :total_tokens,
                        :confidence_score, :coverage_pct,
                        :result_json, :narrative, :schema_version, :prompt_version
                    )
                    ON CONFLICT (project_id, entry_unit_id, schema_version)
                    DO UPDATE SET
                        job_id = EXCLUDED.job_id,
                        tier = EXCLUDED.tier,
                        total_units = EXCLUDED.total_units,
                        total_tokens = EXCLUDED.total_tokens,
                        confidence_score = EXCLUDED.confidence_score,
                        coverage_pct = EXCLUDED.coverage_pct,
                        result_json = EXCLUDED.result_json,
                        narrative = EXCLUDED.narrative,
                        prompt_version = EXCLUDED.prompt_version,
                        analyzed_at = NOW()
                    RETURNING analysis_id
                """),
                {
                    "job_id": UUID(job_id),
                    "pid": pid,
                    "entry_uid": entry_uid,
                    "entry_type": bundle.entry_point.entry_type.value,
                    "tier": bundle.tier.value,
                    "total_units": bundle.total_units,
                    "total_tokens": bundle.total_tokens,
                    "confidence_score": bundle.confidence,
                    "coverage_pct": bundle.coverage * 100.0,
                    "result_json": result_json,
                    "narrative": bundle.narrative,
                    "schema_version": bundle.schema_version,
                    "prompt_version": bundle.prompt_version,
                },
            )
            analysis_id = result.fetchone().analysis_id

            # Populate analysis_units
            tracer = self._tracer or ChainTracer(self._db)
            flat_units = tracer.get_flat_unit_membership(tree)

            for unit in flat_units:
                session.execute(
                    text("""
                        INSERT INTO analysis_units (
                            analysis_id, project_id, unit_id, min_depth, path_count
                        ) VALUES (
                            :aid, :pid, :uid, :min_depth, :path_count
                        )
                        ON CONFLICT (analysis_id, unit_id)
                        DO UPDATE SET
                            min_depth = LEAST(analysis_units.min_depth, EXCLUDED.min_depth),
                            path_count = GREATEST(analysis_units.path_count, EXCLUDED.path_count)
                    """),
                    {
                        "aid": analysis_id,
                        "pid": pid,
                        "uid": UUID(unit["unit_id"]),
                        "min_depth": unit["min_depth"],
                        "path_count": unit["path_count"],
                    },
                )

    def _complete_job(self, job_id: str, status: str, errors: list):
        """Mark job as completed or failed."""
        import json as json_mod

        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = :status,
                        completed_at = :now,
                        error_details = :errors
                    WHERE job_id = :job_id
                      AND worker_id = :worker_id
                """),
                {
                    "job_id": UUID(job_id),
                    "status": status,
                    "now": datetime.utcnow(),
                    "errors": json_mod.dumps(errors) if errors else None,
                    "worker_id": self.worker_id,
                },
            )

        # Update project status
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE projects
                    SET deep_analysis_status = :status
                    WHERE project_id = (
                        SELECT project_id FROM deep_analysis_jobs WHERE job_id = :job_id
                    )
                """),
                {"job_id": UUID(job_id), "status": status},
            )

    def _handle_job_failure(self, job_id: str, error: str):
        """Handle job failure with retry logic."""
        base_backoff_seconds = 15
        with self._db.get_session() as session:
            result = session.execute(
                text("SELECT retry_count FROM deep_analysis_jobs WHERE job_id = :jid"),
                {"jid": UUID(job_id)},
            )
            row = result.fetchone()
            retry_count = row.retry_count if row else 0

            if retry_count >= self.max_retries:
                # Terminal failure
                session.execute(
                    text("""
                        UPDATE deep_analysis_jobs
                        SET status = 'failed',
                            completed_at = :now,
                            error_details = :error
                        WHERE job_id = :jid
                          AND worker_id = :worker_id
                    """),
                    {
                        "jid": UUID(job_id),
                        "now": datetime.utcnow(),
                        "error": error,
                        "worker_id": self.worker_id,
                    },
                )
            else:
                # Back to pending for retry with exponential backoff
                session.execute(
                    text("""
                        UPDATE deep_analysis_jobs
                        SET status = 'pending',
                            worker_id = NULL,
                            next_attempt_at = :next_attempt_at
                        WHERE job_id = :jid
                          AND worker_id = :worker_id
                    """),
                    {
                        "jid": UUID(job_id),
                        "worker_id": self.worker_id,
                        "next_attempt_at": datetime.utcnow() + timedelta(
                            seconds=base_backoff_seconds * (2 ** retry_count)
                        ),
                    },
                )
```

### 1.6 `engine.py` — UnderstandingEngine

```python
"""Understanding Engine — orchestrator for deep code analysis.

Follows the MigrationEngine pattern from core/migration/engine.py.
Provides the public API consumed by API routes.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from ..db import DatabaseManager
from ..db.models import Project
from .chain_tracer import ChainTracer
from .worker import UnderstandingWorker

logger = logging.getLogger(__name__)


class UnderstandingEngine:
    """Orchestrate deep understanding analysis for a project.

    Public API:
        start_analysis(project_id, user_id) -> job_id
        get_job_status(project_id, job_id) -> status dict
        get_entry_points(project_id) -> list of entry points
        get_analysis_results(project_id) -> list of analysis summaries
        get_chain_detail(project_id, analysis_id) -> full detail
    """

    def __init__(self, db_manager: DatabaseManager, pipeline: Any = None):
        self._db = db_manager
        self._pipeline = pipeline
        self._worker: Optional[UnderstandingWorker] = None
        self._tracer: Optional[ChainTracer] = None

    def _ensure_worker(self):
        """Lazily initialize and start the background worker."""
        if self._worker is None:
            self._worker = UnderstandingWorker(self._db)
        if not self._worker._running:
            self._worker.start()

    def _ensure_tracer(self):
        """Lazily initialize the chain tracer."""
        if self._tracer is None:
            self._tracer = ChainTracer(self._db)

    # ── Public API ──────────────────────────────────────────────────────

    def start_analysis(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Start a deep understanding analysis job for a project.

        Creates a DeepAnalysisJob row and ensures the worker is running.

        Returns:
            Dict with job_id and status
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        uid = UUID(user_id) if isinstance(user_id, str) else user_id
        job_id = uuid4()

        with self._db.get_session() as session:
            session.execute(
                text("""
                    INSERT INTO deep_analysis_jobs (job_id, project_id, user_id, status)
                    VALUES (:jid, :pid, :uid, 'pending')
                """),
                {"jid": job_id, "pid": pid, "uid": uid},
            )

            # Update project status
            session.execute(
                text("UPDATE projects SET deep_analysis_status = 'pending' WHERE project_id = :pid"),
                {"pid": pid},
            )

        self._ensure_worker()

        return {"job_id": str(job_id), "status": "pending", "project_id": str(pid)}

    def get_job_status(self, project_id: str, job_id: str) -> Dict[str, Any]:
        """Get the status of an analysis job.

        Returns:
            Dict with status, progress, timestamps, errors
        """
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT job_id, project_id, status,
                           total_entry_points, completed_entry_points,
                           created_at, started_at, completed_at,
                           error_details, retry_count
                    FROM deep_analysis_jobs
                    WHERE job_id = :jid AND project_id = :pid
                """),
                {"jid": UUID(job_id), "pid": UUID(project_id)},
            )
            row = result.fetchone()

        if not row:
            return {"error": "Job not found"}

        return {
            "job_id": str(row.job_id),
            "project_id": str(row.project_id),
            "status": row.status,
            "progress": {
                "total": row.total_entry_points or 0,
                "completed": row.completed_entry_points or 0,
            },
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "retry_count": row.retry_count,
            "errors": row.error_details,
        }

    def get_entry_points(self, project_id: str) -> List[Dict[str, Any]]:
        """Synchronously detect entry points for a project.

        Does NOT require a running analysis job — useful for previewing
        what the analysis will cover.

        Returns:
            List of entry point dicts
        """
        self._ensure_tracer()
        eps = self._tracer.detect_entry_points(project_id)
        return [
            {
                "unit_id": ep.unit_id,
                "name": ep.name,
                "qualified_name": ep.qualified_name,
                "file_path": ep.file_path,
                "entry_type": ep.entry_type.value,
                "language": ep.language,
                "detected_by": ep.detected_by,
            }
            for ep in eps
        ]

    def get_analysis_results(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all analysis results for a project (summaries only).

        Returns:
            List of analysis summary dicts (no full result_json)
        """
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT a.analysis_id, a.entry_unit_id, a.entry_type,
                           a.tier, a.total_units, a.total_tokens,
                           a.confidence_score, a.coverage_pct,
                           a.narrative, a.schema_version, a.analyzed_at,
                           u.name AS entry_name, u.qualified_name AS entry_qualified,
                           f.file_path AS entry_file
                    FROM deep_analyses a
                    JOIN code_units u ON a.entry_unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE a.project_id = :pid
                    ORDER BY f.file_path, u.name
                """),
                {"pid": UUID(project_id)},
            )
            rows = result.fetchall()

        return [
            {
                "analysis_id": str(row.analysis_id),
                "entry_unit_id": str(row.entry_unit_id),
                "entry_name": row.entry_name,
                "entry_qualified_name": row.entry_qualified,
                "entry_file": row.entry_file,
                "entry_type": row.entry_type,
                "tier": row.tier,
                "total_units": row.total_units,
                "total_tokens": row.total_tokens,
                "confidence_score": row.confidence_score,
                "coverage_pct": row.coverage_pct,
                "narrative": row.narrative,
                "schema_version": row.schema_version,
                "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
            }
            for row in rows
        ]

    def get_chain_detail(
        self,
        project_id: str,
        analysis_id: str,
    ) -> Dict[str, Any]:
        """Get full detail for a single analysis.

        Returns:
            Full analysis dict including result_json with evidence refs
        """
        import json

        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT a.*, u.name AS entry_name,
                           u.qualified_name AS entry_qualified,
                           f.file_path AS entry_file
                    FROM deep_analyses a
                    JOIN code_units u ON a.entry_unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE a.analysis_id = :aid AND a.project_id = :pid
                """),
                {"aid": UUID(analysis_id), "pid": UUID(project_id)},
            )
            row = result.fetchone()

        if not row:
            return {"error": "Analysis not found"}

        # Parse result_json
        result_data = {}
        if row.result_json:
            try:
                result_data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
            except (json.JSONDecodeError, TypeError):
                result_data = {"parse_error": "Could not parse result_json"}

        # Get analysis_units
        with self._db.get_session() as session:
            units_result = session.execute(
                text("""
                    SELECT au.unit_id, au.min_depth, au.path_count,
                           u.name, u.qualified_name, u.unit_type,
                           f.file_path
                    FROM analysis_units au
                    JOIN code_units u ON au.unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE au.analysis_id = :aid
                    ORDER BY au.min_depth, u.name
                """),
                {"aid": UUID(analysis_id)},
            )
            units = [
                {
                    "unit_id": str(r.unit_id),
                    "name": r.name,
                    "qualified_name": r.qualified_name,
                    "unit_type": r.unit_type,
                    "file_path": r.file_path,
                    "min_depth": r.min_depth,
                    "path_count": r.path_count,
                }
                for r in units_result.fetchall()
            ]

        return {
            "analysis_id": str(row.analysis_id),
            "entry_point": {
                "unit_id": str(row.entry_unit_id),
                "name": row.entry_name,
                "qualified_name": row.entry_qualified,
                "file_path": row.entry_file,
                "entry_type": row.entry_type,
            },
            "tier": row.tier,
            "total_units": row.total_units,
            "total_tokens": row.total_tokens,
            "confidence_score": row.confidence_score,
            "coverage_pct": row.coverage_pct,
            "narrative": row.narrative,
            "schema_version": row.schema_version,
            "prompt_version": row.prompt_version,
            "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
            "result": result_data,
            "units": units,
        }
```

### 1.7 `frameworks/` — Framework Analyzers

#### `frameworks/base.py`

```python
"""Base classes for framework-specific analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...db import DatabaseManager


@dataclass
class FrameworkContext:
    """Framework-specific context to inject into analysis prompts."""
    framework_name: str           # e.g. "Spring Boot 2.7", "ASP.NET Core 6"
    framework_type: str           # "spring", "aspnet", "django", "express"
    version: Optional[str] = None

    # Discovered configuration
    di_registrations: List[str] = field(default_factory=list)
    middleware_pipeline: List[str] = field(default_factory=list)
    security_config: Dict[str, Any] = field(default_factory=dict)
    transaction_boundaries: List[str] = field(default_factory=list)
    aop_pointcuts: List[str] = field(default_factory=list)

    # Hints for the LLM
    analysis_hints: List[str] = field(default_factory=list)


class FrameworkAnalyzer(ABC):
    """Abstract base for framework-specific analyzers."""

    def __init__(self, db: DatabaseManager):
        self._db = db

    @abstractmethod
    def detect(self, project_id: str) -> bool:
        """Return True if this framework is detected in the project."""
        ...

    @abstractmethod
    def analyze(self, project_id: str) -> FrameworkContext:
        """Analyze framework-specific patterns and return context."""
        ...
```

#### `frameworks/spring.py`

```python
"""Spring Boot / Spring MVC framework analyzer.

Detects:
- XML bean definitions and @Configuration DI
- Spring Security filter chain configuration
- @Transactional boundaries
- AOP pointcuts (@Aspect, @Around, @Before, @After)
- Spring Data repositories
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

from ...db import DatabaseManager
from .base import FrameworkAnalyzer, FrameworkContext

logger = logging.getLogger(__name__)


class SpringAnalyzer(FrameworkAnalyzer):

    def detect(self, project_id: str) -> bool:
        pid = UUID(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM code_units
                    WHERE project_id = :pid
                      AND (
                          source ~ '@SpringBootApplication'
                          OR source ~ '@RestController'
                          OR source ~ '@Controller'
                          OR source ~ 'spring-boot'
                      )
                """),
                {"pid": pid},
            )
            return result.fetchone().cnt > 0

    def analyze(self, project_id: str) -> FrameworkContext:
        pid = UUID(project_id)
        ctx = FrameworkContext(
            framework_name="Spring Boot",
            framework_type="spring",
        )

        with self._db.get_session() as session:
            # DI: @Configuration classes
            di_result = session.execute(
                text("""
                    SELECT name, qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND source ~ '@(Configuration|Component|Service|Repository|Bean)'
                    ORDER BY name LIMIT 50
                """),
                {"pid": pid},
            )
            ctx.di_registrations = [
                row.qualified_name for row in di_result.fetchall()
            ]

            # Security filter chain
            sec_result = session.execute(
                text("""
                    SELECT name, source FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ 'SecurityFilterChain' OR source ~ 'WebSecurityConfigurerAdapter')
                    LIMIT 5
                """),
                {"pid": pid},
            )
            for row in sec_result.fetchall():
                ctx.security_config[row.name] = "Spring Security config detected"

            # @Transactional boundaries
            tx_result = session.execute(
                text("""
                    SELECT qualified_name FROM code_units
                    WHERE project_id = :pid AND source ~ '@Transactional'
                    ORDER BY name LIMIT 50
                """),
                {"pid": pid},
            )
            ctx.transaction_boundaries = [
                row.qualified_name for row in tx_result.fetchall()
            ]

            # AOP
            aop_result = session.execute(
                text("""
                    SELECT qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND source ~ '@(Aspect|Around|Before|After|Pointcut)'
                    LIMIT 20
                """),
                {"pid": pid},
            )
            ctx.aop_pointcuts = [
                row.qualified_name for row in aop_result.fetchall()
            ]

        ctx.analysis_hints = [
            "Spring uses proxy-based AOP — @Transactional only works on public methods called from outside the class",
            "Spring Security filter chain ordering matters — check for antMatchers/requestMatchers precedence",
            "Check for @Lazy and circular dependency patterns in DI registrations",
        ]

        return ctx
```

#### `frameworks/aspnet.py`

```python
"""ASP.NET Core framework analyzer.

Detects:
- DI registration in Startup/Program.cs (AddScoped, AddTransient, AddSingleton)
- Middleware pipeline ordering
- Action filters and authorization attributes
- Entity Framework DbContext patterns
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

from ...db import DatabaseManager
from .base import FrameworkAnalyzer, FrameworkContext

logger = logging.getLogger(__name__)


class AspNetAnalyzer(FrameworkAnalyzer):

    def detect(self, project_id: str) -> bool:
        pid = UUID(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM code_units
                    WHERE project_id = :pid
                      AND (
                          source ~ 'Microsoft\\.AspNetCore'
                          OR source ~ 'WebApplication\\.CreateBuilder'
                          OR source ~ ': ControllerBase'
                          OR source ~ ': Controller'
                      )
                """),
                {"pid": pid},
            )
            return result.fetchone().cnt > 0

    def analyze(self, project_id: str) -> FrameworkContext:
        pid = UUID(project_id)
        ctx = FrameworkContext(
            framework_name="ASP.NET Core",
            framework_type="aspnet",
        )

        with self._db.get_session() as session:
            # DI registrations
            di_result = session.execute(
                text("""
                    SELECT source FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ 'Add(Scoped|Transient|Singleton)' OR source ~ 'builder\\.Services')
                    LIMIT 10
                """),
                {"pid": pid},
            )
            for row in di_result.fetchall():
                for line in (row.source or "").split("\n"):
                    if "Add" in line and ("Scoped" in line or "Transient" in line or "Singleton" in line):
                        ctx.di_registrations.append(line.strip())

            # Middleware pipeline
            mw_result = session.execute(
                text("""
                    SELECT source FROM code_units
                    WHERE project_id = :pid
                      AND source ~ 'app\\.Use'
                    LIMIT 10
                """),
                {"pid": pid},
            )
            for row in mw_result.fetchall():
                for line in (row.source or "").split("\n"):
                    if "app.Use" in line:
                        ctx.middleware_pipeline.append(line.strip())

            # DbContext
            db_result = session.execute(
                text("""
                    SELECT name, qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ ': DbContext' OR source ~ 'DbSet<')
                    LIMIT 20
                """),
                {"pid": pid},
            )
            for row in db_result.fetchall():
                ctx.security_config[row.name] = "EF DbContext"

        ctx.analysis_hints = [
            "ASP.NET middleware order matters — UseAuthentication before UseAuthorization",
            "Check DI lifetime mismatches (Singleton depending on Scoped = captive dependency)",
            "Action filters can short-circuit the pipeline — check for IAuthorizationFilter",
        ]

        return ctx
```

#### `frameworks/__init__.py`

```python
"""Framework detection and analysis registry."""

import logging
from typing import Any, Dict, List

from ...db import DatabaseManager
from .base import FrameworkContext
from .spring import SpringAnalyzer
from .aspnet import AspNetAnalyzer

logger = logging.getLogger(__name__)

# Registry of framework analyzers (order = detection priority)
_ANALYZERS = [
    SpringAnalyzer,
    AspNetAnalyzer,
]


def detect_and_analyze(
    db: DatabaseManager,
    project_id: str,
) -> List[Dict[str, Any]]:
    """Detect frameworks and return analysis contexts.

    Runs each registered analyzer's detect() method. For detected
    frameworks, runs analyze() and returns the contexts.

    Args:
        db: DatabaseManager instance
        project_id: UUID string

    Returns:
        List of serialized FrameworkContext dicts
    """
    contexts = []

    for analyzer_cls in _ANALYZERS:
        try:
            analyzer = analyzer_cls(db)
            if analyzer.detect(project_id):
                ctx = analyzer.analyze(project_id)
                contexts.append({
                    "framework_name": ctx.framework_name,
                    "framework_type": ctx.framework_type,
                    "version": ctx.version,
                    "di_registrations": ctx.di_registrations[:20],
                    "middleware_pipeline": ctx.middleware_pipeline[:10],
                    "security_config": ctx.security_config,
                    "transaction_boundaries": ctx.transaction_boundaries[:20],
                    "aop_pointcuts": ctx.aop_pointcuts[:10],
                    "analysis_hints": ctx.analysis_hints,
                })
                logger.info(f"Detected framework: {ctx.framework_name}")
        except Exception as e:
            logger.warning(f"Framework detection failed for {analyzer_cls.__name__}: {e}")

    return contexts
```

### 1.8 `__init__.py` — Public Exports

```python
"""Deep Understanding Engine — trace execution paths and extract structured knowledge.

Public API:
    UnderstandingEngine  — orchestrator (start_analysis, get_results, etc.)
    ChainTracer          — entry point detection + call tree tracing
    ChainAnalyzer        — tiered LLM analysis
    UnderstandingWorker  — background job processor
"""

from .engine import UnderstandingEngine
from .chain_tracer import ChainTracer
from .analyzer import ChainAnalyzer
from .worker import UnderstandingWorker

__all__ = [
    "UnderstandingEngine",
    "ChainTracer",
    "ChainAnalyzer",
    "UnderstandingWorker",
]
```

---

## 2. Database Design

### 2.1 New Tables

Three new tables extend the existing schema in `core/db/models.py`.
All follow existing conventions: `UUID` TypeDecorator for PKs, `JSONB` for
flexible data, `TIMESTAMP` for audit fields, cascading deletes.

#### 2.1.1 `deep_analysis_jobs` — Job Queue

```python
class DeepAnalysisJob(Base):
    """Job queue for deep understanding analysis.

    Each row represents one analysis run for a project.
    Workers claim jobs via FOR UPDATE SKIP LOCKED.
    """
    __tablename__ = "deep_analysis_jobs"
    __table_args__ = (
        Index('idx_deep_jobs_project_status', 'project_id', 'status'),
        Index('idx_deep_jobs_status_created', 'status', 'created_at'),
    )

    job_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default='pending', nullable=False)
        # pending → running → completed | failed
    worker_id = Column(String(100), nullable=True)  # Identifies claiming worker
    retry_count = Column(Integer, default=0, nullable=False)
    next_attempt_at = Column(TIMESTAMP, nullable=True)  # Retry backoff gate

    # Progress tracking
    total_entry_points = Column(Integer, nullable=True)
    completed_entry_points = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    heartbeat_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

    # Error details (JSON array of {entry_point, error})
    error_details = Column(JSONB, nullable=True)

    # Relationships
    project = relationship("Project")
    analyses = relationship("DeepAnalysis", back_populates="job", cascade="all, delete-orphan")
```

**Index justifications**:
- `(project_id, status)` — API route filters by project + status for polling
- `(status, created_at)` — Worker claims oldest pending jobs first

#### 2.1.2 `deep_analyses` — Per-Entry-Point Results

```python
class DeepAnalysis(Base):
    """Analysis result for a single entry point.

    Stores the complete DeepContextBundle as JSON plus a
    human-readable narrative for chat injection.
    """
    __tablename__ = "deep_analyses"
    __table_args__ = (
        UniqueConstraint('project_id', 'entry_unit_id', 'schema_version',
                         name='uq_deep_analysis_entry'),
        Index('idx_deep_analyses_project', 'project_id'),
        Index('idx_deep_analyses_entry', 'entry_unit_id'),
    )

    analysis_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(), ForeignKey("deep_analysis_jobs.job_id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    entry_unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"), nullable=False)

    entry_type = Column(String(50), nullable=False)    # EntryPointType value
    tier = Column(String(20), nullable=False)           # AnalysisTier value
    total_units = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=True)     # 0..1 extraction confidence
    coverage_pct = Column(Float, nullable=True)         # 0..100 chain coverage estimate

    # The full structured analysis output
    result_json = Column(JSONB, nullable=False)
    # Human-readable narrative (injected into chat context)
    narrative = Column(Text, nullable=True)

    # Versioning
    schema_version = Column(Integer, default=1, nullable=False)
    prompt_version = Column(String(20), default='v1.0', nullable=False)

    analyzed_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("DeepAnalysisJob", back_populates="analyses")
    project = relationship("Project")
    entry_unit = relationship("CodeUnit")
    analysis_units = relationship("AnalysisUnit", back_populates="analysis",
                                  cascade="all, delete-orphan")
```

**Upsert constraint**: `(project_id, entry_unit_id, schema_version)` allows
re-analyzing an entry point (new job) to overwrite previous results within
the same schema version, while preserving results from older schema versions
for comparison.

**Versioning semantics**:
- `schema_version` (integer): Bumped when the result_json structure changes.
  Old analyses with lower schema_version are preserved alongside new ones.
- `prompt_version` (string): Tracks which prompt template produced the result.
  Informational — does not affect the upsert constraint.

#### 2.1.3 `analysis_units` — Junction Table

```python
class AnalysisUnit(Base):
    """Junction table linking analyses to the code units they cover.

    Enables two key queries:
    1. "What analyses cover this unit?" (for chat enrichment)
    2. "What units does this analysis cover?" (for coverage calculation)
    """
    __tablename__ = "analysis_units"
    __table_args__ = (
        # Composite PK serves as the primary unique constraint
        Index('idx_analysis_units_project_unit', 'project_id', 'unit_id'),
    )

    analysis_id = Column(UUID(), ForeignKey("deep_analyses.analysis_id", ondelete="CASCADE"),
                         primary_key=True, nullable=False)
    unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"),
                     primary_key=True, nullable=False)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"),
                        nullable=False)

    # Aggregated context within the call tree
    min_depth = Column(Integer, nullable=False)  # Shallowest path depth
    path_count = Column(Integer, nullable=False, default=1)  # Number of distinct paths

    # Relationships
    analysis = relationship("DeepAnalysis", back_populates="analysis_units")
```

**Index justification**:
- `(project_id, unit_id)` — Chat route looks up "which analyses cover
  the units in my retrieval results?" This index supports that O(1) lookup.
- Composite PK `(analysis_id, unit_id)` prevents duplicate entries and
  enables "which units does this analysis cover?" queries.

### 2.2 Schema Modification — `Project` Table

Add one new column to the existing `Project` model:

```python
# In class Project(Base):
deep_analysis_status = Column(
    String(20), default='none', nullable=False
)
# Values: none | pending | running | completed | failed
```

### 2.3 Alembic Migration DDL

```python
"""Add deep understanding tables.

Revision ID: <auto-generated>
Revises: <current-head>
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


def upgrade():
    # 1. Add column to projects
    op.add_column('projects',
        sa.Column('deep_analysis_status', sa.String(20),
                  server_default='none', nullable=False)
    )

    # 2. Create deep_analysis_jobs
    op.create_table('deep_analysis_jobs',
        sa.Column('job_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True),
                  sa.ForeignKey('projects.project_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('user_id', UUID(as_uuid=True),
                  sa.ForeignKey('users.user_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('status', sa.String(20), server_default='pending', nullable=False),
        sa.Column('worker_id', sa.String(100), nullable=True),
        sa.Column('retry_count', sa.Integer, server_default='0', nullable=False),
        sa.Column('next_attempt_at', sa.TIMESTAMP, nullable=True),
        sa.Column('total_entry_points', sa.Integer, nullable=True),
        sa.Column('completed_entry_points', sa.Integer, server_default='0', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP, nullable=True),
        sa.Column('heartbeat_at', sa.TIMESTAMP, nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP, nullable=True),
        sa.Column('error_details', JSONB, nullable=True),
    )
    op.create_index('idx_deep_jobs_project_status', 'deep_analysis_jobs',
                    ['project_id', 'status'])
    op.create_index('idx_deep_jobs_status_created', 'deep_analysis_jobs',
                    ['status', 'created_at'])

    # 3. Create deep_analyses
    op.create_table('deep_analyses',
        sa.Column('analysis_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True),
                  sa.ForeignKey('deep_analysis_jobs.job_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('project_id', UUID(as_uuid=True),
                  sa.ForeignKey('projects.project_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('entry_unit_id', UUID(as_uuid=True),
                  sa.ForeignKey('code_units.unit_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('entry_type', sa.String(50), nullable=False),
        sa.Column('tier', sa.String(20), nullable=False),
        sa.Column('total_units', sa.Integer, nullable=False),
        sa.Column('total_tokens', sa.Integer, nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('coverage_pct', sa.Float(), nullable=True),
        sa.Column('result_json', JSONB, nullable=False),
        sa.Column('narrative', sa.Text, nullable=True),
        sa.Column('schema_version', sa.Integer, server_default='1', nullable=False),
        sa.Column('prompt_version', sa.String(20), server_default='v1.0', nullable=False),
        sa.Column('analyzed_at', sa.TIMESTAMP, server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint('uq_deep_analysis_entry', 'deep_analyses',
                                ['project_id', 'entry_unit_id', 'schema_version'])
    op.create_index('idx_deep_analyses_project', 'deep_analyses', ['project_id'])
    op.create_index('idx_deep_analyses_entry', 'deep_analyses', ['entry_unit_id'])

    # 4. Create analysis_units
    op.create_table('analysis_units',
        sa.Column('analysis_id', UUID(as_uuid=True),
                  sa.ForeignKey('deep_analyses.analysis_id', ondelete='CASCADE'),
                  primary_key=True, nullable=False),
        sa.Column('unit_id', UUID(as_uuid=True),
                  sa.ForeignKey('code_units.unit_id', ondelete='CASCADE'),
                  primary_key=True, nullable=False),
        sa.Column('project_id', UUID(as_uuid=True),
                  sa.ForeignKey('projects.project_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('min_depth', sa.Integer, nullable=False),
        sa.Column('path_count', sa.Integer, server_default='1', nullable=False),
    )
    op.create_index('idx_analysis_units_project_unit', 'analysis_units',
                    ['project_id', 'unit_id'])


def downgrade():
    op.drop_table('analysis_units')
    op.drop_table('deep_analyses')
    op.drop_table('deep_analysis_jobs')
    op.drop_column('projects', 'deep_analysis_status')
```

**Table creation order**: `deep_analysis_jobs` first (referenced by `deep_analyses`),
then `deep_analyses` (referenced by `analysis_units`).
**Downgrade**: reverse order — drop `analysis_units`, `deep_analyses`, `deep_analysis_jobs`, then column.

---

## 3. Algorithm Specifications

### 3.1 Entry Point Detection

#### Pass 1 — Heuristic (Zero Incoming Calls)

```sql
-- Find units with no incoming 'calls' edges
-- Filters: callable types only
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
```

**Rationale**: Any function/method that is never called by another function
in the codebase is either an entry point (HTTP handler, CLI command, etc.)
or dead code. Pass 2's annotation matching distinguishes between the two.
Nested-function exclusion is intentionally deferred to parser-aware heuristics.

#### Pass 2 — Annotation Patterns

For each detected language in the project, scan `code_units.source` against
the regex patterns in `_ANNOTATION_PATTERNS`. Uses PostgreSQL's `~` regex
operator for server-side matching, avoiding client-side iteration over all units.

#### Merge Logic

```
merged = {}
for ep in pass1_results:
    merged[ep.unit_id] = ep

for ep in pass2_results:
    if ep.unit_id in merged:
        merged[ep.unit_id].entry_type = ep.entry_type  # Pass 2 wins on type
        merged[ep.unit_id].detected_by = "both"
    else:
        merged[ep.unit_id] = ep

return sorted(merged.values(), key=file_path+name)
```

#### Classification Decision Tree

```
if name == "main" and "static" in modifiers:
    → CLI_COMMAND
elif name starts with "test_":
    → UNKNOWN (skip)
elif metadata.is_endpoint:
    → HTTP_ENDPOINT
else:
    → PUBLIC_API  (conservative default)
```

### 3.2 Call Tree CTE

Extends the `_traverse()` pattern from `core/asg_builder/queries.py:436-524`
with two additions:

1. **ARRAY path accumulator** — prevents cycles by checking
   `NOT (tu.unit_id = ANY(ct.path))` before recursing
2. **Full path array returned** — enables tree reconstruction from flat rows

```sql
WITH RECURSIVE call_tree AS (
    -- Seed: the entry point
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
        ARRAY[u.unit_id] AS path      -- ← NEW: path accumulator
    FROM code_units u
    JOIN code_files f ON u.file_id = f.file_id
    WHERE u.unit_id = :uid

    UNION ALL

    -- Recurse: follow outgoing calls and calls_sp edges
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
        ct.path || tu.unit_id          -- ← NEW: append to path
    FROM call_tree ct
    JOIN code_edges e ON e.source_unit_id = ct.unit_id
    JOIN code_units tu ON e.target_unit_id = tu.unit_id
    JOIN code_files tf ON tu.file_id = tf.file_id
    WHERE e.project_id = :pid
      AND e.edge_type IN ('calls', 'calls_sp')
      AND ct.depth < :max_depth
      AND NOT (tu.unit_id = ANY(ct.path))  -- ← NEW: cycle check
)
SELECT unit_id, name, qualified_name, unit_type, language,
       start_line, end_line, source, file_path,
       depth, edge_type, path
FROM call_tree
ORDER BY depth, name
```

**Comparison with `_traverse()`**:

| Feature | `_traverse()` | Call Tree CTE |
|---------|--------------|---------------|
| Cycle prevention | None (relies on max_depth) | ARRAY path check |
| Path reconstruction | Not supported | Full path array |
| Direction | Configurable (in/out) | Outgoing only (calls) |
| Source code | Not returned | Included in SELECT |
| Edge types | Parameterized | Fixed: calls, calls_sp |

### 3.3 Tree Reconstruction Algorithm

Convert flat CTE rows (with path arrays) into nested `CallTreeNode` tree.

```
Input:  List of rows, each with path = [root_id, ..., this_id]
Output: Nested CallTreeNode tree

Algorithm:
1. Build row_data lookup: unit_id → row (keep shallowest depth per unit)
2. Build children_map: for each row's path, parent = path[-2], child = path[-1]
3. Recursively construct: _build(root_id)
     → create node from row_data[root_id]
     → for each child_id in children_map[root_id]:
          → node.children.append(_build(child_id))
     → return node

Complexity: O(N) where N = number of rows
```

### 3.4 Bounded Analysis Tiers

```
total_tokens = sum(token_count for each unit in call tree)

if total_tokens ≤ 100,000:
    Tier 1 — Full source
    Include all source code from every unit in the tree.
    Formatted with file paths, line numbers, depth annotations.

elif total_tokens ≤ 200,000:
    Tier 2 — Depth-prioritized truncation
    1. Full source for depth 0-2
    2. Signature-only for depth 3+
    3. Fill remaining budget with highest-connectivity deep units
       (sorted by len(children), descending)

else:
    Tier 3 — Summarization fallback
    1. Full source for depth 0-1
    2. Group depth 2+ units by their depth-1 parent ("branch")
    3. Summarize each branch via separate LLM call
    4. Include depth 0-1 source + branch summaries
```

**Token counter**: Reuses `TokenCounter` from `core/code_chunker/token_counter.py:9-17`
(tiktoken `cl100k_base` encoding).

### 3.5 Coverage Calculation

Coverage measures what percentage of a project's code units are covered
by at least one deep analysis.

```sql
-- Coverage query
SELECT
    (SELECT COUNT(DISTINCT unit_id)
     FROM analysis_units
     WHERE project_id = :pid) AS analyzed_units,
    (SELECT COUNT(*)
     FROM code_units
     WHERE project_id = :pid
       AND unit_type IN ('function', 'method', 'class')) AS total_units
```

```
coverage_pct = analyzed_units / total_units * 100

Edge cases:
- Empty project (total_units = 0): coverage = 0%
- No analyses yet: coverage = 0%
- Overlapping analyses: COUNT(DISTINCT unit_id) deduplicates
```

**Performance target**: Coverage calculation < 50ms.
The `idx_analysis_units_project_unit` index on `(project_id, unit_id)` ensures
the `COUNT(DISTINCT unit_id)` is an index-only scan.

### 3.6 Config-Enforced Quality Gates

The following config flags are enforced in runtime code paths:

1. `max_entry_points`: cap `detect_entry_points()` output size (stable sort, keep first N).
2. `require_evidence_refs`: reject parsed analysis payloads where any artifact item has empty/missing `evidence`.
3. `min_narrative_length`: if narrative is shorter than threshold, mark analysis as failed validation and retry/fail per policy.

Validation is applied in `ChainAnalyzer` before `_build_bundle()` and surfaced as
structured `AnalysisError(phase="analysis", recoverable=False)` when hard-failing.

---

## 4. Integration Points

All integration points are additive — existing behavior is preserved when
deep analysis has not been run (guard: `deep_analysis_status != 'completed'`).

### 4.1 Chat Route — Narrative Injection

**File**: `codeloom/api/routes/code_chat.py`
**Location**: Between ASG expansion (line 137) and `build_context_with_history()` call (line 142)

**Current flow**:
```python
# Line 126-139: ASG expansion
if retrieval_results and project.get("asg_status") == "complete":
    ...
    retrieval_results = expander.expand(...)

# Line 142: Context building
context = build_context_with_history(
    retrieval_results=retrieval_results,
    conversation_history=conversation_history,
    max_chunks=data.max_sources,
)
```

**Modified flow** — insert between lines 139 and 142:
```python
# Deep analysis narrative enrichment (new)
deep_narrative = ""
if retrieval_results and project.get("deep_analysis_status") == "completed":
    try:
        from codeloom.core.understanding.engine import UnderstandingEngine
        # Collect unit_ids from retrieval results
        result_unit_ids = [
            nws.node.metadata.get("unit_id")
            for nws in retrieval_results
            if nws.node.metadata.get("unit_id")
        ]
        if result_unit_ids:
            # Look up which analyses cover these units
            narratives = _get_relevant_narratives(
                db_manager, project_id, result_unit_ids
            )
            if narratives:
                deep_narrative = "\n\n## FUNCTIONAL NARRATIVE\n" + "\n\n".join(narratives)
    except Exception as e:
        logger.warning(f"Deep analysis narrative lookup failed: {e}")

# Build context with narrative prepended
context = build_context_with_history(
    retrieval_results=retrieval_results,
    conversation_history=conversation_history,
    max_chunks=data.max_sources,
)
if deep_narrative:
    context = deep_narrative + "\n\n" + context
```

**Helper function** (added to `code_chat.py`):
```python
def _get_relevant_narratives(
    db_manager,
    project_id: str,
    unit_ids: list,
    max_narratives: int = 3,
) -> list:
    """Look up deep analysis narratives covering the given units."""
    from uuid import UUID
    from sqlalchemy import text

    pid = UUID(project_id)
    placeholders = ", ".join(f":uid{i}" for i in range(len(unit_ids)))
    params = {"pid": pid, "limit": max_narratives}
    for i, uid in enumerate(unit_ids):
        params[f"uid{i}"] = UUID(uid) if isinstance(uid, str) else uid

    sql = f"""
        SELECT a.narrative
        FROM deep_analyses a
        JOIN (
            SELECT au.analysis_id,
                   COUNT(*) AS overlap_units,
                   MIN(au.min_depth) AS best_depth,
                   SUM(au.path_count) AS overlap_paths
            FROM analysis_units au
            WHERE au.project_id = :pid
              AND au.unit_id IN ({placeholders})
            GROUP BY au.analysis_id
        ) ov ON ov.analysis_id = a.analysis_id
        WHERE a.narrative IS NOT NULL
          AND a.narrative != ''
        ORDER BY ov.overlap_units DESC, ov.best_depth ASC, ov.overlap_paths DESC
        LIMIT :limit
    """

    with db_manager.get_session() as session:
        result = session.execute(text(sql), params)
        return [row.narrative for row in result.fetchall()]
```

### 4.2 Context Builder — Deep Analysis Context for Migration

**File**: `codeloom/core/migration/context_builder.py`
**Location**: New method on `MigrationContextBuilder` class

```python
def get_deep_analysis_context(
    self,
    unit_ids: List[str],
    budget: int = 4_000,
) -> str:
    """Get deep analysis context for migration phases.

    Looks up analyses covering the given unit_ids and formats
    their business rules, data entities, and integrations
    into a context string for the migration LLM prompt.

    Used by phases 3 (Analyze/Transform), 5 (Transform), and 6 (Test).

    Args:
        unit_ids: List of code unit UUID strings
        budget: Token budget for this context section

    Returns:
        Formatted context string, empty if no analyses found
    """
    if not unit_ids:
        return ""

    placeholders = ", ".join(f":uid{i}" for i in range(len(unit_ids)))
    params = {"pid": self._pid}
    for i, uid in enumerate(unit_ids):
        params[f"uid{i}"] = UUID(uid) if isinstance(uid, str) else uid

    sql = f"""
        SELECT a.analysis_id, a.entry_type, a.narrative,
               a.result_json, u.name AS entry_name,
               COUNT(*) AS overlap_units,
               MIN(au.min_depth) AS best_depth,
               SUM(au.path_count) AS overlap_paths
        FROM analysis_units au
        JOIN deep_analyses a ON au.analysis_id = a.analysis_id
        JOIN code_units u ON a.entry_unit_id = u.unit_id
        WHERE au.project_id = :pid
          AND au.unit_id IN ({placeholders})
        GROUP BY a.analysis_id, a.entry_type, a.narrative, a.result_json, u.name
        ORDER BY overlap_units DESC, best_depth ASC, overlap_paths DESC, u.name
    """

    with self._db.get_session() as session:
        result = session.execute(text(sql), params)
        rows = result.fetchall()

    if not rows:
        return ""

    sections = ["## DEEP ANALYSIS CONTEXT\n"]

    for row in rows:
        sections.append(f"### {row.entry_name} ({row.entry_type})")
        if row.narrative:
            sections.append(row.narrative)

        # Extract key facts from result_json
        import json
        try:
            data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
        except (json.JSONDecodeError, TypeError):
            continue

        rules = data.get("business_rules", [])
        if rules:
            sections.append("\n**Business Rules:**")
            for rule in rules[:5]:
                desc = rule.get("description", rule) if isinstance(rule, dict) else str(rule)
                sections.append(f"- {desc}")

        entities = data.get("data_entities", [])
        if entities:
            sections.append("\n**Data Entities:**")
            for ent in entities[:5]:
                desc = ent.get("name", ent) if isinstance(ent, dict) else str(ent)
                sections.append(f"- {desc}")

        integrations = data.get("integrations", [])
        if integrations:
            sections.append("\n**Integrations:**")
            for integ in integrations[:5]:
                desc = integ.get("description", integ) if isinstance(integ, dict) else str(integ)
                sections.append(f"- {desc}")

        sections.append("")

    return self._join_within_budget(sections, budget)
```

**Injection into migration phases**: In `_build_phase_3_context`, `_build_phase_5_context`,
and `_build_phase_6_context`, add after existing MVP context:

```python
# After existing MVP context sections, before return:
try:
    deep_ctx = self.get_deep_analysis_context(unit_ids, budget=budget // 4)
    if deep_ctx:
        sections.append(deep_ctx)
except Exception as e:
    logger.warning("Deep analysis context unavailable: %s", e)
```

### 4.3 Query Intent Extension

**File**: `codeloom/core/engine/retriever.py`
**Location**: `QueryIntent` enum (line 47-53) and `INTENT_PATTERNS` dict (line 56+)

Add two new intent types:

```python
class QueryIntent(Enum):
    SUMMARY = "summary"
    INSIGHTS = "insights"
    QUESTIONS = "questions"
    SEARCH = "search"
    FLOW = "flow"                   # NEW: execution flow queries
    DATA_LIFECYCLE = "data_lifecycle"  # NEW: data entity lifecycle queries
```

```python
INTENT_PATTERNS[QueryIntent.FLOW] = [
    r'\bhow\s+does\s+.*\s+work\b',
    r'\bexecution\s+(flow|path|chain)\b',
    r'\bcall\s+(chain|graph|tree|flow)\b',
    r'\bwhat\s+happens\s+when\b',
    r'\btrace\b.*\b(call|execution|request)\b',
    r'\bend[- ]?to[- ]?end\b',
]
INTENT_PATTERNS[QueryIntent.DATA_LIFECYCLE] = [
    r'\bdata\s+(flow|lifecycle|journey)\b',
    r'\bwhere\s+is\s+.*\b(stored|saved|persisted)\b',
    r'\bhow\s+is\s+.*\b(created|updated|deleted)\b',
    r'\bentity\s+(lifecycle|flow)\b',
    r'\bCRUD\b',
]
```

### 4.4 ASG Expander — Intent-Aware Expansion

**File**: `codeloom/core/asg_builder/expander.py`
**Location**: `ASGExpander.expand()` method (line 31)

Add optional `intent` parameter that adjusts expansion behavior:

```python
def expand(
    self,
    results: List[NodeWithScore],
    project_id: str,
    cached_nodes: List[TextNode],
    max_expansion: int = 12,
    score_decay: float = 0.7,
    intent: Optional[str] = None,  # NEW parameter
) -> List[NodeWithScore]:
```

When `intent == "flow"`:
- Use `depth=2` instead of `depth=1` for callers/callees
- Set `score_decay=0.5` (deeper neighbors get lower priority)
- Follow `calls_sp` edges in addition to `calls`

Default behavior unchanged when `intent` is None.

### 4.5 Migration Phase LLM Override

**File**: `codeloom/core/migration/phases.py`
**Location**: `_call_llm()` function (line 130-138)

**Current**:
```python
def _call_llm(prompt: str) -> str:
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("No LLM configured. Check LLM_PROVIDER settings.")
    logger.info(f"Calling LLM with prompt of {len(prompt)} chars")
    response = llm.complete(prompt)
    return response.text.strip()
```

**Modified** — add `context_type` parameter for LLM routing:
```python
def _call_llm(prompt: str, context_type: Optional[str] = None) -> str:
    """Call the LLM with a prompt and return the response text.

    If context_type is provided, checks config for LLM overrides:
    - "understanding" → uses migration.llm_overrides.understanding_llm
    - "generation"    → uses migration.llm_overrides.generation_llm
    - None / unknown  → uses Settings.llm (default)
    """
    llm = _get_phase_llm(context_type)
    if llm is None:
        raise RuntimeError("No LLM configured. Check LLM_PROVIDER settings.")
    logger.info(f"Calling LLM ({context_type or 'default'}) with prompt of {len(prompt)} chars")
    response = llm.complete(prompt)
    return response.text.strip()


def _get_phase_llm(context_type: Optional[str] = None):
    """Resolve the LLM for a given context type.

    Checks codeloom.yaml → migration.llm_overrides for per-phase LLM config.
    Falls back to Settings.llm if no override is configured.
    """
    if context_type:
        from ...setting import get_settings
        settings = get_settings()
        overrides = getattr(settings, 'migration_llm_overrides', {}) or {}
        override_key = f"{context_type}_llm"
        if override_key in overrides and overrides[override_key]:
            # TODO: Instantiate LLM from override config
            # For now, fall through to default
            pass
    return Settings.llm
```

**Backward compatibility**: All existing callers pass no `context_type`, so they
get `Settings.llm` as before. The override resolution is a no-op until
configuration is explicitly set.

### 4.6 Dependencies Module

**File**: `codeloom/api/deps.py`
**Location**: After `get_migration_engine()` (line 67-75)

```python
async def get_understanding_engine(request: Request):
    """Get or create UnderstandingEngine from app state."""
    if not hasattr(request.app.state, 'understanding_engine') or \
       request.app.state.understanding_engine is None:
        from codeloom.core.understanding.engine import UnderstandingEngine
        request.app.state.understanding_engine = UnderstandingEngine(
            request.app.state.db_manager,
            request.app.state.pipeline,
        )
    return request.app.state.understanding_engine
```

Follows the exact pattern of `get_migration_engine()`: lazy init, inline import,
stored on `app.state`.

### 4.7 App Factory — Router Registration

**File**: `codeloom/api/app.py`
**Location**: After migration router registration (line 78)

```python
from .routes.understanding import router as understanding_router
app.include_router(understanding_router, prefix="/api")
```

### 4.8 Configuration

**File**: `config/codeloom.yaml`
**Location**: After the existing `sql_chat` section (line 362)

See Section 8 for the complete YAML additions.

---

## 5. API Design

### 5.1 Understanding Endpoints

All endpoints prefixed with `/api/understanding/`.

#### `POST /api/understanding/{project_id}/analyze`

Start a deep analysis job.

**Request**: No body required.

**Response** (202 Accepted):
```json
{
  "job_id": "uuid-string",
  "project_id": "uuid-string",
  "status": "pending"
}
```

**Error responses**:
- `404`: Project not found
- `400`: Project ASG not complete (`asg_status != 'complete'`)
- `409`: Analysis already running for this project

#### `GET /api/understanding/{project_id}/status/{job_id}`

Poll job progress.

**Response** (200):
```json
{
  "job_id": "uuid-string",
  "project_id": "uuid-string",
  "status": "running",
  "progress": {
    "total": 42,
    "completed": 15
  },
  "created_at": "2026-02-19T10:00:00Z",
  "started_at": "2026-02-19T10:00:05Z",
  "completed_at": null,
  "retry_count": 0,
  "errors": null
}
```

#### `GET /api/understanding/{project_id}/entry-points`

Synchronous entry point detection (no job required).

**Response** (200):
```json
{
  "entry_points": [
    {
      "unit_id": "uuid-string",
      "name": "placeOrder",
      "qualified_name": "com.acme.OrderService.placeOrder",
      "file_path": "src/main/java/com/acme/OrderService.java",
      "entry_type": "http_endpoint",
      "language": "java",
      "detected_by": "both"
    }
  ],
  "count": 42
}
```

#### `GET /api/understanding/{project_id}/results`

Get analysis summaries for all analyzed entry points.

**Response** (200):
```json
{
  "analyses": [
    {
      "analysis_id": "uuid-string",
      "entry_unit_id": "uuid-string",
      "entry_name": "placeOrder",
      "entry_qualified_name": "com.acme.OrderService.placeOrder",
      "entry_file": "src/main/java/com/acme/OrderService.java",
      "entry_type": "http_endpoint",
      "tier": "tier_1",
      "total_units": 15,
      "total_tokens": 8500,
      "confidence_score": 0.91,
      "coverage_pct": 84.2,
      "narrative": "The placeOrder endpoint validates...",
      "schema_version": 1,
      "analyzed_at": "2026-02-19T10:05:00Z"
    }
  ],
  "count": 42,
  "coverage_pct": 78.5
}
```

#### `GET /api/understanding/{project_id}/chain/{analysis_id}`

Get full analysis detail including evidence references.

**Response** (200):
```json
{
  "analysis_id": "uuid-string",
  "entry_point": {
    "unit_id": "uuid-string",
    "name": "placeOrder",
    "qualified_name": "com.acme.OrderService.placeOrder",
    "file_path": "src/main/java/com/acme/OrderService.java",
    "entry_type": "http_endpoint"
  },
  "tier": "tier_1",
  "total_units": 15,
  "total_tokens": 8500,
  "confidence_score": 0.91,
  "coverage_pct": 84.2,
  "narrative": "The placeOrder endpoint validates...",
  "result": {
    "business_rules": [
      {
        "description": "Orders must have at least one line item",
        "evidence": [
          {
            "unit_id": "uuid-string",
            "qualified_name": "com.acme.OrderValidator.validate",
            "file_path": "src/main/java/com/acme/OrderValidator.java",
            "start_line": 42,
            "end_line": 48,
            "snippet": "if (order.getItems().isEmpty()) { throw new..."
          }
        ]
      }
    ],
    "data_entities": [...],
    "integrations": [...],
    "side_effects": [...],
    "cross_cutting_concerns": [...]
  },
  "units": [
    {
      "unit_id": "uuid-string",
      "name": "placeOrder",
      "qualified_name": "com.acme.OrderService.placeOrder",
      "unit_type": "method",
      "file_path": "src/main/java/com/acme/OrderService.java",
      "min_depth": 0,
      "path_count": 1
    }
  ]
}
```

### 5.2 Settings Endpoints

#### `GET /api/settings/migration-llm`

Get current LLM override configuration.

**Response** (200):
```json
{
  "understanding_llm": null,
  "generation_llm": null,
  "defaults": {
    "provider": "ollama",
    "model": "llama3.1:8b"
  }
}
```

#### `POST /api/settings/migration-llm`

Set LLM overrides for migration phases.

**Request**:
```json
{
  "understanding_llm": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "generation_llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929"
  }
}
```

**Response** (200): Updated configuration echoed back.

### 5.3 Route Implementation

**File**: `codeloom/api/routes/understanding.py`

```python
"""Understanding API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user, get_db_manager, get_understanding_engine, get_project_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["understanding"])


@router.post("/understanding/{project_id}/analyze", status_code=202)
async def start_analysis(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project_for_user(project_id, user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("asg_status") != "complete":
        raise HTTPException(status_code=400, detail="Project ASG not complete")
    if project.get("deep_analysis_status") == "running":
        raise HTTPException(status_code=409, detail="Analysis already running")

    result = engine.start_analysis(project_id, user["user_id"])
    return result


@router.get("/understanding/{project_id}/status/{job_id}")
async def get_status(
    project_id: str,
    job_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project_for_user(project_id, user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = engine.get_job_status(project_id, job_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/understanding/{project_id}/entry-points")
async def get_entry_points(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project_for_user(project_id, user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eps = engine.get_entry_points(project_id)
    return {"entry_points": eps, "count": len(eps)}


@router.get("/understanding/{project_id}/results")
async def get_results(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project_for_user(project_id, user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    analyses = engine.get_analysis_results(project_id)

    # Calculate coverage
    from sqlalchemy import text
    from uuid import UUID
    db = engine._db
    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    (SELECT COUNT(DISTINCT unit_id)
                     FROM analysis_units WHERE project_id = :pid) AS analyzed,
                    (SELECT COUNT(*) FROM code_units
                     WHERE project_id = :pid
                       AND unit_type IN ('function', 'method', 'class')) AS total
            """),
            {"pid": UUID(project_id)},
        )
        row = result.fetchone()
        coverage = (row.analyzed / row.total * 100) if row.total > 0 else 0.0

    return {"analyses": analyses, "count": len(analyses), "coverage_pct": round(coverage, 1)}


@router.get("/understanding/{project_id}/chain/{analysis_id}")
async def get_chain_detail(
    project_id: str,
    analysis_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project_for_user(project_id, user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = engine.get_chain_detail(project_id, analysis_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
```

---

## 6. Prompt Engineering

### 6.1 Chain Analysis Prompt

```python
# In prompts.py

def build_chain_analysis_prompt(
    entry_point,
    source_payload: str,
    tier,
    framework_contexts: list,
) -> str:
    """Build the per-entry-point analysis prompt."""

    framework_section = ""
    if framework_contexts:
        hints = []
        for fc in framework_contexts:
            hints.append(f"Framework: {fc['framework_name']}")
            for hint in fc.get("analysis_hints", []):
                hints.append(f"  - {hint}")
            if fc.get("transaction_boundaries"):
                hints.append(f"  Transaction boundaries: {', '.join(fc['transaction_boundaries'][:5])}")
            if fc.get("di_registrations"):
                hints.append(f"  DI registrations: {len(fc['di_registrations'])} found")
        framework_section = "\n## FRAMEWORK CONTEXT\n" + "\n".join(hints)

    tier_notice = ""
    if tier.value == "tier_2":
        tier_notice = (
            "\nNOTE: Some deeper functions show only signatures (not full source). "
            "Infer their behavior from the signature, name, and how they are called."
        )
    elif tier.value == "tier_3":
        tier_notice = (
            "\nNOTE: Deep branches have been summarized. "
            "Work with the summaries provided; do not assume code details not shown."
        )

    return f"""You are a senior software architect analyzing a complete execution path through a codebase.

## ENTRY POINT
- Name: {entry_point.qualified_name}
- Type: {entry_point.entry_type.value}
- File: {entry_point.file_path}
- Language: {entry_point.language}
{framework_section}

## COMPLETE CALL CHAIN SOURCE CODE
{tier_notice}

{source_payload}

## YOUR TASK

Analyze this execution path and extract structured understanding. For EVERY fact you identify, provide evidence references back to the source code (unit_id, qualified_name, file_path, start_line, end_line).

Return your analysis as a JSON object with this exact structure:

```json
{{
  "business_rules": [
    {{
      "description": "Clear description of the business rule",
      "severity": "critical|important|minor",
      "evidence": [
        {{
          "unit_id": "uuid-string",
          "qualified_name": "com.acme.Service.method",
          "file_path": "src/main/java/com/acme/Service.java",
          "start_line": 42,
          "end_line": 48,
          "snippet": "relevant code excerpt"
        }}
      ]
    }}
  ],
  "data_entities": [
    {{
      "name": "Entity name",
      "operations": ["create", "read", "update", "delete"],
      "evidence": [...]
    }}
  ],
  "integrations": [
    {{
      "description": "What external system is called and why",
      "type": "database|api|queue|file|cache",
      "evidence": [...]
    }}
  ],
  "side_effects": [
    {{
      "description": "What side effect occurs (email, logging, audit, etc.)",
      "trigger": "What triggers this side effect",
      "evidence": [...]
    }}
  ],
  "cross_cutting_concerns": [
    "Authentication check via Spring Security filter",
    "Transaction boundary at service layer"
  ],
  "narrative": "A 2-3 paragraph plain-English description of what this execution path does, suitable for a developer who has never seen this code."
}}
```

IMPORTANT:
- Every business_rule, data_entity, integration, and side_effect MUST have at least one evidence reference
- Every evidence item MUST include `unit_id`
- Use exact qualified_names and line numbers from the source code above
- The narrative should be informative enough to stand alone without the source code
- Be thorough but precise — quality over quantity
"""
```

### 6.2 Cross-Cutting Concern Prompt

```python
def build_cross_cutting_prompt(
    entry_point_summaries: list,
    framework_contexts: list,
) -> str:
    """Build prompt for detecting cross-cutting concerns across entry points."""

    summaries_text = "\n\n".join(
        f"### {s['name']} ({s['entry_type']})\n{s['narrative']}"
        for s in entry_point_summaries
    )

    return f"""You are analyzing cross-cutting concerns across multiple execution paths in a codebase.

## ENTRY POINT SUMMARIES

{summaries_text}

## YOUR TASK

Identify cross-cutting concerns that span multiple execution paths:

1. **Shared business rules** — rules that appear in multiple paths
2. **Common data access patterns** — entities accessed by multiple paths
3. **Shared integrations** — external systems used across paths
4. **Inconsistencies** — places where similar logic is handled differently

Return as JSON:
```json
{{
  "shared_rules": [
    {{"description": "...", "affected_paths": ["path1", "path2"]}}
  ],
  "common_data_patterns": [
    {{"entity": "...", "paths": ["path1", "path2"], "operations": ["read", "write"]}}
  ],
  "shared_integrations": [
    {{"system": "...", "paths": ["path1", "path2"]}}
  ],
  "inconsistencies": [
    {{"description": "...", "path_a": "...", "path_b": "...", "severity": "high|medium|low"}}
  ]
}}
```
"""
```

### 6.3 Branch Summary Prompt (Tier 3)

```python
def build_branch_summary_prompt(branch_name: str, source: str) -> str:
    """Build prompt for summarizing a deep call tree branch."""

    return f"""Summarize the following code branch concisely.
Focus on: what it does, what data it touches, what side effects it has.

Branch: {branch_name}

```
{source[:50000]}
```

Provide a 3-5 sentence summary covering:
1. Primary purpose of this branch
2. Key data entities accessed or modified
3. External integrations or side effects
4. Business rules enforced
"""
```

### 6.4 Framework Detection Prompt (Fallback)

```python
def build_framework_detection_prompt(
    imports: list,
    file_samples: list,
) -> str:
    """Fallback prompt for unrecognized frameworks."""

    imports_text = "\n".join(imports[:50])
    samples_text = "\n\n".join(
        f"// {s['path']}\n{s['source'][:500]}"
        for s in file_samples[:5]
    )

    return f"""Analyze these code imports and samples to identify the application framework.

## IMPORTS
{imports_text}

## FILE SAMPLES
{samples_text}

Identify:
1. Primary web framework (e.g., Spring, Django, Express, ASP.NET)
2. DI framework (if any)
3. ORM/data access framework
4. Security framework
5. Any middleware/filter pipeline

Return as JSON:
```json
{{
  "framework_name": "...",
  "framework_type": "...",
  "version_hints": "...",
  "analysis_hints": ["hint1", "hint2"]
}}
```
"""
```

---

## 7. Worker State Machine

### 7.1 Job Lifecycle

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
              ┌──────────┐     claim (FOR UPDATE     ┌────────┐
  create ───▶ │ pending  │ ──── SKIP LOCKED) ──────▶ │running │
              └──────────┘                           └────────┘
                    ▲                                  │    │
                    │                                  │    │
                    │    stale reclaim                  │    │
                    │    (heartbeat_at stale            │    │
                    │     + retry_count < max_retries)  │    │
                    │                                  │    │
                    └──────────────────────────────────┘    │
                                                           │
                              ┌─────────────────────────────┤
                              │                             │
                              ▼                             ▼
                        ┌───────────┐              ┌────────────┐
                        │ completed │              │   failed   │
                        └───────────┘              └────────────┘
                                                   (retry_count ≥ max_retries
                                                    OR fatal error)
```

### 7.2 State Transitions

| From | To | Trigger | SQL |
|------|-----|---------|-----|
| — | `pending` | `start_analysis()` API call | `INSERT INTO deep_analysis_jobs` |
| `pending` | `pending` | Retry backoff window not elapsed | `next_attempt_at > NOW()` claim skip |
| `pending` | `running` | Worker claims via SKIP LOCKED | See 7.3 |
| `running` | `completed` | All entry points processed | `UPDATE SET status='completed'` |
| `running` | `failed` | Fatal error + retry_count ≥ max | `UPDATE SET status='failed'` |
| `running` | `pending` | Stale heartbeat + retry_count < max | See 7.3 |

### 7.3 SQL Operations

All write transitions are lease-scoped (`WHERE ... AND worker_id = :worker_id`)
once a job is claimed.

#### Claim SQL
```sql
UPDATE deep_analysis_jobs
SET status = 'running',
    worker_id = :worker_id,
    started_at = NOW(),
    heartbeat_at = NOW(),
    retry_count = retry_count + 1,
    next_attempt_at = NULL
WHERE job_id IN (
    SELECT job_id FROM deep_analysis_jobs
    WHERE status = 'pending'
      AND (next_attempt_at IS NULL OR next_attempt_at <= NOW())
    ORDER BY created_at
    LIMIT :limit
    FOR UPDATE SKIP LOCKED
)
RETURNING job_id, project_id, worker_id
```

#### Heartbeat SQL
```sql
UPDATE deep_analysis_jobs
SET heartbeat_at = NOW()
WHERE status = 'running'
  AND worker_id = :worker_id
```

#### Stale Reclaim SQL
```sql
UPDATE deep_analysis_jobs
SET status = 'pending', worker_id = NULL
WHERE status = 'running'
  AND heartbeat_at < NOW() - INTERVAL '120 seconds'
  AND retry_count < :max_retries
```

#### Complete SQL
```sql
UPDATE deep_analysis_jobs
SET status = :status,
    completed_at = NOW(),
    error_details = :errors
WHERE job_id = :job_id
  AND worker_id = :worker_id
```

#### Retry Backoff SQL
```sql
UPDATE deep_analysis_jobs
SET status = 'pending',
    worker_id = NULL,
    next_attempt_at = :now + (:base_backoff_seconds * POWER(2, :retry_count)) * INTERVAL '1 second'
WHERE job_id = :job_id
  AND worker_id = :worker_id
```

### 7.4 Concurrency Model

- **Thread model**: Single daemon thread with asyncio event loop
  (mirrors `RAPTORWorker` from `core/raptor/worker.py:42-110`)
- **Concurrency**: `asyncio.Semaphore(2)` — at most 2 jobs processed simultaneously
- **Queue backpressure**: `asyncio.Queue` with unbounded capacity (bounded by
  `LIMIT` in claim query, typically 2)
- **Thread safety**: All DB access via `DatabaseManager.get_session()` context
  manager, which provides thread-safe session handling
- **Lease ownership**: Every running job is owned by one `worker_id`; heartbeats
  and terminal transitions are filtered by that lease owner
- **Heartbeat**: Separate asyncio task updates `heartbeat_at` every 30s
- **Stale threshold**: 120s — if a worker crashes, its jobs are reclaimed
  after 2 minutes

---

## 8. Configuration Schema

### 8.1 YAML Additions

Add to `config/codeloom.yaml` after the `sql_chat` section:

```yaml
# -----------------------------------------------------------------------------
# MIGRATION — Deep Analysis & LLM Overrides
# Controls the Deep Understanding Engine and per-phase LLM routing
# -----------------------------------------------------------------------------
migration:
  # Deep Analysis settings
  deep_analysis:
    # Token budget thresholds for analysis tiers
    tier_1_max_tokens: 100000       # Full source if ≤ this
    tier_2_max_tokens: 200000       # Depth-prioritized truncation if ≤ this
    # Above tier_2_max → Tier 3 (summarization fallback)

    # Call tree tracing
    max_trace_depth: 10             # Maximum call tree depth
    max_entry_points: 200           # Safety limit on entry points per project

    # Analysis quality
    require_evidence_refs: true     # Reject analysis without evidence refs
    min_narrative_length: 100       # Minimum chars for narrative

    # Worker settings
    worker:
      enabled: true                 # Enable background worker
      poll_interval: 15.0           # Seconds between job polls
      max_concurrent: 2             # Simultaneous jobs (Semaphore size)
      heartbeat_interval: 30.0      # Seconds between heartbeats
      stale_threshold: 120.0        # Seconds before reclaiming stale jobs
      max_retries: 2                # Max retry attempts before terminal failure
      retry_backoff_base_seconds: 15 # Backoff base; attempt N waits base*2^(N-1)

    # Coverage thresholds
    coverage:
      warn_below: 50.0             # Warn in UI if coverage < 50%
      target: 80.0                  # Target coverage percentage

    # Entry point detection
    detection:
      enable_pass1_heuristic: true  # Zero-incoming-calls detection
      enable_pass2_annotations: true # Annotation pattern matching
      skip_test_functions: true     # Exclude test_* functions from Pass 1

  # LLM overrides for different migration phases
  # When set, these override Settings.llm for the specified context
  llm_overrides:
    # LLM for understanding/analysis tasks (deep analysis, chain analysis)
    understanding_llm: null
    # Example:
    # understanding_llm:
    #   provider: openai
    #   model: gpt-4o
    #   temperature: 0.1

    # LLM for code generation tasks (transform, test generation)
    generation_llm: null
    # Example:
    # generation_llm:
    #   provider: anthropic
    #   model: claude-sonnet-4-5-20250929
    #   temperature: 0.2
```

### 8.2 Default Values and Justifications

| Setting | Default | Justification |
|---------|---------|---------------|
| `tier_1_max_tokens` | 100,000 | Fits in most LLM context windows (128K) with room for prompt |
| `tier_2_max_tokens` | 200,000 | Allows depth-prioritized truncation for larger codebases |
| `max_trace_depth` | 10 | Balances thoroughness with CTE performance; most call chains < 10 deep |
| `max_entry_points` | 200 | Safety limit; large projects may have thousands but analysis scales linearly |
| `poll_interval` | 15s | Balances responsiveness with DB load; faster than RAPTOR's 10s due to heavier jobs |
| `max_concurrent` | 2 | LLM calls are the bottleneck, not CPU; 2 avoids rate limiting |
| `heartbeat_interval` | 30s | Frequent enough to detect crashes quickly, infrequent enough to avoid DB chatter |
| `stale_threshold` | 120s | 4x heartbeat interval; allows for temporary network issues |
| `max_retries` | 2 | Transient LLM errors are common; bounded retries avoid infinite churn |
| `retry_backoff_base_seconds` | 15 | Avoids immediate hot-loop retries after provider/transient failures |
| `coverage.target` | 80% | Reasonable default; some utility functions may not need deep analysis |

### 8.3 Runtime Reconfigurability

| Setting | Runtime Reconfigurable | Mechanism |
|---------|----------------------|-----------|
| `tier_*_max_tokens` | Yes | Read on each analysis invocation |
| `max_trace_depth` | Yes | Read on each trace invocation |
| `max_entry_points` | Yes | Read on each detection invocation |
| `worker.max_concurrent` | No | Requires worker restart |
| `worker.poll_interval` | No | Requires worker restart |
| `worker.heartbeat_interval` | No | Requires worker restart |
| `worker.stale_threshold` | No | Requires worker restart |
| `worker.max_retries` | Yes | Read in claim/reclaim/failure paths |
| `worker.retry_backoff_base_seconds` | Yes | Read on each failure transition |
| `coverage.*` | Yes | Read on each coverage query |
| `detection.*` | Yes | Read on each detection invocation |
| `llm_overrides.*` | Yes | Read on each `_call_llm()` invocation |

---

## 9. Testing Strategy

### 9.1 Test File Layout

```
codeloom/tests/test_understanding/
├── __init__.py
├── conftest.py                  # Shared fixtures (mock DB, sample projects)
├── test_chain_tracer.py         # Entry point detection + call tree tracing
├── test_analyzer.py             # Tiered analysis + JSON parsing
├── test_worker.py               # Job lifecycle + retry behavior
├── test_engine.py               # Orchestrator API
├── test_models.py               # Data contract validation
├── test_prompts.py              # Prompt template formatting
├── test_frameworks/
│   ├── __init__.py
│   ├── test_spring.py           # Spring detection + analysis
│   └── test_aspnet.py           # ASP.NET detection + analysis
└── test_integration/
    ├── __init__.py
    ├── test_job_lifecycle.py    # Full job: pending → running → completed
    ├── test_chat_enrichment.py  # Chat with deep analysis narratives
    ├── test_migration_context.py # Migration phases with deep context
    └── test_no_analysis.py      # Fallback when no analysis exists
```

### 9.2 Unit Tests

#### `test_chain_tracer.py`

```python
"""Tests for ChainTracer — entry point detection and call tree tracing."""

class TestEntryPointDetection:
    """Test dual-pass entry point detection."""

    def test_pass1_finds_uncalled_functions(self, mock_db, sample_project):
        """Functions with zero incoming 'calls' edges are detected."""
        tracer = ChainTracer(mock_db)
        eps = tracer.detect_entry_points(sample_project.id)
        uncalled = [ep for ep in eps if ep.detected_by in ("heuristic", "both")]
        assert len(uncalled) > 0
        # Verify none have incoming calls edges
        for ep in uncalled:
            callers = get_callers(mock_db, sample_project.id, ep.unit_id)
            assert len(callers) == 0

    def test_pass2_finds_annotated_endpoints(self, mock_db, spring_project):
        """Spring @GetMapping annotated methods are detected as HTTP_ENDPOINT."""
        tracer = ChainTracer(mock_db)
        eps = tracer.detect_entry_points(spring_project.id)
        http_eps = [ep for ep in eps if ep.entry_type == EntryPointType.HTTP_ENDPOINT]
        assert len(http_eps) > 0

    def test_pass2_overrides_pass1_type(self, mock_db, spring_project):
        """When Pass 2 identifies a specific type, it overrides Pass 1's UNKNOWN."""
        tracer = ChainTracer(mock_db)
        eps = tracer.detect_entry_points(spring_project.id)
        both_detected = [ep for ep in eps if ep.detected_by == "both"]
        for ep in both_detected:
            assert ep.entry_type != EntryPointType.UNKNOWN

    def test_deduplication_by_unit_id(self, mock_db, sample_project):
        """No duplicate unit_ids in results even if detected by both passes."""
        tracer = ChainTracer(mock_db)
        eps = tracer.detect_entry_points(sample_project.id)
        ids = [ep.unit_id for ep in eps]
        assert len(ids) == len(set(ids))

    def test_empty_project_returns_empty(self, mock_db, empty_project):
        """Empty project returns zero entry points."""
        tracer = ChainTracer(mock_db)
        eps = tracer.detect_entry_points(empty_project.id)
        assert eps == []


class TestCallTreeTracing:
    """Test call tree CTE and tree reconstruction."""

    def test_simple_chain(self, mock_db, linear_call_chain):
        """A → B → C produces a tree with depth 2."""
        tracer = ChainTracer(mock_db)
        tree = tracer.trace_call_tree(
            linear_call_chain.project_id,
            linear_call_chain.entry_id,
        )
        assert tree.depth == 0
        assert len(tree.children) == 1
        assert tree.children[0].depth == 1
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].depth == 2

    def test_cycle_prevention(self, mock_db, cyclic_call_graph):
        """A → B → C → A does not infinite-loop; cycle is broken."""
        tracer = ChainTracer(mock_db)
        tree = tracer.trace_call_tree(
            cyclic_call_graph.project_id,
            cyclic_call_graph.entry_id,
            max_depth=10,
        )
        # Tree should contain A, B, C but not recurse back to A
        all_ids = _collect_ids(tree)
        assert len(all_ids) == 3  # A, B, C (no duplicates)

    def test_max_depth_respected(self, mock_db, deep_call_chain):
        """Call tree respects max_depth parameter."""
        tracer = ChainTracer(mock_db)
        tree = tracer.trace_call_tree(
            deep_call_chain.project_id,
            deep_call_chain.entry_id,
            max_depth=3,
        )
        max_seen = _max_depth(tree)
        assert max_seen <= 3

    def test_diamond_dependency(self, mock_db, diamond_graph):
        """A → B, A → C, B → D, C → D: D appears once at minimum depth."""
        tracer = ChainTracer(mock_db)
        tree = tracer.trace_call_tree(
            diamond_graph.project_id,
            diamond_graph.entry_id,
        )
        flat = tracer.get_flat_unit_membership(tree)
        d_entries = [u for u in flat if u["unit_id"] == diamond_graph.d_id]
        assert len(d_entries) == 1  # Deduplicated

    def test_leaf_node_for_no_callees(self, mock_db, isolated_function):
        """Function with no outgoing calls returns a leaf node."""
        tracer = ChainTracer(mock_db)
        tree = tracer.trace_call_tree(
            isolated_function.project_id,
            isolated_function.unit_id,
        )
        assert tree.children == []
        assert tree.source is not None
```

#### `test_analyzer.py`

```python
"""Tests for ChainAnalyzer — tiered analysis and output parsing."""

class TestTierSelection:
    def test_tier1_under_100k(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(50_000) == AnalysisTier.TIER_1

    def test_tier2_between_100k_200k(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(150_000) == AnalysisTier.TIER_2

    def test_tier3_over_200k(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(300_000) == AnalysisTier.TIER_3

    def test_tier1_boundary(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(100_000) == AnalysisTier.TIER_1
        assert analyzer._select_tier(100_001) == AnalysisTier.TIER_2


class TestJsonParsing:
    def test_clean_json(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output('Here is the result:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_invalid_json_returns_error(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output('not json at all')
        assert "parse_error" in result


class TestTokenCounting:
    def test_counts_tree_recursively(self, sample_tree):
        analyzer = ChainAnalyzer()
        total = analyzer._count_tree_tokens(sample_tree)
        assert total > 0
        assert sample_tree.token_count > 0  # Root should have tokens

    def test_empty_source_counts_zero(self):
        analyzer = ChainAnalyzer()
        node = CallTreeNode(
            unit_id="test", name="test", qualified_name="test",
            unit_type="function", language="python", file_path="test.py",
            start_line=1, end_line=1, source=None, depth=0,
        )
        total = analyzer._count_tree_tokens(node)
        assert total == 0
```

#### `test_worker.py`

```python
"""Tests for UnderstandingWorker — job lifecycle and retry behavior."""

class TestJobClaiming:
    def test_claims_oldest_pending_first(self, mock_db, pending_jobs):
        worker = UnderstandingWorker(mock_db)
        jobs = worker._claim_pending_jobs(limit=1)
        assert len(jobs) == 1
        assert jobs[0].job_id == pending_jobs[0].id  # Oldest

    def test_skip_locked_prevents_double_claim(self, mock_db, pending_jobs):
        worker1 = UnderstandingWorker(mock_db)
        worker2 = UnderstandingWorker(mock_db)
        jobs1 = worker1._claim_pending_jobs(limit=1)
        jobs2 = worker2._claim_pending_jobs(limit=1)
        assert jobs1[0].job_id != jobs2[0].job_id

    def test_stale_jobs_reclaimed(self, mock_db, stale_running_job):
        worker = UnderstandingWorker(mock_db, stale_threshold=0)
        jobs = worker._claim_pending_jobs(limit=1)
        assert len(jobs) == 1
        assert jobs[0].job_id == stale_running_job.id
        assert jobs[0].worker_id == worker.worker_id


class TestRetryBehavior:
    def test_retries_on_transient_error(self, mock_db, failed_job_retry_1):
        worker = UnderstandingWorker(mock_db, max_retries=2)
        worker._handle_job_failure(failed_job_retry_1.id, "transient error")
        # Should go back to pending
        status = _get_job_status(mock_db, failed_job_retry_1.id)
        assert status == "pending"
        assert _get_next_attempt_at(mock_db, failed_job_retry_1.id) is not None

    def test_terminal_failure_after_max_retries(self, mock_db, failed_job_retry_2):
        worker = UnderstandingWorker(mock_db, max_retries=2)
        worker._handle_job_failure(failed_job_retry_2.id, "persistent error")
        status = _get_job_status(mock_db, failed_job_retry_2.id)
        assert status == "failed"
```

### 9.3 Integration Tests

#### `test_integration/test_job_lifecycle.py`

```python
"""Full job lifecycle: pending → running → completed."""

class TestFullJobLifecycle:
    def test_end_to_end_analysis(self, db_with_sample_project):
        """Create job → process → verify results stored."""
        engine = UnderstandingEngine(db_with_sample_project)

        # Start analysis
        result = engine.start_analysis(sample_project_id)
        assert result["status"] == "pending"

        # Process synchronously (bypass worker for testing)
        worker = UnderstandingWorker(db_with_sample_project)
        jobs = worker._claim_pending_jobs(limit=1)
        assert len(jobs) == 1
        worker._process_job_sync(jobs[0])

        # Verify results
        status = engine.get_job_status(sample_project_id, result["job_id"])
        assert status["status"] == "completed"
        assert status["progress"]["completed"] > 0

        # Verify analysis results exist
        results = engine.get_analysis_results(sample_project_id)
        assert len(results) > 0
        assert all(r["narrative"] for r in results)
```

#### `test_integration/test_no_analysis.py`

```python
"""Verify fallback behavior when no analysis exists."""

class TestNoAnalysisFallback:
    def test_chat_works_without_analysis(self, db_with_project_no_analysis):
        """Chat endpoint works identically when deep_analysis_status is 'none'."""
        # The narrative lookup returns empty, context is built without it
        narratives = _get_relevant_narratives(db, project_id, unit_ids)
        assert narratives == []

    def test_migration_works_without_analysis(self, db_with_project_no_analysis):
        """Migration context builder returns empty deep context."""
        ctx = MigrationContextBuilder(db, project_id)
        deep = ctx.get_deep_analysis_context(unit_ids)
        assert deep == ""
```

### 9.4 Performance Benchmarks

| Operation | Target | Measurement |
|-----------|--------|-------------|
| `analysis_units` lookup (by project + unit_id) | < 10ms | Index scan on `idx_analysis_units_project_unit` |
| Coverage calculation | < 50ms | Index-only scan with COUNT(DISTINCT) |
| Entry point detection (1000 units) | < 500ms | Two SQL queries + Python merge |
| Call tree CTE (depth=10, 100 nodes) | < 200ms | Single recursive CTE |
| Narrative lookup (3 units) | < 20ms | Index join on analysis_units + deep_analyses |

### 9.5 Mock/Fixture Strategy

- **LLM calls**: Mock `Settings.llm.complete()` to return predefined JSON responses
- **Database**: Use test PostgreSQL database with `alembic upgrade head` in fixture
- **Sample projects**: Create minimal projects with known call graphs (linear, cyclic, diamond)
- **Framework detection**: Seed `code_units` with framework-specific annotations

---

## 10. Deployment & Rollout

### 10.1 Alembic Migration Strategy

**Order**:
1. Add `deep_analysis_status` column to `projects` (with `server_default='none'`)
2. Create `deep_analysis_jobs` table
3. Create `deep_analyses` table (depends on `deep_analysis_jobs`)
4. Create `analysis_units` table (depends on `deep_analyses`)

**Downgrade path**: Reverse order — drop tables, then drop column.

**Pre-migration checklist**:
- Backup database
- Verify no active migration jobs
- Run `alembic upgrade head` during maintenance window

### 10.2 Multi-Worker Compatibility

The `UnderstandingWorker` uses `FOR UPDATE SKIP LOCKED` for job claiming,
which is safe for multiple workers. However, for initial rollout:

```bash
# Disable in multi-worker (Gunicorn) mode:
DISABLE_BACKGROUND_WORKERS=true

# Only enable on a single designated worker:
# In __main__.py, check env before starting worker
```

The worker is disabled when `DISABLE_BACKGROUND_WORKERS=true` (same flag
that controls `RAPTORWorker`).

### 10.3 Resource Estimation

**LLM token consumption per project** (estimates):

| Project Size | Entry Points | Avg Tokens/Chain | Total LLM Tokens | Cost (GPT-4o) |
|-------------|-------------|-------------------|-------------------|---------------|
| Small (50 units) | ~10 | ~5K | ~50K | ~$0.25 |
| Medium (500 units) | ~50 | ~20K | ~1M | ~$5.00 |
| Large (5000 units) | ~200 | ~50K | ~10M | ~$50.00 |

**DB growth per project**:
- `deep_analysis_jobs`: 1 row per run (~1KB)
- `deep_analyses`: 1 row per entry point (~5KB avg with result_json)
- `analysis_units`: ~15 rows per entry point (~100 bytes each)
- Example: 200 entry points → ~1MB deep_analyses + ~300KB analysis_units

### 10.4 Backward Compatibility Proofs

All changes are additive. Existing behavior is preserved:

1. **New column** `projects.deep_analysis_status`: Has `server_default='none'`.
   Existing code never reads this column; no breakage.

2. **New tables**: No existing code references them. Foreign keys use
   `ondelete='CASCADE'` so project deletion cascades cleanly.

3. **Chat route modification**: Guarded by `project.get("deep_analysis_status") == "completed"`.
   For existing projects (status='none'), the guard is False → no change.

4. **Context builder**: `get_deep_analysis_context()` is a new method.
   Existing phase builders don't call it until explicitly modified.
   The injection code uses try/except with fallback to empty string.

5. **Query intent**: New enum values (`FLOW`, `DATA_LIFECYCLE`) are additive.
   Existing `SEARCH` default behavior unchanged.

6. **ASG expander**: New `intent` parameter defaults to `None`.
   Existing callers pass no intent → identical behavior.

7. **`_call_llm()`**: New `context_type` parameter defaults to `None`.
   All existing callers pass no context_type → falls through to `Settings.llm`.

8. **Dependencies**: `get_understanding_engine()` only created when accessed.
   No impact on routes that don't use it.

### 10.5 Phased Rollout

#### Phase A: Foundations (Sprint 1)
- [ ] Alembic migration (3 tables + 1 column)
- [ ] `core/understanding/models.py`
- [ ] `core/understanding/chain_tracer.py`
- [ ] `core/understanding/frameworks/`
- [ ] Unit tests for detection + tracing
- **Verification**: `alembic upgrade head` succeeds; entry point detection returns results

#### Phase B: Analysis Quality (Sprint 2)
- [ ] `core/understanding/analyzer.py`
- [ ] `core/understanding/prompts.py`
- [ ] `core/understanding/worker.py`
- [ ] `core/understanding/engine.py`
- [ ] API routes + `deps.py` + `app.py` registration
- [ ] Integration tests for full job lifecycle
- **Verification**: `POST /api/understanding/{id}/analyze` → poll → results with evidence refs

#### Phase C: Chat Integration (Sprint 3)
- [ ] Chat route narrative injection (`code_chat.py`)
- [ ] Query intent extension (`retriever.py`)
- [ ] ASG expander intent parameter (`expander.py`)
- [ ] Integration tests for chat enrichment
- **Verification**: "How does X work?" query returns FUNCTIONAL NARRATIVE section

#### Phase D: Migration Coupling (Sprint 4)
- [ ] Context builder `get_deep_analysis_context()` method
- [ ] Phase 3/5/6 deep context injection
- [ ] `_call_llm()` LLM override support
- [ ] Configuration schema (`codeloom.yaml`)
- [ ] Settings endpoints
- [ ] Integration tests for migration context
- **Verification**: Transform phase with deep analysis → deep context bundle in prompt

### 10.6 Verification Checklist

| # | Test | Command/Action | Expected |
|---|------|---------------|----------|
| 1 | Schema | `alembic upgrade head` | 3 new tables + 1 new column created |
| 2 | Entry Points | `GET /api/understanding/{id}/entry-points` | Returns classified entry points |
| 3 | Analysis | `POST /api/understanding/{id}/analyze` → poll | Results with business_rules + evidence refs |
| 4 | Chat | Ask "how does X work?" | FUNCTIONAL NARRATIVE section in context |
| 5 | Migration | Run transform phase with deep analysis | Deep context bundle injected into prompt |
| 6 | LLM Override | Set `generation_llm` in config → run transform | Correct LLM used (verify in logs) |
| 7 | Backward Compat | Chat + migration on project WITHOUT analysis | Identical to current behavior |
| 8 | Tests | `pytest codeloom/tests/test_understanding/` | All pass |

---

## Appendix A: Utilities Reused

| Utility | Source | Used For |
|---------|--------|----------|
| `TokenCounter` | `core/code_chunker/token_counter.py:9-17` | Token counting for tier selection |
| `_traverse()` CTE pattern | `core/asg_builder/queries.py:436-524` | Extended for call tree CTE |
| `RAPTORWorker` pattern | `core/raptor/worker.py:42-359` | Worker architecture (daemon thread, asyncio, Queue, Semaphore) |
| `MigrationEngine` pattern | `core/migration/engine.py:62-100` | Orchestrator class structure |
| `get_migration_engine()` | `api/deps.py:67-75` | Dependency injection pattern |
| `DatabaseManager.get_session()` | `core/db/db.py` | Thread-safe DB access |
| `ASGBuilder` ON CONFLICT | `core/asg_builder/builder.py` | Idempotent edge/analysis upserts |

## Appendix B: Critical File Reference

| File | Line(s) | Role in This Design |
|------|---------|-------------------|
| `core/db/models.py` | Full | Extend with 3 new models + 1 column |
| `core/asg_builder/queries.py` | 436-524 | Pattern for recursive CTE (`_traverse()`) |
| `core/asg_builder/expander.py` | 31-50 | Modify for intent-aware expansion |
| `core/raptor/worker.py` | 42-359 | Pattern for background worker architecture |
| `core/migration/engine.py` | 62-100 | Pattern for orchestrator class |
| `core/migration/phases.py` | 130-138 | Modify `_call_llm()` for LLM overrides |
| `core/migration/context_builder.py` | 28-93 | Add `get_deep_analysis_context()` |
| `core/engine/retriever.py` | 47-53 | Extend `QueryIntent` enum |
| `core/code_chunker/token_counter.py` | 9-17 | Reuse for bounded analysis |
| `api/routes/code_chat.py` | 126-146 | Insert narrative enrichment |
| `api/deps.py` | 67-75 | Add `get_understanding_engine()` |
| `api/app.py` | 65-78 | Register understanding router |
| `config/codeloom.yaml` | 362+ | Add migration config sections |
