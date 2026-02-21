"""Build ASG-enriched context for each migration phase.

Each phase needs different levels of detail from the codebase:
- Overview phases (1-3): summarized metrics, module-level relationships
- Detail phases (4-6): full signatures, source code, call sites

Uses the existing ASG query functions and SQLAlchemy models to
assemble context strings within a token budget.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text, func

from ..db import DatabaseManager
from ..db.models import CodeUnit, CodeEdge, CodeFile, Project, MigrationPhase

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for budget management
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 12_000
ANALYSIS_SOURCE_BUDGET = 6_000  # Token budget for source code in analysis phases


class MigrationContextBuilder:
    """Builds rich context for each migration phase by combining ASG data,
    source code, and previous phase outputs.

    Args:
        db: DatabaseManager instance
        project_id: Source project UUID string
    """

    def __init__(self, db: DatabaseManager, project_id: str):
        self._db = db
        self._project_id = project_id
        self._pid = UUID(project_id) if isinstance(project_id, str) else project_id

    # ── Public API ─────────────────────────────────────────────────────

    def build_phase_context(
        self,
        phase_number: int,
        previous_outputs: Dict[int, str],
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        mvp_context: Optional[Dict[str, Any]] = None,
        context_type: Optional[str] = None,
    ) -> str:
        """Build context string for a specific phase.

        Args:
            phase_number: 1-6 (V1) or 1-4 (V2)
            previous_outputs: {phase_number: output_text} for completed phases
            token_budget: Max tokens for the context block
            mvp_context: MVP dict for per-MVP phases. Contains unit_ids,
                file_ids, sp_references, metrics, name, etc.
            context_type: Semantic type override (e.g. "architecture", "discovery").
                Decouples phase number from context builder for V2 pipeline.
                If None, falls back to phase-number dispatch.

        Returns:
            Formatted context string for the LLM prompt
        """
        _type_builders = {
            "discovery": self._build_phase_1_context,
            "architecture": self._build_phase_2_context,
            "analyze": self._build_phase_3_context,
            "design": self._build_phase_4_context,
            "transform": self._build_phase_5_context,
            "test": self._build_phase_6_context,
        }

        if context_type:
            builder = _type_builders.get(context_type)
            if not builder:
                raise ValueError(f"Invalid context_type: {context_type}")
        else:
            _number_builders = {
                1: self._build_phase_1_context,
                2: self._build_phase_2_context,
                3: self._build_phase_3_context,
                4: self._build_phase_4_context,
                5: self._build_phase_5_context,
                6: self._build_phase_6_context,
            }
            builder = _number_builders.get(phase_number)
            if not builder:
                raise ValueError(f"Invalid phase number: {phase_number}")

        return builder(previous_outputs, token_budget, mvp_context)

    # ── Phase 1: Approach ──────────────────────────────────────────────
    # Needs: overview metrics, edge stats, coupling analysis, language distribution

    def _build_phase_1_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        # Project overview
        overview = self._get_project_overview()
        sections.append(self._format_project_overview(overview))

        # Edge statistics (coupling metrics)
        edge_stats = self._get_edge_stats()
        sections.append(self._format_edge_stats(edge_stats))

        # Module complexity (units per file, nesting)
        module_stats = self._get_module_stats()
        sections.append(self._format_module_stats(module_stats))

        # Top coupled units (most incoming edges)
        hot_spots = self._get_hot_spots(limit=15)
        sections.append(self._format_hot_spots(hot_spots))

        return self._join_within_budget(sections, budget)

    # ── Phase 2: Architecture ──────────────────────────────────────────
    # Needs: full dependency graph (module-level), import chains, top-level signatures

    def _build_phase_2_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        overview = self._get_project_overview()
        sections.append(self._format_project_overview(overview))

        # Module-level dependency graph
        dep_graph = self._get_module_dependency_graph()
        sections.append(self._format_dependency_graph(dep_graph))

        # Top-level unit signatures (classes, modules)
        top_units = self._get_top_level_units()
        sections.append(self._format_unit_signatures(top_units, "Top-Level Units"))

        # Class hierarchy
        hierarchy = self._get_class_hierarchy_summary()
        sections.append(self._format_class_hierarchy(hierarchy))

        # Interface contracts
        iface_contracts = self._get_interface_contracts()
        sections.append(self._format_interface_contracts(iface_contracts))

        # Source framework pattern analysis (for technical migration blueprint)
        try:
            patterns = self.get_source_patterns()
            formatted_patterns = self.format_source_patterns(patterns)
            if formatted_patterns.strip():
                sections.append(formatted_patterns)
        except Exception as e:
            logger.warning("Source pattern analysis failed: %s", e)

        return self._join_within_budget(sections, budget)

    # ── Phase 3: Analyze (Per-MVP) ──────────────────────────────────────
    # Needs: MVP-scoped unit analysis, cross-boundary edges, SP references

    def _build_phase_3_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        overview = self._get_project_overview()
        sections.append(self._format_project_overview(overview))

        if mvp_context and mvp_context.get("unit_ids"):
            unit_ids = mvp_context["unit_ids"]
            file_ids = mvp_context.get("file_ids", [])

            # Units in this MVP with enriched metadata (signatures, params, return types)
            mvp_units = self._get_mvp_units_enriched(unit_ids, limit=60)
            sections.append(self._format_unit_details_enriched(mvp_units))

            # Edges crossing MVP boundary (blast radius)
            cross_edges = self._get_mvp_cross_edges(unit_ids, limit=50)
            sections.append(self._format_mvp_cross_edges(cross_edges))

            # SP references for this MVP
            sp_refs = mvp_context.get("sp_references", [])
            if sp_refs:
                sections.append(self._format_sp_references(sp_refs))

            # MVP functional context (business rules, data entities, integrations, validation)
            try:
                functional = self.get_mvp_functional_context(unit_ids)
                formatted_func = self.format_mvp_functional_context(functional)
                if formatted_func.strip():
                    sections.append(formatted_func)
            except Exception as e:
                logger.warning("MVP functional context extraction failed: %s", e)

            # Source code (connectivity-ordered, budget-safe — truncated first by _join_within_budget)
            try:
                source_code = self._get_mvp_source_code_by_connectivity(
                    unit_ids, budget=ANALYSIS_SOURCE_BUDGET
                )
                if source_code:
                    sections.append(self._format_source_code_annotated(source_code))
            except Exception as e:
                logger.warning("MVP source code extraction failed: %s", e)

            # Call paths for integration context
            try:
                call_paths = self._get_mvp_call_paths(unit_ids, limit=20)
                if call_paths:
                    sections.append(self._format_call_paths(call_paths))
            except Exception as e:
                logger.warning("MVP call paths extraction failed: %s", e)
        else:
            # Fallback: project-wide leaf/blast analysis
            leaves = self._get_leaf_units(limit=30)
            sections.append(self._format_leaf_units(leaves))

            blast = self._get_blast_radius_by_file(limit=20)
            sections.append(self._format_blast_radius(blast))

        return self._join_within_budget(sections, budget)

    # ── Phase 4: Design (Per-MVP) ───────────────────────────────────────
    # Needs: full signatures + docstrings for MVP units, their edges, SP refs

    def _build_phase_4_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        if mvp_context and mvp_context.get("unit_ids"):
            unit_ids = mvp_context["unit_ids"]

            # Enriched signatures for MVP units
            units = self._get_mvp_units_enriched(unit_ids, limit=80)
            sections.append(self._format_unit_details_enriched(units))

            # Edges involving these units
            edge_summary = self._get_edge_summary_for_units(
                [u["unit_id"] for u in units]
            )
            sections.append(self._format_edge_summary(edge_summary))

            # SP references with call site detail
            sp_refs = mvp_context.get("sp_references", [])
            if sp_refs:
                sections.append(self._format_sp_references(sp_refs))

            # MVP functional context for design traceability
            try:
                functional = self.get_mvp_functional_context(unit_ids)
                formatted_func = self.format_mvp_functional_context(functional)
                if formatted_func.strip():
                    sections.append(formatted_func)
            except Exception as e:
                logger.warning("MVP functional context extraction failed: %s", e)
        else:
            # Fallback: project-wide
            units = self._get_units_with_signatures(limit=80)
            sections.append(self._format_unit_details_enriched(units))

            edge_summary = self._get_edge_summary_for_units(
                [u["unit_id"] for u in units]
            )
            sections.append(self._format_edge_summary(edge_summary))

        return self._join_within_budget(sections, budget)

    # ── Phase 5: Transform (Per-MVP) ────────────────────────────────────
    # Needs: full source code of MVP units, SP call sites

    def _build_phase_5_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        if mvp_context and mvp_context.get("unit_ids"):
            unit_ids = mvp_context["unit_ids"]

            # Full source code for MVP units only
            sources = self._get_mvp_source_code(unit_ids, budget)
            sections.append(self._format_source_code_annotated(sources))

            # SP references with call sites for stub implementation
            sp_refs = mvp_context.get("sp_references", [])
            if sp_refs:
                sections.append(self._format_sp_references(sp_refs))
        else:
            sources = self._get_source_code(budget)
            sections.append(self._format_source_code_annotated(sources))

        return self._join_within_budget(sections, budget)

    # ── Phase 6: Test (Per-MVP) ─────────────────────────────────────────
    # Needs: MVP call paths, integration points, SP refs for stub tests

    def _build_phase_6_context(
        self, previous_outputs: Dict[int, str], budget: int,
        mvp_context: Optional[Dict] = None,
    ) -> str:
        sections = []

        if mvp_context and mvp_context.get("unit_ids"):
            unit_ids = mvp_context["unit_ids"]

            # Call paths involving MVP units
            call_paths = self._get_mvp_call_paths(unit_ids, limit=30)
            sections.append(self._format_call_paths(call_paths))

            # Integration points between MVP and external code
            integration = self._get_mvp_integration_points(unit_ids)
            sections.append(self._format_integration_points(integration))

            # SP references for stub test generation
            sp_refs = mvp_context.get("sp_references", [])
            if sp_refs:
                sections.append(self._format_sp_references(sp_refs))
        else:
            call_paths = self._get_call_paths(depth=3, limit=30)
            sections.append(self._format_call_paths(call_paths))

            integration = self._get_integration_points()
            sections.append(self._format_integration_points(integration))

        return self._join_within_budget(sections, budget)

    # ── Source Pattern Analysis (Phase 2 — Technical) ────────────────

    def get_source_patterns(self) -> Dict[str, Any]:
        """Analyze source codebase for framework patterns using existing ASG data.

        Detects annotations, DI style, data layer, web layer, config, and test
        frameworks by querying CodeUnit metadata and signatures.
        """
        return {
            "annotations": self._extract_annotations(),
            "di_pattern": self._detect_di_pattern(),
            "data_layer": self._detect_data_layer(),
            "web_layer": self._detect_web_layer(),
            "config_pattern": self._detect_config_pattern(),
            "test_framework": self._detect_test_framework(),
        }

    def _extract_annotations(self) -> Dict[str, int]:
        """Count annotation/decorator occurrences across all code units."""
        with self._db.get_session() as session:
            # Query units that have annotations or decorators in metadata
            rows = session.execute(text("""
                SELECT metadata AS unit_metadata
                FROM code_units
                WHERE project_id = :pid
                  AND metadata IS NOT NULL
                  AND (metadata ? 'annotations' OR metadata ? 'decorators'
                       OR metadata ? 'modifiers')
            """), {"pid": self._pid}).fetchall()

        counts: Dict[str, int] = {}
        for row in rows:
            meta = row.unit_metadata if hasattr(row, 'unit_metadata') else (row[0] if row else {})
            if not isinstance(meta, dict):
                continue
            for key in ("annotations", "decorators", "modifiers"):
                items = meta.get(key) or []
                if isinstance(items, list):
                    for item in items:
                        name = item if isinstance(item, str) else str(item)
                        counts[name] = counts.get(name, 0) + 1
        return counts

    def _detect_di_pattern(self) -> str:
        """Detect dependency injection pattern from annotations and signatures."""
        annotations = self._extract_annotations()

        # Java/Spring field injection
        autowired_count = annotations.get("@Autowired", 0) + annotations.get("Autowired", 0)
        inject_count = annotations.get("@Inject", 0) + annotations.get("Inject", 0)

        # Check for constructor injection via signatures
        constructor_injection = 0
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT COUNT(*) as cnt
                FROM code_units
                WHERE project_id = :pid
                  AND unit_type = 'constructor'
                  AND signature IS NOT NULL
                  AND LENGTH(signature) > 20
            """), {"pid": self._pid}).scalar() or 0
            constructor_injection = result

        if autowired_count > 0 and autowired_count > constructor_injection:
            return "field_injection"
        elif inject_count > 0:
            return "inject_annotation"
        elif constructor_injection > 3:
            return "constructor_injection"

        # Python/Flask/FastAPI patterns
        if annotations.get("@app.route", 0) > 0 or annotations.get("app.route", 0) > 0:
            return "flask_di"

        return "unknown"

    def _detect_data_layer(self) -> str:
        """Detect data access layer pattern."""
        annotations = self._extract_annotations()

        # JPA / Hibernate
        if annotations.get("@Entity", 0) > 0 or annotations.get("Entity", 0) > 0:
            if annotations.get("@Repository", 0) > 0:
                return "spring_data_jpa"
            return "jpa"

        # Check for ORM patterns in signatures/names
        with self._db.get_session() as session:
            orm_count = session.execute(text("""
                SELECT COUNT(*) FROM code_units
                WHERE project_id = :pid
                  AND (name ILIKE '%Repository%' OR name ILIKE '%Dao%'
                       OR qualified_name ILIKE '%.models.%'
                       OR qualified_name ILIKE '%.entities.%')
            """), {"pid": self._pid}).scalar() or 0

        if orm_count > 3:
            # Check for specific ORMs
            if annotations.get("@Table", 0) > 0:
                return "jpa"
            return "orm_repository_pattern"

        # SQLAlchemy / Django ORM patterns
        with self._db.get_session() as session:
            sa_count = session.execute(text("""
                SELECT COUNT(*) FROM code_units
                WHERE project_id = :pid
                  AND (signature ILIKE '%Column(%' OR signature ILIKE '%relationship(%'
                       OR signature ILIKE '%models.Model%')
            """), {"pid": self._pid}).scalar() or 0

        if sa_count > 0:
            return "sqlalchemy_or_django_orm"

        return "unknown"

    def _detect_web_layer(self) -> str:
        """Detect web framework pattern."""
        annotations = self._extract_annotations()

        if annotations.get("@RestController", 0) > 0 or annotations.get("RestController", 0) > 0:
            return "spring_mvc"
        if annotations.get("@Controller", 0) > 0:
            return "spring_mvc"
        if annotations.get("@GetMapping", 0) > 0 or annotations.get("@PostMapping", 0) > 0:
            return "spring_mvc"

        # Express/Koa patterns
        with self._db.get_session() as session:
            express_count = session.execute(text("""
                SELECT COUNT(*) FROM code_units
                WHERE project_id = :pid
                  AND (signature ILIKE '%router.get%' OR signature ILIKE '%router.post%'
                       OR signature ILIKE '%app.get(%' OR signature ILIKE '%app.post(%')
            """), {"pid": self._pid}).scalar() or 0

        if express_count > 0:
            return "express_or_koa"

        # FastAPI / Flask
        if annotations.get("@app.route", 0) > 0:
            return "flask"
        if annotations.get("@router.get", 0) > 0 or annotations.get("@router.post", 0) > 0:
            return "fastapi"

        return "unknown"

    def _detect_config_pattern(self) -> str:
        """Detect configuration pattern."""
        annotations = self._extract_annotations()

        if annotations.get("@Value", 0) > 0 or annotations.get("@Configuration", 0) > 0:
            return "spring_properties"
        if annotations.get("@ConfigurationProperties", 0) > 0:
            return "spring_config_properties"

        # Check for config file references in source
        with self._db.get_session() as session:
            config_count = session.execute(text("""
                SELECT COUNT(*) FROM code_files
                WHERE project_id = :pid
                  AND (file_path ILIKE '%application.properties%'
                       OR file_path ILIKE '%application.yml%'
                       OR file_path ILIKE '%.env%'
                       OR file_path ILIKE '%config.yaml%'
                       OR file_path ILIKE '%settings.py%'
                       OR file_path ILIKE '%appsettings.json%')
            """), {"pid": self._pid}).scalar() or 0

        if config_count > 0:
            return "file_based_config"

        return "unknown"

    def _detect_test_framework(self) -> str:
        """Detect test framework from test files and annotations."""
        annotations = self._extract_annotations()

        if annotations.get("@Test", 0) > 0:
            if annotations.get("@SpringBootTest", 0) > 0:
                return "junit_spring"
            return "junit"

        with self._db.get_session() as session:
            test_files = session.execute(text("""
                SELECT file_path FROM code_files
                WHERE project_id = :pid
                  AND (file_path ILIKE '%test%' OR file_path ILIKE '%spec%')
                LIMIT 5
            """), {"pid": self._pid}).fetchall()

        paths = [r.file_path for r in test_files]
        if any("pytest" in p or "test_" in p for p in paths):
            return "pytest"
        if any(".spec." in p or ".test." in p for p in paths):
            return "jest_or_mocha"
        if any("Test.java" in p or "Tests.java" in p for p in paths):
            return "junit"
        if any("Test.cs" in p or "Tests.cs" in p for p in paths):
            return "xunit_or_nunit"

        return "unknown"

    def format_source_patterns(self, patterns: Dict[str, Any]) -> str:
        """Format source patterns into a markdown section for the LLM prompt."""
        if not patterns or all(v in (None, "unknown", {}) for v in patterns.values()):
            return "## Source Framework Patterns\nNo framework patterns detected."

        lines = ["## Source Framework Patterns"]

        # Annotations
        annotations = patterns.get("annotations", {})
        if annotations:
            total = sum(annotations.values())
            lines.append(f"\n### Annotations/Decorators ({total} occurrences)")
            sorted_anns = sorted(annotations.items(), key=lambda x: x[1], reverse=True)
            groups = []
            for ann, count in sorted_anns[:20]:
                groups.append(f"{ann} ({count})")
            lines.append("- " + ", ".join(groups))

        # DI
        di = patterns.get("di_pattern", "unknown")
        if di != "unknown":
            di_labels = {
                "field_injection": "Field Injection via @Autowired",
                "inject_annotation": "Injection via @Inject",
                "constructor_injection": "Constructor Injection",
                "flask_di": "Flask/module-level injection",
            }
            lines.append(f"\n### DI Pattern: {di_labels.get(di, di)}")

        # Data layer
        data = patterns.get("data_layer", "unknown")
        if data != "unknown":
            data_labels = {
                "spring_data_jpa": "Spring Data JPA (Repository interfaces + @Entity)",
                "jpa": "JPA / Hibernate (@Entity annotations)",
                "orm_repository_pattern": "ORM with Repository Pattern",
                "sqlalchemy_or_django_orm": "SQLAlchemy or Django ORM",
            }
            lines.append(f"\n### Data Layer: {data_labels.get(data, data)}")

        # Web layer
        web = patterns.get("web_layer", "unknown")
        if web != "unknown":
            web_labels = {
                "spring_mvc": "Spring MVC (@RestController / @Controller)",
                "express_or_koa": "Express.js or Koa",
                "flask": "Flask (@app.route)",
                "fastapi": "FastAPI (@router)",
            }
            lines.append(f"\n### Web Layer: {web_labels.get(web, web)}")

        # Config
        config = patterns.get("config_pattern", "unknown")
        if config != "unknown":
            config_labels = {
                "spring_properties": "Spring @Value / @Configuration",
                "spring_config_properties": "Spring @ConfigurationProperties",
                "file_based_config": "File-based configuration",
            }
            lines.append(f"\n### Configuration: {config_labels.get(config, config)}")

        # Test framework
        test = patterns.get("test_framework", "unknown")
        if test != "unknown":
            test_labels = {
                "junit": "JUnit",
                "junit_spring": "JUnit + Spring Boot Test",
                "pytest": "pytest",
                "jest_or_mocha": "Jest or Mocha",
                "xunit_or_nunit": "xUnit or NUnit",
            }
            lines.append(f"\n### Test Framework: {test_labels.get(test, test)}")

        return "\n".join(lines)

    # ── Deep Understanding Context ────────────────────────────────

    def get_deep_analysis_context(
        self,
        unit_ids: List[str],
        max_narratives: int = 5,
    ) -> str:
        """Build context from deep analysis results overlapping with the given units.

        Queries analysis_units for overlap with unit_ids, ranks by overlap count
        and depth, and composes a DEEP UNDERSTANDING section with narratives,
        business rules, and integration summaries.

        Args:
            unit_ids: List of code unit UUIDs in the current MVP
            max_narratives: Maximum number of analyses to include

        Returns:
            Formatted markdown context string (empty if no analyses found)
        """
        if not unit_ids:
            return ""

        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]

        # Query overlapping analyses ranked by relevance
        placeholders = ", ".join(f":uid{i}" for i in range(len(uids)))
        params: Dict[str, Any] = {"pid": self._pid, "limit": max_narratives}
        for i, uid in enumerate(uids):
            params[f"uid{i}"] = uid

        sql = f"""
            SELECT
                a.analysis_id,
                a.entry_point_name,
                a.narrative,
                a.confidence_score,
                a.coverage_pct,
                a.result_json,
                ov.overlap_units,
                ov.best_depth,
                ov.overlap_paths
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

        with self._db.get_session() as session:
            rows = session.execute(text(sql), params).fetchall()

        if not rows:
            return ""

        # Build context section
        sections = ["## DEEP UNDERSTANDING (from Deep Analysis Engine)"]

        # Coverage summary
        total_mvp_units = len(unit_ids)
        covered_unit_ids = set()
        for row in rows:
            result_json = row.result_json if hasattr(row, "result_json") else {}
            if isinstance(result_json, dict):
                for br in result_json.get("business_rules", []):
                    for ref in (br.get("evidence_refs") or br.get("evidence") or []):
                        uid = ref.get("unit_id")
                        if uid:
                            covered_unit_ids.add(uid)

        overlap_count = len(covered_unit_ids & set(str(u) for u in uids))
        coverage_pct = (overlap_count / total_mvp_units * 100) if total_mvp_units else 0

        # Check coverage thresholds
        warn_below = 50.0
        try:
            from ..config.config_loader import get_deep_analysis_coverage_config
            cov_cfg = get_deep_analysis_coverage_config()
            warn_below = cov_cfg.get("warn_below", 50.0)
        except Exception:
            pass

        coverage_note = ""
        if coverage_pct < warn_below:
            coverage_note = f" (WARNING: below {warn_below}% threshold)"

        sections.append(
            f"\n**Deep Analysis Coverage**: {overlap_count}/{total_mvp_units} "
            f"MVP units covered ({coverage_pct:.0f}%){coverage_note}"
        )

        # Add narratives and business rules from each analysis
        for row in rows:
            ep_name = row.entry_point_name if hasattr(row, "entry_point_name") else "Unknown"
            narrative = row.narrative if hasattr(row, "narrative") else ""
            confidence = row.confidence_score if hasattr(row, "confidence_score") else 0
            overlap = row.overlap_units if hasattr(row, "overlap_units") else 0

            sections.append(f"\n### Entry Point: {ep_name}")
            sections.append(
                f"Confidence: {confidence:.0%} | "
                f"Overlap with MVP: {overlap} units"
            )

            if narrative:
                sections.append(f"\n{narrative}")

            # Include business rules summary if available
            result_json = row.result_json if hasattr(row, "result_json") else {}
            if isinstance(result_json, dict):
                rules = result_json.get("business_rules", [])
                if rules:
                    sections.append(f"\n**Business Rules** ({len(rules)}):")
                    for rule in rules[:10]:
                        rule_id = rule.get("id", "?")
                        desc = rule.get("description") or rule.get("name") or ""
                        sections.append(f"- [{rule_id}] {desc[:200]}")

                integrations = result_json.get("integrations", [])
                if integrations:
                    sections.append(f"\n**Integrations** ({len(integrations)}):")
                    for integ in integrations[:5]:
                        name = integ.get("name") or integ.get("type") or ""
                        desc = integ.get("description") or ""
                        sections.append(f"- {name}: {desc[:150]}")

        return "\n".join(sections)

    # ── MVP Functional Context (Phases 3-4) ────────────────────────

    def get_mvp_functional_context(self, unit_ids: List[str]) -> Dict[str, Any]:
        """Extract business domain context scoped to a specific MVP's units.

        Returns data entities, business rules (service methods), external
        integrations, and validation logic for the given MVP unit set.
        """
        if not unit_ids:
            return {}
        return {
            "data_entities": self._get_mvp_data_entities(unit_ids),
            "business_rules": self._get_mvp_business_methods(unit_ids),
            "integrations": self._get_mvp_external_integrations(unit_ids),
            "validation_logic": self._get_mvp_validation_units(unit_ids),
        }

    def _get_mvp_data_entities(self, unit_ids: List[str]) -> List[Dict]:
        """Get entity/model classes within this MVP's units."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cu.docstring,
                    cu.metadata,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                  AND cu.unit_type IN ('class', 'interface', 'struct')
                  AND (
                    cu.metadata::text ILIKE '%Entity%'
                    OR cu.metadata::text ILIKE '%Table%'
                    OR cu.metadata::text ILIKE '%model%'
                    OR cu.name ILIKE '%Entity%'
                    OR cu.name ILIKE '%Model%'
                    OR cu.qualified_name ILIKE '%.models.%'
                    OR cu.qualified_name ILIKE '%.entities.%'
                    OR cu.qualified_name ILIKE '%.entity.%'
                  )
                ORDER BY cu.name
            """), {"uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_business_methods(self, unit_ids: List[str]) -> List[Dict]:
        """Get service-layer methods that represent business rules."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cu.docstring,
                    cu.start_line,
                    cu.end_line,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                  AND cu.unit_type IN ('method', 'function')
                  AND (
                    cu.qualified_name ILIKE '%Service%'
                    OR cu.qualified_name ILIKE '%Manager%'
                    OR cu.qualified_name ILIKE '%Handler%'
                    OR cu.qualified_name ILIKE '%UseCase%'
                    OR cu.qualified_name ILIKE '%Interactor%'
                    OR cu.metadata::text ILIKE '%Service%'
                  )
                ORDER BY cu.qualified_name
            """), {"uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_external_integrations(self, unit_ids: List[str]) -> List[Dict]:
        """Get units in this MVP with HTTP client, message queue, or external API patterns."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cu.docstring,
                    cu.metadata,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                  AND (
                    cu.metadata::text ILIKE '%FeignClient%'
                    OR cu.metadata::text ILIKE '%RestTemplate%'
                    OR cu.metadata::text ILIKE '%HttpClient%'
                    OR cu.metadata::text ILIKE '%WebClient%'
                    OR cu.metadata::text ILIKE '%KafkaListener%'
                    OR cu.metadata::text ILIKE '%RabbitListener%'
                    OR cu.name ILIKE '%Client%'
                    OR cu.name ILIKE '%Gateway%'
                    OR cu.name ILIKE '%Adapter%'
                    OR cu.name ILIKE '%Proxy%'
                    OR cu.signature ILIKE '%HttpClient%'
                    OR cu.signature ILIKE '%RestTemplate%'
                    OR cu.signature ILIKE '%fetch(%'
                    OR cu.signature ILIKE '%axios%'
                  )
                ORDER BY cu.name
            """), {"uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_validation_units(self, unit_ids: List[str]) -> List[Dict]:
        """Get validation-related units in this MVP."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cu.docstring,
                    cu.metadata,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                  AND (
                    cu.metadata::text ILIKE '%Valid%'
                    OR cu.metadata::text ILIKE '%NotNull%'
                    OR cu.metadata::text ILIKE '%NotBlank%'
                    OR cu.metadata::text ILIKE '%Size%'
                    OR cu.metadata::text ILIKE '%Pattern%'
                    OR cu.name ILIKE '%Validator%'
                    OR cu.name ILIKE '%Validation%'
                    OR cu.qualified_name ILIKE '%validation%'
                  )
                ORDER BY cu.name
            """), {"uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def format_mvp_functional_context(self, functional: Dict[str, Any]) -> str:
        """Format MVP functional context into a markdown section."""
        if not functional:
            return "## MVP Functional Context\nNo functional context extracted."

        lines = ["## MVP Functional Context"]

        # Data entities
        entities = functional.get("data_entities", [])
        if entities:
            lines.append("\n### Data Entities")
            for e in entities:
                meta = e.get("metadata") or {}
                params = meta.get("parsed_params", [])
                fields = ", ".join(p.get("name", "?") for p in params) if params else ""
                doc = (e.get("docstring") or "")[:300]
                lines.append(f"- **{e['name']}** ({e['file_path']})")
                if fields:
                    lines.append(f"  Fields: {fields}")
                if doc:
                    lines.append(f"  {doc}")
        else:
            lines.append("\n### Data Entities\nNone detected in this MVP.")

        # Business methods
        methods = functional.get("business_rules", [])
        if methods:
            lines.append("\n### Business Methods (Service Layer)")
            for m in methods:
                sig = m.get("signature") or m.get("name", "")
                doc = (m.get("docstring") or "")[:500]
                loc = f"{m.get('file_path', '?')}:{m.get('start_line', '?')}"
                lines.append(f"- **{m['qualified_name']}** — `{sig}`")
                lines.append(f"  Location: {loc}")
                if doc:
                    lines.append(f"  Purpose: {doc}")
        else:
            lines.append("\n### Business Methods\nNo service-layer methods detected.")

        # Integrations
        integrations = functional.get("integrations", [])
        if integrations:
            lines.append("\n### External Integrations")
            for i in integrations:
                meta = i.get("metadata") or {}
                annotations = meta.get("annotations", []) or meta.get("decorators", []) or []
                ann_str = ", ".join(str(a) for a in annotations[:3]) if annotations else ""
                lines.append(f"- **{i['name']}** ({i['unit_type']}) — {i['file_path']}")
                if ann_str:
                    lines.append(f"  Annotations: {ann_str}")
                if i.get("docstring"):
                    lines.append(f"  {i['docstring'][:300]}")
        else:
            lines.append("\n### External Integrations\nNone detected in this MVP.")

        # Validation
        validation = functional.get("validation_logic", [])
        if validation:
            lines.append("\n### Validation Rules")
            for v in validation:
                meta = v.get("metadata") or {}
                annotations = meta.get("annotations", []) or meta.get("decorators", []) or []
                ann_str = ", ".join(str(a) for a in annotations[:5]) if annotations else ""
                lines.append(f"- **{v['qualified_name']}** ({v['unit_type']})")
                if ann_str:
                    lines.append(f"  Constraints: {ann_str}")
        else:
            lines.append("\n### Validation Rules\nNone detected in this MVP.")

        return "\n".join(lines)

    # ── Data queries ───────────────────────────────────────────────────

    def _get_project_overview(self) -> Dict[str, Any]:
        with self._db.get_session() as session:
            project = session.query(Project).filter(
                Project.project_id == self._pid
            ).first()
            if not project:
                return {}

            file_count = session.query(func.count(CodeFile.file_id)).filter(
                CodeFile.project_id == self._pid
            ).scalar() or 0

            unit_count = session.query(func.count(CodeUnit.unit_id)).filter(
                CodeUnit.project_id == self._pid
            ).scalar() or 0

            edge_count = session.query(func.count(CodeEdge.id)).filter(
                CodeEdge.project_id == self._pid
            ).scalar() or 0

            # Language distribution
            lang_dist = session.execute(text("""
                SELECT language, COUNT(*) as cnt
                FROM code_files
                WHERE project_id = :pid AND language IS NOT NULL
                GROUP BY language ORDER BY cnt DESC
            """), {"pid": self._pid}).fetchall()

            # Unit type distribution
            type_dist = session.execute(text("""
                SELECT unit_type, COUNT(*) as cnt
                FROM code_units
                WHERE project_id = :pid
                GROUP BY unit_type ORDER BY cnt DESC
            """), {"pid": self._pid}).fetchall()

        return {
            "name": project.name,
            "description": project.description or "",
            "primary_language": project.primary_language,
            "file_count": file_count,
            "unit_count": unit_count,
            "edge_count": edge_count,
            "total_lines": project.total_lines or 0,
            "languages": {r.language: r.cnt for r in lang_dist},
            "unit_types": {r.unit_type: r.cnt for r in type_dist},
        }

    def _get_edge_stats(self) -> Dict[str, int]:
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT edge_type, COUNT(*) AS cnt
                FROM code_edges
                WHERE project_id = :pid
                GROUP BY edge_type ORDER BY cnt DESC
            """), {"pid": self._pid})
            return {r.edge_type: r.cnt for r in result.fetchall()}

    def _get_module_stats(self) -> List[Dict]:
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cf.file_path,
                    cf.language,
                    cf.line_count,
                    COUNT(cu.unit_id) AS unit_count
                FROM code_files cf
                LEFT JOIN code_units cu ON cf.file_id = cu.file_id
                WHERE cf.project_id = :pid
                GROUP BY cf.file_id, cf.file_path, cf.language, cf.line_count
                ORDER BY unit_count DESC
                LIMIT 30
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_hot_spots(self, limit: int = 15) -> List[Dict]:
        """Units with the most incoming edges (highest coupling)."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    COUNT(ce.id) AS incoming_count
                FROM code_units cu
                JOIN code_edges ce ON ce.target_unit_id = cu.unit_id
                WHERE cu.project_id = :pid
                GROUP BY cu.unit_id, cu.name, cu.qualified_name, cu.unit_type, cu.language
                ORDER BY incoming_count DESC
                LIMIT :lim
            """), {"pid": self._pid, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_module_dependency_graph(self) -> List[Dict]:
        """File-level dependency graph (aggregated from unit edges)."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    sf.file_path AS source_file,
                    tf.file_path AS target_file,
                    ce.edge_type,
                    COUNT(*) AS edge_count
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                JOIN code_files sf ON su.file_id = sf.file_id
                JOIN code_files tf ON tu.file_id = tf.file_id
                WHERE ce.project_id = :pid
                  AND sf.file_id != tf.file_id
                GROUP BY sf.file_path, tf.file_path, ce.edge_type
                ORDER BY edge_count DESC
                LIMIT 100
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_top_level_units(self) -> List[Dict]:
        """Top-level units: classes, modules, interfaces."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.unit_id,
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.signature,
                    cu.docstring,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.project_id = :pid
                  AND cu.unit_type IN ('class', 'module', 'interface', 'enum')
                ORDER BY cu.qualified_name
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_class_hierarchy_summary(self) -> List[Dict]:
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    su.name AS child,
                    su.qualified_name AS child_qn,
                    tu.name AS parent,
                    tu.qualified_name AS parent_qn,
                    ce.edge_type
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                WHERE ce.project_id = :pid AND ce.edge_type IN ('inherits', 'implements')
                ORDER BY ce.edge_type, tu.name, su.name
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_interface_contracts(self) -> List[Dict]:
        """Get interface -> implementor relationships for architecture context."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    tu.name AS interface_name,
                    tu.qualified_name AS interface_qn,
                    su.name AS implementor_name,
                    su.qualified_name AS implementor_qn,
                    su.unit_type AS implementor_type
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                WHERE ce.project_id = :pid AND ce.edge_type = 'implements'
                ORDER BY tu.name, su.name
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_leaf_units(self, limit: int = 30) -> List[Dict]:
        """Units with zero or few incoming edges — safest to migrate first."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cf.file_path,
                    COALESCE(inc.cnt, 0) AS incoming_count,
                    COALESCE(out.cnt, 0) AS outgoing_count
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.project_id = :pid
                  AND cu.unit_type NOT IN ('module')
                ORDER BY incoming_count ASC, outgoing_count DESC
                LIMIT :lim
            """), {"pid": self._pid, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_blast_radius_by_file(self, limit: int = 20) -> List[Dict]:
        """Files ranked by total incoming edges to their units (blast radius)."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cf.file_path,
                    cf.language,
                    COUNT(DISTINCT ce.id) AS total_dependents,
                    COUNT(DISTINCT cu.unit_id) AS unit_count
                FROM code_files cf
                JOIN code_units cu ON cf.file_id = cu.file_id
                LEFT JOIN code_edges ce ON ce.target_unit_id = cu.unit_id
                WHERE cf.project_id = :pid
                GROUP BY cf.file_id, cf.file_path, cf.language
                ORDER BY total_dependents DESC
                LIMIT :lim
            """), {"pid": self._pid, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_units_with_signatures(self, limit: int = 80) -> List[Dict]:
        """Units with signatures, docstrings, and enriched metadata, ordered by connectivity."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.unit_id,
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.signature,
                    cu.docstring,
                    cu.metadata,
                    cf.file_path,
                    (
                        COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)
                    ) AS connectivity
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.project_id = :pid
                ORDER BY connectivity DESC
                LIMIT :lim
            """), {"pid": self._pid, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_edge_summary_for_units(self, unit_ids: List[str]) -> List[Dict]:
        """Get edge summary for a set of units."""
        if not unit_ids:
            return []
        with self._db.get_session() as session:
            # Use a simple approach — query edges for these units
            uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids[:50]]
            result = session.execute(text("""
                SELECT
                    su.qualified_name AS source,
                    tu.qualified_name AS target,
                    ce.edge_type
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                WHERE ce.project_id = :pid
                  AND (ce.source_unit_id = ANY(:uids) OR ce.target_unit_id = ANY(:uids))
                ORDER BY ce.edge_type, su.qualified_name
                LIMIT 200
            """), {"pid": self._pid, "uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_source_code(self, budget: int) -> List[Dict]:
        """Get source code for units, prioritized by connectivity, within budget."""
        char_budget = budget * CHARS_PER_TOKEN
        used = 0
        results = []

        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.source,
                    cu.signature,
                    cf.file_path,
                    (
                        COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)
                    ) AS connectivity
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.project_id = :pid AND cu.source IS NOT NULL
                ORDER BY connectivity DESC
            """), {"pid": self._pid}).fetchall()

            for row in rows:
                source = row.source or ""
                entry_len = len(source) + len(row.qualified_name or "") + 50
                if used + entry_len > char_budget:
                    break
                used += entry_len
                results.append(dict(row._mapping))

        return results

    def _get_call_paths(self, depth: int = 3, limit: int = 30) -> List[Dict]:
        """Get call chain starting points for integration testing context."""
        with self._db.get_session() as session:
            # Find entry points: units that are called but call many others
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cf.file_path,
                    COALESCE(out.cnt, 0) AS calls_made,
                    COALESCE(inc.cnt, 0) AS called_by
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges
                    WHERE project_id = :pid AND edge_type = 'calls'
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges
                    WHERE project_id = :pid AND edge_type = 'calls'
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                WHERE cu.project_id = :pid
                  AND COALESCE(out.cnt, 0) > 0
                ORDER BY (COALESCE(out.cnt, 0) + COALESCE(inc.cnt, 0)) DESC
                LIMIT :lim
            """), {"pid": self._pid, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_integration_points(self) -> List[Dict]:
        """Get units that serve as integration boundaries (high fan-in + fan-out)."""
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cf.file_path,
                    COALESCE(inc.cnt, 0) AS fan_in,
                    COALESCE(out.cnt, 0) AS fan_out
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.project_id = :pid
                  AND (COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)) >= 3
                ORDER BY (COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)) DESC
                LIMIT 25
            """), {"pid": self._pid})
            return [dict(r._mapping) for r in result.fetchall()]

    # ── MVP-scoped queries ────────────────────────────────────────────

    def _get_mvp_units(self, unit_ids: List[str], limit: int = 60) -> List[Dict]:
        """Get units belonging to this MVP with basic signatures."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids[:limit]]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.unit_id,
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.signature,
                    cu.docstring,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids)
                ORDER BY cu.qualified_name
            """), {"uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_units_enriched(self, unit_ids: List[str], limit: int = 80) -> List[Dict]:
        """Get MVP units with enriched metadata (parsed_params, return_type)."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids[:limit]]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.unit_id,
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.signature,
                    cu.docstring,
                    cu.metadata,
                    cf.file_path,
                    (
                        COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)
                    ) AS connectivity
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.unit_id = ANY(:uids)
                ORDER BY connectivity DESC
            """), {"pid": self._pid, "uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_cross_edges(self, unit_ids: List[str], limit: int = 50) -> List[Dict]:
        """Get edges crossing the MVP boundary (one end inside, one outside)."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    su.qualified_name AS source,
                    tu.qualified_name AS target,
                    ce.edge_type,
                    CASE
                        WHEN ce.source_unit_id = ANY(:uids) AND ce.target_unit_id = ANY(:uids) THEN 'internal'
                        WHEN ce.source_unit_id = ANY(:uids) THEN 'outbound'
                        ELSE 'inbound'
                    END AS direction
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                JOIN code_units tu ON ce.target_unit_id = tu.unit_id
                WHERE ce.project_id = :pid
                  AND (ce.source_unit_id = ANY(:uids) OR ce.target_unit_id = ANY(:uids))
                ORDER BY direction, ce.edge_type, su.qualified_name
                LIMIT :lim
            """), {"pid": self._pid, "uids": uids, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_source_code(self, unit_ids: List[str], budget: int) -> List[Dict]:
        """Get source code for MVP units only, within budget."""
        if not unit_ids:
            return []
        char_budget = budget * CHARS_PER_TOKEN
        used = 0
        results = []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]

        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.source,
                    cu.signature,
                    cf.file_path
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cu.unit_id = ANY(:uids) AND cu.source IS NOT NULL
                ORDER BY cu.qualified_name
            """), {"uids": uids}).fetchall()

            for row in rows:
                source = row.source or ""
                entry_len = len(source) + len(row.qualified_name or "") + 50
                if used + entry_len > char_budget:
                    break
                used += entry_len
                results.append(dict(row._mapping))

        return results

    def _get_mvp_source_code_by_connectivity(self, unit_ids: List[str], budget: int) -> List[Dict]:
        """Get source code for MVP units ordered by connectivity (fan-in + fan-out).

        Prioritizes the most connected units so the LLM sees the most
        architecturally significant code first.
        """
        if not unit_ids:
            return []
        char_budget = budget * CHARS_PER_TOKEN
        used = 0
        results = []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]

        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.language,
                    cu.source,
                    cu.signature,
                    cf.file_path,
                    (
                        COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)
                    ) AS connectivity
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.unit_id = ANY(:uids) AND cu.source IS NOT NULL
                ORDER BY connectivity DESC
            """), {"pid": self._pid, "uids": uids}).fetchall()

            for row in rows:
                source = row.source or ""
                entry_len = len(source) + len(row.qualified_name or "") + 50
                if used + entry_len > char_budget:
                    break
                used += entry_len
                results.append(dict(row._mapping))

        return results

    def get_inter_mvp_edges(self, clusters: List[Dict]) -> List[Dict]:
        """Get edge counts between MVP clusters for coherence evaluation.

        Builds unit_id -> cluster_idx mapping, queries cross-cluster edges,
        returns aggregated edge counts by cluster pair and edge type.
        """
        # Build unit_id -> cluster_idx mapping
        uid_to_cluster: Dict[str, int] = {}
        all_uids = []
        for idx, c in enumerate(clusters):
            for uid in (c.get("unit_ids") or []):
                uid_to_cluster[uid] = idx
                all_uids.append(uid)

        if not all_uids:
            return []

        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in all_uids]

        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT
                    ce.source_unit_id,
                    ce.target_unit_id,
                    ce.edge_type
                FROM code_edges ce
                WHERE ce.project_id = :pid
                  AND ce.source_unit_id = ANY(:uids)
                  AND ce.target_unit_id = ANY(:uids)
            """), {"pid": self._pid, "uids": uids}).fetchall()

        # Aggregate by cluster pair
        from collections import defaultdict
        pair_edges: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            src_uid = str(row.source_unit_id)
            tgt_uid = str(row.target_unit_id)
            src_cluster = uid_to_cluster.get(src_uid)
            tgt_cluster = uid_to_cluster.get(tgt_uid)
            if src_cluster is not None and tgt_cluster is not None and src_cluster != tgt_cluster:
                key = (src_cluster, tgt_cluster)
                pair_edges[key][row.edge_type] += 1

        result = []
        for (src_mvp, tgt_mvp), edge_types in pair_edges.items():
            result.append({
                "source_mvp": src_mvp,
                "target_mvp": tgt_mvp,
                "edge_count": sum(edge_types.values()),
                "edge_types": dict(edge_types),
            })

        return sorted(result, key=lambda x: x["edge_count"], reverse=True)

    def _get_mvp_call_paths(self, unit_ids: List[str], limit: int = 30) -> List[Dict]:
        """Get call path entry points involving MVP units."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cf.file_path,
                    COALESCE(out.cnt, 0) AS calls_made,
                    COALESCE(inc.cnt, 0) AS called_by
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges
                    WHERE project_id = :pid AND edge_type = 'calls'
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges
                    WHERE project_id = :pid AND edge_type = 'calls'
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                WHERE cu.unit_id = ANY(:uids)
                  AND COALESCE(out.cnt, 0) > 0
                ORDER BY (COALESCE(out.cnt, 0) + COALESCE(inc.cnt, 0)) DESC
                LIMIT :lim
            """), {"pid": self._pid, "uids": uids, "lim": limit})
            return [dict(r._mapping) for r in result.fetchall()]

    def _get_mvp_integration_points(self, unit_ids: List[str]) -> List[Dict]:
        """Get units at MVP boundary with high fan-in/fan-out."""
        if not unit_ids:
            return []
        uids = [UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT
                    cu.name,
                    cu.qualified_name,
                    cu.unit_type,
                    cu.signature,
                    cf.file_path,
                    COALESCE(inc.cnt, 0) AS fan_in,
                    COALESCE(out.cnt, 0) AS fan_out
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                LEFT JOIN (
                    SELECT target_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY target_unit_id
                ) inc ON inc.target_unit_id = cu.unit_id
                LEFT JOIN (
                    SELECT source_unit_id, COUNT(*) AS cnt
                    FROM code_edges WHERE project_id = :pid
                    GROUP BY source_unit_id
                ) out ON out.source_unit_id = cu.unit_id
                WHERE cu.unit_id = ANY(:uids)
                  AND (COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)) >= 2
                ORDER BY (COALESCE(inc.cnt, 0) + COALESCE(out.cnt, 0)) DESC
                LIMIT 25
            """), {"pid": self._pid, "uids": uids})
            return [dict(r._mapping) for r in result.fetchall()]

    # ── Formatters ─────────────────────────────────────────────────────

    def _format_project_overview(self, data: Dict) -> str:
        if not data:
            return "## Project Overview\nNo project data available."
        langs = ", ".join(f"{k} ({v} files)" for k, v in data.get("languages", {}).items())
        types = ", ".join(f"{k}: {v}" for k, v in data.get("unit_types", {}).items())
        return f"""## Project Overview
- **Name**: {data.get('name', 'Unknown')}
- **Primary Language**: {data.get('primary_language', 'Unknown')}
- **Files**: {data.get('file_count', 0)}
- **Code Units**: {data.get('unit_count', 0)}
- **ASG Edges**: {data.get('edge_count', 0)}
- **Total Lines**: {data.get('total_lines', 0)}
- **Languages**: {langs}
- **Unit Types**: {types}"""

    def _format_edge_stats(self, stats: Dict[str, int]) -> str:
        if not stats:
            return "## Edge Statistics\nNo edges found."
        lines = ["## Edge Statistics (Coupling Metrics)"]
        total = sum(stats.values())
        lines.append(f"Total edges: {total}")
        for edge_type, count in stats.items():
            pct = (count / total * 100) if total else 0
            lines.append(f"- {edge_type}: {count} ({pct:.1f}%)")
        return "\n".join(lines)

    def _format_module_stats(self, stats: List[Dict]) -> str:
        if not stats:
            return "## Module Complexity\nNo module data."
        lines = ["## Module Complexity (Top Files by Unit Count)"]
        lines.append("| File | Language | Lines | Units |")
        lines.append("|------|----------|-------|-------|")
        for s in stats[:20]:
            lines.append(f"| {s['file_path']} | {s['language']} | {s['line_count'] or 0} | {s['unit_count']} |")
        return "\n".join(lines)

    def _format_hot_spots(self, spots: List[Dict]) -> str:
        if not spots:
            return "## Hot Spots\nNo high-coupling units found."
        lines = ["## Hot Spots (Most Referenced Units)"]
        lines.append("| Unit | Type | Language | Incoming Edges |")
        lines.append("|------|------|----------|----------------|")
        for s in spots:
            lines.append(f"| {s['qualified_name']} | {s['unit_type']} | {s['language']} | {s['incoming_count']} |")
        return "\n".join(lines)

    def _format_dependency_graph(self, deps: List[Dict]) -> str:
        if not deps:
            return "## Module Dependency Graph\nNo cross-file dependencies found."
        lines = ["## Module Dependency Graph (File-Level)"]
        lines.append("| Source File | Target File | Edge Type | Count |")
        lines.append("|------------|-------------|-----------|-------|")
        for d in deps[:50]:
            lines.append(f"| {d['source_file']} | {d['target_file']} | {d['edge_type']} | {d['edge_count']} |")
        return "\n".join(lines)

    def _format_unit_signatures(self, units: List[Dict], title: str) -> str:
        if not units:
            return f"## {title}\nNo units found."
        lines = [f"## {title}"]
        for u in units:
            sig = u.get("signature", "") or u.get("name", "")
            doc = u.get("docstring", "")
            doc_line = f"  # {doc[:100]}..." if doc and len(doc) > 100 else f"  # {doc}" if doc else ""
            lines.append(f"- [{u['unit_type']}] {u['file_path']}: {sig}{doc_line}")
        return "\n".join(lines)

    def _format_class_hierarchy(self, hierarchy: List[Dict]) -> str:
        if not hierarchy:
            return "## Class Hierarchy\nNo inheritance relationships found."
        lines = ["## Class Hierarchy"]
        for h in hierarchy:
            edge_type = h.get("edge_type", "inherits")
            verb = "implements" if edge_type == "implements" else "extends"
            lines.append(f"- {h['child_qn']} {verb} {h['parent_qn']}")
        return "\n".join(lines)

    def _format_interface_contracts(self, contracts: List[Dict]) -> str:
        if not contracts:
            return "## Interface Contracts\nNo interface implementations found."
        lines = ["## Interface Contracts"]
        # Group by interface
        by_iface: Dict[str, List[str]] = {}
        for c in contracts:
            iface = c["interface_qn"]
            by_iface.setdefault(iface, []).append(
                f"{c['implementor_qn']} ({c['implementor_type']})"
            )
        for iface, implementors in by_iface.items():
            lines.append(f"\n**{iface}**:")
            for impl in implementors:
                lines.append(f"  - {impl}")
        return "\n".join(lines)

    def _format_leaf_units(self, leaves: List[Dict]) -> str:
        if not leaves:
            return "## Leaf Units (Low Blast Radius)\nNo leaf units found."
        lines = ["## Leaf Units (Low Blast Radius — Safe to Migrate First)"]
        lines.append("| Unit | Type | File | Incoming | Outgoing |")
        lines.append("|------|------|------|----------|----------|")
        for u in leaves:
            lines.append(
                f"| {u['qualified_name']} | {u['unit_type']} | {u['file_path']} | "
                f"{u['incoming_count']} | {u['outgoing_count']} |"
            )
        return "\n".join(lines)

    def _format_blast_radius(self, blast: List[Dict]) -> str:
        if not blast:
            return "## Blast Radius by File\nNo dependency data."
        lines = ["## Blast Radius by File"]
        lines.append("| File | Language | Units | Total Dependents |")
        lines.append("|------|----------|-------|-----------------|")
        for b in blast:
            lines.append(f"| {b['file_path']} | {b['language']} | {b['unit_count']} | {b['total_dependents']} |")
        return "\n".join(lines)

    def _format_unit_details(self, units: List[Dict]) -> str:
        if not units:
            return "## Unit Signatures & Contracts\nNo unit data."
        lines = ["## Unit Signatures & Contracts (Ordered by Connectivity)"]
        for u in units:
            sig = u.get("signature") or u.get("name", "")
            doc = u.get("docstring") or ""
            lines.append(f"\n### {u.get('qualified_name', u.get('name', ''))}")
            lines.append(f"- **Type**: {u['unit_type']} | **Language**: {u['language']}")
            lines.append(f"- **File**: {u['file_path']}")
            lines.append(f"- **Signature**: `{sig}`")
            if doc:
                lines.append(f"- **Docstring**: {doc[:300]}")
        return "\n".join(lines)

    def _format_unit_details_enriched(self, units: List[Dict]) -> str:
        """Format unit details using enriched metadata (parsed_params, return_type)."""
        if not units:
            return "## Unit Signatures & Contracts\nNo unit data."
        lines = ["## Unit Signatures & Contracts (Ordered by Connectivity)"]
        for u in units:
            sig = u.get("signature") or u.get("name", "")
            doc = u.get("docstring") or ""
            lines.append(f"\n### {u.get('qualified_name', u.get('name', ''))}")
            lines.append(f"- **Type**: {u['unit_type']} | **Language**: {u['language']}")
            lines.append(f"- **File**: {u['file_path']}")
            lines.append(f"- **Signature**: `{sig}`")
            if doc:
                lines.append(f"- **Docstring**: {doc[:300]}")

            # Use enriched metadata if available
            meta = u.get("metadata") or {}
            parsed_params = meta.get("parsed_params")
            if parsed_params:
                param_strs = []
                for p in parsed_params:
                    pstr = p.get("name", "?")
                    if p.get("type"):
                        pstr += f": {p['type']}"
                    if p.get("default"):
                        pstr += f" = {p['default']}"
                    param_strs.append(pstr)
                lines.append(f"- **Parameters**: ({', '.join(param_strs)})")

            ret_type = meta.get("return_type")
            if ret_type:
                lines.append(f"- **Returns**: `{ret_type}`")

            modifiers = meta.get("modifiers")
            if modifiers:
                lines.append(f"- **Modifiers**: {', '.join(modifiers)}")

        return "\n".join(lines)

    def _format_source_code_annotated(self, sources: List[Dict]) -> str:
        """Format source code with override/implements annotations."""
        if not sources:
            return "## Source Code\nNo source code available."

        # Build set of override unit names for annotation
        override_units = set()
        with self._db.get_session() as session:
            result = session.execute(text("""
                SELECT su.qualified_name
                FROM code_edges ce
                JOIN code_units su ON ce.source_unit_id = su.unit_id
                WHERE ce.project_id = :pid AND ce.edge_type = 'overrides'
            """), {"pid": self._pid})
            override_units = {r.qualified_name for r in result.fetchall()}

        lines = ["## Source Code (Ordered by Connectivity)"]
        for s in sources:
            qn = s.get('qualified_name', s.get('name', ''))
            override_marker = " [OVERRIDES]" if qn in override_units else ""
            lines.append(f"\n### {qn}{override_marker}")
            lines.append(f"File: {s['file_path']} | Type: {s['unit_type']}")
            lines.append(f"```{s.get('language', '')}")
            lines.append(s.get("source", "# source not available"))
            lines.append("```")
        return "\n".join(lines)

    def _format_edge_summary(self, edges: List[Dict]) -> str:
        if not edges:
            return "## Relationships\nNo edges for selected units."
        lines = ["## Relationships Between Units"]
        for e in edges:
            lines.append(f"- {e['source']} --[{e['edge_type']}]--> {e['target']}")
        return "\n".join(lines)

    def _format_source_code(self, sources: List[Dict]) -> str:
        if not sources:
            return "## Source Code\nNo source code available."
        lines = ["## Source Code (Ordered by Connectivity)"]
        for s in sources:
            lines.append(f"\n### {s.get('qualified_name', s.get('name', ''))}")
            lines.append(f"File: {s['file_path']} | Type: {s['unit_type']}")
            lines.append(f"```{s.get('language', '')}")
            lines.append(s.get("source", "# source not available"))
            lines.append("```")
        return "\n".join(lines)

    def _format_call_paths(self, paths: List[Dict]) -> str:
        if not paths:
            return "## Call Path Entry Points\nNo call paths found."
        lines = ["## Call Path Entry Points (Integration Boundaries)"]
        lines.append("| Unit | Type | File | Calls Made | Called By |")
        lines.append("|------|------|------|-----------|----------|")
        for p in paths:
            sig = p.get("signature") or p.get("name", "")
            lines.append(
                f"| {p['qualified_name']} | {p['unit_type']} | {p['file_path']} | "
                f"{p['calls_made']} | {p['called_by']} |"
            )
        return "\n".join(lines)

    def _format_integration_points(self, points: List[Dict]) -> str:
        if not points:
            return "## Integration Points\nNo integration points identified."
        lines = ["## Integration Points (High Fan-In + Fan-Out)"]
        for p in points:
            sig = p.get("signature") or p.get("name", "")
            lines.append(f"- **{p['qualified_name']}** ({p['unit_type']}) — fan_in={p['fan_in']}, fan_out={p['fan_out']}")
            lines.append(f"  File: {p['file_path']} | Signature: `{sig}`")
        return "\n".join(lines)

    # ── MVP Formatters ──────────────────────────────────────────────────

    def _format_mvp_cross_edges(self, edges: List[Dict]) -> str:
        """Format edges crossing the MVP boundary."""
        if not edges:
            return "## MVP Boundary Edges\nNo cross-boundary edges found."
        lines = ["## MVP Boundary Edges (Blast Radius)"]

        inbound = [e for e in edges if e.get("direction") == "inbound"]
        outbound = [e for e in edges if e.get("direction") == "outbound"]
        internal = [e for e in edges if e.get("direction") == "internal"]

        if internal:
            lines.append(f"\n**Internal edges**: {len(internal)}")
            for e in internal[:10]:
                lines.append(f"  - {e['source']} --[{e['edge_type']}]--> {e['target']}")

        if outbound:
            lines.append(f"\n**Outbound edges** (MVP depends on external): {len(outbound)}")
            for e in outbound[:15]:
                lines.append(f"  - {e['source']} --[{e['edge_type']}]--> {e['target']}")

        if inbound:
            lines.append(f"\n**Inbound edges** (external depends on MVP): {len(inbound)}")
            for e in inbound[:15]:
                lines.append(f"  - {e['source']} --[{e['edge_type']}]--> {e['target']}")

        return "\n".join(lines)

    def _format_sp_references(self, sp_refs: List[Dict]) -> str:
        """Format stored procedure references for LLM context."""
        if not sp_refs:
            return "## Stored Procedure References\nNo SP references."
        lines = ["## Stored Procedure References"]
        for ref in sp_refs:
            sp_name = ref.get("sp_name", "Unknown")
            call_sites = ref.get("call_sites", [])
            lines.append(f"\n### SP: {sp_name}")
            lines.append(f"- **Call sites**: {len(call_sites)}")
            for site in call_sites[:10]:
                caller = site.get("caller_name", "?")
                file_path = site.get("file_path", "?")
                line = site.get("line", "?")
                lines.append(f"  - `{caller}` in `{file_path}:{line}`")
        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _join_within_budget(self, sections: List[str], token_budget: int) -> str:
        """Join sections, truncating if needed to stay within token budget."""
        char_budget = token_budget * CHARS_PER_TOKEN
        result = []
        used = 0
        for section in sections:
            if used + len(section) > char_budget:
                remaining = char_budget - used
                if remaining > 200:
                    result.append(section[:remaining] + "\n\n[... truncated for token budget]")
                break
            result.append(section)
            used += len(section) + 2  # +2 for newlines
        return "\n\n".join(result)
