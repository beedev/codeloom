"""Migration Engine — MVP-centric migration pipeline orchestrator.

Supports two pipeline versions:

V1 (6-phase, legacy):
  Plan-level: Phase 1 Discovery, Phase 2 Architecture
  Per-MVP:    Phase 3 Analyze, Phase 4 Design, Phase 5 Transform, Phase 6 Test

V2 (4-phase, default for new plans):
  Plan-level: Phase 1 Architecture, Phase 2 Discovery
  Per-MVP:    Phase 3 Transform, Phase 4 Test
  On-demand:  analyze_mvp() merges old Analyze+Design into FunctionalMVP.analysis_output

Each phase has a human approval gate. Pipeline version is set on MigrationPlan.pipeline_version.
"""

import io
import logging
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from ..db import DatabaseManager
from ..db.models import CodeFile, CodeUnit, FunctionalMVP, MigrationPlan, MigrationPhase, Project
from .context_builder import MigrationContextBuilder
from .doc_enricher import DocEnricher
from .mvp_clusterer import MvpClusterer, _MAX_CLUSTER_SIZE
from .phases import execute_phase, execute_mvp_analysis, get_phase_type, _describe_mvp, _evaluate_mvp_coherence

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:40]


# V1: 6-phase pipeline (Discovery -> Architecture -> Analyze -> Design -> Transform -> Test)
PLAN_PHASES_V1 = (1, 2)
MVP_PHASES_V1 = (3, 4, 5, 6)

# V2: 4-phase pipeline (Architecture -> Discovery -> Transform -> Test)
PLAN_PHASES_V2 = (1, 2)
MVP_PHASES_V2 = (3, 4)

# Backward compat defaults (V1)
PLAN_PHASES = PLAN_PHASES_V1
MVP_PHASES = MVP_PHASES_V1


def _plan_phases(v: int) -> tuple:
    return PLAN_PHASES_V2 if v == 2 else PLAN_PHASES_V1


def _mvp_phases(v: int) -> tuple:
    return MVP_PHASES_V2 if v == 2 else MVP_PHASES_V1


class MigrationEngine:
    """Orchestrate the MVP-centric migration pipeline.

    Args:
        db_manager: DatabaseManager instance
        pipeline: LocalRAGPipeline (for LLM access via Settings.llm)
    """

    def __init__(self, db_manager: DatabaseManager, pipeline: Any = None):
        self._db = db_manager
        self._pipeline = pipeline
        self._clusterer = MvpClusterer(db_manager)

    # ── Plan Lifecycle ─────────────────────────────────────────────────

    def create_plan(
        self,
        user_id: str,
        source_project_id: str,
        target_brief: str,
        target_stack: Dict,
        constraints: Optional[Dict] = None,
        migration_type: str = "framework_migration",
    ) -> Dict:
        """Create a new migration plan with 2 plan-level phases.

        V2 pipeline (default): Architecture (1) then Discovery (2).
        Per-MVP phases are created after the second plan-level phase is approved.

        Args:
            migration_type: One of "version_upgrade", "framework_migration", "rewrite".

        Returns:
            Plan dict with plan_id and phase summaries.
        """
        valid_types = {"version_upgrade", "framework_migration", "rewrite"}
        if migration_type not in valid_types:
            migration_type = "framework_migration"

        plan_id = uuid4()
        uid = UUID(user_id) if isinstance(user_id, str) else user_id
        pid = UUID(source_project_id) if isinstance(source_project_id, str) else source_project_id
        version = 2  # New plans always use V2 pipeline

        with self._db.get_session() as session:
            plan = MigrationPlan(
                plan_id=plan_id,
                user_id=uid,
                source_project_id=pid,
                target_brief=target_brief,
                target_stack=target_stack,
                constraints=constraints or {},
                status="draft",
                current_phase=0,
                migration_type=migration_type,
                pipeline_version=version,
            )
            session.add(plan)

            # Create plan-level phases with version-aware types
            for n in _plan_phases(version):
                phase = MigrationPhase(
                    phase_id=uuid4(),
                    plan_id=plan_id,
                    phase_number=n,
                    phase_type=get_phase_type(n, version),
                    status="pending",
                    mvp_id=None,
                )
                session.add(phase)

        logger.info(f"Created migration plan {plan_id} for project {source_project_id}")
        return self.get_plan_status(str(plan_id))

    # ── Asset Inventory ───────────────────────────────────────────────

    def get_asset_inventory(self, plan_id: str) -> Dict:
        """Return file-type breakdown with rule-based strategy suggestions.

        Queries code_files and code_units grouped by language, then applies
        deterministic strategy rules based on migration_type and target_stack.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")

            project_id = plan.source_project_id
            migration_type = plan.migration_type or "framework_migration"
            target_stack = plan.target_stack or {}

            # Aggregate file stats by language
            from sqlalchemy import func
            file_stats = (
                session.query(
                    CodeFile.language,
                    func.count(CodeFile.file_id).label("file_count"),
                    func.sum(CodeFile.line_count).label("total_lines"),
                )
                .filter(CodeFile.project_id == project_id)
                .group_by(CodeFile.language)
                .all()
            )

            # Aggregate unit counts by language
            unit_stats = dict(
                session.query(
                    CodeUnit.language,
                    func.count(CodeUnit.unit_id),
                )
                .filter(CodeUnit.project_id == project_id)
                .group_by(CodeUnit.language)
                .all()
            )

            # Gather sample paths per language (up to 3)
            sample_paths: Dict[str, List[str]] = {}
            for lang_row in file_stats:
                lang = lang_row.language
                if not lang:
                    continue
                samples = (
                    session.query(CodeFile.file_path)
                    .filter(CodeFile.project_id == project_id, CodeFile.language == lang)
                    .limit(3)
                    .all()
                )
                sample_paths[lang] = [s[0] for s in samples]

        # Build inventory items
        inventory = []
        source_languages = []
        for row in file_stats:
            lang = row.language
            if not lang:
                continue
            source_languages.append(lang)
            inventory.append({
                "language": lang,
                "file_count": row.file_count,
                "unit_count": unit_stats.get(lang, 0),
                "total_lines": int(row.total_lines or 0),
                "sample_paths": sample_paths.get(lang, []),
            })

        # Sort by unit count descending (primary language first)
        inventory.sort(key=lambda x: x["unit_count"], reverse=True)

        suggested = self._rule_based_strategies(
            migration_type, source_languages, target_stack
        )

        return {
            "inventory": inventory,
            "suggested_strategies": suggested,
            "llm_refined": False,
        }

    @staticmethod
    def _rule_based_strategies(
        migration_type: str,
        source_languages: List[str],
        target_stack: Dict,
    ) -> Dict[str, Dict]:
        """Pure-function: deterministic strategy suggestions per language.

        Returns {lang: {strategy, target, reason}}.
        """
        target_languages = [l.lower() for l in (target_stack.get("languages") or [])]
        target_frameworks = [f.lower() for f in (target_stack.get("frameworks") or [])]
        target_versions = target_stack.get("versions") or {}

        # Detect primary language — first target language found in source
        primary_lang = None
        for tl in target_languages:
            for sl in source_languages:
                if tl in sl.lower() or sl.lower() in tl:
                    primary_lang = sl
                    break
            if primary_lang:
                break
        if not primary_lang and source_languages:
            primary_lang = source_languages[0]

        config_types = {"xml", "json", "yaml", "yml", "toml", "properties", "ini", "conf"}
        data_types = {"sql", "graphql", "proto", "protobuf"}

        strategies: Dict[str, Dict] = {}
        for lang in source_languages:
            ll = lang.lower()

            if ll in config_types:
                if migration_type == "version_upgrade":
                    strategies[lang] = {"strategy": "no_change", "target": None, "reason": None}
                elif migration_type in ("framework_migration", "rewrite"):
                    strategies[lang] = {"strategy": "convert", "target": "YAML", "reason": None}
                else:
                    strategies[lang] = {"strategy": "no_change", "target": None, "reason": None}

            elif ll in data_types:
                if migration_type == "version_upgrade":
                    strategies[lang] = {"strategy": "no_change", "target": None, "reason": None}
                else:
                    strategies[lang] = {"strategy": "keep_as_is", "target": None, "reason": None}

            elif lang == primary_lang:
                # Primary language gets the plan's migration type
                target_label = None
                if migration_type == "version_upgrade":
                    target_label = target_versions.get(ll) or (
                        target_languages[0].title() if target_languages else None
                    )
                elif migration_type == "framework_migration":
                    target_label = (
                        target_frameworks[0].title() if target_frameworks else
                        target_languages[0].title() if target_languages else None
                    )
                elif migration_type == "rewrite":
                    target_label = target_languages[0].title() if target_languages else None

                strategies[lang] = {
                    "strategy": migration_type,
                    "target": target_label,
                    "reason": None,
                }

            else:
                # Non-primary languages
                if migration_type == "rewrite":
                    target_label = target_languages[0].title() if target_languages else None
                    strategies[lang] = {"strategy": "rewrite", "target": target_label, "reason": None}
                else:
                    strategies[lang] = {"strategy": "keep_as_is", "target": None, "reason": None}

        return strategies

    def refine_asset_strategies(self, plan_id: str) -> Dict:
        """LLM-refine the rule-based strategy suggestions.

        Calls the LLM with asset inventory context for nuanced overrides.
        Falls back to rule-based defaults on any failure.
        """
        from .prompts import ASSET_REFINEMENT_PROMPT

        base = self.get_asset_inventory(plan_id)
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")
            migration_type = plan.migration_type or "framework_migration"
            target_brief = plan.target_brief or ""
            target_stack = plan.target_stack or {}

        # Build inventory text for the prompt
        inv_lines = []
        for item in base["inventory"]:
            paths_str = ", ".join(item["sample_paths"][:3])
            inv_lines.append(
                f"- {item['language']}: {item['file_count']} files, "
                f"{item['unit_count']} units, {item['total_lines']} lines "
                f"(samples: {paths_str})"
            )

        import json
        prompt = ASSET_REFINEMENT_PROMPT.format(
            migration_type=migration_type,
            target_brief=target_brief,
            target_stack=json.dumps(target_stack),
            inventory_text="\n".join(inv_lines),
            rule_based_json=json.dumps(base["suggested_strategies"], indent=2),
        )

        try:
            from codeloom.setting import Settings
            llm = Settings.llm
            if not llm:
                logger.warning("No LLM available for asset refinement, using rule-based defaults")
                return base

            response = llm.complete(prompt)
            text = response.text.strip()

            # Extract JSON from potential markdown fences
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            refined = json.loads(text)

            # Merge refinements onto rule-based defaults
            strategies = base["suggested_strategies"]
            for lang, override in refined.items():
                if lang in strategies and isinstance(override, dict):
                    if "strategy" in override:
                        strategies[lang]["strategy"] = override["strategy"]
                    if "target" in override:
                        strategies[lang]["target"] = override["target"]
                    if "reason" in override:
                        strategies[lang]["reason"] = override["reason"]

            base["suggested_strategies"] = strategies
            base["llm_refined"] = True
            return base

        except Exception as exc:
            logger.warning("LLM asset refinement failed, using rule-based defaults: %s", exc)
            return base

    def save_asset_strategies(self, plan_id: str, strategies: Dict) -> Dict:
        """Persist user-confirmed asset strategies on the plan."""
        valid_strategies = {
            "version_upgrade", "framework_migration", "rewrite",
            "convert", "keep_as_is", "no_change",
        }
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        # Validate
        for lang, spec in strategies.items():
            if not isinstance(spec, dict) or "strategy" not in spec:
                raise ValueError(f"Invalid strategy spec for '{lang}': must have 'strategy' key")
            if spec["strategy"] not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{spec['strategy']}' for '{lang}'. "
                    f"Must be one of: {', '.join(sorted(valid_strategies))}"
                )

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")
            plan.asset_strategies = strategies

            # Derive target_stack from confirmed strategies so the user
            # doesn't have to type languages/frameworks manually.
            derived = self._derive_target_stack_from_strategies(
                strategies, plan.migration_type or "framework_migration",
            )
            existing = plan.target_stack or {}
            plan.target_stack = {**existing, **derived}

        # Generate Foundation MVP with LLM-identified prep activities
        try:
            self._generate_foundation_mvp(plan_id, strategies)
        except Exception:
            logger.warning("Foundation MVP generation failed", exc_info=True)

        return {"status": "saved", "plan_id": plan_id, "strategy_count": len(strategies)}

    def _generate_foundation_mvp(self, plan_id: str, strategies: Dict) -> None:
        """Generate Foundation MVP 0 with LLM-identified prerequisite activities.

        Called after asset strategies are saved — that's when we know the exact
        migration scope. Only strategies with active migration work (not
        no_change/keep_as_is) are included in the context.
        """
        import json

        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid,
            ).first()
            if not plan:
                return

            # Gather source context — use full languages list, not just primary
            project = session.query(Project).filter(
                Project.project_id == plan.source_project_id,
            ).first()
            source_langs = (
                (project.languages or [project.primary_language or "unknown"])
                if project else ["unknown"]
            )

            # Only include strategies with actual migration work
            active_strategies = {
                lang: spec for lang, spec in strategies.items()
                if spec.get("strategy") not in ("no_change", "keep_as_is")
            }
            if not active_strategies:
                logger.info("Foundation: no active strategies, skipping")
                return

            # Extract discovery metadata for richer context
            discovery = plan.discovery_metadata or {}
            shared_concerns = discovery.get("shared_concerns", [])
            sp_analysis = discovery.get("sp_analysis", {})
            total_mvps = discovery.get("total_mvps", 0)

            # Build strategy lines with from→to context
            strategy_lines = []
            for lang, spec in active_strategies.items():
                target = spec.get("target", "N/A")
                reason = spec.get("reason", "")
                line = f"  - {lang}: {spec['strategy']} -> {target}"
                if reason:
                    line += f" ({reason})"
                strategy_lines.append(line)

            # Build LLM prompt with full codebase context
            prompt = (
                "You are a senior migration architect. Given the migration plan below, "
                "produce a categorized list of **foundational prerequisite activities** "
                "that must be completed BEFORE any code migration begins.\n\n"
                "## Migration Context\n"
                f"- Type: {plan.migration_type or 'framework_migration'}\n"
                f"- Brief: {plan.target_brief}\n"
                f"- Source languages: {', '.join(source_langs)}\n"
                f"- Target stack: {json.dumps(plan.target_stack or {})}\n"
                f"- Total functional MVPs: {total_mvps}\n"
                "- Asset strategies:\n"
                + "\n".join(strategy_lines) + "\n"
            )

            if shared_concerns:
                concern_names = [
                    c.get("name", str(c)) if isinstance(c, dict) else str(c)
                    for c in shared_concerns[:10]
                ]
                prompt += f"\n- Shared/cross-cutting concerns discovered: {', '.join(concern_names)}\n"

            if sp_analysis.get("total_sps", 0) > 0:
                prompt += (
                    f"\n- Stored procedures: {sp_analysis['total_sps']} total, "
                    f"{sp_analysis.get('sps_with_callers', 0)} with callers, "
                    f"{sp_analysis.get('orphan_sps', 0)} orphan\n"
                )

            prompt += (
                "\n## Instructions\n"
                "- Group activities by category (e.g., Build System, Dependencies, "
                "Testing Infrastructure, Project Setup)\n"
                "- Be specific to the actual technologies listed above — do NOT invent "
                "technologies or frameworks not mentioned in the context\n"
                "- Each activity should be actionable and concrete\n"
                "- Exclude anything related to assets staying unchanged\n"
                "- Focus on PREREQUISITES only — not the migration work itself\n"
                "- Use markdown with ## headers for categories and bullet points for activities\n"
            )

            # Call LLM
            from codeloom.core.migration.phases import _call_llm
            description = _call_llm(prompt, context_type="generation", temperature=0.3)

            # Check if Foundation MVP already exists (idempotent on re-save)
            existing = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid,
                FunctionalMVP.name.ilike("MVP 0%Foundation%"),
            ).first()

            if existing:
                existing.description = description
                logger.info("Foundation MVP updated for plan %s", plan_id)
            else:
                # Priority 0 is reserved for Foundation; regular MVPs start at 1
                foundation = FunctionalMVP(
                    plan_id=pid,
                    name="MVP 0 \u2014 Foundation & Prerequisites",
                    description=description,
                    status="discovered",
                    priority=0,
                    file_ids=[],
                    unit_ids=[],
                    sp_references=[],
                    metrics={
                        "size": 0,
                        "cohesion": 1.0,
                        "coupling": 0.0,
                        "readiness": 1.0,
                        "complexity": "low",
                    },
                    current_phase=0,
                )
                session.add(foundation)
                logger.info("Foundation MVP created for plan %s", plan_id)

    @staticmethod
    def _derive_target_stack_from_strategies(
        strategies: Dict, migration_type: str,
    ) -> Dict:
        """Derive target_stack languages/frameworks from confirmed asset strategies.

        Extracts target labels from strategy entries:
        - rewrite targets → languages
        - framework_migration targets → frameworks
        - version_upgrade → keeps the source language name
        """
        languages: set = set()
        frameworks: set = set()
        for lang, info in strategies.items():
            target = info.get("target")
            strategy = info.get("strategy", "")
            if strategy == "rewrite" and target:
                languages.add(target.lower())
            elif strategy == "framework_migration" and target:
                frameworks.add(target.lower())
            elif strategy == "version_upgrade":
                languages.add(lang.lower())
        return {
            "languages": sorted(languages) if languages else [],
            "frameworks": sorted(frameworks) if frameworks else [],
        }

    # ── Discovery ──────────────────────────────────────────────────────

    def run_discovery(
        self,
        plan_id: str,
        clustering_params: Optional[Dict] = None,
    ) -> Dict:
        """Run MVP clustering + LLM Discovery phase.

        V1: Discovery is Phase 1.
        V2: Discovery is Phase 2 (Architecture runs first).

        Returns:
            Discovery result dict with mvps and sp_analysis.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")

            project_id = str(plan.source_project_id)
            version = plan.pipeline_version or 1
            asset_strategies = plan.asset_strategies  # None = include everything (backward compat)

        # Step 1: Run the clustering algorithm
        cluster_result = self._clusterer.cluster(
            project_id, clustering_params, asset_strategies=asset_strategies
        )

        # Step 1.5: Agentic MVP refinement — describe, evaluate, merge/split
        try:
            refined = self._refine_mvps_agentic(
                clusters=cluster_result["mvps"],
                project_id=project_id,
            )
            cluster_result["mvps"] = refined
        except Exception as exc:
            logger.warning(
                "Agentic MVP refinement failed, using algorithmic clusters: %s",
                exc,
            )

        # Step 1.75: Deduplicate MVPs (merge same-name, remove mega-clusters)
        cluster_result["mvps"] = self._deduplicate_mvps(cluster_result["mvps"])

        # Step 2: Persist MVP candidates
        mvp_rows = []
        index_to_mvp_id = {}  # cluster index -> DB mvp_id
        with self._db.get_session() as session:
            for i, mvp_data in enumerate(cluster_result["mvps"]):
                mvp = FunctionalMVP(
                    plan_id=pid,
                    name=mvp_data.get("name", f"MVP {i + 1}"),
                    description=mvp_data.get("description"),
                    status="discovered",
                    priority=mvp_data.get("priority", i),
                    file_ids=mvp_data.get("file_ids", []),
                    unit_ids=mvp_data.get("unit_ids", []),
                    sp_references=mvp_data.get("sp_references", []),
                    metrics=mvp_data.get("metrics", {}),
                    current_phase=0,
                )
                session.add(mvp)
                session.flush()  # Get mvp_id
                index_to_mvp_id[i] = mvp.mvp_id
                mvp_rows.append({
                    "mvp_id": mvp.mvp_id,
                    "name": mvp.name,
                    "priority": mvp.priority,
                    "metrics": mvp.metrics,
                    "unit_count": len(mvp.unit_ids) if mvp.unit_ids is not None else 0,
                    "sp_count": len(mvp.sp_references) if mvp.sp_references is not None else 0,
                })

            # Populate depends_on_mvp_ids using the cluster dependency info
            for i, mvp_data in enumerate(cluster_result["mvps"]):
                dep_indices = mvp_data.get("depends_on", [])
                if dep_indices:
                    dep_mvp_ids = [
                        index_to_mvp_id[idx]
                        for idx in dep_indices
                        if idx in index_to_mvp_id
                    ]
                    if dep_mvp_ids:
                        mvp = session.query(FunctionalMVP).filter(
                            FunctionalMVP.mvp_id == index_to_mvp_id[i]
                        ).first()
                        if mvp:
                            mvp.depends_on_mvp_ids = dep_mvp_ids

            # Store discovery metadata on the plan
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            plan.discovery_metadata = {
                "clustering_params": clustering_params or {},
                "total_mvps": len(mvp_rows),
                "shared_concerns": cluster_result.get("shared_concerns", []),
                "sp_analysis": cluster_result.get("sp_analysis", {}),
            }

        # Step 3: Execute the LLM Discovery phase
        # V1: Discovery is Phase 1. V2: Discovery is Phase 2.
        discovery_phase_number = 2 if version == 2 else 1
        phase_result = self.execute_phase(plan_id, discovery_phase_number)

        # Step 4: Auto-analyze all MVPs
        try:
            analysis_results = self._analyze_all_mvps(plan_id, mvp_rows)
            logger.info(
                "Auto-analysis complete: %d/%d MVPs analyzed",
                sum(1 for r in analysis_results.values() if r["status"] == "completed"),
                len(analysis_results),
            )
        except Exception as exc:
            logger.warning("Auto-analysis step failed: %s", exc)
            analysis_results = {}

        logger.info(f"Discovery complete for plan {plan_id}: {len(mvp_rows)} MVPs found")
        return {
            "phase_output": phase_result,
            "mvps": mvp_rows,
            "sp_analysis": cluster_result.get("sp_analysis", {}),
            "shared_concerns": cluster_result.get("shared_concerns", []),
            "auto_analysis": analysis_results,
        }

    def _analyze_all_mvps(self, plan_id: str, mvp_rows: List[Dict]) -> Dict:
        """Run deep analysis for all discovered MVPs.

        Called at end of discovery or via background task.
        Errors are logged per-MVP but do not fail the overall batch.
        Status is tracked on each FunctionalMVP.analysis_status.
        """
        results = {}
        total = len(mvp_rows)
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        # Mark all target MVPs as "analyzing" upfront
        with self._db.get_session() as session:
            for mvp_row in mvp_rows:
                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_row["mvp_id"],
                    FunctionalMVP.plan_id == pid,
                ).first()
                if mvp:
                    mvp.analysis_status = "analyzing"
                    mvp.analysis_error = None

        for i, mvp_row in enumerate(mvp_rows):
            mvp_id = mvp_row["mvp_id"]
            logger.info(
                "Auto-analyzing MVP %d/%d: %s (id=%s)",
                i + 1, total, mvp_row.get("name", "?"), mvp_id,
            )
            try:
                result = self.analyze_mvp(str(plan_id), mvp_id)
                results[mvp_id] = {"status": "completed", "analysis_at": result["analysis_at"]}
            except Exception as exc:
                logger.warning("Auto-analysis failed for MVP %s: %s", mvp_id, exc)
                results[mvp_id] = {"status": "failed", "error": str(exc)}
                # analyze_mvp already marks failed status, but ensure it
                with self._db.get_session() as session:
                    mvp = session.query(FunctionalMVP).filter(
                        FunctionalMVP.mvp_id == mvp_id,
                        FunctionalMVP.plan_id == pid,
                    ).first()
                    if mvp and mvp.analysis_status != "failed":
                        mvp.analysis_status = "failed"
                        mvp.analysis_error = str(exc)
        return results

    # ── Agentic MVP Refinement ──────────────────────────────────────────

    def _refine_mvps_agentic(
        self,
        clusters: List[Dict],
        project_id: str,
        max_iterations: int = 3,
    ) -> List[Dict]:
        """Agentic MVP refinement: per-MVP functional description + merge/split.

        Multi-pass LLM refinement loop:
          A) Describe each MVP functionally (name + description)
          B) Evaluate coherence across all MVPs
          C) Apply merge/split suggestions
          D) Repeat until stable or max_iterations reached

        All LLM calls use temperature=0 for deterministic output —
        same codebase should produce the same clusters.

        Falls back to original clusters on any critical failure.
        """
        ctx = MigrationContextBuilder(self._db, project_id)

        for iteration in range(max_iterations):
            # Pass A: Describe each MVP functionally
            for cluster in clusters:
                if not cluster.get("_described") or cluster.get("_needs_redescribe"):
                    try:
                        desc = _describe_mvp(ctx, cluster, token_budget=6_000)
                        cluster["name"] = desc.get("name", cluster.get("name", "Unknown"))
                        cluster["description"] = desc.get("description", "")
                        cluster["_functional_detail"] = desc
                        cluster["_described"] = True
                        cluster["_needs_redescribe"] = False
                    except Exception as e:
                        logger.warning(
                            "Agentic description failed for cluster %s: %s",
                            cluster.get("name", "?"), e,
                        )
                        cluster["_described"] = True
                        cluster["_needs_redescribe"] = False

            # Pass B: Evaluate coherence across all MVPs
            try:
                inter_edges = ctx.get_inter_mvp_edges(clusters)
                suggestions = _evaluate_mvp_coherence(
                    [
                        {
                            "idx": i,
                            "name": c.get("name", f"MVP {i}"),
                            "description": c.get("description", ""),
                            "metrics": c.get("metrics", {}),
                            "unit_count": len(c.get("unit_ids", [])),
                        }
                        for i, c in enumerate(clusters)
                    ],
                    inter_edges,
                )
            except Exception as e:
                logger.warning("Agentic coherence evaluation failed: %s", e)
                break

            # Pass C: Apply merge/split suggestions
            merges = suggestions.get("merge_suggestions", [])
            splits = suggestions.get("split_suggestions", [])

            if not merges and not splits:
                logger.info(
                    "Agentic MVP refinement converged at iteration %d", iteration
                )
                break

            try:
                clusters = self._apply_agentic_suggestions(clusters, merges, splits)
                # Mark affected clusters for re-description
                for c in clusters:
                    if c.get("_merged") or c.get("_split"):
                        c["_needs_redescribe"] = True
                        c.pop("_merged", None)
                        c.pop("_split", None)
            except Exception as e:
                logger.warning("Applying agentic suggestions failed: %s", e)
                break

        # Clean internal flags before returning
        for c in clusters:
            c.pop("_described", None)
            c.pop("_needs_redescribe", None)
            c.pop("_functional_detail", None)
            c.pop("_merged", None)
            c.pop("_split", None)

        return clusters

    def _apply_agentic_suggestions(
        self,
        clusters: List[Dict],
        merges: List[Dict],
        splits: List[Dict],
    ) -> List[Dict]:
        """Apply merge/split suggestions from the coherence evaluation.

        Operates on in-memory cluster dicts (not persisted).
        Reuses the merge pattern from merge_mvps() but on dicts.
        """
        result = list(clusters)

        # Apply merges (process in reverse to keep indices valid)
        # Guard: reject merges that would create oversized MVPs (>200 units)
        _MERGE_MAX_UNITS = 200
        merged_indices = set()
        for merge in merges:
            ids = merge.get("ids", [])
            if len(ids) < 2:
                continue
            # Validate indices
            valid_ids = [i for i in ids if 0 <= i < len(result) and i not in merged_indices]
            if len(valid_ids) < 2:
                continue

            # Size guard: check merged size before committing
            merged_unit_count = len(set().union(
                *(set(result[i].get("unit_ids", [])) for i in valid_ids)
            ))
            if merged_unit_count > _MERGE_MAX_UNITS:
                logger.info(
                    "Agentic merge rejected: MVPs %s would create %d-unit cluster (max %d)",
                    valid_ids, merged_unit_count, _MERGE_MAX_UNITS,
                )
                continue

            primary_idx = valid_ids[0]
            primary = result[primary_idx]

            for other_idx in valid_ids[1:]:
                other = result[other_idx]
                # Merge other into primary
                primary["unit_ids"] = list(
                    set(primary.get("unit_ids", [])) | set(other.get("unit_ids", []))
                )
                primary["file_ids"] = list(
                    set(primary.get("file_ids", [])) | set(other.get("file_ids", []))
                )
                # Merge SP references (deduplicate by sp_name)
                existing_sp_names = {
                    r.get("sp_name") for r in primary.get("sp_references", [])
                }
                for ref in other.get("sp_references", []):
                    if ref.get("sp_name") not in existing_sp_names:
                        primary.setdefault("sp_references", []).append(ref)
                        existing_sp_names.add(ref.get("sp_name"))
                merged_indices.add(other_idx)

            primary["_merged"] = True
            logger.info(
                "Agentic merge: MVPs %s → %s (reason: %s)",
                valid_ids, primary.get("name", "?"), merge.get("reason", "?"),
            )

        # Remove merged clusters (reverse order to preserve indices)
        for idx in sorted(merged_indices, reverse=True):
            result.pop(idx)

        # Apply splits
        new_clusters = []
        split_indices = set()
        for split_req in splits:
            idx = split_req.get("id")
            if idx is None or idx < 0 or idx >= len(result):
                continue
            if idx in split_indices:
                continue

            cluster = result[idx]
            unit_ids = cluster.get("unit_ids", [])

            # Simple split: divide units roughly in half
            # A more sophisticated approach would use the split_hint, but for now
            # we do a basic partition by file_id grouping
            if len(unit_ids) < 4:
                continue  # Too small to split meaningfully

            mid = len(unit_ids) // 2
            first_half = unit_ids[:mid]
            second_half = unit_ids[mid:]

            # Update original cluster
            cluster["unit_ids"] = first_half
            cluster["file_ids"] = list({
                fid for fid in cluster.get("file_ids", [])
            })  # Will be recalculated during persist
            cluster["_split"] = True

            # Create new cluster
            new_cluster = {
                "package": cluster.get("package", "unknown") + ".split",
                "unit_ids": second_half,
                "file_ids": [],
                "units": [],
                "name": cluster.get("name", "Unknown") + " (Split)",
                "description": None,
                "metrics": {},
                "sp_references": [],
                "depends_on": [],
                "_split": True,
            }
            new_clusters.append(new_cluster)
            split_indices.add(idx)

            logger.info(
                "Agentic split: MVP %d '%s' into %d + %d units (reason: %s)",
                idx, cluster.get("name", "?"), len(first_half), len(second_half),
                split_req.get("reason", "?"),
            )

        result.extend(new_clusters)
        return result

    @staticmethod
    def _deduplicate_mvps(mvps: List[Dict]) -> List[Dict]:
        """Merge same-name MVPs, remove mega-clusters, deduplicate unit_ids.

        Fixes three issues:
        1. Duplicate names: merge unit_ids/file_ids into the first occurrence
        2. Mega-clusters: skip any MVP containing >60% of the total unique units
        3. Unit overlap: ensure each unit appears in exactly one MVP
        """
        if not mvps:
            return mvps

        # Count total unique units
        all_units = set()
        for m in mvps:
            all_units.update(m.get("unit_ids", []))
        total_units = len(all_units)
        mega_threshold = max(total_units * 0.6, _MAX_CLUSTER_SIZE * 2)

        # Pass 1: Merge MVPs with identical names
        seen_names: Dict[str, int] = {}
        merged: List[Dict] = []

        for m in mvps:
            name = m.get("name", "Unknown")
            if name in seen_names:
                # Merge into existing
                target = merged[seen_names[name]]
                target["unit_ids"] = list(
                    set(target.get("unit_ids", [])) | set(m.get("unit_ids", []))
                )
                target["file_ids"] = list(
                    set(target.get("file_ids", [])) | set(m.get("file_ids", []))
                )
                # Keep the longer description
                if len(m.get("description") or "") > len(target.get("description") or ""):
                    target["description"] = m["description"]
                logger.info("Dedup: merged duplicate MVP '%s'", name)
            else:
                seen_names[name] = len(merged)
                merged.append(m)

        # Pass 2: Remove mega-clusters (>60% of all units)
        result = []
        mega_units: Set[str] = set()
        for m in merged:
            unit_count = len(m.get("unit_ids", []))
            if unit_count > mega_threshold:
                logger.info(
                    "Dedup: dropping mega-cluster '%s' (%d units, threshold=%d)",
                    m.get("name", "?"), unit_count, int(mega_threshold),
                )
                mega_units.update(m.get("unit_ids", []))
            else:
                result.append(m)

        # Pass 3: Ensure each unit appears in only one MVP (first-writer wins)
        claimed: Set[str] = set()
        for m in result:
            original = m.get("unit_ids", [])
            unique = [uid for uid in original if uid not in claimed]
            claimed.update(unique)
            m["unit_ids"] = unique
            # Recalculate file_ids if units changed
            if len(unique) != len(original):
                m["file_ids"] = list({fid for fid in m.get("file_ids", [])})

        # Remove any MVP that became empty after dedup
        result = [m for m in result if m.get("unit_ids")]

        # Reassign priorities to close gaps left by removed MVPs
        for i, m in enumerate(result):
            m["priority"] = i

        if len(result) != len(mvps):
            logger.info(
                "Dedup: %d MVPs → %d MVPs (removed %d duplicates/mega-clusters)",
                len(mvps), len(result), len(mvps) - len(result),
            )

        return result

    # ── MVP Refinement ─────────────────────────────────────────────────

    def update_mvp(
        self,
        plan_id: str,
        mvp_id: int,
        updates: Dict[str, Any],
    ) -> Dict:
        """Update an MVP's name, description, or unit assignments.

        Allowed updates: name, description, unit_ids, file_ids, priority.

        Returns:
            Updated MVP dict.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")

            allowed = {"name", "description", "unit_ids", "file_ids", "priority"}
            for key, value in updates.items():
                if key in allowed:
                    setattr(mvp, key, value)

            if "unit_ids" in updates or "file_ids" in updates:
                mvp.status = "refined"
                mvp.diagrams = None  # Invalidate cached diagrams on membership change

            return self._mvp_to_dict(mvp)

    def merge_mvps(
        self,
        plan_id: str,
        mvp_ids: List[int],
        new_name: Optional[str] = None,
    ) -> Dict:
        """Merge multiple MVPs into one.

        The first MVP absorbs the others; others are deleted.

        Returns:
            Merged MVP dict.
        """
        if len(mvp_ids) < 2:
            raise ValueError("Need at least 2 MVPs to merge")

        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id.in_(mvp_ids),
                FunctionalMVP.plan_id == pid,
            ).order_by(FunctionalMVP.priority).all()

            if len(mvps) < 2:
                raise ValueError("Some MVPs not found")

            primary = mvps[0]
            merged_unit_ids = set(primary.unit_ids or [])
            merged_file_ids = set(primary.file_ids or [])
            merged_sp_refs = list(primary.sp_references or [])
            sp_names_seen = {r.get("sp_name") for r in merged_sp_refs}

            for other in mvps[1:]:
                merged_unit_ids.update(other.unit_ids or [])
                merged_file_ids.update(other.file_ids or [])
                for ref in (other.sp_references or []):
                    if ref.get("sp_name") not in sp_names_seen:
                        merged_sp_refs.append(ref)
                        sp_names_seen.add(ref.get("sp_name"))
                session.delete(other)

            primary.unit_ids = sorted(merged_unit_ids)
            primary.file_ids = sorted(merged_file_ids)
            primary.sp_references = merged_sp_refs
            primary.status = "refined"
            primary.analysis_output = None  # Scope changed — force fresh re-discovery
            primary.analysis_at = None
            primary.diagrams = None  # Invalidate cached diagrams after merge
            if new_name:
                primary.name = new_name

            # Get project_id for auto-naming (before session closes)
            plan = session.query(MigrationPlan).filter(MigrationPlan.plan_id == pid).first()
            project_id = str(plan.source_project_id) if plan else None

            result = self._mvp_to_dict(primary)
            mvp_id = primary.mvp_id

        # Auto-name from merged source code if no explicit name was provided.
        # LLM call runs OUTSIDE the merge session to avoid holding DB connections.
        if not new_name and project_id:
            try:
                ctx = MigrationContextBuilder(self._db, project_id)
                cluster = {"unit_ids": result["unit_ids"], "metrics": result.get("metrics", {})}
                desc = _describe_mvp(ctx, cluster, token_budget=4_000)
                auto_name = desc.get("name", result["name"])
                auto_desc = desc.get("description", result.get("description", ""))
                with self._db.get_session() as session:
                    mvp = session.query(FunctionalMVP).filter(
                        FunctionalMVP.mvp_id == mvp_id
                    ).first()
                    if mvp:
                        mvp.name = auto_name
                        mvp.description = auto_desc
                result["name"] = auto_name
                result["description"] = auto_desc
                logger.info("Auto-named merged MVP %d: %s", mvp_id, auto_name)
            except Exception as e:
                logger.warning("Auto-naming merged MVP failed: %s", e)

        return result

    def split_mvp(
        self,
        plan_id: str,
        mvp_id: int,
        split_unit_ids: List[str],
        new_name: str,
    ) -> Dict:
        """Split units from one MVP into a new MVP.

        Args:
            plan_id: Plan UUID string
            mvp_id: Source MVP ID
            split_unit_ids: Unit IDs to move to the new MVP
            new_name: Name for the new MVP

        Returns:
            Dict with 'original' and 'new_mvp' dicts.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id
        split_set = set(split_unit_ids)

        with self._db.get_session() as session:
            source = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not source:
                raise ValueError(f"MVP {mvp_id} not found")

            remaining_units = [uid for uid in (source.unit_ids or []) if uid not in split_set]
            if not remaining_units:
                raise ValueError("Cannot move all units — original MVP would be empty")

            # Create new MVP
            new_mvp = FunctionalMVP(
                plan_id=pid,
                name=new_name,
                status="refined",
                priority=source.priority + 1,
                unit_ids=sorted(split_set),
                file_ids=[],  # Will be recalculated
                sp_references=[],
                metrics={},
            )
            session.add(new_mvp)

            source.unit_ids = remaining_units
            source.status = "refined"
            source.diagrams = None  # Invalidate cached diagrams after split
            session.flush()

            result = {
                "original": self._mvp_to_dict(source),
                "new_mvp": self._mvp_to_dict(new_mvp),
            }

        return result

    # ── MVP Phase Creation ─────────────────────────────────────────────

    def create_mvp_phases(self, plan_id: str) -> List[Dict]:
        """Create per-MVP phases for all discovered/refined MVPs.

        V1: Creates phases 3-6 (Analyze, Design, Transform, Test).
        V2: Creates phases 3-4 (Transform, Test).

        Should be called after the second plan-level phase is approved
        and the user has finished refining MVPs.

        Returns:
            List of created phase summaries.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Plan {plan_id} not found")

            version = plan.pipeline_version or 1

            # Verify both plan-level phases are approved
            for pn in _plan_phases(version):
                ph = session.query(MigrationPhase).filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.phase_number == pn,
                    MigrationPhase.mvp_id.is_(None),
                ).first()
                if not ph or not ph.approved:
                    phase_type = get_phase_type(pn, version)
                    raise ValueError(
                        f"Phase {pn} ({phase_type}) must be approved before creating MVP phases"
                    )

            mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid
            ).order_by(FunctionalMVP.priority).all()

            if not mvps:
                raise ValueError("No MVPs found — run discovery first")

            mvp_ph = _mvp_phases(version)
            created = []
            for mvp in mvps:
                for n in mvp_ph:
                    phase = MigrationPhase(
                        phase_id=uuid4(),
                        plan_id=pid,
                        mvp_id=mvp.mvp_id,
                        phase_number=n,
                        phase_type=get_phase_type(n, version),
                        status="pending",
                    )
                    session.add(phase)
                    created.append({
                        "phase_id": str(phase.phase_id),
                        "mvp_id": mvp.mvp_id,
                        "mvp_name": mvp.name,
                        "phase_number": n,
                        "phase_type": get_phase_type(n, version),
                    })

        logger.info(f"Created {len(created)} per-MVP phases for plan {plan_id}")
        return created

    # ── Phase Execution ────────────────────────────────────────────────

    def execute_phase(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
    ) -> Dict:
        """Execute a migration phase.

        Version-aware: reads plan.pipeline_version to determine context_type
        and enrichment triggers.

        V1: Phase 1=Discovery, 2=Architecture, 3-6=per-MVP
        V2: Phase 1=Architecture, 2=Discovery, 3-4=per-MVP

        Returns:
            Phase output dict.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")

            version = plan.pipeline_version or 1
            plan_ph = _plan_phases(version)
            mvp_ph = _mvp_phases(version)

            if phase_number in mvp_ph and mvp_id is None:
                raise ValueError(f"Phase {phase_number} requires an mvp_id")
            if phase_number in plan_ph and mvp_id is not None:
                raise ValueError(f"Plan-level phase {phase_number} does not accept mvp_id")

            # Check prerequisites
            self._validate_prerequisites(session, pid, phase_number, mvp_id, version)

            # Get the target phase row
            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(
                    f"Phase {phase_number} not found "
                    f"{'for MVP ' + str(mvp_id) if mvp_id else '(plan-level)'}"
                )

            # Set status to running
            phase.status = "running"
            plan.status = "in_progress"
            plan.current_phase = phase_number

            # Collect previous phase outputs for context chaining
            previous_outputs = self._collect_previous_outputs(session, pid, phase_number, mvp_id)

            # Resolve project name for disk output folder
            project_name = None
            if plan.source_project_id:
                proj = session.query(Project).filter(
                    Project.project_id == plan.source_project_id
                ).first()
                if proj:
                    project_name = proj.name

            # Plan data for prompt building
            plan_data = {
                "target_brief": plan.target_brief,
                "target_stack": plan.target_stack or {},
                "constraints": plan.constraints or {},
                "migration_type": plan.migration_type or "framework_migration",
                "discovery_metadata": plan.discovery_metadata or {},
                "framework_docs": plan.framework_docs or {},
                "_project_name": project_name,
            }
            project_id = str(plan.source_project_id)

            # MVP context for per-MVP phases
            mvp_data = None
            if mvp_id:
                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id
                ).first()
                if mvp:
                    mvp_data = self._mvp_to_dict(mvp)
                    # V2: inject on-demand analysis output into plan_data for Transform
                    if mvp.analysis_output:
                        plan_data["_analysis_output"] = mvp.analysis_output.get("output", "")

            # Inject MVP functional summaries for Discovery prompt
            all_mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid
            ).order_by(FunctionalMVP.priority).all()
            if all_mvps:
                plan_data["_mvp_summaries"] = [
                    {
                        "name": m.name,
                        "description": m.description or "",
                        "unit_count": len(m.unit_ids or []),
                    }
                    for m in all_mvps
                ]

        # Determine context_type for semantic dispatch (V2 decouples phase# from executor)
        context_type = get_phase_type(phase_number, version) if version == 2 else None

        # Execute outside the session (LLM call can be slow)
        try:
            ctx_builder = MigrationContextBuilder(self._db, project_id)

            # Architecture enrichment: auto-enrich with source patterns + framework docs
            # V1: triggers at Phase 2 (architecture). V2: triggers at Phase 1 (architecture).
            arch_phase = 1 if version == 2 else 2
            if phase_number == arch_phase:
                self._enrich_for_phase_2(plan_data, ctx_builder, pid)

            result = execute_phase(
                phase_number=phase_number,
                plan=plan_data,
                previous_outputs=previous_outputs,
                context_builder=ctx_builder,
                mvp_context=mvp_data,
                context_type=context_type,
            )

            # Persist result
            with self._db.get_session() as session:
                phase = self._get_phase(session, pid, phase_number, mvp_id)
                phase.status = "complete"
                phase.output = result.get("output", "")
                phase.output_files = result.get("output_files", [])

                if mvp_id:
                    mvp = session.query(FunctionalMVP).filter(
                        FunctionalMVP.mvp_id == mvp_id
                    ).first()
                    if mvp:
                        mvp.current_phase = phase_number
                        # Mark MVP in_progress on first per-MVP phase
                        first_mvp_phase = mvp_ph[0]
                        if phase_number == first_mvp_phase:
                            mvp.status = "in_progress"

            # Write to disk (fire-and-forget — DB is source of truth)
            disk_path = None
            try:
                disk_path = self._write_phase_to_disk(
                    plan_id=str(pid),
                    phase_number=phase_number,
                    phase_type=context_type or get_phase_type(phase_number, version),
                    output=result.get("output", ""),
                    output_files=result.get("output_files", []),
                    mvp_id=mvp_id,
                    project_name=plan_data.get("_project_name"),
                )
            except Exception as disk_err:
                logger.warning(f"Failed to write phase output to disk: {disk_err}")

            # Store disk path in phase metadata for UI display
            if disk_path:
                with self._db.get_session() as session:
                    phase = self._get_phase(session, pid, phase_number, mvp_id)
                    if phase:
                        meta = dict(phase.phase_metadata or {})
                        meta["output_path"] = disk_path
                        phase.phase_metadata = meta

            logger.info(
                f"Phase {phase_number} complete for plan {plan_id}"
                + (f" MVP {mvp_id}" if mvp_id else "")
                + (f" → {disk_path}" if disk_path else "")
            )

        except Exception as e:
            logger.error(f"Phase {phase_number} failed: {e}")
            with self._db.get_session() as session:
                phase = self._get_phase(session, pid, phase_number, mvp_id)
                if phase:
                    phase.status = "error"
                    phase.output = f"Error: {str(e)}"
            raise

        return self.get_phase_output(str(pid), phase_number, mvp_id)

    def approve_phase(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
    ) -> Dict:
        """Approve a completed phase. Unlocks the next phase.

        Version-aware: V1 completes at MVP Phase 6, V2 at MVP Phase 4.

        Returns:
            Updated plan status dict.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Plan {plan_id} not found")

            version = plan.pipeline_version or 1
            final_mvp_phase = _mvp_phases(version)[-1]  # V1=6, V2=4

            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(
                    f"Phase {phase_number} not found"
                    + (f" for MVP {mvp_id}" if mvp_id else "")
                )
            if phase.status != "complete":
                raise ValueError(f"Phase {phase_number} is not complete (status: {phase.status})")

            phase.approved = True
            phase.approved_at = datetime.utcnow()

            # Check if this is the final MVP phase → mark MVP migrated + check plan completion
            if mvp_id and phase_number == final_mvp_phase:
                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id
                ).first()
                if mvp:
                    mvp.status = "migrated"
                    mvp.current_phase = final_mvp_phase

                # Check if ALL MVPs are migrated
                all_mvps = session.query(FunctionalMVP).filter(
                    FunctionalMVP.plan_id == pid
                ).all()
                if all(m.status == "migrated" for m in all_mvps):
                    plan.status = "complete"
                    logger.info(f"Plan {plan_id} complete — all MVPs migrated")

        logger.info(
            f"Phase {phase_number} approved for plan {plan_id}"
            + (f" MVP {mvp_id}" if mvp_id else "")
        )
        return self.get_plan_status(str(pid))

    def reject_phase(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> Dict:
        """Reject a completed phase. Allows re-execution with optional feedback.

        Returns:
            Updated phase output dict.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(f"Phase {phase_number} not found")

            phase.status = "rejected"
            phase.approved = False
            if feedback:
                meta = phase.phase_metadata or {}
                meta["rejection_feedback"] = feedback
                phase.phase_metadata = meta

        return self.get_phase_output(str(pid), phase_number, mvp_id)

    def delete_plan(self, plan_id: str) -> None:
        """Delete a migration plan, all phases, and all MVPs (via CASCADE)."""
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")
            session.delete(plan)

        logger.info(f"Deleted migration plan {plan_id}")

    # ── Status Queries ─────────────────────────────────────────────────

    def get_plan_status(self, plan_id: str) -> Dict:
        """Full plan status with plan-level phases, MVPs, and per-MVP phases."""
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")

            # Plan-level phases
            plan_phases = session.query(MigrationPhase).filter(
                MigrationPhase.plan_id == pid,
                MigrationPhase.mvp_id.is_(None),
            ).order_by(MigrationPhase.phase_number).all()

            # MVPs with their phases
            mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid
            ).order_by(FunctionalMVP.priority).all()

            mvp_summaries = []
            for mvp in mvps:
                mvp_phases = session.query(MigrationPhase).filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.mvp_id == mvp.mvp_id,
                ).order_by(MigrationPhase.phase_number).all()

                mvp_summaries.append({
                    **self._mvp_to_dict(mvp),
                    "phases": [self._phase_summary(p) for p in mvp_phases],
                })

            # Build source_stack from source project's detected languages
            source_stack = None
            if plan.source_project_id:
                src_proj = session.query(Project).filter(
                    Project.project_id == plan.source_project_id
                ).first()
                if src_proj:
                    source_stack = {
                        "primary_language": src_proj.primary_language,
                        "languages": src_proj.languages or [],
                    }

            return {
                "plan_id": str(plan.plan_id),
                "source_project_id": str(plan.source_project_id) if plan.source_project_id else None,
                "source_stack": source_stack,
                "target_brief": plan.target_brief,
                "target_stack": plan.target_stack,
                "constraints": plan.constraints,
                "status": plan.status,
                "current_phase": plan.current_phase,
                "migration_type": plan.migration_type or "framework_migration",
                "pipeline_version": plan.pipeline_version or 1,
                "asset_strategies": plan.asset_strategies,
                "discovery_metadata": plan.discovery_metadata,
                "created_at": plan.created_at.isoformat() if plan.created_at else None,
                "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
                "plan_phases": [self._phase_summary(p) for p in plan_phases],
                "mvps": mvp_summaries,
            }

    def get_phase_output(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
    ) -> Dict:
        """Detailed phase output including generated content."""
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(
                    f"Phase {phase_number} not found"
                    + (f" for MVP {mvp_id}" if mvp_id else "")
                )

            return {
                "phase_id": str(phase.phase_id),
                "phase_number": phase.phase_number,
                "phase_type": phase.phase_type,
                "status": phase.status,
                "output": phase.output,
                "output_files": phase.output_files or [],
                "approved": phase.approved,
                "approved_at": phase.approved_at.isoformat() if phase.approved_at else None,
                "input_summary": phase.input_summary,
                "mvp_id": phase.mvp_id,
                "phase_metadata": phase.phase_metadata,
            }

    def get_mvp_detail(self, plan_id: str, mvp_id: int) -> Dict:
        """Get detailed MVP info with resolved files, units, phases, and analysis."""
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Plan {plan_id} not found")

            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")

            phases = session.query(MigrationPhase).filter(
                MigrationPhase.plan_id == pid,
                MigrationPhase.mvp_id == mvp_id,
            ).order_by(MigrationPhase.phase_number).all()

            # Resolve file_ids → actual file info
            resolved_files = []
            if mvp.file_ids:
                files = session.query(CodeFile).filter(
                    CodeFile.file_id.in_(mvp.file_ids)
                ).order_by(CodeFile.file_path).all()
                resolved_files = [
                    {
                        "file_id": str(f.file_id),
                        "file_path": f.file_path,
                        "language": f.language,
                        "line_count": f.line_count or 0,
                    }
                    for f in files
                ]

            # Resolve unit_ids → actual unit info with file paths
            resolved_units = []
            if mvp.unit_ids:
                units = session.query(CodeUnit).filter(
                    CodeUnit.unit_id.in_(mvp.unit_ids)
                ).order_by(CodeUnit.qualified_name).all()

                # Build file_id → file_path lookup
                unit_file_ids = {u.file_id for u in units}
                file_map = {}
                if unit_file_ids:
                    unit_files = session.query(CodeFile).filter(
                        CodeFile.file_id.in_(unit_file_ids)
                    ).all()
                    file_map = {f.file_id: f.file_path for f in unit_files}

                resolved_units = [
                    {
                        "unit_id": str(u.unit_id),
                        "name": u.name,
                        "qualified_name": u.qualified_name,
                        "unit_type": u.unit_type,
                        "language": u.language,
                        "file_path": file_map.get(u.file_id, ""),
                        "start_line": u.start_line,
                        "end_line": u.end_line,
                        "signature": u.signature,
                    }
                    for u in units
                ]

            # Extract architecture mapping from plan-level architecture phase output
            architecture_mapping = []
            version = plan.pipeline_version or 1
            arch_phase_num = 1 if version == 2 else 2
            arch_phase = self._get_phase(session, pid, arch_phase_num, None)
            if arch_phase and arch_phase.output:
                mvp_file_paths = [f["file_path"] for f in resolved_files]
                architecture_mapping = self._extract_mvp_mapping(
                    arch_phase.output, mvp_file_paths
                )

            return {
                **self._mvp_to_dict(mvp),
                "files": resolved_files,
                "units": resolved_units,
                "architecture_mapping": architecture_mapping,
                "phases": [
                    {
                        "phase_id": str(p.phase_id),
                        "phase_number": p.phase_number,
                        "phase_type": p.phase_type,
                        "status": p.status,
                        "approved": p.approved,
                        "approved_at": p.approved_at.isoformat() if p.approved_at else None,
                        "output_preview": (p.output[:200] + "...") if p.output and len(p.output) > 200 else p.output,
                    }
                    for p in phases
                ],
            }

    def list_plans(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Dict]:
        """List migration plans, optionally filtered by user or project."""
        with self._db.get_session() as session:
            query = session.query(MigrationPlan)

            if user_id:
                uid = UUID(user_id) if isinstance(user_id, str) else user_id
                query = query.filter(MigrationPlan.user_id == uid)
            if project_id:
                pid = UUID(project_id) if isinstance(project_id, str) else project_id
                query = query.filter(MigrationPlan.source_project_id == pid)

            query = query.order_by(MigrationPlan.updated_at.desc())
            plans = query.all()

            results = []
            for plan in plans:
                plan_phases = session.query(MigrationPhase).filter(
                    MigrationPhase.plan_id == plan.plan_id,
                    MigrationPhase.mvp_id.is_(None),
                ).order_by(MigrationPhase.phase_number).all()

                mvp_count = session.query(FunctionalMVP).filter(
                    FunctionalMVP.plan_id == plan.plan_id
                ).count()

                # Source stack from project
                source_stack = None
                if plan.source_project_id:
                    src_proj = session.query(Project).filter(
                        Project.project_id == plan.source_project_id
                    ).first()
                    if src_proj:
                        source_stack = {
                            "primary_language": src_proj.primary_language,
                            "languages": src_proj.languages or [],
                        }

                results.append({
                    "plan_id": str(plan.plan_id),
                    "source_project_id": str(plan.source_project_id) if plan.source_project_id else None,
                    "source_stack": source_stack,
                    "target_brief": plan.target_brief,
                    "target_stack": plan.target_stack,
                    "constraints": plan.constraints,
                    "status": plan.status,
                    "current_phase": plan.current_phase,
                    "pipeline_version": plan.pipeline_version or 1,
                    "mvp_count": mvp_count,
                    "created_at": plan.created_at.isoformat() if plan.created_at else None,
                    "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
                    "plan_phases": [self._phase_summary(p) for p in plan_phases],
                })

            return results

    # ── On-Demand MVP Analysis ────────────────────────────────────────

    def analyze_mvp(self, plan_id: str, mvp_id: int) -> Dict:
        """Run on-demand deep analysis for an MVP (merges Analyze + Design).

        V2 pipeline only. Result is stored on FunctionalMVP.analysis_output,
        not as a pipeline phase. Can be re-run to refresh the analysis.

        Tracks status via FunctionalMVP.analysis_status:
        pending -> analyzing -> completed | failed

        Returns:
            Dict with 'output' (markdown), 'output_files', and 'analysis_at'.
        """
        import re
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        # Mark as analyzing
        with self._db.get_session() as session:
            mvp_row = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if mvp_row:
                mvp_row.analysis_status = "analyzing"
                mvp_row.analysis_error = None

        try:
            with self._db.get_session() as session:
                plan = session.query(MigrationPlan).filter(
                    MigrationPlan.plan_id == pid
                ).first()
                if not plan:
                    raise ValueError(f"Migration plan {plan_id} not found")

                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id,
                    FunctionalMVP.plan_id == pid,
                ).first()
                if not mvp:
                    raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")

                project_id = str(plan.source_project_id)

                # Get project name for disk output path
                src_proj = session.query(Project).filter(
                    Project.project_id == plan.source_project_id
                ).first()
                project_name = src_proj.name if src_proj else None

                # Collect plan-level phase outputs for context
                previous_outputs = {}
                plan_phases = session.query(MigrationPhase).filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.mvp_id.is_(None),
                    MigrationPhase.approved == True,
                ).order_by(MigrationPhase.phase_number).all()
                for pp in plan_phases:
                    previous_outputs[pp.phase_number] = pp.output or ""

                plan_data = {
                    "target_brief": plan.target_brief,
                    "target_stack": plan.target_stack or {},
                    "constraints": plan.constraints or {},
                    "migration_type": plan.migration_type or "framework_migration",
                    "discovery_metadata": plan.discovery_metadata or {},
                    "framework_docs": plan.framework_docs or {},
                    "_project_name": project_name,
                }
                mvp_data = self._mvp_to_dict(mvp)

            # Execute outside the session (LLM call can be slow)
            ctx_builder = MigrationContextBuilder(self._db, project_id)
            result = execute_mvp_analysis(
                plan=plan_data,
                previous_outputs=previous_outputs,
                context_builder=ctx_builder,
                mvp_context=mvp_data,
            )

            # Store result on the MVP row — mark completed
            now = datetime.utcnow()
            with self._db.get_session() as session:
                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id,
                    FunctionalMVP.plan_id == pid,
                ).first()
                if mvp:
                    mvp.analysis_output = {"output": result.get("output", "")}
                    mvp.analysis_at = now
                    mvp.analysis_status = "completed"
                    mvp.analysis_error = None

            # Write MVP feature documents to disk (fire-and-forget)
            try:
                self._write_mvp_documents(
                    plan_id=str(pid),
                    mvp_id=mvp_id,
                    project_name=plan_data.get("_project_name"),
                )
            except Exception as disk_err:
                logger.warning("Failed to write MVP documents to disk: %s", disk_err)

            logger.info(f"Deep analysis complete for MVP {mvp_id} in plan {plan_id}")
            return {
                "output": result.get("output", ""),
                "output_files": result.get("output_files", []),
                "analysis_at": now.isoformat(),
            }

        except Exception as e:
            # Mark as failed
            with self._db.get_session() as session:
                mvp_row = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id,
                    FunctionalMVP.plan_id == pid,
                ).first()
                if mvp_row:
                    mvp_row.analysis_status = "failed"
                    mvp_row.analysis_error = str(e)
            raise

    @staticmethod
    def _extract_mvp_mapping(
        architecture_output: str, mvp_file_paths: List[str]
    ) -> List[Dict]:
        """Parse Architecture output's Module Structure Mapping table for this MVP's files.

        Looks for markdown table rows:
          | Source Path | Source Class | Target Path | Target Class | Changes |

        Returns matching rows filtered by the MVP's file paths.
        """
        import re

        if not architecture_output or not mvp_file_paths:
            return []

        # Normalize MVP file paths for matching (basename without extension)
        mvp_basenames = set()
        mvp_path_set = set()
        for fp in mvp_file_paths:
            mvp_path_set.add(fp)
            base = os.path.basename(fp)
            stem, _ = os.path.splitext(base)
            mvp_basenames.add(stem.lower())

        # Parse markdown table rows: | col1 | col2 | col3 | col4 | col5 |
        # Skip header and separator lines
        table_row_re = re.compile(
            r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
            re.MULTILINE,
        )

        mappings = []
        for match in table_row_re.finditer(architecture_output):
            source_path = match.group(1).strip()
            source_class = match.group(2).strip()
            target_path = match.group(3).strip()
            target_class = match.group(4).strip()
            changes = match.group(5).strip()

            # Skip header/separator rows
            if source_path.startswith("-") or source_path.lower() == "source path":
                continue

            # Check if source_path matches any MVP file (by full path or basename)
            source_basename = os.path.splitext(os.path.basename(source_path))[0].lower()
            if source_path in mvp_path_set or source_basename in mvp_basenames:
                mappings.append({
                    "source_path": source_path,
                    "source_class": source_class,
                    "target_path": target_path,
                    "target_class": target_class,
                    "changes": changes,
                })

        return mappings

    # ── Diff Context & Download ──────────────────────────────────────

    def get_diff_context(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
    ) -> Dict:
        """Build diff context: pair original source files with migrated output.

        Reconstructs original source from CodeUnit rows (same pattern as
        projects.py:get_file_content) and matches them to migrated files
        by basename similarity.

        Returns:
            Dict with migrated_files, source_files, and file_mapping.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(
                    f"Phase {phase_number} not found"
                    + (f" for MVP {mvp_id}" if mvp_id else "")
                )
            migrated_files = phase.output_files or []

            # Get source files from MVP's file_ids
            source_files: List[Dict] = []
            if mvp_id:
                mvp = session.query(FunctionalMVP).filter(
                    FunctionalMVP.mvp_id == mvp_id,
                    FunctionalMVP.plan_id == pid,
                ).first()
                if mvp and mvp.file_ids:
                    for fid in mvp.file_ids:
                        try:
                            file_uuid = UUID(fid) if isinstance(fid, str) else fid
                        except (ValueError, AttributeError):
                            continue
                        code_file = session.query(CodeFile).filter(
                            CodeFile.file_id == file_uuid,
                        ).first()
                        if not code_file:
                            continue

                        # Reconstruct source from code units ordered by line
                        units = session.query(CodeUnit).filter(
                            CodeUnit.file_id == code_file.file_id,
                        ).order_by(CodeUnit.start_line).all()

                        content = "\n".join(u.source for u in units if u.source)
                        source_files.append({
                            "file_path": code_file.file_path,
                            "language": code_file.language or "",
                            "content": content,
                        })

        # Build file mapping by basename similarity
        file_mapping = self._build_file_mapping(source_files, migrated_files)

        return {
            "migrated_files": migrated_files,
            "source_files": source_files,
            "file_mapping": file_mapping,
        }

    def get_phase_files_download(
        self,
        plan_id: str,
        phase_number: int,
        mvp_id: Optional[int] = None,
        fmt: str = "zip",
        file_path: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """Package phase output files for download.

        Args:
            fmt: 'zip' for all files, 'single' for one file (requires file_path)
            file_path: Required when fmt='single'

        Returns:
            Tuple of (content_bytes, content_type, filename).
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            phase = self._get_phase(session, pid, phase_number, mvp_id)
            if not phase:
                raise ValueError(
                    f"Phase {phase_number} not found"
                    + (f" for MVP {mvp_id}" if mvp_id else "")
                )
            output_files = phase.output_files or []

        if not output_files:
            raise ValueError("No output files available for download")

        if fmt == "single":
            if not file_path:
                raise ValueError("file_path is required for single-file download")
            matched = next(
                (f for f in output_files if f.get("file_path") == file_path), None
            )
            if not matched:
                raise ValueError(f"File not found: {file_path}")
            content = matched.get("content", "").encode("utf-8")
            filename = os.path.basename(file_path)
            return content, "text/plain; charset=utf-8", filename

        # ZIP format
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_files:
                fp = f.get("file_path", "unknown")
                zf.writestr(fp, f.get("content", ""))
        buf.seek(0)
        zip_name = f"phase-{phase_number}"
        if mvp_id:
            zip_name += f"-mvp-{mvp_id}"
        zip_name += ".zip"
        return buf.getvalue(), "application/zip", zip_name

    @staticmethod
    def _build_file_mapping(
        source_files: List[Dict], migrated_files: List[Dict]
    ) -> List[Dict]:
        """Match source files to migrated files by basename similarity."""
        mapping = []

        def _stem(path: str) -> str:
            """Strip extension and return lowered basename."""
            base = os.path.basename(path)
            root, _ = os.path.splitext(base)
            return root.lower()

        source_stems = {_stem(f["file_path"]): f["file_path"] for f in source_files}

        for mf in migrated_files:
            target_path = mf.get("file_path", "")
            target_stem = _stem(target_path)

            # Exact stem match
            if target_stem in source_stems:
                mapping.append({
                    "source_path": source_stems[target_stem],
                    "target_path": target_path,
                    "confidence": 0.9,
                })
            else:
                # Partial match: check if target stem starts with or contains source stem
                best_match = None
                best_score = 0.0
                for s_stem, s_path in source_stems.items():
                    if s_stem in target_stem or target_stem in s_stem:
                        score = min(len(s_stem), len(target_stem)) / max(len(s_stem), len(target_stem))
                        if score > best_score:
                            best_score = score
                            best_match = s_path
                if best_match and best_score > 0.3:
                    mapping.append({
                        "source_path": best_match,
                        "target_path": target_path,
                        "confidence": round(best_score * 0.8, 2),
                    })

        return mapping

    # ── Framework Doc Enrichment ──────────────────────────────────────

    def _enrich_for_phase_2(
        self,
        plan_data: Dict,
        ctx_builder: MigrationContextBuilder,
        plan_id,
    ) -> None:
        """Auto-enrich plan_data with source patterns + framework docs at Phase 2.

        Source patterns are detected from the ASG. Framework docs are fetched
        via Tavily and cached on the plan row for subsequent phases.
        """
        # 1. Detect source patterns from ASG metadata
        try:
            source_patterns = ctx_builder.get_source_patterns()
            plan_data["_source_patterns"] = source_patterns
        except Exception as e:
            logger.warning("Source pattern detection failed: %s", e)
            source_patterns = None

        # 2. Fetch framework docs if not already cached
        if not plan_data.get("framework_docs"):
            frameworks = plan_data.get("target_stack", {}).get("frameworks", [])
            if frameworks:
                try:
                    enricher = DocEnricher()
                    fw_docs = enricher.enrich_plan(frameworks, source_patterns)
                    if fw_docs:
                        plan_data["framework_docs"] = fw_docs
                        # Cache on the plan row
                        with self._db.get_session() as session:
                            plan = session.query(MigrationPlan).filter(
                                MigrationPlan.plan_id == plan_id
                            ).first()
                            if plan:
                                plan.framework_docs = fw_docs
                        logger.info(
                            "Enriched plan %s with docs for %d framework(s)",
                            plan_id, len(fw_docs),
                        )
                except Exception as e:
                    logger.warning("Framework doc enrichment failed: %s", e)

    def enrich_framework_docs(self, plan_id: str) -> Dict:
        """Re-fetch framework docs for a plan's target stack.

        Called by the API endpoint to refresh cached docs.

        Returns:
            Dict with enriched frameworks and any failures.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                raise ValueError(f"Migration plan {plan_id} not found")

            frameworks = (plan.target_stack or {}).get("frameworks", [])
            project_id = str(plan.source_project_id)

        if not frameworks:
            return {"enriched": [], "failed": [], "total_tokens": 0}

        # Detect source patterns for targeted search
        source_patterns = None
        try:
            ctx_builder = MigrationContextBuilder(self._db, project_id)
            source_patterns = ctx_builder.get_source_patterns()
        except Exception as e:
            logger.warning("Source pattern detection failed during enrichment: %s", e)

        enricher = DocEnricher()
        fw_docs = enricher.enrich_plan(frameworks, source_patterns)

        # Cache on plan
        enriched = list(fw_docs.keys())
        failed = [f for f in frameworks if f not in fw_docs]

        if fw_docs:
            with self._db.get_session() as session:
                plan = session.query(MigrationPlan).filter(
                    MigrationPlan.plan_id == pid
                ).first()
                if plan:
                    plan.framework_docs = fw_docs

        total_chars = sum(len(d.get("content", "")) for d in fw_docs.values())

        return {
            "enriched": enriched,
            "failed": failed,
            "total_tokens": total_chars // 4,
        }

    # ── Private Helpers ────────────────────────────────────────────────

    def _get_phase(self, session, plan_id, phase_number, mvp_id=None):
        """Get a phase row by plan_id + phase_number + mvp_id."""
        q = session.query(MigrationPhase).filter(
            MigrationPhase.plan_id == plan_id,
            MigrationPhase.phase_number == phase_number,
        )
        if mvp_id is not None:
            q = q.filter(MigrationPhase.mvp_id == mvp_id)
        else:
            q = q.filter(MigrationPhase.mvp_id.is_(None))
        return q.first()

    def _validate_prerequisites(self, session, plan_id, phase_number, mvp_id, version=1):
        """Validate that prerequisite phases are approved.

        Version-aware rules:
          V1: Phase 1 always OK, Phase 2 needs Phase 1, Phase 3 needs Phase 2,
              Phases 4-6 need prior MVP phase.
          V2: Phase 1 (architecture) always OK, Phase 2 (discovery) needs Phase 1,
              Phase 3 (transform) needs Phase 2 approved + MVP phases created,
              Phase 4 (test) needs Phase 3 for this MVP approved.
        """
        if phase_number == 1:
            return  # Phase 1 always allowed in both versions

        if phase_number == 2:
            # Phase 1 must be approved
            phase_1 = self._get_phase(session, plan_id, 1, None)
            if not phase_1 or not phase_1.approved:
                type_name = get_phase_type(1, version)
                raise ValueError(f"Phase 1 ({type_name}) must be approved before Phase 2")
            return

        # Per-MVP phases
        first_mvp_phase = _mvp_phases(version)[0]  # V1=3, V2=3

        if phase_number == first_mvp_phase:
            # Both plan-level phases must be approved
            phase_2 = self._get_phase(session, plan_id, 2, None)
            if not phase_2 or not phase_2.approved:
                type_name = get_phase_type(2, version)
                raise ValueError(f"Phase 2 ({type_name}) must be approved before per-MVP phases")
            return

        # Subsequent MVP phases: previous phase for this MVP must be approved
        prev_phase = self._get_phase(session, plan_id, phase_number - 1, mvp_id)
        if not prev_phase or not prev_phase.approved:
            raise ValueError(
                f"Phase {phase_number - 1} for MVP {mvp_id} must be approved "
                f"before executing phase {phase_number}"
            )

    def _collect_previous_outputs(self, session, plan_id, phase_number, mvp_id):
        """Collect approved phase outputs for context chaining."""
        previous_outputs = {}

        # Always include plan-level phase outputs
        plan_phases = session.query(MigrationPhase).filter(
            MigrationPhase.plan_id == plan_id,
            MigrationPhase.mvp_id.is_(None),
            MigrationPhase.approved == True,
        ).order_by(MigrationPhase.phase_number).all()

        for pp in plan_phases:
            previous_outputs[pp.phase_number] = pp.output or ""

        # For per-MVP phases, also include previous MVP phase outputs
        if mvp_id:
            mvp_phases = session.query(MigrationPhase).filter(
                MigrationPhase.plan_id == plan_id,
                MigrationPhase.mvp_id == mvp_id,
                MigrationPhase.phase_number < phase_number,
                MigrationPhase.approved == True,
            ).order_by(MigrationPhase.phase_number).all()

            for pp in mvp_phases:
                previous_outputs[pp.phase_number] = pp.output or ""

        return previous_outputs

    def _phase_summary(self, phase: MigrationPhase) -> Dict:
        """Convert a phase row to a summary dict."""
        return {
            "phase_id": str(phase.phase_id),
            "phase_number": phase.phase_number,
            "phase_type": phase.phase_type,
            "status": phase.status,
            "approved": phase.approved,
            "approved_at": phase.approved_at.isoformat() if phase.approved_at else None,
            "mvp_id": phase.mvp_id,
            "output_preview": (
                (phase.output[:200] + "...") if phase.output and len(phase.output) > 200
                else phase.output
            ),
        }

    @staticmethod
    def _mvp_to_dict(mvp: FunctionalMVP) -> Dict:
        """Convert a FunctionalMVP row to a dict.

        Note: JSONB columns (file_ids, unit_ids, depends_on_mvp_ids,
        sp_references, metrics, analysis_output, diagrams) must use
        `is not None` — SQLAlchemy JSONB raises TypeError on truthiness.
        """
        return {
            "mvp_id": mvp.mvp_id,
            "name": mvp.name,
            "description": mvp.description,
            "status": mvp.status,
            "priority": mvp.priority,
            "file_ids": mvp.file_ids if mvp.file_ids is not None else [],
            "unit_ids": mvp.unit_ids if mvp.unit_ids is not None else [],
            "depends_on_mvp_ids": mvp.depends_on_mvp_ids if mvp.depends_on_mvp_ids is not None else [],
            "sp_references": mvp.sp_references if mvp.sp_references is not None else [],
            "metrics": mvp.metrics if mvp.metrics is not None else {},
            "current_phase": mvp.current_phase or 0,
            "analysis_output": mvp.analysis_output if mvp.analysis_output is not None else None,
            "analysis_at": mvp.analysis_at.isoformat() if mvp.analysis_at else None,
            "has_cached_diagrams": isinstance(mvp.diagrams, dict) and len(mvp.diagrams) > 0,
            "created_at": mvp.created_at.isoformat() if mvp.created_at else None,
            "updated_at": mvp.updated_at.isoformat() if mvp.updated_at else None,
        }

    @staticmethod
    def _get_plan_dir(plan_id: str, project_name: Optional[str] = None) -> str:
        """Build the disk output directory path for a migration plan."""
        base_dir = os.environ.get("MIGRATION_OUTPUT_DIR", "outputs/migrations")
        short_id = str(plan_id)[:8]
        proj_slug = _slugify(project_name or "unknown")
        return os.path.join(base_dir, f"{short_id}-{proj_slug}")

    def _write_phase_to_disk(
        self,
        plan_id: str,
        phase_number: int,
        phase_type: str,
        output: str,
        output_files: List[Dict],
        mvp_id: Optional[int] = None,
        project_name: Optional[str] = None,
    ) -> Optional[str]:
        """Write phase output to disk.

        Code files go to a unified code/ folder (the actual migrated codebase);
        LLM markdown goes to _plans/ for reference. Returns the plan root
        directory path, or None if nothing was written.
        """
        if not output and not output_files:
            return None

        plan_dir = self._get_plan_dir(plan_id, project_name)

        # Write LLM markdown output to _plans/
        if output:
            plans_dir = os.path.join(plan_dir, "_plans")
            os.makedirs(plans_dir, exist_ok=True)
            if mvp_id is not None:
                md_name = f"mvp-{mvp_id}-phase-{phase_number}-{phase_type}.md"
            else:
                md_name = f"phase-{phase_number}-{phase_type}.md"
            with open(os.path.join(plans_dir, md_name), "w") as f:
                f.write(output)

        # Write generated code files to code/ (unified target project)
        if output_files:
            code_dir = os.path.join(plan_dir, "code")
            for fd in output_files:
                fp = fd.get("file_path", "unknown")
                full_path = os.path.join(code_dir, fp)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(fd.get("content", ""))

        return plan_dir

    def _write_mvp_documents(
        self,
        plan_id: str,
        mvp_id: int,
        project_name: Optional[str] = None,
    ) -> Optional[str]:
        """Write MVP feature documents to disk.

        Creates a per-MVP subfolder with:
          - summary.md — MVP overview (name, description, metrics, files)
          - analysis.md — Deep analysis output (Functional Requirements Register)
          - diagrams/*.puml + *.svg — Cached UML diagrams

        Fire-and-forget: DB is source of truth. Returns MVP dir path or None.
        """
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                return None

            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid,
            ).first()
            if not plan:
                return None

            # Collect all data while session is open
            mvp_data = self._mvp_to_dict(mvp)
            analysis_output = mvp.analysis_output
            diagrams = mvp.diagrams
            target_brief = plan.target_brief
            target_stack = plan.target_stack or {}

            # Get source file paths for this MVP
            file_paths = []
            if mvp_data.get("file_ids"):
                from sqlalchemy import text as sa_text
                uids = [UUID(f) if isinstance(f, str) else f for f in mvp_data["file_ids"]]
                rows = session.execute(sa_text(
                    "SELECT file_path, language FROM code_files WHERE file_id = ANY(:fids)"
                ), {"fids": uids})
                file_paths = [(r.file_path, r.language) for r in rows.fetchall()]

        # Build output directory
        plan_dir = self._get_plan_dir(plan_id, project_name)
        mvp_dir = os.path.join(plan_dir, "_plans", f"mvp-{mvp_id}")
        os.makedirs(mvp_dir, exist_ok=True)

        # 1. Write summary.md
        summary_lines = [
            f"# MVP {mvp_id}: {mvp_data.get('name', 'Unnamed')}",
            "",
            f"**Status**: {mvp_data.get('status', 'unknown')}",
            f"**Priority**: {mvp_data.get('priority', '?')}",
            "",
            f"**Migration Target**: {target_brief}",
            f"**Target Stack**: {', '.join(target_stack.get('languages', []))} / {', '.join(target_stack.get('frameworks', []))}",
            "",
        ]

        if mvp_data.get("description"):
            summary_lines.extend([
                "## Description",
                "",
                mvp_data["description"],
                "",
            ])

        metrics = mvp_data.get("metrics", {})
        if metrics:
            summary_lines.extend([
                "## Metrics",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Size (units) | {metrics.get('size', '?')} |",
                f"| Cohesion | {metrics.get('cohesion', '?')} |",
                f"| Coupling | {metrics.get('coupling', '?')} |",
                f"| Readiness | {metrics.get('readiness', '?')} |",
                "",
            ])

        if file_paths:
            summary_lines.extend([
                "## Source Files",
                "",
                "| File Path | Language |",
                "|-----------|----------|",
            ])
            for fp, lang in sorted(file_paths):
                summary_lines.append(f"| {fp} | {lang or '?'} |")
            summary_lines.append("")

        deps = mvp_data.get("depends_on_mvp_ids", [])
        if deps:
            summary_lines.extend([
                "## Dependencies",
                "",
                f"Depends on MVPs: {', '.join(str(d) for d in deps)}",
                "",
            ])

        with open(os.path.join(mvp_dir, "summary.md"), "w") as f:
            f.write("\n".join(summary_lines))

        # 2. Write analysis.md (deep analysis / functional requirements register)
        if analysis_output:
            analysis_text = ""
            if isinstance(analysis_output, dict):
                analysis_text = analysis_output.get("output", "")
            elif isinstance(analysis_output, str):
                analysis_text = analysis_output

            if analysis_text:
                with open(os.path.join(mvp_dir, "analysis.md"), "w") as f:
                    f.write(f"# Deep Analysis: {mvp_data.get('name', 'MVP')}\n\n")
                    f.write(analysis_text)

        # 3. Write diagrams (PlantUML source + SVG)
        if diagrams and isinstance(diagrams, dict):
            diag_dir = os.path.join(mvp_dir, "diagrams")
            os.makedirs(diag_dir, exist_ok=True)
            for dtype, diag_data in diagrams.items():
                if not isinstance(diag_data, dict):
                    continue
                puml = diag_data.get("puml")
                svg = diag_data.get("svg")
                if puml:
                    with open(os.path.join(diag_dir, f"{dtype}.puml"), "w") as f:
                        f.write(puml)
                if svg:
                    with open(os.path.join(diag_dir, f"{dtype}.svg"), "w") as f:
                        f.write(svg)

        logger.info(
            "MVP %d feature documents written to %s",
            mvp_id, mvp_dir,
        )
        return mvp_dir
