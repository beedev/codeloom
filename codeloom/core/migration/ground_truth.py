"""Codebase ground truth — verified facts for output validation.

Extracts codebase facts from the database once per migration run and
provides validation methods used at every pipeline boundary. This closes
the feedback loop between what the migration engine *knows* about the
codebase and what it *produces*.

The grounding layer addresses a systemic gap: the engine has rich
infrastructure for reading codebase facts (pattern detectors, context
builders, edge statistics) but previously never used them to validate
outputs. This module provides that validation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class ClusterIssue:
    """An issue found during post-clustering validation."""
    cluster_index: int
    issue_type: str          # self_ref | mixed_language | sub_minimum | circular_dep
    severity: str            # error | warning
    message: str
    auto_fixed: bool = False


@dataclass
class OutputIssue:
    """An issue found during phase output validation."""
    phase: str
    issue_type: str          # ungrounded_tech | hallucinated_entity | missing_layer
    severity: str            # error | warning
    message: str
    evidence: str = ""


@dataclass
class UnitFact:
    """Verified facts about a single code unit."""
    unit_id: str
    name: str
    qualified_name: str
    unit_type: str
    language: str
    file_id: str
    file_path: str


# ── Framework-Language Compatibility Map ─────────────────────────────

_FW_LANGS = {
    "express": {"javascript", "typescript"},
    "express.js": {"javascript", "typescript"},
    "koa": {"javascript", "typescript"},
    "fastify": {"javascript", "typescript"},
    "next.js": {"javascript", "typescript"},
    "nestjs": {"javascript", "typescript"},
    "spring": {"java", "kotlin"},
    "spring boot": {"java", "kotlin"},
    "django": {"python"},
    "flask": {"python"},
    "fastapi": {"python"},
    "asp.net": {"csharp", "c#"},
    ".net": {"csharp", "c#"},
    "rails": {"ruby"},
    "gin": {"go"},
}


# ── Ground Truth ────────────────────────────────────────────────────

class CodebaseGroundTruth:
    """Codebase facts extracted from DB for output validation.

    Instantiated once per migration run. Provides grounding data and
    validation methods used at every pipeline boundary.

    Usage:
        gt = CodebaseGroundTruth(db_manager, project_id)
        issues = gt.validate_clusters(clusters)
        context = gt.build_src_gap_context(unit_ids, existing_ids)
        summary = gt.format_layer_summary()
    """

    def __init__(self, db: DatabaseManager, project_id: str):
        self._db = db
        self._pid = UUID(project_id) if isinstance(project_id, str) else project_id

        # One-time extraction from DB
        self.languages: Dict[str, Dict[str, int]] = {}
        self.layers: Dict[str, Optional[str]] = {}
        self.units_by_id: Dict[str, UnitFact] = {}
        self.file_paths: Dict[str, str] = {}        # file_id → file_path
        self.patterns: Dict[str, str] = {}

        self._extract()

    # ── Extraction (one-time) ───────────────────────────────────────

    def _extract(self) -> None:
        """Extract all facts from DB. Called once at construction."""
        self._load_languages()
        self._load_units()
        self._load_file_paths()
        self._detect_patterns()
        self._detect_layers()
        logger.info(
            "Ground truth extracted: %d languages, %d units, %d files, layers=%s",
            len(self.languages), len(self.units_by_id),
            len(self.file_paths), self.layers,
        )

    def _load_languages(self) -> None:
        """Load language distribution from code_files and code_units."""
        with self._db.get_session() as session:
            file_langs = session.execute(text("""
                SELECT language, COUNT(*) as cnt
                FROM code_files
                WHERE project_id = :pid AND language IS NOT NULL
                GROUP BY language
            """), {"pid": self._pid}).fetchall()

            unit_langs = session.execute(text("""
                SELECT language, COUNT(*) as cnt
                FROM code_units
                WHERE project_id = :pid AND language IS NOT NULL
                GROUP BY language
            """), {"pid": self._pid}).fetchall()

        file_map = {r.language: r.cnt for r in file_langs}
        unit_map = {r.language: r.cnt for r in unit_langs}

        all_langs = set(file_map) | set(unit_map)
        self.languages = {
            lang: {"files": file_map.get(lang, 0), "units": unit_map.get(lang, 0)}
            for lang in all_langs
        }

    def _load_units(self) -> None:
        """Load unit index: unit_id → UnitFact."""
        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
                       u.language, u.file_id,
                       f.file_path
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.project_id = :pid
            """), {"pid": self._pid}).fetchall()

        for r in rows:
            uid = str(r.unit_id)
            self.units_by_id[uid] = UnitFact(
                unit_id=uid,
                name=r.name or "",
                qualified_name=r.qualified_name or r.name or "",
                unit_type=r.unit_type or "unknown",
                language=r.language or "unknown",
                file_id=str(r.file_id),
                file_path=r.file_path or "",
            )

    def _load_file_paths(self) -> None:
        """Load file_id → file_path mapping."""
        with self._db.get_session() as session:
            rows = session.execute(text("""
                SELECT file_id, file_path
                FROM code_files
                WHERE project_id = :pid
            """), {"pid": self._pid}).fetchall()

        self.file_paths = {str(r.file_id): r.file_path for r in rows}

    def _detect_patterns(self) -> None:
        """Detect source code patterns (DI, data layer, web, config, test).

        Reuses the same detection logic as MigrationContextBuilder but
        stores results for validation, not just prompt context.
        """
        annotations = self._extract_annotations()

        self.patterns["di"] = self._detect_di(annotations)
        self.patterns["data"] = self._detect_data(annotations)
        self.patterns["web"] = self._detect_web(annotations)
        self.patterns["config"] = self._detect_config(annotations)

    def _detect_layers(self) -> None:
        """Derive infrastructure layer presence from detected patterns."""
        self.layers = {
            "api": self.patterns.get("web") if self.patterns.get("web") != "unknown" else None,
            "database": self.patterns.get("data") if self.patterns.get("data") != "unknown" else None,
            "config": self.patterns.get("config") if self.patterns.get("config") != "unknown" else None,
        }

    def _extract_annotations(self) -> Dict[str, int]:
        """Count annotation/decorator occurrences across all code units."""
        with self._db.get_session() as session:
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

    def _detect_di(self, annotations: Dict[str, int]) -> str:
        autowired = annotations.get("@Autowired", 0) + annotations.get("Autowired", 0)
        inject = annotations.get("@Inject", 0) + annotations.get("Inject", 0)
        if autowired > 0:
            return "field_injection"
        if inject > 0:
            return "inject_annotation"
        if annotations.get("@app.route", 0) > 0:
            return "flask_di"
        return "unknown"

    def _detect_data(self, annotations: Dict[str, int]) -> str:
        if annotations.get("@Entity", 0) > 0 or annotations.get("Entity", 0) > 0:
            if annotations.get("@Repository", 0) > 0:
                return "spring_data_jpa"
            return "jpa"
        if annotations.get("@Table", 0) > 0:
            return "jpa"
        # Check for ORM naming patterns
        orm_units = sum(
            1 for u in self.units_by_id.values()
            if any(kw in u.name.lower() for kw in ("repository", "dao"))
            or any(kw in u.qualified_name.lower() for kw in (".models.", ".entities."))
        )
        if orm_units > 3:
            return "orm_repository_pattern"
        return "unknown"

    def _detect_web(self, annotations: Dict[str, int]) -> str:
        if any(annotations.get(a, 0) > 0 for a in (
            "@RestController", "RestController", "@Controller",
            "@GetMapping", "@PostMapping",
        )):
            return "spring_mvc"
        # Check for Express/Koa patterns in unit signatures
        express_units = sum(
            1 for u in self.units_by_id.values()
            if any(kw in u.qualified_name.lower() for kw in ("router.get", "router.post", "app.get(", "app.post("))
        )
        if express_units > 0:
            return "express_or_koa"
        if annotations.get("@app.route", 0) > 0:
            return "flask"
        if any(annotations.get(a, 0) > 0 for a in ("@router.get", "@router.post")):
            return "fastapi"
        return "unknown"

    def _detect_config(self, annotations: Dict[str, int]) -> str:
        if any(annotations.get(a, 0) > 0 for a in ("@Value", "@Configuration", "@ConfigurationProperties")):
            return "spring_properties"
        config_files = sum(
            1 for fp in self.file_paths.values()
            if any(kw in fp.lower() for kw in (
                "application.properties", "application.yml", ".env",
                "config.yaml", "settings.py", "appsettings.json",
            ))
        )
        if config_files > 0:
            return "file_based_config"
        return "unknown"

    # ── Cluster Validation (Step 2) ─────────────────────────────────

    def validate_clusters(self, clusters: List[Dict[str, Any]]) -> List[ClusterIssue]:
        """Validate clustering output against codebase ground truth.

        Checks:
        - Self-referencing dependencies
        - Mixed-language clusters
        - Sub-minimum size clusters with no merge target

        Returns list of issues found (some may be auto-fixed).
        """
        issues: List[ClusterIssue] = []

        for i, cluster in enumerate(clusters):
            # Check self-referencing dependencies
            deps = cluster.get("depends_on", [])
            priority = cluster.get("priority", i)
            if priority in deps or i in deps:
                issues.append(ClusterIssue(
                    cluster_index=i,
                    issue_type="self_ref",
                    severity="error",
                    message=f"Cluster {i} ({cluster.get('name', '?')}) references itself in depends_on",
                ))

            # Check language homogeneity
            unit_ids = cluster.get("unit_ids", [])
            langs = set()
            for uid in unit_ids:
                u = self.units_by_id.get(uid)
                if u:
                    langs.add(u.language)
            if len(langs) > 1:
                issues.append(ClusterIssue(
                    cluster_index=i,
                    issue_type="mixed_language",
                    severity="warning",
                    message=(
                        f"Cluster {i} ({cluster.get('name', '?')}) contains "
                        f"{len(langs)} languages: {sorted(langs)}"
                    ),
                ))

            # Check sub-minimum size
            size = cluster.get("metrics", {}).get("size", len(unit_ids))
            if size < 3 and len(clusters) > 1:
                issues.append(ClusterIssue(
                    cluster_index=i,
                    issue_type="sub_minimum",
                    severity="warning",
                    message=(
                        f"Cluster {i} ({cluster.get('name', '?')}) has only "
                        f"{size} units (minimum recommended: 3)"
                    ),
                ))

        if issues:
            logger.warning(
                "Cluster validation found %d issues: %s",
                len(issues),
                ", ".join(f"{iss.issue_type}@{iss.cluster_index}" for iss in issues),
            )

        return issues

    def fix_self_references(self, clusters: List[Dict[str, Any]]) -> int:
        """Remove self-referencing dependencies from clusters. Returns count fixed."""
        fixed = 0
        for i, cluster in enumerate(clusters):
            deps = cluster.get("depends_on", [])
            priority = cluster.get("priority", i)
            clean = [d for d in deps if d != priority and d != i]
            if len(clean) < len(deps):
                cluster["depends_on"] = clean
                fixed += 1
                logger.info(
                    "Fixed self-reference in cluster %d (%s): removed %d self-deps",
                    i, cluster.get("name", "?"), len(deps) - len(clean),
                )
        return fixed

    def split_mixed_language_clusters(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split clusters containing multiple languages into per-language sub-clusters."""
        result: List[Dict[str, Any]] = []

        for cluster in clusters:
            unit_ids = cluster.get("unit_ids", [])

            # Group units by language
            by_lang: Dict[str, List[str]] = {}
            for uid in unit_ids:
                u = self.units_by_id.get(uid)
                lang = u.language if u else "unknown"
                by_lang.setdefault(lang, []).append(uid)

            if len(by_lang) <= 1:
                result.append(cluster)
                continue

            # Split into per-language sub-clusters
            logger.info(
                "Splitting mixed-language cluster '%s' into %d sub-clusters: %s",
                cluster.get("name", "?"), len(by_lang), list(by_lang.keys()),
            )
            for lang, lang_uids in by_lang.items():
                sub = {
                    **cluster,
                    "unit_ids": lang_uids,
                    "file_ids": list({
                        self.units_by_id[uid].file_id
                        for uid in lang_uids
                        if uid in self.units_by_id
                    }),
                    "name": f"{cluster.get('name', 'Unnamed')} ({lang})",
                    "units": [
                        u for u in cluster.get("units", [])
                        if u.get("unit_id") in set(lang_uids)
                    ],
                    "metrics": {},  # Will be recomputed by caller
                }
                result.append(sub)

        return result

    def rescue_orphan_clusters(
        self, clusters: List[Dict[str, Any]], min_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """Attempt to merge sub-minimum clusters into nearest neighbor by directory.

        Clusters with no suitable neighbor are flagged with
        metrics["suggested_action"] = "review_for_drop".
        """
        if len(clusters) <= 1:
            return clusters

        result = list(clusters)

        for i in range(len(result) - 1, -1, -1):
            size = result[i].get("metrics", {}).get("size", len(result[i].get("unit_ids", [])))
            if size >= min_size:
                continue

            # Find nearest cluster by shared directory prefix
            orphan_dirs = self._get_cluster_dirs(result[i])
            best_target = None
            best_overlap = 0

            for j, candidate in enumerate(result):
                if j == i:
                    continue
                candidate_dirs = self._get_cluster_dirs(candidate)
                overlap = len(orphan_dirs & candidate_dirs)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_target = j

            if best_target is not None and best_overlap > 0:
                # Merge into target
                target = result[best_target]
                orphan = result[i]
                target["unit_ids"].extend(orphan.get("unit_ids", []))
                target["file_ids"] = list(set(target.get("file_ids", [])) | set(orphan.get("file_ids", [])))
                target["units"].extend(orphan.get("units", []))
                result.pop(i)
                logger.info(
                    "Merged orphan cluster '%s' (%d units) into '%s' (shared dirs: %d)",
                    orphan.get("name", "?"), size,
                    target.get("name", "?"), best_overlap,
                )
            else:
                # Flag for human review
                result[i].setdefault("metrics", {})["suggested_action"] = "review_for_drop"
                logger.info(
                    "Flagged orphan cluster '%s' (%d units) for review — no suitable merge target",
                    result[i].get("name", "?"), size,
                )

        return result

    def _get_cluster_dirs(self, cluster: Dict[str, Any]) -> Set[str]:
        """Get the set of parent directories for a cluster's files."""
        dirs = set()
        for uid in cluster.get("unit_ids", []):
            u = self.units_by_id.get(uid)
            if u and u.file_path:
                # Get parent directory
                parts = u.file_path.rsplit("/", 1)
                if len(parts) > 1:
                    dirs.add(parts[0])
        return dirs

    def reconcile_file_ids(self, clusters: List[Dict[str, Any]]) -> int:
        """Recalculate ALL clusters' file_ids from unit_ids. Enforce exclusive ownership.

        1. Derive file_ids from unit_ids for every cluster (not just empty ones)
        2. For files appearing in multiple clusters, assign to the cluster with
           the most units from that file (break ties by cluster index)
        3. Returns count of clusters modified

        Replaces derive_file_ids() which only filled empty lists.
        """
        # Pass 1: compute file→unit_count per cluster
        cluster_files: List[Dict[str, int]] = []  # [{file_id: unit_count}, ...]
        for cluster in clusters:
            file_counts: Dict[str, int] = {}
            for uid in cluster.get("unit_ids", []):
                u = self.units_by_id.get(uid)
                if u:
                    file_counts[u.file_id] = file_counts.get(u.file_id, 0) + 1
            cluster_files.append(file_counts)

        # Pass 2: for contested files, assign to the cluster with highest unit count
        file_owner: Dict[str, int] = {}  # file_id → winning cluster index
        for cidx, file_counts in enumerate(cluster_files):
            for fid, count in file_counts.items():
                if fid not in file_owner:
                    file_owner[fid] = cidx
                else:
                    prev_idx = file_owner[fid]
                    prev_count = cluster_files[prev_idx].get(fid, 0)
                    if count > prev_count:
                        file_owner[fid] = cidx

        # Pass 3: rewrite each cluster's file_ids to the exclusively-assigned set
        modified = 0
        for cidx, cluster in enumerate(clusters):
            new_file_ids = sorted(
                fid for fid, owner in file_owner.items() if owner == cidx
            )
            old_file_ids = sorted(cluster.get("file_ids", []))
            if new_file_ids != old_file_ids:
                cluster["file_ids"] = new_file_ids
                modified += 1
                if old_file_ids:
                    logger.info(
                        "Reconciled file_ids for cluster '%s': %d → %d files",
                        cluster.get("name", "?"), len(old_file_ids), len(new_file_ids),
                    )
                else:
                    logger.info(
                        "Derived %d file_ids for cluster '%s'",
                        len(new_file_ids), cluster.get("name", "?"),
                    )

        return modified

    def format_language_guidance(self, target_stack: Dict[str, Any]) -> str:
        """Format language-aware migration guidance from ground truth.

        Compares project languages against target stack to identify which
        languages should be converted vs. preserved. Prevents impossible
        migrations like "Python → Express.js".

        Args:
            target_stack: Plan's target stack dict with 'languages' and 'frameworks'.

        Returns:
            Markdown section for prompt injection.
        """
        target_langs = set(
            lang.lower() for lang in (target_stack.get("languages") or [])
        )
        target_fws = set(
            fw.lower() for fw in (target_stack.get("frameworks") or [])
        )

        # Derive target-compatible languages from frameworks
        compatible_langs = set(target_langs)
        for fw in target_fws:
            fw_lower = fw.lower()
            for pattern, langs in _FW_LANGS.items():
                if pattern in fw_lower:
                    compatible_langs |= langs

        # Classify project languages
        source_langs = sorted(self.languages.keys())
        if not source_langs or not compatible_langs:
            return ""

        convertible = []
        preserve = []
        for lang in source_langs:
            stats = self.languages[lang]
            count_str = f"{stats.get('files', 0)} files, {stats.get('units', 0)} units"
            if lang.lower() in compatible_langs:
                convertible.append(f"- **{lang}** ({count_str}): compatible with target — MIGRATE")
            else:
                preserve.append(f"- **{lang}** ({count_str}): NOT compatible with target — KEEP AS-IS or EXCLUDE")

        if not preserve:
            return ""  # All languages are compatible, no guidance needed

        lines = ["### Source Language Guidance (from Codebase Analysis)\n"]
        if convertible:
            lines.append("**Languages to migrate** (compatible with target framework):")
            lines.extend(convertible)
        lines.append("")
        lines.append("**Languages to preserve** (NOT compatible with target framework):")
        lines.extend(preserve)
        lines.append("")
        lines.append(
            "CRITICAL: Do NOT map preserved languages to the target framework. "
            "Files in preserved languages should be kept as-is, excluded from migration, "
            "or placed in a separate 'keep' MVP. Do NOT generate Express.js/Spring/etc. "
            "equivalents for code written in incompatible languages."
        )
        return "\n".join(lines)

    def format_language_summary(self) -> str:
        """Format language distribution as a prompt-friendly summary.

        Placed early in codebase_context so the LLM sees correct language
        stats before any other project metadata.
        """
        if not self.languages:
            return ""
        total_files = sum(v["files"] for v in self.languages.values())
        if total_files == 0:
            return ""
        sorted_langs = sorted(
            self.languages.items(), key=lambda x: x[1]["files"], reverse=True,
        )
        primary = sorted_langs[0][0]
        pct = sorted_langs[0][1]["files"] / total_files * 100
        lines = ["### Codebase Language Distribution (Verified)\n"]
        lines.append(f"**Primary language: {primary}** ({pct:.0f}% of files)\n")
        for lang, stats in sorted_langs:
            lines.append(f"- {lang}: {stats['files']} files, {stats['units']} units")
        return "\n".join(lines)

    def get_mvp_dominant_language(self, unit_ids: List[str]) -> Optional[str]:
        """Get the dominant language for an MVP's units."""
        lang_counts: Dict[str, int] = {}
        for uid in unit_ids:
            u = self.units_by_id.get(uid)
            if u:
                lang_counts[u.language] = lang_counts.get(u.language, 0) + 1
        if not lang_counts:
            return None
        return max(lang_counts, key=lang_counts.get)

    def is_language_compatible(self, language: str, target_stack: Dict[str, Any]) -> bool:
        """Check if a language is compatible with the plan's target frameworks."""
        target_langs = {
            lang.lower() for lang in (target_stack.get("languages") or [])
        }
        target_fws = {
            fw.lower() for fw in (target_stack.get("frameworks") or [])
        }
        compatible = set(target_langs)
        for fw in target_fws:
            for pattern, langs in _FW_LANGS.items():
                if pattern in fw:
                    compatible |= langs
        return language.lower() in compatible

    # ── SRC Grounding (Step 3) ──────────────────────────────────────

    def build_src_gap_context(
        self,
        mvp_unit_ids: List[str],
        existing_register_ids: Set[str],
    ) -> str:
        """Build grounded context for SRC gap-filling.

        Returns a compact listing of actual unit names, types, and files
        from verified DB facts — not from LLM output. Used to prevent
        hallucination in SRC iteration prompts.

        Args:
            mvp_unit_ids: Unit IDs belonging to this MVP.
            existing_register_ids: Already-covered register IDs (e.g. {"BR-1", "DE-1"}).

        Returns:
            Formatted string with unit listing + existing IDs.
        """
        lines = [
            "## Actual Code Units in This MVP (from codebase — use ONLY these)\n",
        ]

        # Group by file for readability
        by_file: Dict[str, List[UnitFact]] = {}
        for uid in mvp_unit_ids:
            u = self.units_by_id.get(uid)
            if u:
                by_file.setdefault(u.file_path, []).append(u)

        count = 0
        for file_path in sorted(by_file.keys()):
            units = by_file[file_path]
            lines.append(f"### `{file_path}`")
            for u in sorted(units, key=lambda x: x.name):
                lines.append(f"- **{u.qualified_name}** ({u.unit_type}, {u.language})")
                count += 1
                if count >= 100:
                    lines.append(f"\n[... {len(mvp_unit_ids) - count} more units truncated]")
                    break
            if count >= 100:
                break

        if existing_register_ids:
            existing_sorted = ", ".join(sorted(existing_register_ids))
            lines.append(f"\n## Already Covered Register IDs\n{existing_sorted}")

        return "\n".join(lines)

    # ── Phase Output Validation (Step 4) ────────────────────────────

    def format_layer_summary(self) -> str:
        """Format detected infrastructure layers for prompt injection.

        Returns a markdown section that constrains LLM tech recommendations
        to layers that actually exist in the source codebase.
        """
        layer_names = {
            "api": "API / Web Framework",
            "database": "Database / ORM",
            "config": "Configuration",
        }

        # Separate detected vs absent layers for emphasis
        detected_layers = []
        absent_layers = []
        for key, label in layer_names.items():
            detected = self.layers.get(key)
            if detected:
                detected_layers.append(f"- **{label}**: {detected} (detected)")
            else:
                absent_layers.append(label)

        lines = ["### Source Infrastructure Layers (Verified from Codebase Analysis)\n"]
        lines.extend(detected_layers)

        if absent_layers:
            absent_str = ", ".join(absent_layers)
            lines.append(f"\n**NOT PRESENT in source codebase**: {absent_str}")
            lines.append("")
            lines.append(
                f"CRITICAL CONSTRAINT: The source codebase has NO {absent_str}. "
                f"Your output MUST NOT include:"
            )
            if "Database / ORM" in absent_layers:
                lines.append(
                    "- Any database ORM (TypeORM, Sequelize, Prisma, Hibernate, JPA, etc.)")
                lines.append(
                    "- Any database migration tools or database connection configuration")
                lines.append(
                    "- Any 'Data Layer Mapping' or 'Database' sections in the architecture")
                lines.append(
                    "- Any 'Raw SQL/ORM' references — there is NO SQL in this project")
            if "API / Web Framework" in absent_layers:
                lines.append(
                    "- Any web framework recommendations (Express, Koa, Spring, etc.)")
            lines.append("")
            lines.append(
                "If a layer does not exist in the source, it does not need a target equivalent. "
                "Omit it entirely from your mapping."
            )
        else:
            lines.append("")
            lines.append(
                "All detected layers should be mapped to appropriate target equivalents."
            )

        return "\n".join(lines)

    def validate_phase_output(
        self,
        phase_type: str,
        output: str,
    ) -> List[OutputIssue]:
        """Spot-check phase output against ground truth.

        Advisory warnings only — logged but not blocking.
        """
        issues: List[OutputIssue] = []

        if phase_type in ("architecture", "discovery"):
            issues.extend(self._check_ungrounded_tech(output))

        if phase_type in ("analyze", "design"):
            issues.extend(self._check_hallucinated_entities(output))

        return issues

    def _check_ungrounded_tech(self, output: str) -> List[OutputIssue]:
        """Check if output recommends technologies for non-existent layers."""
        issues = []
        output_lower = output.lower()

        # If no database detected, flag ORM and SQL recommendations
        if self.layers.get("database") is None:
            orm_keywords = [
                "typeorm", "sequelize", "prisma", "hibernate", "jpa",
                "sqlalchemy", "django orm", "knex", "drizzle", "mikro-orm",
                "raw sql", "raw sql/orm", "database migration",
                "database connection", "connection pool",
            ]
            for kw in orm_keywords:
                if kw in output_lower:
                    issues.append(OutputIssue(
                        phase="architecture",
                        issue_type="ungrounded_tech",
                        severity="warning",
                        message=f"Output recommends '{kw}' but no database layer was detected in the source codebase",
                        evidence=f"layers.database = {self.layers.get('database')}",
                    ))
                    break  # One warning per category is enough

        return issues

    _QUALIFIED_NAME_RE = re.compile(r"`([A-Z][a-zA-Z0-9]+(?:\.[a-zA-Z][a-zA-Z0-9]+)+)`")

    def _check_hallucinated_entities(self, output: str) -> List[OutputIssue]:
        """Check if output references class names not in the codebase."""
        issues = []

        # Extract qualified names from backtick-quoted code references
        refs = self._QUALIFIED_NAME_RE.findall(output)
        known_names = {u.qualified_name for u in self.units_by_id.values()}
        known_simple = {u.name for u in self.units_by_id.values()}

        for ref in refs:
            # Check if any part of the ref matches a known unit
            parts = ref.split(".")
            base_name = parts[0]
            if (ref not in known_names
                    and base_name not in known_simple
                    and not any(ref in qn for qn in known_names)):
                issues.append(OutputIssue(
                    phase="analyze",
                    issue_type="hallucinated_entity",
                    severity="warning",
                    message=f"Output references '{ref}' which is not a known code unit",
                    evidence=f"Not found in {len(known_names)} known units",
                ))

        # Cap warnings to avoid noise
        if len(issues) > 10:
            count = len(issues)
            issues = issues[:10]
            issues.append(OutputIssue(
                phase="analyze",
                issue_type="hallucinated_entity",
                severity="warning",
                message=f"... and {count - 10} more hallucinated entity references",
            ))

        return issues
