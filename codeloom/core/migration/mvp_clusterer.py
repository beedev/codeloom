"""MVP Clustering — groups code units into functional vertical slices.

Primary algorithm (RAPTOR-driven semantic clustering):
1. Load RAPTOR L1 summaries (per-file semantic summaries with embeddings)
2. Cluster L1 summaries using UMAP+GMM → L2 functional groups (~15-25 MVPs)
3. Assign orphan files (no L1 summary) to nearest L2 cluster by embedding similarity
4. Map L2 clusters back to code units via source_id → file_id → code_units
5. Compute cohesion/coupling metrics from ASG edges
6. Attach SP references, rank by migration readiness

Fallback algorithm (package-based, used when no RAPTOR tree exists):
1. Seed clusters from package/namespace structure
2. Compute cohesion/coupling, merge/split mechanically
3. Attach SP references, rank

Input:  Project with parsed code units, ASG edges, and (ideally) RAPTOR tree
Output: List of MVP candidate dicts (not yet persisted)
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)

# Fallback thresholds (used only when RAPTOR tree is unavailable)
_COHESION_MERGE_THRESHOLD = 0.3
_COUPLING_SPLIT_THRESHOLD = 0.7
_MIN_CLUSTER_SIZE = 3
_MAX_CLUSTER_SIZE = 120
_SHARED_PACKAGE_KEYWORDS = frozenset({
    "shared", "common", "util", "utils", "helper", "helpers",
    "base", "core", "infrastructure", "framework", "lib",
})


class MvpClusterer:
    """Groups code units into functional MVPs for incremental migration."""

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    # ── Public API ──────────────────────────────────────────────────

    def cluster(
        self,
        project_id: str,
        params: Optional[Dict[str, Any]] = None,
        asset_strategies: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """Run the full clustering pipeline.

        Tries RAPTOR-driven semantic clustering first. Falls back to
        package-based clustering if no RAPTOR tree exists for the project.

        When asset_strategies is provided, only languages with active strategies
        (version_upgrade, framework_migration, rewrite) are clustered. Passive
        languages (keep_as_is, convert) get synthetic MVPs. Languages marked
        no_change are excluded entirely.

        Returns:
            Dict with 'mvps', 'shared_concerns', and 'sp_analysis'.
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        # Partition languages by strategy (supports sub-type granularity)
        active_langs = None   # None = all languages (backward compat)
        passive_langs = []
        # Track sub-type level overrides: {lang: {unit_type: strategy}}
        _sub_type_strategies: Dict[str, Dict[str, str]] = {}
        if asset_strategies:
            _active = {"version_upgrade", "framework_migration", "rewrite", "convert"}
            _passive = {"keep_as_is"}

            active_langs = []
            for lang, s in asset_strategies.items():
                sub_types = s.get("sub_types")
                if sub_types:
                    # Language has sub-type level strategies — include in active
                    # if ANY sub-type has an active strategy
                    has_active = any(
                        st.get("strategy", s.get("strategy", "")) in _active
                        for st in sub_types.values()
                    )
                    if has_active:
                        active_langs.append(lang)
                    # Also check if any sub-types are passive (for synthetic MVPs later)
                    has_passive = any(
                        st.get("strategy") in _passive
                        for st in sub_types.values()
                    )
                    if has_passive:
                        passive_langs.append(lang)
                    # Store per-sub-type strategies for unit-level filtering
                    _sub_type_strategies[lang] = {
                        ut: st.get("strategy", s.get("strategy", ""))
                        for ut, st in sub_types.items()
                    }
                elif s.get("strategy") in _active:
                    active_langs.append(lang)
                elif s.get("strategy") in _passive:
                    passive_langs.append(lang)
                # no_change languages are excluded entirely

        # Load units and edges (needed for both paths)
        units = self._load_units(pid, languages=active_langs)

        # Filter out units whose sub-type strategy is passive or no_change
        if _sub_type_strategies:
            _passive_strats = {"keep_as_is", "no_change"}
            units = [
                u for u in units
                if u["language"] not in _sub_type_strategies
                or _sub_type_strategies[u["language"]].get(
                    u.get("unit_type", ""), "convert"
                ) not in _passive_strats
            ]

        edges = self._load_edges(pid)

        if not units:
            return {
                "mvps": [],
                "shared_concerns": [],
                "sp_analysis": {"total_sps": 0, "sps_with_callers": 0, "orphan_sps": 0},
            }

        edge_lookup = self._build_edge_lookup(edges)

        # Try RAPTOR-driven clustering first
        raptor_summaries = self._load_raptor_l1_summaries(pid)
        if raptor_summaries:
            logger.info(
                "Using RAPTOR-driven semantic clustering (%d L1 summaries)",
                len(raptor_summaries),
            )
            clusters = self._cluster_raptor(raptor_summaries, units, edge_lookup)
        else:
            logger.info("No RAPTOR tree found, falling back to package-based clustering")
            p = params or {}
            clusters = self._cluster_package_based(units, edges, edge_lookup, p)

        # Common post-processing: metrics, SP refs, ranking
        for c in clusters:
            self._compute_metrics(c, edge_lookup)

        sp_edges = [e for e in edges if e["edge_type"] == "calls_sp"]
        sp_units = {u["unit_id"]: u for u in units if u["unit_type"] in ("stored_procedure", "sql_function")}
        self._attach_sp_references(clusters, sp_edges, sp_units)
        self._rank_clusters(clusters)
        initial_cluster_count = len(clusters)

        # Compute inter-MVP dependencies and reorder via topological sort
        self._compute_mvp_dependencies(clusters, edges)
        clusters = self._topological_sort_with_readiness(clusters)

        # Name the clusters
        for i, cluster in enumerate(clusters):
            if not cluster.get("name"):
                cluster["name"] = self._generate_name(cluster)
            cluster["priority"] = i + 1  # Reserve 0 for Foundation MVP

        # Create synthetic MVPs for passive languages/sub-types (keep_as_is)
        passive_count = 0
        if passive_langs:
            passive_units = self._load_units(pid, languages=passive_langs)

            # For languages with sub-type strategies, filter to only passive units
            if _sub_type_strategies:
                _passive_strats = {"keep_as_is"}
                filtered = []
                for u in passive_units:
                    lang = u["language"]
                    if lang in _sub_type_strategies:
                        ut = u.get("unit_type", "")
                        if _sub_type_strategies[lang].get(ut, "") in _passive_strats:
                            filtered.append(u)
                    else:
                        filtered.append(u)
                passive_units = filtered

            by_lang: Dict[str, List[Dict]] = defaultdict(list)
            for u in passive_units:
                by_lang[u["language"]].append(u)

            for lang in passive_langs:
                lang_units = by_lang.get(lang, [])
                if not lang_units:
                    continue
                strategy = asset_strategies[lang].get("strategy", "keep_as_is")
                target = asset_strategies[lang].get("target")
                label = strategy.replace("_", " ").title()
                name = f"{lang.title()} — {label}"
                if target:
                    name += f" → {target}"
                clusters.append({
                    "name": name,
                    "unit_ids": [u["unit_id"] for u in lang_units],
                    "file_ids": list({u["file_id"] for u in lang_units}),
                    "strategy": strategy,
                    "metrics": {
                        "size": len(lang_units),
                        "cohesion": 1.0,
                        "coupling": 0.0,
                    },
                    "priority": len(clusters) + 1,  # Reserve 0 for Foundation
                })
                passive_count += 1

        logger.info(
            "Final MVP count: %d (clustering: %d, passive: %d)",
            len(clusters),
            initial_cluster_count,
            passive_count,
        )

        # SP analysis summary
        all_sp_ids = set(sp_units.keys())
        sps_with_callers = {e["target_unit_id"] for e in sp_edges}
        orphan_sps = all_sp_ids - sps_with_callers

        return {
            "mvps": clusters,
            "shared_concerns": [],
            "sp_analysis": {
                "total_sps": len(all_sp_ids),
                "sps_with_callers": len(sps_with_callers),
                "orphan_sps": len(orphan_sps),
                "orphan_sp_names": [sp_units[sid]["name"] for sid in orphan_sps],
            },
        }

    # ── RAPTOR-Driven Clustering ────────────────────────────────────

    def _load_raptor_l1_summaries(self, project_id: UUID) -> List[Dict[str, Any]]:
        """Load RAPTOR level-1 summary nodes with their embeddings.

        Each L1 node is a semantic summary of code chunks from a single source file,
        created by RAPTOR's UMAP+GMM clustering + LLM summarization during ingestion.
        """
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT id, text, embedding, metadata_,
                           metadata_->>'source_id' as source_id
                    FROM data_embeddings
                    WHERE metadata_->>'project_id' = :pid
                    AND metadata_->>'tree_level' = '1'
                """),
                {"pid": str(project_id)},
            )
            rows = result.fetchall()

        summaries = []
        for r in rows:
            emb = r.embedding
            if emb is None:
                continue
            # Parse embedding from pgvector format if needed
            if isinstance(emb, str):
                emb = np.array([float(x) for x in emb.strip("[]").split(",")])
            elif isinstance(emb, (list, tuple)):
                emb = np.array(emb, dtype=np.float32)
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)

            summaries.append({
                "node_id": str(r.id),
                "text": r.text,
                "embedding": emb,
                "source_id": r.source_id,
                "metadata": r.metadata_ if r.metadata_ else {},
            })

        logger.info("Loaded %d RAPTOR L1 summaries for project %s", len(summaries), project_id)
        return summaries

    def _load_orphan_file_embeddings(
        self,
        project_id: UUID,
        covered_source_ids: set,
    ) -> List[Dict[str, Any]]:
        """Load average embeddings for files that have no L1 summary.

        For each orphan file, averages the L0 chunk embeddings to produce
        a single representative vector that can be compared to L2 cluster centroids.
        """
        with self._db.get_session() as session:
            # Get all source_ids with L0 chunks
            result = session.execute(
                text("""
                    SELECT metadata_->>'source_id' as source_id,
                           AVG(embedding) as avg_embedding
                    FROM data_embeddings
                    WHERE metadata_->>'project_id' = :pid
                    AND (metadata_->>'tree_level' = '0' OR metadata_->>'tree_level' IS NULL)
                    GROUP BY metadata_->>'source_id'
                """),
                {"pid": str(project_id)},
            )
            rows = result.fetchall()

        # pgvector AVG may not work directly — fall back to manual averaging
        # if the above returns NULLs for avg_embedding
        orphans = []
        needs_manual = False
        for r in rows:
            sid = r.source_id
            if not sid or sid in covered_source_ids:
                continue
            if r.avg_embedding is None:
                needs_manual = True
                break
            emb = r.avg_embedding
            if isinstance(emb, str):
                emb = np.array([float(x) for x in emb.strip("[]").split(",")])
            elif isinstance(emb, (list, tuple)):
                emb = np.array(emb, dtype=np.float32)
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            orphans.append({"source_id": sid, "embedding": emb})

        if needs_manual:
            orphans = self._load_orphan_embeddings_manual(project_id, covered_source_ids)

        logger.info(
            "Loaded %d orphan file embeddings (files without L1 summaries)",
            len(orphans),
        )
        return orphans

    def _load_orphan_embeddings_manual(
        self,
        project_id: UUID,
        covered_source_ids: set,
    ) -> List[Dict[str, Any]]:
        """Manual fallback: load all L0 embeddings and average per source_id."""
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT metadata_->>'source_id' as source_id, embedding
                    FROM data_embeddings
                    WHERE metadata_->>'project_id' = :pid
                    AND (metadata_->>'tree_level' = '0' OR metadata_->>'tree_level' IS NULL)
                """),
                {"pid": str(project_id)},
            )
            rows = result.fetchall()

        # Group embeddings by source_id
        source_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        for r in rows:
            sid = r.source_id
            if not sid or sid in covered_source_ids:
                continue
            emb = r.embedding
            if emb is None:
                continue
            if isinstance(emb, str):
                emb = np.array([float(x) for x in emb.strip("[]").split(",")])
            elif isinstance(emb, (list, tuple)):
                emb = np.array(emb, dtype=np.float32)
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            source_embeddings[sid].append(emb)

        orphans = []
        for sid, embs in source_embeddings.items():
            if embs:
                avg = np.mean(embs, axis=0)
                orphans.append({"source_id": sid, "embedding": avg})

        return orphans

    def _cluster_raptor(
        self,
        summaries: List[Dict[str, Any]],
        units: List[Dict[str, Any]],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """RAPTOR-driven semantic clustering pipeline.

        1. Cluster L1 summary embeddings via UMAP+GMM → L2 functional groups
        2. Assign orphan files to nearest L2 cluster
        3. Map clusters to code units
        """
        # Extract embeddings matrix from L1 summaries
        embeddings = np.array([s["embedding"] for s in summaries], dtype=np.float32)
        source_ids = [s["source_id"] for s in summaries]

        # Step 1: UMAP dimensionality reduction + GMM clustering on L1 embeddings
        labels = self._umap_gmm_cluster(embeddings, len(summaries))

        # Group summaries by cluster label
        l2_groups: Dict[int, List[str]] = defaultdict(list)  # label -> [source_ids]
        l2_texts: Dict[int, List[str]] = defaultdict(list)   # label -> [summary texts]
        l2_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        for i, label in enumerate(labels):
            l2_groups[label].append(source_ids[i])
            l2_texts[label].append(summaries[i]["text"])
            l2_embeddings[label].append(embeddings[i])

        logger.info(
            "RAPTOR L1→L2 clustering: %d L1 summaries → %d L2 groups",
            len(summaries), len(l2_groups),
        )

        # Step 2: Assign orphan files to nearest L2 cluster
        covered_source_ids = set(source_ids)
        project_id = self._get_project_id_from_units(units)
        orphans = self._load_orphan_file_embeddings(
            project_id,
            covered_source_ids,
        )

        if orphans:
            # Compute centroids for each L2 cluster
            centroids = {}
            for label, embs in l2_embeddings.items():
                centroids[label] = np.mean(embs, axis=0)

            centroid_labels = sorted(centroids.keys())
            centroid_matrix = np.array([centroids[l] for l in centroid_labels])

            for orphan in orphans:
                # Cosine similarity to each centroid
                orph_emb = orphan["embedding"]
                orph_norm = orph_emb / (np.linalg.norm(orph_emb) + 1e-10)
                centroid_norms = centroid_matrix / (
                    np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10
                )
                similarities = centroid_norms @ orph_norm
                best_idx = int(np.argmax(similarities))
                best_label = centroid_labels[best_idx]
                l2_groups[best_label].append(orphan["source_id"])

            logger.info("Assigned %d orphan files to L2 clusters", len(orphans))

        # Step 3: Map L2 clusters to code units
        file_to_units = defaultdict(list)
        for u in units:
            file_to_units[u["file_id"]].append(u)

        clusters = []
        for label in sorted(l2_groups.keys()):
            group_source_ids = l2_groups[label]
            cluster_units = []
            cluster_file_ids = set()

            for sid in group_source_ids:
                for u in file_to_units.get(sid, []):
                    cluster_units.append(u)
                    cluster_file_ids.add(u["file_id"])

            if not cluster_units:
                continue

            # Use the L2 summary texts to derive a representative package name
            # for display and naming purposes
            representative_pkg = self._derive_package_from_units(cluster_units)

            clusters.append({
                "package": representative_pkg,
                "unit_ids": [u["unit_id"] for u in cluster_units],
                "file_ids": list(cluster_file_ids),
                "units": cluster_units,
                "name": "",
                "description": None,
                "metrics": {},
                "sp_references": [],
                "depends_on": [],
                "_raptor_summaries": [t[:200] for t in l2_texts.get(label, [])],
            })

        logger.info(
            "RAPTOR clustering produced %d MVPs covering %d units",
            len(clusters),
            sum(len(c["unit_ids"]) for c in clusters),
        )
        return clusters

    def _get_project_id_from_units(self, units: List[Dict[str, Any]]) -> UUID:
        """Get project_id by querying the first unit's file."""
        if not units:
            raise ValueError("No units to derive project_id from")
        file_id = units[0]["file_id"]
        with self._db.get_session() as session:
            result = session.execute(
                text("SELECT project_id FROM code_files WHERE file_id = :fid"),
                {"fid": file_id},
            )
            row = result.fetchone()
            if row:
                return row.project_id
        raise ValueError(f"No project found for file_id {file_id}")

    def _umap_gmm_cluster(self, embeddings: np.ndarray, n_items: int) -> List[int]:
        """UMAP dimensionality reduction + GMM soft clustering.

        Reuses the same approach as RAPTOR's own clustering but applied
        to L1 summary embeddings to produce L2 functional groups.

        Target: ~3-5 L1 summaries per cluster → n_items/4, bounded to [10, 30].
        """
        try:
            import umap
            from sklearn.mixture import GaussianMixture
        except ImportError:
            logger.error("umap-learn or scikit-learn not available, using simple clustering")
            return self._fallback_kmeans(embeddings, n_items)

        # Each L1 summary covers ~20 code units on average.
        # For MVPs of 60-100 units we want 3-5 L1s per cluster.
        n_clusters = max(10, min(n_items // 4, 30))

        # UMAP reduction
        n_neighbors = min(15, n_items - 1)
        if n_neighbors < 2:
            return list(range(n_items))  # Each item its own cluster

        n_components = min(10, n_items - 1, embeddings.shape[1])
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )
        reduced = reducer.fit_transform(embeddings)
        logger.info("UMAP reduced %s → %s", embeddings.shape, reduced.shape)

        # GMM clustering with soft assignments
        n_clusters = min(n_clusters, n_items // 2)  # Need at least 2 items per cluster
        if n_clusters < 2:
            return [0] * n_items

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        gmm.fit(reduced)

        # Hard assignment (highest probability cluster)
        labels = gmm.predict(reduced).tolist()

        # Log cluster size distribution
        from collections import Counter
        dist = Counter(labels)
        logger.info(
            "GMM produced %d clusters: sizes %s",
            len(dist),
            sorted(dist.values(), reverse=True),
        )

        return labels

    def _fallback_kmeans(self, embeddings: np.ndarray, n_items: int) -> List[int]:
        """Simple fallback when UMAP/GMM unavailable."""
        try:
            from sklearn.cluster import KMeans
            n_clusters = max(10, min(n_items // 4, 30))
            n_clusters = min(n_clusters, n_items // 2)  # At least 2 items per cluster
            if n_clusters < 2:
                return [0] * n_items
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return km.fit_predict(embeddings).tolist()
        except ImportError:
            # Last resort: single cluster
            return [0] * n_items

    @staticmethod
    def _derive_package_from_units(units: List[Dict[str, Any]]) -> str:
        """Derive a representative package name from the most common package prefix."""
        pkg_counts: Dict[str, int] = defaultdict(int)
        for u in units:
            qn = u["qualified_name"]
            if "::" in qn:
                qn = qn.split("::")[0]
            parts = qn.replace("/", ".").replace("\\", ".").split(".")
            # Use varying depths to find the most representative prefix
            for depth in range(min(4, len(parts)), 0, -1):
                pkg = ".".join(parts[:depth])
                pkg_counts[pkg] += 1

        if not pkg_counts:
            return "default"

        # Find the most specific prefix that covers majority of units
        threshold = len(units) * 0.5
        candidates = [
            (pkg, count) for pkg, count in pkg_counts.items()
            if count >= threshold
        ]
        if candidates:
            # Pick the longest (most specific) qualifying prefix
            candidates.sort(key=lambda x: (-len(x[0].split(".")), -x[1]))
            return candidates[0][0]

        # Fallback: most frequent prefix
        return max(pkg_counts, key=pkg_counts.get)

    # ── Fallback: Package-Based Clustering ──────────────────────────

    def _cluster_package_based(
        self,
        units: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Original package-based clustering (fallback when no RAPTOR tree)."""
        cohesion_threshold = params.get("cohesion_merge_threshold", _COHESION_MERGE_THRESHOLD)
        coupling_threshold = params.get("coupling_split_threshold", _COUPLING_SPLIT_THRESHOLD)
        min_size = params.get("min_cluster_size", _MIN_CLUSTER_SIZE)
        max_size = params.get("max_cluster_size", _MAX_CLUSTER_SIZE)

        clusters, shared = self._seed_clusters(units)
        logger.info(f"Seeded {len(clusters)} clusters + {len(shared)} shared concerns")

        for cluster in clusters:
            self._compute_metrics(cluster, edge_lookup)

        clusters = self._merge_clusters(clusters, edge_lookup, cohesion_threshold, min_size)
        clusters = self._split_clusters(clusters, edge_lookup, coupling_threshold, units, max_size)

        for cluster in clusters:
            self._compute_metrics(cluster, edge_lookup)

        return clusters

    # ── Data Loading ────────────────────────────────────────────────

    def _load_units(
        self, project_id: UUID, languages: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        with self._db.get_session() as session:
            if languages is not None:
                result = session.execute(
                    text("""
                        SELECT unit_id, file_id, name, qualified_name, unit_type,
                               language, metadata
                        FROM code_units
                        WHERE project_id = :pid AND language = ANY(:langs)
                    """),
                    {"pid": project_id, "langs": languages},
                )
            else:
                result = session.execute(
                    text("""
                        SELECT unit_id, file_id, name, qualified_name, unit_type,
                               language, metadata
                        FROM code_units
                        WHERE project_id = :pid
                    """),
                    {"pid": project_id},
                )
            return [
                {
                    "unit_id": str(r.unit_id),
                    "file_id": str(r.file_id),
                    "name": r.name,
                    "qualified_name": r.qualified_name or r.name,
                    "unit_type": r.unit_type,
                    "language": r.language,
                    "metadata": r.metadata or {},
                }
                for r in result.fetchall()
            ]

    def _load_edges(self, project_id: UUID) -> List[Dict[str, Any]]:
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT source_unit_id, target_unit_id, edge_type, metadata
                    FROM code_edges
                    WHERE project_id = :pid
                """),
                {"pid": project_id},
            )
            return [
                {
                    "source_unit_id": str(r.source_unit_id),
                    "target_unit_id": str(r.target_unit_id),
                    "edge_type": r.edge_type,
                    "metadata": r.metadata or {},
                }
                for r in result.fetchall()
            ]

    # ── Seed Clusters (fallback path) ───────────────────────────────

    def _seed_clusters(
        self, units: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Group units by top-level package/namespace."""
        package_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for u in units:
            pkg = self._extract_package(u["qualified_name"], u["language"])
            package_groups[pkg].append(u)

        clusters = []
        shared = []

        for pkg, pkg_units in package_groups.items():
            pkg_lower = pkg.lower().rsplit(".", 1)[-1] if "." in pkg else pkg.lower()
            if pkg_lower in _SHARED_PACKAGE_KEYWORDS:
                shared.append(pkg)
                continue

            unit_ids = [u["unit_id"] for u in pkg_units]
            file_ids = list({u["file_id"] for u in pkg_units})

            clusters.append({
                "package": pkg,
                "unit_ids": unit_ids,
                "file_ids": file_ids,
                "units": pkg_units,
                "name": "",
                "description": None,
                "metrics": {},
                "sp_references": [],
                "depends_on": [],
            })

        return clusters, shared

    @staticmethod
    def _extract_package(qualified_name: str, language: Optional[str]) -> str:
        """Extract the top-level package/namespace from a qualified name."""
        if "::" in qualified_name:
            qualified_name = qualified_name.split("::")[0]

        parts = qualified_name.replace("/", ".").replace("\\", ".").split(".")

        if language in ("java", "csharp") and len(parts) > 3:
            return ".".join(parts[:3])

        if len(parts) >= 2:
            return ".".join(parts[:-1])

        return parts[0] if parts else "default"

    # ── Metrics ─────────────────────────────────────────────────────

    def _build_edge_lookup(
        self, edges: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build outgoing edge lookup by source_unit_id."""
        lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            lookup[e["source_unit_id"]].append(e)
        return lookup

    def _compute_metrics(
        self,
        cluster: Dict[str, Any],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Compute cohesion, coupling, and size metrics for a cluster."""
        unit_set = set(cluster["unit_ids"])
        size = len(unit_set)

        if size < 2:
            cluster["metrics"] = {
                "cohesion": 1.0,
                "coupling": 0.0,
                "size": size,
                "internal_edges": 0,
                "external_edges": 0,
            }
            return

        internal_edges = 0
        external_edges = 0
        semantic_types = ("calls", "calls_sp", "imports", "inherits", "implements", "overrides", "type_dep")

        for uid in unit_set:
            for e in edge_lookup.get(uid, []):
                if e["edge_type"] not in semantic_types:
                    continue
                if e["target_unit_id"] in unit_set:
                    internal_edges += 1
                else:
                    external_edges += 1

        total_edges = internal_edges + external_edges
        max_internal = size * (size - 1)

        cohesion = internal_edges / max_internal if max_internal > 0 else 0.0
        coupling = external_edges / total_edges if total_edges > 0 else 0.0

        cluster["metrics"] = {
            "cohesion": round(cohesion, 3),
            "coupling": round(coupling, 3),
            "size": size,
            "internal_edges": internal_edges,
            "external_edges": external_edges,
        }

    # ── Merge (fallback path) ───────────────────────────────────────

    def _merge_clusters(
        self,
        clusters: List[Dict[str, Any]],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
        cohesion_threshold: float,
        min_size: int,
    ) -> List[Dict[str, Any]]:
        """Merge clusters that are too small or have low cohesion."""
        if len(clusters) <= 1:
            return clusters

        merged = list(clusters)
        changed = True

        while changed:
            changed = False
            for i in range(len(merged) - 1, -1, -1):
                c = merged[i]
                should_merge = (
                    c["metrics"].get("size", 0) < min_size
                    or c["metrics"].get("cohesion", 1.0) < cohesion_threshold
                )
                if not should_merge:
                    continue

                best_target = self._find_most_coupled_neighbor(c, merged, edge_lookup, i)
                if best_target is not None and best_target != i:
                    target_cluster = merged[best_target]
                    target_cluster["unit_ids"].extend(c["unit_ids"])
                    target_cluster["file_ids"] = list(
                        set(target_cluster["file_ids"]) | set(c["file_ids"])
                    )
                    target_cluster["units"].extend(c.get("units", []))
                    merged.pop(i)
                    self._compute_metrics(target_cluster, edge_lookup)
                    changed = True
                    break

        return merged

    def _find_most_coupled_neighbor(
        self,
        cluster: Dict[str, Any],
        all_clusters: List[Dict[str, Any]],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
        self_idx: int,
    ) -> Optional[int]:
        """Find the cluster index with the most edges to/from the given cluster."""
        unit_set = set(cluster["unit_ids"])
        coupling_count: Dict[int, int] = defaultdict(int)

        uid_to_cluster: Dict[str, int] = {}
        for idx, c in enumerate(all_clusters):
            if idx == self_idx:
                continue
            for uid in c["unit_ids"]:
                uid_to_cluster[uid] = idx

        for uid in unit_set:
            for e in edge_lookup.get(uid, []):
                target_cluster = uid_to_cluster.get(e["target_unit_id"])
                if target_cluster is not None:
                    coupling_count[target_cluster] += 1

        if not coupling_count:
            return None
        return max(coupling_count, key=coupling_count.get)

    # ── Split (fallback path) ───────────────────────────────────────

    def _split_clusters(
        self,
        clusters: List[Dict[str, Any]],
        edge_lookup: Dict[str, List[Dict[str, Any]]],
        coupling_threshold: float,
        all_units: List[Dict[str, Any]],
        max_size: int = _MAX_CLUSTER_SIZE,
    ) -> List[Dict[str, Any]]:
        """Recursively split clusters at sub-package boundaries."""
        unit_lookup = {u["unit_id"]: u for u in all_units}
        result = []
        queue = list(clusters)
        max_depth = 10

        while queue:
            c = queue.pop(0)
            depth = c.get("_split_depth", 0)

            should_split = (
                c["metrics"].get("coupling", 0) > coupling_threshold
                or c["metrics"].get("size", 0) > max_size
            )
            if not should_split or c["metrics"].get("size", 0) <= _MIN_CLUSTER_SIZE or depth >= max_depth:
                c.pop("_split_depth", None)
                result.append(c)
                continue

            sub_groups: Dict[str, List[str]] = defaultdict(list)
            for uid in c["unit_ids"]:
                u = unit_lookup.get(uid)
                if u:
                    sub_pkg = self._extract_sub_package(u["qualified_name"], c.get("package", ""))
                    sub_groups[sub_pkg].append(uid)

            if len(sub_groups) >= 2:
                for sub_pkg, sub_uids in sub_groups.items():
                    sub_file_ids = list({unit_lookup[uid]["file_id"] for uid in sub_uids if uid in unit_lookup})
                    sub_units = [unit_lookup[uid] for uid in sub_uids if uid in unit_lookup]
                    new_cluster = {
                        "package": sub_pkg,
                        "unit_ids": sub_uids,
                        "file_ids": sub_file_ids,
                        "units": sub_units,
                        "name": "",
                        "description": None,
                        "metrics": {},
                        "sp_references": [],
                        "depends_on": [],
                        "_split_depth": depth + 1,
                    }
                    self._compute_metrics(new_cluster, edge_lookup)
                    queue.append(new_cluster)
            else:
                c.pop("_split_depth", None)
                result.append(c)

        return result

    @staticmethod
    def _extract_sub_package(qualified_name: str, parent_package: str) -> str:
        """Get one level deeper than the parent package."""
        if "::" in qualified_name:
            qualified_name = qualified_name.split("::")[0]
        parts = qualified_name.replace("/", ".").replace("\\", ".").split(".")
        parent_parts = parent_package.split(".") if parent_package else []

        is_prefix = (
            len(parts) > len(parent_parts)
            and parts[:len(parent_parts)] == parent_parts
        )

        if is_prefix:
            depth = len(parent_parts) + 1
        else:
            depth = min(4, len(parts) - 1) if len(parts) > 4 else len(parts) - 1

        if depth <= 0:
            depth = 1

        return ".".join(parts[:depth]) if len(parts) > depth else ".".join(parts[:-1]) if len(parts) > 1 else parts[0]

    # ── SP References ───────────────────────────────────────────────

    def _attach_sp_references(
        self,
        clusters: List[Dict[str, Any]],
        sp_edges: List[Dict[str, Any]],
        sp_units: Dict[str, Dict[str, Any]],
    ) -> None:
        """Attach SP reference info to each cluster."""
        for cluster in clusters:
            unit_set = set(cluster["unit_ids"])
            sp_refs: Dict[str, Dict[str, Any]] = {}

            for e in sp_edges:
                if e["source_unit_id"] in unit_set:
                    sp_id = e["target_unit_id"]
                    sp = sp_units.get(sp_id, {})
                    sp_name = e["metadata"].get("sp_name", sp.get("name", "unknown"))

                    if sp_name not in sp_refs:
                        sp_refs[sp_name] = {
                            "sp_name": sp_name,
                            "sp_unit_id": sp_id,
                            "call_sites": [],
                        }
                    sp_refs[sp_name]["call_sites"].append({
                        "caller_id": e["source_unit_id"],
                        "pattern": e["metadata"].get("call_pattern", "unknown"),
                    })

            cluster["sp_references"] = list(sp_refs.values())

    # ── Dependency Computation ───────────────────────────────────────

    def _compute_mvp_dependencies(
        self,
        clusters: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        """Compute inter-MVP dependency edges from ASG edges.

        For each edge (calls, imports, inherits, implements), if the source
        and target units belong to different clusters, the source cluster
        depends on the target cluster. Mutates clusters in place, populating
        the 'depends_on' set for each cluster.
        """
        # Build unit_id → cluster index map
        uid_to_cluster: Dict[str, int] = {}
        for idx, cluster in enumerate(clusters):
            for uid in cluster["unit_ids"]:
                uid_to_cluster[uid] = idx

        # Initialize depends_on as sets
        for cluster in clusters:
            cluster["_depends_on_indices"] = set()

        dependency_types = ("calls", "imports", "inherits", "implements", "overrides", "type_dep")

        for edge in edges:
            if edge["edge_type"] not in dependency_types:
                continue
            src_idx = uid_to_cluster.get(edge["source_unit_id"])
            tgt_idx = uid_to_cluster.get(edge["target_unit_id"])
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                # Source cluster depends on target cluster
                clusters[src_idx]["_depends_on_indices"].add(tgt_idx)

        logger.info(
            "Computed MVP dependencies: %d clusters with %d total dependency edges",
            len(clusters),
            sum(len(c["_depends_on_indices"]) for c in clusters),
        )

    def _topological_sort_with_readiness(
        self,
        clusters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Topological sort using Kahn's algorithm with readiness as tiebreaker.

        Clusters with no remaining dependencies are ordered by readiness score
        (highest first). Foundation MVP (if present) is always kept at index 0.

        Returns a new list in the correct migration order.
        """
        n = len(clusters)
        if n <= 1:
            return clusters

        # Build adjacency and in-degree from _depends_on_indices
        in_degree = [0] * n
        dependents: Dict[int, List[int]] = defaultdict(list)

        for idx, cluster in enumerate(clusters):
            deps = cluster.get("_depends_on_indices", set())
            in_degree[idx] = len(deps)
            for dep_idx in deps:
                dependents[dep_idx].append(idx)

        # Kahn's algorithm: start with zero in-degree nodes
        # Use negative readiness for min-heap behavior (highest readiness first)
        queue: List[Tuple[float, int]] = []
        for idx in range(n):
            if in_degree[idx] == 0:
                readiness = clusters[idx].get("metrics", {}).get("readiness", 0)
                queue.append((-readiness, idx))
        queue.sort()

        result = []
        while queue:
            _, idx = queue.pop(0)
            result.append(clusters[idx])

            for dependent_idx in dependents.get(idx, []):
                in_degree[dependent_idx] -= 1
                if in_degree[dependent_idx] == 0:
                    readiness = clusters[dependent_idx].get("metrics", {}).get("readiness", 0)
                    queue.append((-readiness, dependent_idx))
                    queue.sort()

        # Handle cycles: append any remaining clusters (not reached by topo sort)
        if len(result) < n:
            seen = {id(c) for c in result}
            for cluster in clusters:
                if id(cluster) not in seen:
                    result.append(cluster)
            logger.warning(
                "Dependency cycle detected: %d clusters could not be topologically sorted",
                n - len([c for c in result if id(c) in seen]),
            )

        # Map old indices to new indices for depends_on
        old_to_new: Dict[int, int] = {}
        for new_idx, cluster in enumerate(result):
            for old_idx, orig in enumerate(clusters):
                if cluster is orig:
                    old_to_new[old_idx] = new_idx
                    break

        # Assign final priorities and convert depends_on indices
        for i, cluster in enumerate(result):
            cluster["priority"] = i + 1  # Reserve 0 for Foundation
            old_deps = cluster.pop("_depends_on_indices", set())
            cluster["depends_on"] = sorted(
                old_to_new[d] for d in old_deps if d in old_to_new
            )

        return result

    def _load_file_paths(self, project_id: UUID) -> Dict[str, str]:
        """Load file_id -> file_path mapping for a project."""
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT file_id, file_path
                    FROM code_files
                    WHERE project_id = :pid
                """),
                {"pid": project_id},
            )
            return {str(r.file_id): r.file_path for r in result.fetchall()}

    @staticmethod
    def _get_file_id_for_units(cluster: Dict[str, Any], unit_id: str) -> List[str]:
        """Get the file_id(s) for a unit within a cluster."""
        for u in cluster.get("units", []):
            if u["unit_id"] == unit_id:
                return [u["file_id"]]
        return []

    # ── Ranking ─────────────────────────────────────────────────────

    def _rank_clusters(self, clusters: List[Dict[str, Any]]) -> None:
        """Sort clusters by migration readiness score (highest first)."""
        for c in clusters:
            m = c["metrics"]
            cohesion = m.get("cohesion", 0)
            coupling = m.get("coupling", 1)
            sp_count = len(c.get("sp_references", []))
            size = m.get("size", 1)

            sp_ratio = sp_count / max(size, 1)

            readiness = (
                cohesion * 0.3
                + (1 - coupling) * 0.3
                + (1 - min(sp_ratio, 1.0)) * 0.2
                + min(1.0, 10 / max(size, 1)) * 0.2
            )
            m["readiness"] = round(readiness, 3)
            m["complexity"] = "low" if readiness > 0.7 else "medium" if readiness > 0.4 else "high"

        clusters.sort(key=lambda c: c["metrics"].get("readiness", 0), reverse=True)

    # ── Naming ──────────────────────────────────────────────────────

    @staticmethod
    def _generate_name(cluster: Dict[str, Any]) -> str:
        """Generate a human-readable name from the package path."""
        pkg = cluster.get("package", "unknown")
        parts = pkg.replace("/", ".").split(".")
        skip = {"com", "org", "net", "io", "src", "main", "java", "app", "lib", "pkg"}
        meaningful = [p for p in parts if p.lower() not in skip and p]
        if len(meaningful) >= 2:
            name = ".".join(meaningful[-2:])
        elif meaningful:
            name = meaningful[-1]
        else:
            name = parts[-1] if parts else "unknown"
        return ".".join(
            segment.replace("_", " ").replace("-", " ").title()
            for segment in name.split(".")
        )
