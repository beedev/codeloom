"""RAPTOR Clustering Module.

Implements GMM (Gaussian Mixture Model) clustering with UMAP dimensionality reduction
for grouping semantically similar document chunks.

Key Features:
- UMAP reduces high-dimensional embeddings to manageable dimensions
- GMM provides soft clustering (nodes can belong to multiple clusters)
- Probability thresholds control cluster membership
- Handles edge cases (small datasets, single clusters)

Reference: https://arxiv.org/abs/2401.18059
"""

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from llama_index.core.schema import BaseNode, TextNode

from .config import ClusteringConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Represents a cluster of semantically similar nodes."""
    cluster_id: str
    node_ids: List[str]
    nodes: List[BaseNode]
    centroid: Optional[np.ndarray] = None
    probability_scores: Optional[List[float]] = None

    @property
    def size(self) -> int:
        return len(self.nodes)

    def get_texts(self) -> List[str]:
        """Get text content from all nodes in cluster."""
        return [node.get_content() for node in self.nodes]


class RAPTORClustering:
    """
    Semantic clustering for RAPTOR tree construction.

    Uses UMAP for dimensionality reduction followed by GMM for soft clustering.
    Nodes can belong to multiple clusters based on probability thresholds.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize clustering with configuration.

        Args:
            config: Clustering configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG.clustering
        self._umap_model = None
        self._gmm_model = None

    def cluster_nodes(
        self,
        nodes: List[BaseNode],
        embeddings: Optional[np.ndarray] = None
    ) -> List[Cluster]:
        """
        Cluster nodes by semantic similarity.

        Args:
            nodes: List of nodes to cluster
            embeddings: Pre-computed embeddings (extracted from nodes if None)

        Returns:
            List of Cluster objects containing grouped nodes
        """
        if len(nodes) < self.config.min_cluster_size:
            logger.info(f"Too few nodes ({len(nodes)}) for clustering, returning single cluster")
            return [self._create_single_cluster(nodes)]

        # Extract embeddings if not provided
        if embeddings is None:
            embeddings = self._extract_embeddings(nodes)

        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings available, returning single cluster")
            return [self._create_single_cluster(nodes)]

        # Reduce dimensionality with UMAP
        reduced_embeddings = self._reduce_dimensions(embeddings)

        if reduced_embeddings is None:
            logger.warning("Dimensionality reduction failed, returning single cluster")
            return [self._create_single_cluster(nodes)]

        # Perform GMM clustering
        clusters = self._gmm_cluster(nodes, reduced_embeddings)

        logger.info(f"Created {len(clusters)} clusters from {len(nodes)} nodes")
        return clusters

    def _extract_embeddings(self, nodes: List[BaseNode]) -> Optional[np.ndarray]:
        """Extract embeddings from nodes."""
        embeddings = []

        for node in nodes:
            if hasattr(node, 'embedding') and node.embedding is not None:
                embeddings.append(node.embedding)
            else:
                logger.warning(f"Node {node.node_id} has no embedding")
                return None

        if not embeddings:
            return None

        return np.array(embeddings)

    def _reduce_dimensions(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce embedding dimensions using UMAP.

        UMAP preserves both local and global structure better than PCA/t-SNE
        for clustering purposes.
        """
        try:
            # Lazy import to avoid slow startup
            import umap

            n_samples = embeddings.shape[0]

            # Adjust n_neighbors if we have few samples
            n_neighbors = min(self.config.umap_n_neighbors, n_samples - 1)
            if n_neighbors < 2:
                logger.warning("Too few samples for UMAP, skipping reduction")
                return embeddings  # Return original embeddings

            # Adjust n_components if needed
            n_components = min(self.config.umap_n_components, n_samples - 1, embeddings.shape[1])

            self._umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric=self.config.umap_metric,
                random_state=self.config.random_state,
                low_memory=True,  # Better for larger datasets
            )

            reduced = self._umap_model.fit_transform(embeddings)
            logger.debug(f"UMAP reduced {embeddings.shape} -> {reduced.shape}")
            return reduced

        except ImportError:
            logger.error("umap-learn not installed. Install with: pip install umap-learn")
            return None
        except Exception as e:
            logger.error(f"UMAP reduction failed: {e}")
            return None

    def _gmm_cluster(
        self,
        nodes: List[BaseNode],
        embeddings: np.ndarray
    ) -> List[Cluster]:
        """
        Perform GMM clustering with soft assignments.

        GMM allows nodes to belong to multiple clusters based on probability,
        which is better for overlapping topics than hard clustering.
        """
        try:
            from sklearn.mixture import GaussianMixture

            n_samples = embeddings.shape[0]

            # Determine optimal number of clusters
            n_clusters = self._estimate_n_clusters(n_samples)

            if n_clusters <= 1:
                return [self._create_single_cluster(nodes)]

            # Fit GMM
            self._gmm_model = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                n_init=self.config.n_init,
                random_state=self.config.random_state,
            )

            # Get soft cluster assignments (probabilities)
            self._gmm_model.fit(embeddings)
            probabilities = self._gmm_model.predict_proba(embeddings)

            # Build clusters using probability threshold
            clusters = self._build_clusters_from_probabilities(
                nodes, probabilities, embeddings
            )

            return clusters

        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            return [self._create_single_cluster(nodes)]
        except Exception as e:
            logger.error(f"GMM clustering failed: {e}")
            return [self._create_single_cluster(nodes)]

    def _estimate_n_clusters(self, n_samples: int) -> int:
        """
        Estimate optimal number of clusters.

        Uses heuristic based on sample count and configured limits.
        """
        # Heuristic: sqrt(n/2) clusters, bounded by config
        import math

        ideal = int(math.sqrt(n_samples / 2))

        # Apply bounds
        min_clusters = 2
        max_clusters = min(
            self.config.max_clusters,
            n_samples // self.config.min_cluster_size
        )

        n_clusters = max(min_clusters, min(ideal, max_clusters))

        logger.debug(f"Estimated {n_clusters} clusters for {n_samples} samples")
        return n_clusters

    def _build_clusters_from_probabilities(
        self,
        nodes: List[BaseNode],
        probabilities: np.ndarray,
        embeddings: np.ndarray
    ) -> List[Cluster]:
        """
        Build clusters using soft probability assignments.

        Nodes can belong to multiple clusters if probability exceeds threshold.
        """
        n_clusters = probabilities.shape[1]
        threshold = self.config.gmm_probability_threshold

        clusters = []

        for cluster_idx in range(n_clusters):
            cluster_probs = probabilities[:, cluster_idx]

            # Get nodes that belong to this cluster (above threshold)
            member_indices = np.where(cluster_probs >= threshold)[0]

            if len(member_indices) < self.config.min_cluster_size:
                # Skip clusters that are too small
                continue

            # Limit cluster size
            if len(member_indices) > self.config.max_cluster_size:
                # Keep nodes with highest probability
                sorted_indices = np.argsort(cluster_probs[member_indices])[::-1]
                member_indices = member_indices[sorted_indices[:self.config.max_cluster_size]]

            cluster_nodes = [nodes[i] for i in member_indices]
            cluster_node_ids = [nodes[i].node_id for i in member_indices]
            cluster_probs_list = [float(cluster_probs[i]) for i in member_indices]

            # Calculate centroid
            cluster_embeddings = embeddings[member_indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            cluster = Cluster(
                cluster_id=str(uuid.uuid4()),
                node_ids=cluster_node_ids,
                nodes=cluster_nodes,
                centroid=centroid,
                probability_scores=cluster_probs_list
            )

            clusters.append(cluster)

        # Handle edge case: no clusters met the size threshold
        if not clusters:
            logger.warning("No clusters met size threshold, creating single cluster")
            return [self._create_single_cluster(nodes)]

        # Ensure all nodes are in at least one cluster
        all_clustered_ids = set()
        for cluster in clusters:
            all_clustered_ids.update(cluster.node_ids)

        orphan_nodes = [n for n in nodes if n.node_id not in all_clustered_ids]
        if orphan_nodes:
            # Add orphans to their highest-probability cluster
            for node in orphan_nodes:
                node_idx = nodes.index(node)
                best_cluster_idx = np.argmax(probabilities[node_idx])

                # Find or create the target cluster
                for cluster in clusters:
                    if cluster.nodes[0].node_id in [nodes[i].node_id for i in
                        np.where(probabilities[:, best_cluster_idx] >= threshold)[0]]:
                        cluster.nodes.append(node)
                        cluster.node_ids.append(node.node_id)
                        break
                else:
                    # Fallback: add to first cluster
                    clusters[0].nodes.append(node)
                    clusters[0].node_ids.append(node.node_id)

            logger.debug(f"Added {len(orphan_nodes)} orphan nodes to existing clusters")

        return clusters

    def _create_single_cluster(self, nodes: List[BaseNode]) -> Cluster:
        """Create a single cluster containing all nodes."""
        return Cluster(
            cluster_id=str(uuid.uuid4()),
            node_ids=[n.node_id for n in nodes],
            nodes=nodes,
            centroid=None,
            probability_scores=[1.0] * len(nodes)
        )

    def split_large_cluster(self, cluster: Cluster) -> List[Cluster]:
        """
        Split a cluster that exceeds max_cluster_size.

        Args:
            cluster: Cluster to split

        Returns:
            List of smaller clusters
        """
        if cluster.size <= self.config.max_cluster_size:
            return [cluster]

        # Re-cluster the nodes within this cluster
        return self.cluster_nodes(cluster.nodes)


def cluster_nodes(
    nodes: List[BaseNode],
    config: Optional[ClusteringConfig] = None
) -> List[Cluster]:
    """
    Convenience function to cluster nodes.

    Args:
        nodes: List of nodes to cluster
        config: Optional clustering configuration

    Returns:
        List of clusters
    """
    clustering = RAPTORClustering(config)
    return clustering.cluster_nodes(nodes)
