"""RAPTOR Configuration.

Configuration for Recursive Abstractive Processing for Tree-Organized Retrieval.
Loads settings from config/raptor.yaml with Python defaults as fallback.

Reference: https://arxiv.org/abs/2401.18059
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict

from dbnotebook.core.config.config_loader import (
    get_clustering_config,
    get_summarization_config,
    get_tree_building_config,
    get_retrieval_config,
    get_raptor_presets,
    load_raptor_config,
)


def _get(config: Dict[str, Any], key: str, default: Any) -> Any:
    """Get config value with fallback to default."""
    return config.get(key, default)


@dataclass
class ClusteringConfig:
    """Configuration for GMM clustering with UMAP dimensionality reduction."""

    # UMAP dimensionality reduction
    umap_n_components: int = field(default_factory=lambda: _get(get_clustering_config(), "umap_n_components", 10))
    umap_n_neighbors: int = field(default_factory=lambda: _get(get_clustering_config(), "umap_n_neighbors", 15))
    umap_min_dist: float = field(default_factory=lambda: _get(get_clustering_config(), "umap_min_dist", 0.1))
    umap_metric: str = field(default_factory=lambda: _get(get_clustering_config(), "umap_metric", "cosine"))

    # GMM clustering
    gmm_probability_threshold: float = field(default_factory=lambda: _get(get_clustering_config(), "gmm_probability_threshold", 0.3))
    min_cluster_size: int = field(default_factory=lambda: _get(get_clustering_config(), "min_cluster_size", 3))
    max_cluster_size: int = field(default_factory=lambda: _get(get_clustering_config(), "max_cluster_size", 10))
    max_clusters: int = field(default_factory=lambda: _get(get_clustering_config(), "max_clusters", 50))

    # Clustering behavior
    random_state: int = field(default_factory=lambda: _get(get_clustering_config(), "random_state", 42))
    n_init: int = field(default_factory=lambda: _get(get_clustering_config(), "n_init", 10))


# Default prompts (used if not in YAML)
_DEFAULT_CLUSTER_PROMPT = """You are an expert summarizer. Below are related text chunks from a document that have been grouped together by semantic similarity.

Create a comprehensive summary that:
1. Captures the main themes and key points across all chunks
2. Preserves important details, facts, and figures
3. Maintains logical flow and coherence
4. Is self-contained and understandable without the original chunks

CHUNKS TO SUMMARIZE:
{chunks}

COMPREHENSIVE SUMMARY:"""

_DEFAULT_ROOT_PROMPT = """You are an expert summarizer. Below are summaries from different sections of a document.

Create a high-level executive summary that:
1. Provides an overview of the entire document's content
2. Highlights the most important themes and conclusions
3. Notes any key relationships between different sections
4. Is suitable for answering "summarize this document" queries

SECTION SUMMARIES:
{summaries}

DOCUMENT SUMMARY:"""


@dataclass
class SummarizationConfig:
    """Configuration for LLM-based cluster summarization."""

    # Token limits
    max_input_tokens: int = field(default_factory=lambda: _get(get_summarization_config(), "max_input_tokens", 6000))
    summary_max_tokens: int = field(default_factory=lambda: _get(get_summarization_config(), "summary_max_tokens", 500))

    # Content limits
    max_chunks_per_summary: int = field(default_factory=lambda: _get(get_summarization_config(), "max_chunks_per_summary", 10))

    # LLM prompts
    cluster_summary_prompt: str = field(default_factory=lambda: _get(get_summarization_config(), "cluster_summary_prompt", _DEFAULT_CLUSTER_PROMPT))
    root_summary_prompt: str = field(default_factory=lambda: _get(get_summarization_config(), "root_summary_prompt", _DEFAULT_ROOT_PROMPT))


@dataclass
class TreeBuildingConfig:
    """Configuration for RAPTOR tree construction."""

    # Tree structure
    max_tree_depth: int = field(default_factory=lambda: _get(get_tree_building_config(), "max_tree_depth", 4))
    min_nodes_to_cluster: int = field(default_factory=lambda: _get(get_tree_building_config(), "min_nodes_to_cluster", 5))

    # Processing
    batch_size: int = field(default_factory=lambda: _get(get_tree_building_config(), "batch_size", 50))
    embedding_batch_size: int = field(default_factory=lambda: _get(get_tree_building_config(), "embedding_batch_size", 8))

    # Concurrency
    max_concurrent_summaries: int = field(default_factory=lambda: _get(get_tree_building_config(), "max_concurrent_summaries", 3))

    # Retry behavior
    max_retries: int = field(default_factory=lambda: _get(get_tree_building_config(), "max_retries", 3))
    retry_delay_seconds: float = field(default_factory=lambda: _get(get_tree_building_config(), "retry_delay_seconds", 1.0))


# Default keywords for query type detection
_DEFAULT_SUMMARY_KEYWORDS = [
    "summarize", "summary", "overview", "main points",
    "key takeaways", "themes", "gist", "brief",
    "what is this about", "tldr", "highlights",
    "main ideas", "general idea", "big picture"
]

_DEFAULT_DETAIL_KEYWORDS = [
    "specific", "detail", "exactly", "quote",
    "what does it say about", "where", "when",
    "how many", "what is the", "explain",
    "section", "page", "paragraph"
]


@dataclass
class RetrievalConfig:
    """Configuration for RAPTOR-aware retrieval."""

    # Level selection for different query types
    summary_query_levels: List[int] = field(default_factory=lambda: _get(get_retrieval_config(), "summary_query_levels", [0, 1, 2, 3]))
    detail_query_levels: List[int] = field(default_factory=lambda: _get(get_retrieval_config(), "detail_query_levels", [0, 1]))

    # Intent detection keywords
    summary_keywords: List[str] = field(default_factory=lambda: _get(get_retrieval_config(), "summary_keywords", _DEFAULT_SUMMARY_KEYWORDS))
    detail_keywords: List[str] = field(default_factory=lambda: _get(get_retrieval_config(), "detail_keywords", _DEFAULT_DETAIL_KEYWORDS))

    # Retrieval parameters
    top_k_per_level: int = field(default_factory=lambda: _get(get_retrieval_config(), "top_k_per_level", 6))
    rerank_top_k: int = field(default_factory=lambda: _get(get_retrieval_config(), "rerank_top_k", 6))
    min_similarity_threshold: float = field(default_factory=lambda: _get(get_retrieval_config(), "min_similarity_threshold", 0.3))

    # Score boosting for level-aware retrieval
    summary_level_boost: float = field(default_factory=lambda: _get(get_retrieval_config(), "summary_level_boost", 1.5))
    detail_level_boost: float = field(default_factory=lambda: _get(get_retrieval_config(), "detail_level_boost", 1.3))

    # Hybrid retrieval (BM25 + Vector fusion)
    use_hybrid_retrieval: bool = field(default_factory=lambda: _get(get_retrieval_config(), "use_hybrid_retrieval", True))
    bm25_weight: float = field(default_factory=lambda: _get(get_retrieval_config(), "bm25_weight", 0.5))
    vector_weight: float = field(default_factory=lambda: _get(get_retrieval_config(), "vector_weight", 0.5))

    # Query expansion for hybrid retrieval
    num_query_variations: int = field(default_factory=lambda: _get(get_retrieval_config(), "num_query_variations", 3))


def _load_feature_flags() -> Dict[str, Any]:
    """Load feature flags from YAML config."""
    config = load_raptor_config()
    return {
        "enabled": config.get("enabled", True),
        "auto_build_on_upload": config.get("auto_build_on_upload", True),
        "fallback_to_flat_retrieval": config.get("fallback_to_flat_retrieval", True),
    }


@dataclass
class RAPTORConfig:
    """Master configuration for RAPTOR system."""

    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    tree_building: TreeBuildingConfig = field(default_factory=TreeBuildingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Feature flags (loaded from YAML)
    enabled: bool = field(default_factory=lambda: _load_feature_flags()["enabled"])
    auto_build_on_upload: bool = field(default_factory=lambda: _load_feature_flags()["auto_build_on_upload"])
    fallback_to_flat_retrieval: bool = field(default_factory=lambda: _load_feature_flags()["fallback_to_flat_retrieval"])

    @classmethod
    def default(cls) -> "RAPTORConfig":
        """Create default configuration (loads from YAML)."""
        return cls()

    @classmethod
    def fast(cls) -> "RAPTORConfig":
        """Create configuration optimized for speed.

        Uses 'fast' preset from config/raptor.yaml if available,
        otherwise falls back to hardcoded fast defaults.
        """
        presets = get_raptor_presets()
        fast_preset = presets.get("fast", {})

        # Get preset values with fallbacks
        tree_config = fast_preset.get("tree_building", {})
        cluster_config = fast_preset.get("clustering", {})
        summary_config = fast_preset.get("summarization", {})

        return cls(
            clustering=ClusteringConfig(
                max_cluster_size=cluster_config.get("max_cluster_size", 15),
                umap_n_components=cluster_config.get("umap_n_components", 5),
            ),
            summarization=SummarizationConfig(
                summary_max_tokens=summary_config.get("summary_max_tokens", 300),
                max_chunks_per_summary=summary_config.get("max_chunks_per_summary", 15),
            ),
            tree_building=TreeBuildingConfig(
                max_tree_depth=tree_config.get("max_tree_depth", 3),
                min_nodes_to_cluster=tree_config.get("min_nodes_to_cluster", 8),
            ),
        )

    @classmethod
    def thorough(cls) -> "RAPTORConfig":
        """Create configuration for maximum quality.

        Uses 'thorough' preset from config/raptor.yaml if available,
        otherwise falls back to hardcoded thorough defaults.
        """
        presets = get_raptor_presets()
        thorough_preset = presets.get("thorough", {})

        # Get preset values with fallbacks
        tree_config = thorough_preset.get("tree_building", {})
        cluster_config = thorough_preset.get("clustering", {})
        summary_config = thorough_preset.get("summarization", {})

        return cls(
            clustering=ClusteringConfig(
                min_cluster_size=cluster_config.get("min_cluster_size", 2),
                max_cluster_size=cluster_config.get("max_cluster_size", 6),
                gmm_probability_threshold=cluster_config.get("gmm_probability_threshold", 0.2),
            ),
            summarization=SummarizationConfig(
                summary_max_tokens=summary_config.get("summary_max_tokens", 800),
                max_chunks_per_summary=summary_config.get("max_chunks_per_summary", 6),
            ),
            tree_building=TreeBuildingConfig(
                max_tree_depth=tree_config.get("max_tree_depth", 5),
                min_nodes_to_cluster=tree_config.get("min_nodes_to_cluster", 3),
            ),
        )


# Default configuration instance
DEFAULT_CONFIG = RAPTORConfig.default()
