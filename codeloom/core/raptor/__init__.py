"""RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval.

This module implements RAPTOR for DBNotebook, enabling hierarchical document
summarization and multi-level retrieval.

Key Components:
- config: Configuration dataclasses for clustering, summarization, tree building
- clustering: GMM clustering with UMAP dimensionality reduction
- summarizer: LLM-based cluster summarization
- tree_builder: Orchestrates tree construction from document chunks
- worker: Background job for async tree building
- retriever: Level-aware retrieval for RAPTOR trees

Usage:
    from dbnotebook.core.raptor import RAPTORConfig, TreeBuilder, RAPTORRetriever

    # Build tree for a document
    config = RAPTORConfig.default()
    builder = TreeBuilder(config, llm, embed_model)
    tree = await builder.build_tree(chunks, source_id)

    # Retrieve with level awareness
    retriever = RAPTORRetriever(config, vector_store)
    results = retriever.retrieve(query, source_ids, query_intent="summary")

Reference Paper: https://arxiv.org/abs/2401.18059
"""

from .config import (
    ClusteringConfig,
    SummarizationConfig,
    TreeBuildingConfig,
    RetrievalConfig,
    RAPTORConfig,
    DEFAULT_CONFIG,
)

from .clustering import (
    Cluster,
    RAPTORClustering,
    cluster_nodes,
)

from .summarizer import (
    SummaryNode,
    RAPTORSummarizer,
    create_summarizer,
)

from .tree_builder import (
    TreeBuildResult,
    RAPTORTreeBuilder,
    create_tree_builder,
)

from .worker import (
    RAPTORJob,
    RAPTORWorker,
    build_raptor_tree_sync,
)

from .retriever import (
    RAPTORQueryType,
    RAPTORRetrievalResult,
    RAPTORRetriever,
    create_raptor_retriever,
    has_raptor_tree,
)

__all__ = [
    # Configuration
    "ClusteringConfig",
    "SummarizationConfig",
    "TreeBuildingConfig",
    "RetrievalConfig",
    "RAPTORConfig",
    "DEFAULT_CONFIG",
    # Clustering
    "Cluster",
    "RAPTORClustering",
    "cluster_nodes",
    # Summarization
    "SummaryNode",
    "RAPTORSummarizer",
    "create_summarizer",
    # Tree Building
    "TreeBuildResult",
    "RAPTORTreeBuilder",
    "create_tree_builder",
    # Worker
    "RAPTORJob",
    "RAPTORWorker",
    "build_raptor_tree_sync",
    # Retrieval
    "RAPTORQueryType",
    "RAPTORRetrievalResult",
    "RAPTORRetriever",
    "create_raptor_retriever",
    "has_raptor_tree",
]
