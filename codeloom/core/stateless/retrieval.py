"""Fast retrieval functions extracted from /api/query pattern.

These functions create per-request retrievers without global state,
making them safe for multi-user concurrent access.
"""

import logging
from typing import List, Optional, Tuple, Any

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore

from ..utils import unwrap_llm

logger = logging.getLogger(__name__)


def create_retriever(
    nodes: List[TextNode],
    project_id: str,
    vector_store: Any,
    llm: Optional[Any] = None,
    source_ids: Optional[List[str]] = None,
    language: str = "eng",
    retriever_factory: Optional[Any] = None,
) -> Any:
    """Create a per-request chunk-only retriever (no RAPTOR in retrieval).

    This is thread-safe because it creates a new retriever instance
    for each request rather than reusing a global one.

    Design: Retrieval uses chunks only via get_retrievers() for fast, precise
    results. RAPTOR summaries are fetched separately and added to LLM context.

    Args:
        nodes: Cached nodes for the project
        project_id: UUID of the project to query
        vector_store: PGVectorStore instance
        llm: LLM instance (defaults to Settings.llm)
        source_ids: Optional list of source IDs to filter (unused, kept for API compat)
        language: Language code for prompts (default: "eng")
        retriever_factory: LocalRetriever instance with get_retrievers method

    Returns:
        Chunk-only retriever instance
    """
    if llm is None:
        llm = Settings.llm

    # Unwrap LLM wrappers (e.g., GroqWithBackoff) for LlamaIndex compatibility
    llm = unwrap_llm(llm)

    if retriever_factory is None:
        raise ValueError("retriever_factory is required to create per-request retrievers")

    # Use get_retrievers() for chunk-only retrieval (matches API pattern)
    # RAPTOR summaries are fetched separately via get_raptor_summaries()
    return retriever_factory.get_retrievers(
        llm=llm,
        language=language,
        nodes=nodes,
        vector_store=vector_store,
        project_id=project_id,
    )


def fast_retrieve(
    nodes: List[TextNode],
    query: str,
    project_id: str,
    vector_store: Any,
    retriever_factory: Any,
    llm: Optional[Any] = None,
    source_ids: Optional[List[str]] = None,
    top_k: int = 6,
    language: str = "eng",
) -> List[NodeWithScore]:
    """Fast retrieval using the API pattern - no global state.

    This is the core retrieval function extracted from /api/query.
    It creates a per-request retriever and returns relevant chunks.

    Args:
        nodes: Cached nodes for the project
        query: User's query string
        project_id: UUID of the project to query
        vector_store: PGVectorStore instance
        retriever_factory: LocalRetriever instance
        llm: LLM instance (defaults to Settings.llm)
        source_ids: Optional list of source IDs to filter
        top_k: Maximum number of results to return
        language: Language code for prompts

    Returns:
        List of NodeWithScore containing relevant chunks

    Example:
        nodes = pipeline._get_cached_nodes(project_id)
        results = fast_retrieve(
            nodes=nodes,
            query="What are the key findings?",
            project_id=project_id,
            vector_store=pipeline._vector_store,
            retriever_factory=pipeline._engine._retriever,
        )
    """
    if not nodes:
        logger.debug(f"No nodes found for project {project_id}")
        return []

    try:
        # Create per-request retriever (thread-safe)
        retriever = create_retriever(
            nodes=nodes,
            project_id=project_id,
            vector_store=vector_store,
            llm=llm,
            source_ids=source_ids,
            language=language,
            retriever_factory=retriever_factory,
        )

        # Retrieve relevant chunks
        query_bundle = QueryBundle(query_str=query)
        retrieval_results = retriever.retrieve(query_bundle)

        # Limit to top_k
        return retrieval_results[:top_k]

    except Exception as e:
        logger.warning(f"Retrieval failed [{type(e).__name__}]: {e}", exc_info=True)
        return []


def enhanced_retrieve(
    nodes: List[TextNode],
    query: str,
    project_id: str,
    vector_store: Any,
    retriever_factory: Any,
    llm: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    source_ids: Optional[List[str]] = None,
    top_k: int = 6,
    use_raptor: bool = True,
    use_reranker: bool = True,
    raptor_top_k: int = 5,
    min_raptor_score: float = 0.3,
    language: str = "eng",
) -> Tuple[List[NodeWithScore], List[Tuple[TextNode, float]], dict]:
    """Enhanced retrieval with RAPTOR-aware reranking.

    Unlike fast_retrieve(), this function:
    1. Retrieves both chunks and RAPTOR summaries
    2. Combines them for unified reranking
    3. Returns both types with proper relevance scores

    This is the recommended retrieval function for Chat V2 where
    quality matters more than raw speed.

    Args:
        nodes: Cached nodes for the project
        query: User's query string
        project_id: UUID of the project to query
        vector_store: PGVectorStore instance
        retriever_factory: LocalRetriever instance
        llm: LLM instance (defaults to Settings.llm)
        embed_model: Embedding model (defaults to Settings.embed_model)
        source_ids: Optional list of source IDs to filter
        top_k: Maximum number of chunk results to return
        use_raptor: Whether to include RAPTOR summaries in retrieval
        use_reranker: Whether to apply reranker
        raptor_top_k: Maximum number of RAPTOR summaries
        min_raptor_score: Minimum score for RAPTOR summaries
        language: Language code for prompts

    Returns:
        Tuple of (chunks, raptor_summaries, metadata_dict)
        - chunks: List of NodeWithScore for document chunks
        - raptor_summaries: List of (TextNode, score) for RAPTOR summaries
        - metadata: Dict with timing info and strategy used

    Example:
        nodes = pipeline._get_cached_nodes(project_id)
        chunks, summaries, metadata = enhanced_retrieve(
            nodes=nodes,
            query="What are the key findings?",
            project_id=project_id,
            vector_store=pipeline._vector_store,
            retriever_factory=pipeline._engine._retriever,
            use_raptor=True,
            use_reranker=True,
        )
    """
    from codeloom.core.services.retrieval_service import (
        RetrievalService,
        RetrievalRequest,
    )

    # Create request
    request = RetrievalRequest(
        query=query,
        project_id=project_id,
        use_raptor=use_raptor,
        use_reranker=use_reranker,
        top_k=top_k,
        raptor_top_k=raptor_top_k,
        min_raptor_score=min_raptor_score,
        source_ids=source_ids,
        language=language,
    )

    # Create service and execute
    service = RetrievalService()
    result = service.retrieve(
        request=request,
        nodes=nodes,
        llm=llm or Settings.llm,
        vector_store=vector_store,
        retriever_factory=retriever_factory,
        embed_model=embed_model,
    )

    # Return in the expected format
    metadata = {
        "strategy_used": result.strategy_used,
        "reranker_applied": result.reranker_applied,
        **result.timings,
    }

    return result.chunks, result.raptor_summaries, metadata


def get_raptor_summaries(
    query: str,
    project_id: str,
    vector_store: Any,
    embed_model: Optional[Any] = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> List[Tuple[TextNode, float]]:
    """Get RAPTOR hierarchical summaries for query context.

    RAPTOR summaries provide high-level document understanding
    that complements chunk-level retrieval.

    Args:
        query: User's query string
        project_id: UUID of the project
        vector_store: PGVectorStore instance with get_top_raptor_summaries method
        embed_model: Embedding model (defaults to Settings.embed_model)
        top_k: Maximum number of summaries to return
        min_score: Minimum relevance score threshold

    Returns:
        List of (TextNode, score) tuples for relevant summaries

    Example:
        summaries = get_raptor_summaries(
            query="What are the key findings?",
            project_id=project_id,
            vector_store=pipeline._vector_store,
        )
    """
    if not hasattr(vector_store, 'get_top_raptor_summaries'):
        logger.debug("Vector store does not support RAPTOR summaries")
        return []

    try:
        if embed_model is None:
            embed_model = Settings.embed_model

        # Get query embedding for similarity search
        query_embedding = embed_model.get_query_embedding(query)

        # Bounded lookup: max top_k summaries, tree_level >= 1
        raptor_results = vector_store.get_top_raptor_summaries(
            project_id=project_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Filter by relevance threshold
        filtered_summaries = [
            (node, score) for node, score in raptor_results
            if score >= min_score
        ]

        if filtered_summaries:
            logger.debug(f"Retrieved {len(filtered_summaries)} RAPTOR summaries for query")

        return filtered_summaries

    except Exception as e:
        logger.debug(f"RAPTOR summaries unavailable: {e}")
        return []
