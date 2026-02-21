"""Unified Retrieval Service for Chat V2 and Content Studio.

This service provides RAPTOR-aware retrieval with explicit reranker control.
It combines chunk retrieval with RAPTOR summaries in a single ranking pass.

Usage:
    from codeloom.core.services import RetrievalService, RetrievalRequest

    service = RetrievalService()
    request = RetrievalRequest.for_chat_v2(query, project_id, max_sources=6)
    result = service.retrieve(request, nodes, llm, vector_store, retriever_factory)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from codeloom.core.providers.reranker_provider import (
    get_shared_reranker,
    is_reranker_enabled,
    set_reranker_config,
)
from codeloom.core.config import get_config_value
from .base import BaseService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRequest:
    """Configuration for a single retrieval operation.

    Encapsulates all retrieval parameters for clean API and type safety.
    Use factory methods for common use cases.
    """

    query: str
    project_id: str
    use_raptor: bool = True
    use_reranker: bool = True
    reranker_model: Optional[str] = None
    top_k: int = 6
    raptor_top_k: int = 5
    min_raptor_score: float = 0.3
    source_ids: Optional[List[str]] = None
    language: str = "eng"

    @classmethod
    def for_chat_v2(
        cls,
        query: str,
        project_id: str,
        max_sources: int = 6,
        use_raptor: bool = True,
        use_reranker: bool = True,
    ) -> "RetrievalRequest":
        """Factory for Chat V2 retrieval with sensible defaults.

        Chat V2 prioritizes quality: RAPTOR + reranker enabled by default.

        Args:
            query: User's query string
            project_id: UUID of the project to query
            max_sources: Maximum number of source chunks to return
            use_raptor: Whether to include RAPTOR summaries in retrieval
            use_reranker: Whether to apply reranker

        Returns:
            Configured RetrievalRequest
        """
        # Load config-based defaults
        config_use_raptor = get_config_value(
            "ingestion", "chat_v2", "use_raptor_in_retrieval", default=True
        )
        config_force_reranker = get_config_value(
            "ingestion", "chat_v2", "force_reranker", default=True
        )
        config_raptor_top_k = get_config_value(
            "ingestion", "chat_v2", "raptor_top_k", default=5
        )
        config_min_raptor_score = get_config_value(
            "ingestion", "chat_v2", "min_raptor_score", default=0.3
        )

        return cls(
            query=query,
            project_id=project_id,
            use_raptor=use_raptor and config_use_raptor,
            use_reranker=use_reranker and config_force_reranker,
            top_k=max_sources,
            raptor_top_k=config_raptor_top_k,
            min_raptor_score=config_min_raptor_score,
        )

    @classmethod
    def for_content_studio(
        cls,
        query: str,
        project_id: str,
        top_k: int = 10,
    ) -> "RetrievalRequest":
        """Factory for Content Studio retrieval.

        Content Studio needs broader context for generation tasks.

        Args:
            query: Content generation prompt
            project_id: UUID of the project to query
            top_k: Number of chunks to retrieve

        Returns:
            Configured RetrievalRequest
        """
        return cls(
            query=query,
            project_id=project_id,
            use_raptor=True,
            use_reranker=True,
            top_k=top_k,
            raptor_top_k=3,  # Fewer summaries for generation context
            min_raptor_score=0.4,  # Higher threshold for quality
        )

    @classmethod
    def for_sql_chat(
        cls,
        query: str,
        project_id: str,
        top_k: int = 5,
    ) -> "RetrievalRequest":
        """Factory for SQL Chat dictionary context retrieval.

        SQL Chat needs precise schema information, so we use reranking
        but skip RAPTOR (schema dictionaries don't have RAPTOR trees).

        Args:
            query: Natural language query + table names
            project_id: UUID of the SQL dictionary project
            top_k: Number of dictionary chunks to retrieve

        Returns:
            Configured RetrievalRequest
        """
        return cls(
            query=query,
            project_id=project_id,
            use_raptor=False,  # Schema docs don't have RAPTOR trees
            use_reranker=True,
            top_k=top_k,
        )


@dataclass
class RetrievalResult:
    """Result of a unified retrieval operation.

    Contains chunks, RAPTOR summaries, and metadata about the retrieval.
    """

    chunks: List[NodeWithScore] = field(default_factory=list)
    raptor_summaries: List[Tuple[TextNode, float]] = field(default_factory=list)
    strategy_used: str = "simple"  # "raptor_aware" | "hybrid" | "simple"
    reranker_applied: bool = False
    timings: dict = field(default_factory=dict)

    @property
    def all_nodes(self) -> List[NodeWithScore]:
        """Get all nodes (chunks + RAPTOR summaries) as NodeWithScore list."""
        all_nodes = list(self.chunks)
        for node, score in self.raptor_summaries:
            all_nodes.append(NodeWithScore(node=node, score=score))
        return all_nodes

    @property
    def total_count(self) -> int:
        """Total number of retrieved items."""
        return len(self.chunks) + len(self.raptor_summaries)


class RetrievalService(BaseService):
    """Unified retrieval service with RAPTOR and reranker control.

    This service provides a single entry point for retrieval across:
    - Chat V2 (browser-facing chat)
    - Content Studio (multimodal generation)
    - SQL Chat (dictionary context)

    Features:
    - RAPTOR-aware retrieval (summaries participate in ranking)
    - Explicit reranker control per request
    - Detailed timing metrics
    - Thread-safe operation

    Note: This does NOT modify the programmatic API (/api/query) which
    has its own per-request config via set_reranker_config().
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        db_manager: Optional[Any] = None,
        project_manager: Optional[Any] = None,
    ):
        """Initialize RetrievalService.

        Args:
            pipeline: Optional LocalRAGPipeline instance
            db_manager: Optional DatabaseManager instance
            project_manager: Optional ProjectManager instance
        """
        super().__init__(pipeline, db_manager, project_manager)
        logger.debug("RetrievalService initialized")

    def retrieve(
        self,
        request: RetrievalRequest,
        nodes: List[TextNode],
        llm: Any,
        vector_store: Any,
        retriever_factory: Any,
        embed_model: Optional[Any] = None,
    ) -> RetrievalResult:
        """Execute unified retrieval with RAPTOR and reranker control.

        This method combines chunk retrieval with RAPTOR summaries in a
        single ranking pass, then applies reranking if enabled.

        Args:
            request: RetrievalRequest with query and configuration
            nodes: Cached nodes for the project
            llm: LLM instance for query expansion
            vector_store: PGVectorStore instance
            retriever_factory: LocalRetriever instance
            embed_model: Embedding model (defaults to Settings.embed_model)

        Returns:
            RetrievalResult with chunks, summaries, and metadata
        """
        timings = {}
        start_time = time.time()

        if embed_model is None:
            embed_model = Settings.embed_model

        # Early return if no nodes
        if not nodes:
            logger.debug(f"No nodes for project {request.project_id}")
            return RetrievalResult(
                strategy_used="empty",
                timings={"total_ms": 0},
            )

        # Step 1: Get chunks via fast_retrieve pattern
        t1 = time.time()
        chunks = self._retrieve_chunks(
            request=request,
            nodes=nodes,
            llm=llm,
            vector_store=vector_store,
            retriever_factory=retriever_factory,
        )
        timings["chunk_retrieval_ms"] = int((time.time() - t1) * 1000)

        # Step 2: Get RAPTOR summaries if enabled
        raptor_summaries = []
        if request.use_raptor:
            t2 = time.time()
            raptor_summaries = self._retrieve_raptor_summaries(
                request=request,
                vector_store=vector_store,
                embed_model=embed_model,
            )
            timings["raptor_retrieval_ms"] = int((time.time() - t2) * 1000)

        # Step 3: Combine and optionally rerank
        strategy_used = "hybrid" if chunks else "simple"
        reranker_applied = False

        if request.use_reranker and (chunks or raptor_summaries):
            t3 = time.time()
            chunks, raptor_summaries, reranker_applied = self._apply_reranking(
                request=request,
                chunks=chunks,
                raptor_summaries=raptor_summaries,
            )
            timings["reranking_ms"] = int((time.time() - t3) * 1000)
            if reranker_applied:
                strategy_used = "raptor_aware" if raptor_summaries else "hybrid_reranked"

        timings["total_ms"] = int((time.time() - start_time) * 1000)

        logger.info(
            f"Retrieval: {len(chunks)} chunks, {len(raptor_summaries)} RAPTOR summaries, "
            f"strategy={strategy_used}, reranker={reranker_applied}, "
            f"time={timings['total_ms']}ms"
        )

        return RetrievalResult(
            chunks=chunks,
            raptor_summaries=raptor_summaries,
            strategy_used=strategy_used,
            reranker_applied=reranker_applied,
            timings=timings,
        )

    def _retrieve_chunks(
        self,
        request: RetrievalRequest,
        nodes: List[TextNode],
        llm: Any,
        vector_store: Any,
        retriever_factory: Any,
    ) -> List[NodeWithScore]:
        """Retrieve chunks using the fast_retrieve pattern.

        Args:
            request: RetrievalRequest with query and configuration
            nodes: Cached nodes for the project
            llm: LLM instance
            vector_store: PGVectorStore instance
            retriever_factory: LocalRetriever instance

        Returns:
            List of NodeWithScore for retrieved chunks
        """
        try:
            # Import here to avoid circular imports
            from codeloom.core.stateless import fast_retrieve

            return fast_retrieve(
                nodes=nodes,
                query=request.query,
                project_id=request.project_id,
                vector_store=vector_store,
                retriever_factory=retriever_factory,
                llm=llm,
                source_ids=request.source_ids,
                top_k=request.top_k,
                language=request.language,
            )
        except Exception as e:
            logger.warning(f"Chunk retrieval failed: {e}", exc_info=True)
            return []

    def _retrieve_raptor_summaries(
        self,
        request: RetrievalRequest,
        vector_store: Any,
        embed_model: Any,
    ) -> List[Tuple[TextNode, float]]:
        """Retrieve RAPTOR hierarchical summaries.

        Args:
            request: RetrievalRequest with query and configuration
            vector_store: PGVectorStore with get_top_raptor_summaries method
            embed_model: Embedding model for query embedding

        Returns:
            List of (TextNode, score) tuples
        """
        if not hasattr(vector_store, "get_top_raptor_summaries"):
            logger.debug("Vector store does not support RAPTOR summaries")
            return []

        try:
            # Get query embedding
            query_embedding = embed_model.get_query_embedding(request.query)

            # Retrieve RAPTOR summaries
            raptor_results = vector_store.get_top_raptor_summaries(
                project_id=request.project_id,
                query_embedding=query_embedding,
                top_k=request.raptor_top_k,
            )

            # Filter by minimum score threshold
            filtered = [
                (node, score)
                for node, score in raptor_results
                if score >= request.min_raptor_score
            ]

            if filtered:
                logger.debug(f"Retrieved {len(filtered)} RAPTOR summaries (filtered from {len(raptor_results)})")

            return filtered

        except Exception as e:
            logger.debug(f"RAPTOR retrieval failed: {e}")
            return []

    def _apply_reranking(
        self,
        request: RetrievalRequest,
        chunks: List[NodeWithScore],
        raptor_summaries: List[Tuple[TextNode, float]],
    ) -> Tuple[List[NodeWithScore], List[Tuple[TextNode, float]], bool]:
        """Apply reranking to combined chunks and RAPTOR summaries.

        When RAPTOR is enabled, summaries participate in the reranking
        alongside chunks, allowing them to compete for relevance.

        Args:
            request: RetrievalRequest with reranker configuration
            chunks: Retrieved chunks with scores
            raptor_summaries: RAPTOR summaries with scores

        Returns:
            Tuple of (reranked_chunks, reranked_summaries, reranker_was_applied)
        """
        # Check if reranker is globally enabled
        if not is_reranker_enabled():
            logger.debug("Reranker disabled globally")
            return chunks, raptor_summaries, False

        # Get shared reranker
        reranker = get_shared_reranker(
            model=request.reranker_model or "base",
            top_n=request.top_k + request.raptor_top_k,  # Room for both types
        )

        if reranker is None:
            logger.debug("Reranker not available")
            return chunks, raptor_summaries, False

        # If no RAPTOR summaries, just rerank chunks
        if not raptor_summaries:
            query_bundle = QueryBundle(query_str=request.query)
            reranked = reranker.postprocess_nodes(chunks, query_bundle)
            return reranked[: request.top_k], [], True

        # Combine chunks and RAPTOR summaries for unified ranking
        all_nodes = list(chunks)
        raptor_node_ids = set()
        for node, score in raptor_summaries:
            node_with_score = NodeWithScore(node=node, score=score)
            all_nodes.append(node_with_score)
            raptor_node_ids.add(node.node_id)

        # Rerank combined list
        query_bundle = QueryBundle(query_str=request.query)
        reranked = reranker.postprocess_nodes(all_nodes, query_bundle)

        # Separate back into chunks and summaries, preserving order
        final_chunks = []
        final_summaries = []
        chunk_count = 0
        summary_count = 0

        for node_with_score in reranked:
            node_id = node_with_score.node.node_id
            if node_id in raptor_node_ids:
                if summary_count < request.raptor_top_k:
                    final_summaries.append((node_with_score.node, node_with_score.score))
                    summary_count += 1
            else:
                if chunk_count < request.top_k:
                    final_chunks.append(node_with_score)
                    chunk_count += 1

            # Stop if we have enough of both
            if chunk_count >= request.top_k and summary_count >= request.raptor_top_k:
                break

        logger.debug(
            f"Reranked: {len(final_chunks)} chunks, {len(final_summaries)} RAPTOR summaries"
        )

        return final_chunks, final_summaries, True


def create_retrieval_service(
    pipeline: Optional[Any] = None,
    db_manager: Optional[Any] = None,
    project_manager: Optional[Any] = None,
) -> RetrievalService:
    """Factory function to create a RetrievalService instance.

    Args:
        pipeline: Optional LocalRAGPipeline instance
        db_manager: Optional DatabaseManager instance
        project_manager: Optional ProjectManager instance

    Returns:
        Configured RetrievalService instance
    """
    return RetrievalService(pipeline, db_manager, project_manager)
