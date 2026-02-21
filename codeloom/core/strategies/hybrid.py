"""Hybrid retrieval strategy combining BM25 and vector search."""

import logging
from typing import List, Optional, Dict, Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

from ..interfaces import RetrievalStrategy
from ..prompt import get_query_gen_prompt
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class TwoStageRetriever(QueryFusionRetriever):
    """
    Two-stage retriever that combines fusion retrieval with reranking.

    Stage 1: BM25 + Vector retrieval with query fusion
    Stage 2: Rerank results using cross-encoder model
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        setting: RAGSettings,
        rerank_top_n: int = 6,
        **kwargs
    ) -> None:
        super().__init__(retrievers, **kwargs)
        self._setting = setting
        self._rerank_model = SentenceTransformerRerank(
            top_n=rerank_top_n,
            model=self._setting.retriever.rerank_llm,
        )
        logger.debug(f"TwoStageRetriever initialized with rerank model: {self._setting.retriever.rerank_llm}")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid retrieval combining BM25 keyword search with vector similarity.

    Features:
    - BM25 for exact term matching
    - Vector search for semantic similarity
    - Query fusion for ambiguous queries
    - Two-stage retrieval with reranking for precise queries
    - Automatic strategy selection based on query clarity
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        setting: Optional[RAGSettings] = None,
        language: str = "eng",
        similarity_top_k: int = 20,
        rerank_top_k: int = 6,
        use_reranking: bool = True,
        fusion_weights: Optional[List[float]] = None,
    ):
        self._setting = setting or get_settings()
        self._language = language
        self._similarity_top_k = similarity_top_k
        self._rerank_top_k = rerank_top_k
        self._use_reranking = use_reranking
        self._fusion_weights = fusion_weights or [0.5, 0.5]

        self._nodes: List[BaseNode] = nodes or []
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[BaseRetriever] = None

        if nodes:
            self._build_index(nodes)

    def configure(self, **kwargs) -> None:
        """Configure strategy parameters."""
        if "similarity_top_k" in kwargs:
            self._similarity_top_k = kwargs["similarity_top_k"]
        if "rerank_top_k" in kwargs:
            self._rerank_top_k = kwargs["rerank_top_k"]
        if "use_reranking" in kwargs:
            self._use_reranking = kwargs["use_reranking"]
        if "fusion_weights" in kwargs:
            self._fusion_weights = kwargs["fusion_weights"]
        if "language" in kwargs:
            self._language = kwargs["language"]
        if "nodes" in kwargs:
            self._build_index(kwargs["nodes"])

    def _build_index(self, nodes: List[BaseNode]) -> None:
        """Build vector index from nodes."""
        if not nodes:
            logger.warning("No nodes provided for indexing")
            return

        self._nodes = nodes
        self._index = VectorStoreIndex(nodes=nodes)
        self._retriever = self._create_retriever()
        logger.debug(f"Built hybrid index with {len(nodes)} nodes")

    def _create_retriever(self) -> BaseRetriever:
        """Create the hybrid retriever."""
        if self._index is None:
            raise ValueError("Index not built. Call configure(nodes=...) first.")

        llm = Settings.llm

        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=False
        )

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            index=self._index,
            similarity_top_k=self._similarity_top_k,
            verbose=False
        )

        retrievers = [bm25_retriever, vector_retriever]

        if self._use_reranking:
            # Two-stage with reranking
            return TwoStageRetriever(
                retrievers=retrievers,
                retriever_weights=self._fusion_weights,
                setting=self._setting,
                rerank_top_n=self._rerank_top_k,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=self._similarity_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                verbose=False
            )
        else:
            # Simple fusion without reranking
            return QueryFusionRetriever(
                retrievers=retrievers,
                retriever_weights=self._fusion_weights,
                llm=llm,
                query_gen_prompt=get_query_gen_prompt(self._language),
                similarity_top_k=self._rerank_top_k,
                num_queries=self._setting.retriever.num_queries,
                mode=self._setting.retriever.fusion_mode,
                verbose=False
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """Retrieve relevant nodes using hybrid search."""
        if self._retriever is None:
            raise ValueError("Retriever not initialized. Provide nodes via configure().")

        # Apply metadata filters if provided
        if filters:
            filtered_nodes = self._filter_nodes(filters)
            if filtered_nodes:
                # Rebuild index with filtered nodes
                self._index = VectorStoreIndex(nodes=filtered_nodes)
                self._retriever = self._create_retriever()

        results = self._retriever.retrieve(query)

        # Apply top_k limit
        return results[:top_k]

    def _filter_nodes(self, filters: Dict[str, Any]) -> List[BaseNode]:
        """Filter nodes by metadata."""
        filtered = []
        for node in self._nodes:
            metadata = node.metadata or {}
            match = all(
                metadata.get(key) == value
                for key, value in filters.items()
            )
            if match:
                filtered.append(node)

        logger.debug(f"Filtered {len(filtered)} nodes from {len(self._nodes)} using {filters}")
        return filtered

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def description(self) -> str:
        return "Hybrid retrieval combining BM25 keyword search with vector similarity"
