"""Pure keyword/BM25 retrieval strategy."""

import logging
from typing import List, Optional, Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from ..interfaces import RetrievalStrategy
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class KeywordRetrievalStrategy(RetrievalStrategy):
    """
    Pure keyword retrieval using BM25 algorithm.

    Best for:
    - Queries with specific technical terms
    - Finding exact keyword matches
    - When semantic similarity is less important

    Features:
    - Fast BM25 term-frequency matching
    - Good for technical documentation
    - Works well with domain-specific vocabulary
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        setting: Optional[RAGSettings] = None,
        similarity_top_k: int = 10,
    ):
        self._setting = setting or get_settings()
        self._similarity_top_k = similarity_top_k

        self._nodes: List[BaseNode] = nodes or []
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[BM25Retriever] = None

        if nodes:
            self._build_index(nodes)

    def configure(self, **kwargs) -> None:
        """Configure strategy parameters."""
        if "similarity_top_k" in kwargs:
            self._similarity_top_k = kwargs["similarity_top_k"]
        if "nodes" in kwargs:
            self._build_index(kwargs["nodes"])

    def _build_index(self, nodes: List[BaseNode]) -> None:
        """Build BM25 index from nodes."""
        if not nodes:
            logger.warning("No nodes provided for indexing")
            return

        self._nodes = nodes
        # BM25 requires a VectorStoreIndex for initialization
        self._index = VectorStoreIndex(nodes=nodes)
        self._retriever = self._create_retriever()
        logger.debug(f"Built keyword index with {len(nodes)} nodes")

    def _create_retriever(self) -> BM25Retriever:
        """Create the BM25 retriever."""
        if self._index is None:
            raise ValueError("Index not built. Call configure(nodes=...) first.")

        return BM25Retriever.from_defaults(
            index=self._index,
            similarity_top_k=self._similarity_top_k,
            verbose=False
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """Retrieve relevant nodes using BM25 keyword search."""
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
        return "keyword"

    @property
    def description(self) -> str:
        return "Pure keyword retrieval using BM25 algorithm"
