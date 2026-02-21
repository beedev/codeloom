"""Pure semantic/vector retrieval strategy."""

import logging
from typing import List, Optional, Dict, Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, NodeWithScore

from ..interfaces import RetrievalStrategy
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class SemanticRetrievalStrategy(RetrievalStrategy):
    """
    Pure semantic retrieval using vector similarity search.

    Best for:
    - Queries requiring semantic understanding
    - Finding conceptually similar content
    - When exact keyword matching is not important

    Features:
    - Fast vector similarity search
    - Embedding-based semantic matching
    - Configurable similarity threshold
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        setting: Optional[RAGSettings] = None,
        similarity_top_k: int = 10,
        similarity_cutoff: Optional[float] = None,
    ):
        self._setting = setting or get_settings()
        self._similarity_top_k = similarity_top_k
        self._similarity_cutoff = similarity_cutoff

        self._nodes: List[BaseNode] = nodes or []
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[VectorIndexRetriever] = None

        if nodes:
            self._build_index(nodes)

    def configure(self, **kwargs) -> None:
        """Configure strategy parameters."""
        if "similarity_top_k" in kwargs:
            self._similarity_top_k = kwargs["similarity_top_k"]
        if "similarity_cutoff" in kwargs:
            self._similarity_cutoff = kwargs["similarity_cutoff"]
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
        logger.debug(f"Built semantic index with {len(nodes)} nodes")

    def _create_retriever(self) -> VectorIndexRetriever:
        """Create the vector retriever."""
        if self._index is None:
            raise ValueError("Index not built. Call configure(nodes=...) first.")

        return VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=False
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """Retrieve relevant nodes using semantic search."""
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

        # Apply similarity cutoff if configured
        if self._similarity_cutoff is not None:
            results = [
                r for r in results
                if r.score is not None and r.score >= self._similarity_cutoff
            ]

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
        return "semantic"

    @property
    def description(self) -> str:
        return "Pure semantic retrieval using vector similarity search"
