"""Abstract interface for retrieval strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from llama_index.core.schema import NodeWithScore


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.

    Implementations can include:
    - Hybrid (BM25 + Vector)
    - Pure semantic/vector search
    - Pure keyword/BM25 search
    - Custom retrieval strategies
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.

        Args:
            query: The search query string
            top_k: Number of top results to return
            filters: Optional metadata filters (e.g., notebook_id)

        Returns:
            List of NodeWithScore objects ranked by relevance
        """
        pass

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """
        Configure strategy parameters.

        Args:
            **kwargs: Strategy-specific configuration options
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for logging/display."""
        pass

    @property
    def description(self) -> str:
        """Return a description of the strategy."""
        return f"{self.name} retrieval strategy"
