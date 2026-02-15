"""
Abstract interface for vector store implementations.

Provides a consistent API for different vector store backends
(PGVectorStore, ChromaDB, etc.) to enable dependency injection
and easier testing.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from llama_index.core.schema import BaseNode


class IVectorStore(ABC):
    """
    Abstract interface for vector store implementations.

    This interface defines the contract that all vector store implementations
    must fulfill, enabling:
    - Dependency injection for better testability
    - Easy swapping of vector store backends
    - Consistent API across different implementations
    """

    @abstractmethod
    def add_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: Optional[str] = None
    ) -> int:
        """
        Store nodes with embeddings.

        Args:
            nodes: List of nodes to store
            notebook_id: Optional notebook ID to associate with all nodes

        Returns:
            Number of nodes successfully added (after deduplication)
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Query for similar nodes using vector similarity.

        Args:
            query_embedding: Query vector embedding
            similarity_top_k: Maximum number of results to return
            filters: Optional metadata filters to apply

        Returns:
            List of (node, similarity_score) tuples, sorted by relevance
        """
        pass

    @abstractmethod
    def delete(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[dict] = None
    ) -> bool:
        """
        Delete nodes by ID or metadata filter.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion

        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """
        Get specific nodes by their IDs.

        Args:
            node_ids: List of node IDs to retrieve

        Returns:
            List of nodes matching the given IDs
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the vector store is connected and ready to use.

        Returns:
            True if connected and operational, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the vector store (clear cache and collection).

        This method should:
        - Clear all cached data
        - Clear the vector store collection
        - Reset internal state
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with statistics (count, size, etc.)
        """
        pass
