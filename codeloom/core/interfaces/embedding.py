"""Abstract interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Any


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations can include:
    - HuggingFace embeddings (local)
    - OpenAI embeddings
    - Cohere embeddings
    - Custom embedding models
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Some embedding models use different embeddings for
        queries vs documents.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the embedding dimension.

        Returns:
            Integer dimension of the embedding vectors
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the current model name."""
        pass

    @abstractmethod
    def get_llama_index_embedding(self) -> Any:
        """
        Return a LlamaIndex-compatible embedding instance.

        This allows the provider to be used with LlamaIndex's
        vector stores and indexing.

        Returns:
            A LlamaIndex BaseEmbedding instance
        """
        pass

    def validate(self) -> bool:
        """
        Validate that the embedding model is working.

        Returns:
            True if model is working, False otherwise
        """
        try:
            result = self.embed(["test"])
            return len(result) == 1 and len(result[0]) == self.dimension
        except Exception:
            return False
