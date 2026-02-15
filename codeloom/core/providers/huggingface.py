"""HuggingFace embedding provider implementation."""

import logging
import os
from typing import List, Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..interfaces import EmbeddingProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace embedding provider for local embeddings.

    Supports many sentence-transformer models:
    - nomic-ai/nomic-embed-text-v1.5 (768 dims)
    - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    - BAAI/bge-small-en-v1.5 (384 dims)
    - And many more
    """

    # Common models with their dimensions
    MODEL_DIMENSIONS = {
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "thenlper/gte-small": 384,
        "thenlper/gte-base": 768,
        "thenlper/gte-large": 1024,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        trust_remote_code: bool = True,
        setting: Optional[RAGSettings] = None,
    ):
        self._setting = setting or get_settings()

        self._model = (
            model or
            os.getenv("EMBEDDING_MODEL") or
            self._setting.embedding.name
        )
        self._trust_remote_code = trust_remote_code

        self._embedding: Optional[HuggingFaceEmbedding] = None
        self._dimension: Optional[int] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the embedding model."""
        self._embedding = HuggingFaceEmbedding(
            model_name=self._model,
            trust_remote_code=self._trust_remote_code
        )

        # Get dimension from known models or detect dynamically
        if self._model in self.MODEL_DIMENSIONS:
            self._dimension = self.MODEL_DIMENSIONS[self._model]
        else:
            # Detect dimension by embedding a test string
            test_embedding = self._embedding.get_text_embedding("test")
            self._dimension = len(test_embedding)

        logger.debug(f"Initialized HuggingFace embedding: {self._model} (dim={self._dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self._embedding is None:
            raise RuntimeError("Embedding model not initialized")

        return [self._embedding.get_text_embedding(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        if self._embedding is None:
            raise RuntimeError("Embedding model not initialized")

        return self._embedding.get_query_embedding(query)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Dimension not initialized")
        return self._dimension

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model

    def get_llama_index_embedding(self) -> Any:
        """Return LlamaIndex-compatible embedding instance."""
        return self._embedding

    def validate(self) -> bool:
        """Validate that the embedding model is working."""
        try:
            result = self.embed(["test"])
            return len(result) == 1 and len(result[0]) == self._dimension
        except Exception as e:
            logger.warning(f"Embedding validation failed: {e}")
            return False

    @classmethod
    def list_popular_models(cls) -> List[str]:
        """Return list of popular embedding models."""
        return sorted(cls.MODEL_DIMENSIONS.keys())

    @classmethod
    def get_model_dimension(cls, model: str) -> Optional[int]:
        """Get the dimension for a known model."""
        return cls.MODEL_DIMENSIONS.get(model)
