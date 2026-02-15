import os
import logging
from functools import lru_cache
from typing import Optional

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

from ...setting import get_settings, RAGSettings

load_dotenv()

logger = logging.getLogger(__name__)

# Cache for embedding models to avoid re-initialization
_embedding_cache: dict = {}


class LocalEmbedding:
    """Manages embedding model initialization and caching."""

    @staticmethod
    def set(
        model_name: Optional[str] = None,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None
    ):
        """
        Get or create embedding model with caching.

        Args:
            model_name: Override model name (uses settings default if None)
            host: Ollama host for model availability checks
            setting: RAGSettings instance

        Returns:
            Embedding model instance
        """
        setting = setting or get_settings()
        model_name = model_name or setting.ingestion.embed_llm

        # Check cache first
        cache_key = f"{model_name}_{host}"
        if cache_key in _embedding_cache:
            logger.debug(f"Using cached embedding model: {model_name}")
            return _embedding_cache[cache_key]

        # Create new embedding model
        # OpenAI embedding models (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
        openai_embed_models = {"text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"}
        if model_name in openai_embed_models:
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning(
                    "OPENAI_API_KEY not set. OpenAI embeddings may fail."
                )
            model = OpenAIEmbedding(model=model_name)
        else:
            cache_folder = os.path.join(
                os.getcwd(),
                setting.ingestion.cache_folder
            )
            model = HuggingFaceEmbedding(
                model_name=model_name,
                cache_folder=cache_folder,
                trust_remote_code=True,
                embed_batch_size=setting.ingestion.embed_batch_size,
                max_length=512  # Ensure chunks don't exceed model's token limit
            )

        # Cache the model
        _embedding_cache[cache_key] = model
        logger.debug(f"Created and cached embedding model: {model_name}")

        return model

    @staticmethod
    def pull(host: str, model_name: Optional[str] = None):
        """
        Pull embedding model from Ollama.

        Args:
            host: Ollama host
            model_name: Model to pull (uses settings default if None)

        Returns:
            Response object from Ollama API
        """
        setting = get_settings()
        model_name = model_name or setting.ingestion.embed_llm

        payload = {"name": model_name}
        url = f"http://{host}:{setting.ollama.port}/api/pull"

        try:
            return requests.post(url, json=payload, stream=True, timeout=30)
        except requests.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise

    @staticmethod
    def check_model_exist(
        host: str,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Check if embedding model exists on Ollama server.

        Args:
            host: Ollama host
            model_name: Model to check (uses settings default if None)

        Returns:
            True if model exists, False otherwise
        """
        setting = get_settings()
        model_name = model_name or setting.ingestion.embed_llm

        url = f"http://{host}:{setting.ollama.port}/api/tags"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            if not models:
                return False

            model_names = [m.get("name", "") for m in models]
            return model_name in model_names

        except requests.RequestException as e:
            logger.warning(f"Error checking model existence: {e}")
            return False
        except (KeyError, ValueError) as e:
            logger.warning(f"Error parsing Ollama response: {e}")
            return False

    @staticmethod
    def clear_cache() -> None:
        """Clear the embedding model cache."""
        global _embedding_cache
        _embedding_cache.clear()
        logger.info("Embedding model cache cleared")
