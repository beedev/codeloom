from .embedding import LocalEmbedding
from .model import LocalRAGModel
from .ingestion import LocalDataIngestion
from .vector_store import LocalVectorStore, PGVectorStore
from .engine import LocalChatEngine
from .prompt import get_system_prompt

# Plugin architecture (MVP 5)
from .registry import PluginRegistry
from .interfaces import (
    RetrievalStrategy,
    LLMProvider,
    EmbeddingProvider,
    ContentProcessor,
    ImageGenerationProvider,
)
from .plugins import (
    register_default_plugins,
    get_configured_llm,
    get_configured_embedding,
    get_configured_strategy,
    get_configured_image_provider,
    list_available_plugins,
    get_plugin_info,
)

__all__ = [
    # Legacy exports
    "LocalEmbedding",
    "LocalRAGModel",
    "LocalDataIngestion",
    "LocalVectorStore",
    "PGVectorStore",
    "LocalChatEngine",
    "get_system_prompt",
    # Plugin architecture
    "PluginRegistry",
    "RetrievalStrategy",
    "LLMProvider",
    "EmbeddingProvider",
    "ContentProcessor",
    "ImageGenerationProvider",
    "register_default_plugins",
    "get_configured_llm",
    "get_configured_embedding",
    "get_configured_strategy",
    "get_configured_image_provider",
    "list_available_plugins",
    "get_plugin_info",
]
