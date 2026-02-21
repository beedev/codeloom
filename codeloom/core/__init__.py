# Lazy imports to avoid triggering full dependency chain.
# This allows targeted imports like `from codeloom.core.db.models import Base`
# without pulling in embedding, LLM, vector store, etc.

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

_IMPORT_MAP = {
    "LocalEmbedding": ".embedding",
    "LocalRAGModel": ".model",
    "LocalDataIngestion": ".ingestion",
    "LocalVectorStore": ".vector_store",
    "PGVectorStore": ".vector_store",
    "LocalChatEngine": ".engine",
    "get_system_prompt": ".prompt",
    "PluginRegistry": ".registry",
    "RetrievalStrategy": ".interfaces",
    "LLMProvider": ".interfaces",
    "EmbeddingProvider": ".interfaces",
    "ContentProcessor": ".interfaces",
    "ImageGenerationProvider": ".interfaces",
    "register_default_plugins": ".plugins",
    "get_configured_llm": ".plugins",
    "get_configured_embedding": ".plugins",
    "get_configured_strategy": ".plugins",
    "get_configured_image_provider": ".plugins",
    "list_available_plugins": ".plugins",
    "get_plugin_info": ".plugins",
}


def __getattr__(name):
    if name in _IMPORT_MAP:
        import importlib
        module = importlib.import_module(_IMPORT_MAP[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module 'codeloom.core' has no attribute {name}")
