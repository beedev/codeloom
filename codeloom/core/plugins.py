"""Plugin auto-registration and initialization."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from . import defaults
from .registry import PluginRegistry
from .interfaces import (
    RetrievalStrategy,
    LLMProvider,
    EmbeddingProvider,
    ImageGenerationProvider,
    VisionProvider,
)
from .strategies import (
    HybridRetrievalStrategy,
    SemanticRetrievalStrategy,
    KeywordRetrievalStrategy,
)
from .providers import (
    OllamaLLMProvider,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    GroqLLMProvider,
    HuggingFaceEmbeddingProvider,
    GeminiImageProvider,
    GeminiVisionProvider,
    OpenAIVisionProvider,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Flag to track if plugins have been initialized
_plugins_initialized = False


def register_default_plugins() -> None:
    """Register all default plugins with the registry."""
    global _plugins_initialized

    if _plugins_initialized:
        logger.debug("Plugins already initialized, skipping")
        return

    # Register retrieval strategies
    PluginRegistry.register_strategy("hybrid", HybridRetrievalStrategy)
    PluginRegistry.register_strategy("semantic", SemanticRetrievalStrategy)
    PluginRegistry.register_strategy("keyword", KeywordRetrievalStrategy)

    # Register LLM providers
    PluginRegistry.register_llm_provider("ollama", OllamaLLMProvider)
    PluginRegistry.register_llm_provider("openai", OpenAILLMProvider)
    PluginRegistry.register_llm_provider("anthropic", AnthropicLLMProvider)
    PluginRegistry.register_llm_provider("groq", GroqLLMProvider)

    # Register embedding providers
    PluginRegistry.register_embedding_provider("huggingface", HuggingFaceEmbeddingProvider)

    # Register image generation providers
    PluginRegistry.register_image_provider("gemini", GeminiImageProvider)

    # Register vision providers
    PluginRegistry.register_vision_provider("gemini", GeminiVisionProvider)
    PluginRegistry.register_vision_provider("openai", OpenAIVisionProvider)

    _plugins_initialized = True
    logger.info(f"Registered plugins: {PluginRegistry.get_registry_stats()}")


def get_configured_llm(**override_kwargs) -> LLMProvider:
    """
    Get LLM provider configured from environment.

    Uses centralized defaults from codeloom.core.defaults.

    Returns:
        Configured LLMProvider instance
    """
    register_default_plugins()

    provider = defaults.LLM_PROVIDER.lower()
    model = defaults.LLM_MODEL

    kwargs = {}
    if model:
        kwargs["model"] = model
    kwargs.update(override_kwargs)

    return PluginRegistry.get_llm_provider(provider, **kwargs)


def get_configured_embedding(**override_kwargs) -> EmbeddingProvider:
    """
    Get embedding provider configured from environment.

    Uses centralized defaults from codeloom.core.defaults.

    Returns:
        Configured EmbeddingProvider instance
    """
    register_default_plugins()

    provider = defaults.EMBEDDING_PROVIDER.lower()
    model = defaults.EMBEDDING_MODEL

    kwargs = {}
    if model:
        kwargs["model"] = model
    kwargs.update(override_kwargs)

    return PluginRegistry.get_embedding_provider(provider, **kwargs)


def get_configured_strategy(**override_kwargs) -> RetrievalStrategy:
    """
    Get retrieval strategy configured from environment.

    Environment variables:
        RETRIEVAL_STRATEGY: Strategy name (hybrid, semantic, keyword)

    Returns:
        Configured RetrievalStrategy instance
    """
    register_default_plugins()

    strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid").lower()
    return PluginRegistry.get_strategy(strategy, **override_kwargs)


def get_configured_image_provider(**override_kwargs) -> ImageGenerationProvider:
    """
    Get image generation provider configured from environment.

    Uses centralized defaults from codeloom.core.defaults.

    Returns:
        Configured ImageGenerationProvider instance
    """
    register_default_plugins()

    provider = defaults.IMAGE_GENERATION_PROVIDER.lower()
    model = os.getenv("GEMINI_IMAGE_MODEL")

    kwargs = {}
    if model:
        kwargs["model"] = model
    kwargs.update(override_kwargs)

    return PluginRegistry.get_image_provider(provider, **kwargs)


def get_configured_vision_provider(**override_kwargs) -> VisionProvider:
    """
    Get vision provider configured from environment.

    Uses centralized defaults from codeloom.core.defaults.

    Returns:
        Configured VisionProvider instance
    """
    register_default_plugins()

    provider = defaults.VISION_PROVIDER.lower()

    kwargs = {}
    if provider == "gemini":
        model = os.getenv("GEMINI_VISION_MODEL")
        if model:
            kwargs["model"] = model
    elif provider == "openai":
        model = os.getenv("OPENAI_VISION_MODEL")
        if model:
            kwargs["model"] = model

    kwargs.update(override_kwargs)

    return PluginRegistry.get_vision_provider(provider, **kwargs)


def list_available_plugins() -> dict:
    """
    List all available plugins.

    Returns:
        Dictionary with plugin types and available options
    """
    register_default_plugins()
    return PluginRegistry.discover_plugins()


def get_plugin_info() -> dict:
    """
    Get detailed information about current configuration.

    Returns:
        Dictionary with current configuration and available options
    """
    register_default_plugins()

    return {
        "current_config": {
            "llm_provider": defaults.LLM_PROVIDER,
            "llm_model": defaults.LLM_MODEL or "(auto-detect from provider)",
            "embedding_provider": defaults.EMBEDDING_PROVIDER,
            "embedding_model": defaults.EMBEDDING_MODEL,
            "retrieval_strategy": os.getenv("RETRIEVAL_STRATEGY", "hybrid"),
            "image_provider": defaults.IMAGE_GENERATION_PROVIDER,
            "image_model": os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp"),
            "vision_provider": defaults.VISION_PROVIDER,
            "vision_model": os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash-exp"),
        },
        "available_plugins": PluginRegistry.discover_plugins(),
        "registry_stats": PluginRegistry.get_registry_stats(),
    }
