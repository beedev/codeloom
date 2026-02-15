"""Plugin registry for managing swappable RAG components."""

import logging
import os
from typing import Dict, Type, Optional, Any, List

from .interfaces import (
    RetrievalStrategy,
    LLMProvider,
    EmbeddingProvider,
    ContentProcessor,
    ImageGenerationProvider,
    VisionProvider,
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for RAG component plugins.

    Provides factory methods for creating configured instances
    of retrieval strategies, LLM providers, embedding providers,
    and content processors.

    Usage:
        # Register a provider
        PluginRegistry.register_llm_provider("openai", OpenAIProvider)

        # Get an instance
        llm = PluginRegistry.get_llm_provider("openai", api_key="...")

        # Get configured from environment
        llm = PluginRegistry.get_configured_llm()
    """

    # Class-level registries
    _strategies: Dict[str, Type[RetrievalStrategy]] = {}
    _llm_providers: Dict[str, Type[LLMProvider]] = {}
    _embedding_providers: Dict[str, Type[EmbeddingProvider]] = {}
    _processors: Dict[str, Type[ContentProcessor]] = {}
    _image_providers: Dict[str, Type[ImageGenerationProvider]] = {}
    _vision_providers: Dict[str, Type[VisionProvider]] = {}

    # =========================================================================
    # Retrieval Strategy Registration
    # =========================================================================

    @classmethod
    def register_strategy(cls, name: str, strategy: Type[RetrievalStrategy]) -> None:
        """
        Register a retrieval strategy.

        Args:
            name: Strategy identifier (e.g., "hybrid", "semantic")
            strategy: Strategy class (not instance)
        """
        cls._strategies[name.lower()] = strategy
        logger.debug(f"Registered retrieval strategy: {name}")

    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> RetrievalStrategy:
        """
        Get a retrieval strategy instance.

        Args:
            name: Strategy identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured RetrievalStrategy instance

        Raises:
            KeyError: If strategy not found
        """
        name = name.lower()
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise KeyError(
                f"Unknown retrieval strategy: {name}. "
                f"Available: {available}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def get_configured_strategy(cls, **override_kwargs) -> RetrievalStrategy:
        """
        Get a retrieval strategy configured from environment.

        Environment variables:
            RETRIEVAL_STRATEGY: Strategy name (default: "hybrid")

        Args:
            **override_kwargs: Override environment configuration

        Returns:
            Configured RetrievalStrategy instance
        """
        strategy_name = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
        return cls.get_strategy(strategy_name, **override_kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Return list of registered strategy names."""
        return list(cls._strategies.keys())

    # =========================================================================
    # LLM Provider Registration
    # =========================================================================

    @classmethod
    def register_llm_provider(cls, name: str, provider: Type[LLMProvider]) -> None:
        """
        Register an LLM provider.

        Args:
            name: Provider identifier (e.g., "openai", "ollama")
            provider: Provider class (not instance)
        """
        cls._llm_providers[name.lower()] = provider
        logger.debug(f"Registered LLM provider: {name}")

    @classmethod
    def get_llm_provider(cls, name: str, **kwargs) -> LLMProvider:
        """
        Get an LLM provider instance.

        Args:
            name: Provider identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured LLMProvider instance

        Raises:
            KeyError: If provider not found
        """
        name = name.lower()
        if name not in cls._llm_providers:
            available = ", ".join(cls._llm_providers.keys())
            raise KeyError(
                f"Unknown LLM provider: {name}. "
                f"Available: {available}"
            )
        return cls._llm_providers[name](**kwargs)

    @classmethod
    def get_configured_llm(cls, **override_kwargs) -> LLMProvider:
        """
        Get an LLM provider configured from environment.

        Environment variables:
            LLM_PROVIDER: Provider name (default: "ollama")
            LLM_MODEL: Model name (default: provider-specific)

        Args:
            **override_kwargs: Override environment configuration

        Returns:
            Configured LLMProvider instance
        """
        provider_name = os.getenv("LLM_PROVIDER", "ollama")
        model = os.getenv("LLM_MODEL")

        kwargs = {}
        if model:
            kwargs["model"] = model
        kwargs.update(override_kwargs)

        return cls.get_llm_provider(provider_name, **kwargs)

    @classmethod
    def list_llm_providers(cls) -> List[str]:
        """Return list of registered LLM provider names."""
        return list(cls._llm_providers.keys())

    # =========================================================================
    # Embedding Provider Registration
    # =========================================================================

    @classmethod
    def register_embedding_provider(
        cls, name: str, provider: Type[EmbeddingProvider]
    ) -> None:
        """
        Register an embedding provider.

        Args:
            name: Provider identifier (e.g., "huggingface", "openai")
            provider: Provider class (not instance)
        """
        cls._embedding_providers[name.lower()] = provider
        logger.debug(f"Registered embedding provider: {name}")

    @classmethod
    def get_embedding_provider(cls, name: str, **kwargs) -> EmbeddingProvider:
        """
        Get an embedding provider instance.

        Args:
            name: Provider identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured EmbeddingProvider instance

        Raises:
            KeyError: If provider not found
        """
        name = name.lower()
        if name not in cls._embedding_providers:
            available = ", ".join(cls._embedding_providers.keys())
            raise KeyError(
                f"Unknown embedding provider: {name}. "
                f"Available: {available}"
            )
        return cls._embedding_providers[name](**kwargs)

    @classmethod
    def get_configured_embedding(cls, **override_kwargs) -> EmbeddingProvider:
        """
        Get an embedding provider configured from environment.

        Environment variables:
            EMBEDDING_PROVIDER: Provider name (default: "huggingface")
            EMBEDDING_MODEL: Model name (default: provider-specific)

        Args:
            **override_kwargs: Override environment configuration

        Returns:
            Configured EmbeddingProvider instance
        """
        provider_name = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        model = os.getenv("EMBEDDING_MODEL")

        kwargs = {}
        if model:
            kwargs["model"] = model
        kwargs.update(override_kwargs)

        return cls.get_embedding_provider(provider_name, **kwargs)

    @classmethod
    def list_embedding_providers(cls) -> List[str]:
        """Return list of registered embedding provider names."""
        return list(cls._embedding_providers.keys())

    # =========================================================================
    # Content Processor Registration
    # =========================================================================

    @classmethod
    def register_processor(cls, name: str, processor: Type[ContentProcessor]) -> None:
        """
        Register a content processor.

        Args:
            name: Processor identifier (e.g., "pdf", "docx")
            processor: Processor class (not instance)
        """
        cls._processors[name.lower()] = processor
        logger.debug(f"Registered content processor: {name}")

    @classmethod
    def get_processor(cls, name: str, **kwargs) -> ContentProcessor:
        """
        Get a content processor instance.

        Args:
            name: Processor identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured ContentProcessor instance

        Raises:
            KeyError: If processor not found
        """
        name = name.lower()
        if name not in cls._processors:
            available = ", ".join(cls._processors.keys())
            raise KeyError(
                f"Unknown content processor: {name}. "
                f"Available: {available}"
            )
        return cls._processors[name](**kwargs)

    @classmethod
    def get_processor_for_file(cls, filename: str, **kwargs) -> Optional[ContentProcessor]:
        """
        Get a processor that can handle the given file.

        Args:
            filename: File name to find processor for
            **kwargs: Configuration options passed to constructor

        Returns:
            Matching ContentProcessor instance, or None if no match
        """
        for name, processor_cls in cls._processors.items():
            processor = processor_cls(**kwargs)
            if processor.can_process(filename):
                return processor
        return None

    @classmethod
    def list_processors(cls) -> List[str]:
        """Return list of registered processor names."""
        return list(cls._processors.keys())

    @classmethod
    def list_supported_file_types(cls) -> List[str]:
        """Return list of all supported file extensions."""
        extensions = set()
        for processor_cls in cls._processors.values():
            # Create temporary instance to get supported types
            try:
                processor = processor_cls()
                extensions.update(processor.supported_types)
            except Exception:
                pass
        return sorted(extensions)

    # =========================================================================
    # Image Generation Provider Registration
    # =========================================================================

    @classmethod
    def register_image_provider(
        cls, name: str, provider: Type[ImageGenerationProvider]
    ) -> None:
        """
        Register an image generation provider.

        Args:
            name: Provider identifier (e.g., "gemini", "dalle")
            provider: Provider class (not instance)
        """
        cls._image_providers[name.lower()] = provider
        logger.debug(f"Registered image provider: {name}")

    @classmethod
    def get_image_provider(cls, name: str, **kwargs) -> ImageGenerationProvider:
        """
        Get an image generation provider instance.

        Args:
            name: Provider identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured ImageGenerationProvider instance

        Raises:
            KeyError: If provider not found
        """
        name = name.lower()
        if name not in cls._image_providers:
            available = ", ".join(cls._image_providers.keys())
            raise KeyError(
                f"Unknown image provider: {name}. "
                f"Available: {available}"
            )
        return cls._image_providers[name](**kwargs)

    @classmethod
    def get_configured_image_provider(cls, **override_kwargs) -> ImageGenerationProvider:
        """
        Get an image generation provider configured from environment.

        Environment variables:
            IMAGE_GENERATION_PROVIDER: Provider name (default: "gemini")
            GEMINI_IMAGE_MODEL: Model name (default: gemini-2.0-flash-exp)

        Args:
            **override_kwargs: Override environment configuration

        Returns:
            Configured ImageGenerationProvider instance
        """
        provider_name = os.getenv("IMAGE_GENERATION_PROVIDER", "gemini")
        model = os.getenv("GEMINI_IMAGE_MODEL")

        kwargs = {}
        if model:
            kwargs["model"] = model
        kwargs.update(override_kwargs)

        return cls.get_image_provider(provider_name, **kwargs)

    @classmethod
    def list_image_providers(cls) -> List[str]:
        """Return list of registered image provider names."""
        return list(cls._image_providers.keys())

    # =========================================================================
    # Vision Provider Registration
    # =========================================================================

    @classmethod
    def register_vision_provider(
        cls, name: str, provider: Type[VisionProvider]
    ) -> None:
        """
        Register a vision provider.

        Args:
            name: Provider identifier (e.g., "gemini", "openai")
            provider: Provider class (not instance)
        """
        cls._vision_providers[name.lower()] = provider
        logger.debug(f"Registered vision provider: {name}")

    @classmethod
    def get_vision_provider(cls, name: str, **kwargs) -> VisionProvider:
        """
        Get a vision provider instance.

        Args:
            name: Provider identifier
            **kwargs: Configuration options passed to constructor

        Returns:
            Configured VisionProvider instance

        Raises:
            KeyError: If provider not found
        """
        name = name.lower()
        if name not in cls._vision_providers:
            available = ", ".join(cls._vision_providers.keys())
            raise KeyError(
                f"Unknown vision provider: {name}. "
                f"Available: {available}"
            )
        return cls._vision_providers[name](**kwargs)

    @classmethod
    def get_configured_vision_provider(cls, **override_kwargs) -> VisionProvider:
        """
        Get a vision provider configured from environment.

        Environment variables:
            VISION_PROVIDER: Provider name (default: "gemini")
            GEMINI_VISION_MODEL: Gemini model name
            OPENAI_VISION_MODEL: OpenAI model name

        Args:
            **override_kwargs: Override environment configuration

        Returns:
            Configured VisionProvider instance
        """
        provider_name = os.getenv("VISION_PROVIDER", "gemini")

        kwargs = {}
        if provider_name == "gemini":
            model = os.getenv("GEMINI_VISION_MODEL")
            if model:
                kwargs["model"] = model
        elif provider_name == "openai":
            model = os.getenv("OPENAI_VISION_MODEL")
            if model:
                kwargs["model"] = model

        kwargs.update(override_kwargs)
        return cls.get_vision_provider(provider_name, **kwargs)

    @classmethod
    def list_vision_providers(cls) -> List[str]:
        """Return list of registered vision provider names."""
        return list(cls._vision_providers.keys())

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registries. Useful for testing."""
        cls._strategies.clear()
        cls._llm_providers.clear()
        cls._embedding_providers.clear()
        cls._processors.clear()
        cls._image_providers.clear()
        cls._vision_providers.clear()
        logger.debug("Cleared all plugin registries")

    @classmethod
    def get_registry_stats(cls) -> Dict[str, int]:
        """Return counts of registered plugins by type."""
        return {
            "strategies": len(cls._strategies),
            "llm_providers": len(cls._llm_providers),
            "embedding_providers": len(cls._embedding_providers),
            "processors": len(cls._processors),
            "image_providers": len(cls._image_providers),
            "vision_providers": len(cls._vision_providers),
        }

    @classmethod
    def discover_plugins(cls) -> Dict[str, List[str]]:
        """
        Return all registered plugins organized by type.

        Returns:
            Dictionary mapping plugin type to list of names
        """
        return {
            "retrieval_strategies": cls.list_strategies(),
            "llm_providers": cls.list_llm_providers(),
            "embedding_providers": cls.list_embedding_providers(),
            "content_processors": cls.list_processors(),
            "image_providers": cls.list_image_providers(),
            "vision_providers": cls.list_vision_providers(),
        }
