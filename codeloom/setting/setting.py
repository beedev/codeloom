"""RAG Settings Module.

Loads configuration from YAML files with environment variable and Python defaults as fallback.
Config files: config/ingestion.yaml, config/models.yaml
"""

import os
import yaml
from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from codeloom.core.config.config_loader import (
    load_ingestion_config,
    get_chunking_config,
    get_embedding_config,
    get_contextual_retrieval_config,
    get_retriever_settings,
    get_llm_settings,
)

load_dotenv()


# Find config directory relative to project root
def _get_config_path() -> Path:
    """Get path to config directory."""
    # Try relative to this file first
    current_dir = Path(__file__).parent.parent.parent
    config_path = current_dir / "config"
    if config_path.exists():
        return config_path
    # Fallback to current working directory
    return Path.cwd() / "config"


def _get(config: Dict[str, Any], key: str, default: Any) -> Any:
    """Get config value with fallback to default."""
    return config.get(key, default)


class OllamaSettings(BaseModel):
    """Ollama LLM settings (loaded from config/ingestion.yaml llm section)."""

    llm: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4.1"),
        description="LLM model"
    )
    keep_alive: str = Field(
        default_factory=lambda: _get(get_llm_settings(), "keep_alive", "1h"),
        description="Keep alive time for the server"
    )
    tfs_z: float = Field(
        default_factory=lambda: _get(get_llm_settings(), "tfs_z", 1.0),
        description="TFS normalization factor"
    )
    top_k: int = Field(
        default_factory=lambda: _get(get_llm_settings(), "top_k", 40),
        description="Top k sampling"
    )
    top_p: float = Field(
        default_factory=lambda: _get(get_llm_settings(), "top_p", 0.9),
        description="Top p sampling"
    )
    repeat_last_n: int = Field(
        default_factory=lambda: _get(get_llm_settings(), "repeat_last_n", 64),
        description="Repeat last n tokens"
    )
    repeat_penalty: float = Field(
        default_factory=lambda: _get(get_llm_settings(), "repeat_penalty", 1.1),
        description="Repeat penalty"
    )
    request_timeout: float = Field(
        default_factory=lambda: _get(get_llm_settings(), "request_timeout", 300),
        description="Request timeout"
    )
    port: int = Field(
        default=11434, description="Port number"
    )
    context_window: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_WINDOW", "128000")),
        description="Context window size"
    )
    temperature: float = Field(
        default_factory=lambda: _get(get_llm_settings(), "temperature", 0.1),
        description="Temperature"
    )
    chat_token_limit: int = Field(
        default_factory=lambda: int(os.getenv("CHAT_TOKEN_LIMIT", "32000")),
        description="Chat memory limit"
    )


class RetrieverSettings(BaseModel):
    """Retriever settings (loaded from config/ingestion.yaml retriever section)."""

    num_queries: int = Field(
        default_factory=lambda: _get(get_retriever_settings(), "num_queries", 3),
        description="Number of generated queries"
    )
    similarity_top_k: int = Field(
        default_factory=lambda: _get(get_retriever_settings(), "similarity_top_k", 20),
        description="Top k documents for initial retrieval"
    )
    retriever_weights: List[float] = Field(
        default_factory=lambda: _get(get_retriever_settings(), "retriever_weights", [0.5, 0.5]),
        description="Weights for retriever (BM25, Vector)"
    )
    top_k_rerank: int = Field(
        default_factory=lambda: _get(get_retriever_settings(), "top_k_rerank", 10),
        description="Top k after reranking (final results to LLM)"
    )
    rerank_llm: str = Field(
        default_factory=lambda: _get(get_retriever_settings(), "rerank_model", "mixedbread-ai/mxbai-rerank-large-v1"),
        description="Rerank LLM model"
    )
    fusion_mode: str = Field(
        default_factory=lambda: _get(get_retriever_settings(), "fusion_mode", "dist_based_score"),
        description="Fusion mode"
    )


class IngestionSettings(BaseModel):
    """Ingestion settings (loaded from config/ingestion.yaml chunking/embedding sections)."""

    embed_llm: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        description="Embedding LLM model"
    )
    embed_batch_size: int = Field(
        default_factory=lambda: _get(get_embedding_config(), "batch_size", 8),
        description="Embedding batch size"
    )
    cache_folder: str = Field(
        default_factory=lambda: _get(get_embedding_config(), "cache_folder", "data/huggingface"),
        description="Cache folder"
    )
    chunk_size: int = Field(
        default_factory=lambda: _get(get_chunking_config(), "chunk_size", 512),
        description="Document chunk size"
    )
    chunk_overlap: int = Field(
        default_factory=lambda: _get(get_chunking_config(), "chunk_overlap", 32),
        description="Document chunk overlap"
    )
    chunking_regex: str = Field(
        default_factory=lambda: _get(get_chunking_config(), "chunking_regex", "[^,.;。？！]+[,.;。？！]?"),
        description="Chunking regex"
    )
    paragraph_sep: str = Field(
        default_factory=lambda: _get(get_chunking_config(), "paragraph_sep", "\n \n"),
        description="Paragraph separator"
    )


class ContextualRetrievalSettings(BaseModel):
    """Settings for Contextual Retrieval (loaded from config/ingestion.yaml contextual_retrieval section).

    Enriches chunks with LLM-generated context during ingestion to improve
    retrieval for structured content like tables, lists, and technical data.
    """
    enabled: bool = Field(
        default_factory=lambda: _get(get_contextual_retrieval_config(), "enabled", False),
        description="Enable contextual retrieval enrichment during ingestion"
    )
    batch_size: int = Field(
        default_factory=lambda: _get(get_contextual_retrieval_config(), "batch_size", 5),
        description="Number of chunks to process in each batch"
    )
    max_concurrency: int = Field(
        default_factory=lambda: _get(get_contextual_retrieval_config(), "max_concurrency", 3),
        description="Maximum concurrent LLM calls for context generation"
    )


class AnthropicSettings(BaseModel):
    """Settings for Anthropic/Claude models."""
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key"
    )
    model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        description="Default Claude model"
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
        description="Temperature for Claude models"
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        description="Maximum tokens for Claude models"
    )


class GeminiSettings(BaseModel):
    """Settings for Google Gemini models."""
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY"),
        description="Google API key for Gemini"
    )
    model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        description="Default Gemini model"
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
        description="Temperature for Gemini models"
    )
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
        description="Maximum output tokens for Gemini models"
    )


class ImageGenerationSettings(BaseModel):
    """Settings for image generation."""
    provider: str = Field(
        default_factory=lambda: os.getenv("IMAGE_GENERATION_PROVIDER", "imagen"),
        description="Image generation provider (dalle|imagen)"
    )
    imagen_model: str = Field(
        default_factory=lambda: os.getenv("IMAGEN_MODEL", "imagen-3.0-generate-001"),
        description="Imagen model name"
    )
    num_images: int = Field(
        default_factory=lambda: int(os.getenv("IMAGEN_NUM_IMAGES", "1")),
        description="Number of images to generate"
    )
    aspect_ratio: str = Field(
        default_factory=lambda: os.getenv("IMAGEN_ASPECT_RATIO", "1:1"),
        description="Image aspect ratio"
    )
    output_format: str = Field(
        default_factory=lambda: os.getenv("IMAGEN_OUTPUT_FORMAT", "png"),
        description="Image output format"
    )
    output_dir: str = Field(
        default_factory=lambda: os.getenv("IMAGE_OUTPUT_DIR", "outputs/images"),
        description="Output directory for generated images"
    )
    max_image_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
        description="Maximum image file size in MB"
    )
    supported_formats: str = Field(
        default_factory=lambda: os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,webp"),
        description="Supported image formats (comma-separated)"
    )


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str
    display_name: Optional[str] = None
    enabled: bool = True


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""
    enabled: bool = True
    type: str = "api"  # "local" or "api"
    requires_api_key: bool = False
    env_key: Optional[str] = None
    models: List[ModelConfig] = []


class ModelsSettings(BaseModel):
    """Settings for available models loaded from config/models.yaml."""
    providers: Dict[str, ProviderConfig] = {}
    default_model: str = "llama3.1:latest"
    default_provider: str = "ollama"

    @classmethod
    def load_from_yaml(cls, config_path: Optional[Path] = None) -> "ModelsSettings":
        """Load models configuration from YAML file."""
        if config_path is None:
            config_path = _get_config_path() / "models.yaml"

        if not config_path.exists():
            # Return empty settings if config doesn't exist
            return cls()

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Parse providers
            providers = {}
            for provider_name, provider_data in data.get('providers', {}).items():
                if provider_data is None:
                    continue

                # Parse models for this provider
                models = []
                for model_data in provider_data.get('models', []):
                    if isinstance(model_data, dict):
                        models.append(ModelConfig(**model_data))
                    elif isinstance(model_data, str):
                        models.append(ModelConfig(name=model_data))

                providers[provider_name] = ProviderConfig(
                    enabled=provider_data.get('enabled', True),
                    type=provider_data.get('type', 'api'),
                    requires_api_key=provider_data.get('requires_api_key', False),
                    env_key=provider_data.get('env_key'),
                    models=models
                )

            return cls(
                providers=providers,
                default_model=data.get('default_model', 'llama3.1:latest'),
                default_provider=data.get('default_provider', 'ollama')
            )
        except Exception as e:
            print(f"Warning: Failed to load models config: {e}")
            return cls()

    def get_models_for_provider(self, provider: str) -> List[ModelConfig]:
        """Get list of models for a specific provider."""
        if provider in self.providers and self.providers[provider].enabled:
            return [m for m in self.providers[provider].models if m.enabled]
        return []

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled in config."""
        return provider in self.providers and self.providers[provider].enabled

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for a provider."""
        if provider not in self.providers:
            return False

        config = self.providers[provider]
        if not config.requires_api_key:
            return True

        if config.env_key:
            key = os.getenv(config.env_key, "")
            return bool(key and key != f"your_{provider}_api_key_here")

        return False


# Singleton for models settings
_models_settings_instance: ModelsSettings | None = None


def get_models_settings() -> ModelsSettings:
    """Get singleton ModelsSettings instance."""
    global _models_settings_instance
    if _models_settings_instance is None:
        _models_settings_instance = ModelsSettings.load_from_yaml()
    return _models_settings_instance


def reload_models_settings() -> ModelsSettings:
    """Reload models settings from config file."""
    global _models_settings_instance
    _models_settings_instance = ModelsSettings.load_from_yaml()
    return _models_settings_instance


class QueryTimeSettings(BaseModel):
    """Per-request query settings from UI controls.

    These settings are passed with each query and override the defaults
    loaded from config files. They are NOT persisted.

    Frontend → Backend Mapping:
    - search_style (0-100): 0=keyword (BM25), 100=semantic (vector)
    - result_depth: focused=10 results, balanced=20, comprehensive=40
    - temperature (0-100): maps to 0.0-2.0 for LLM
    """

    # Retrieval weights (computed from search_style)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 keyword weight")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Vector semantic weight")

    # Result depth (similarity_top_k)
    similarity_top_k: int = Field(default=20, ge=5, le=100, description="Number of results to retrieve")

    # LLM temperature (low = deterministic, high = creative)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM response temperature")

    @classmethod
    def from_frontend(
        cls,
        search_style: int = 50,
        result_depth: str = "balanced",
        temperature: int = 5
    ) -> "QueryTimeSettings":
        """Create settings from frontend UI values.

        Args:
            search_style: 0-100 slider (0=keyword, 100=semantic)
            result_depth: "focused" | "balanced" | "comprehensive"
            temperature: 0-100 slider (maps to 0.0-2.0)

        Returns:
            QueryTimeSettings instance
        """
        # Convert search_style to weights
        bm25_weight = (100 - search_style) / 100
        vector_weight = search_style / 100

        # Convert result_depth to similarity_top_k
        top_k_map = {"focused": 10, "balanced": 20, "comprehensive": 40}
        similarity_top_k = top_k_map.get(result_depth, 20)

        # Convert temperature slider to float (0-100 → 0.0-2.0)
        temp_float = temperature / 50  # 0→0, 50→1.0, 100→2.0

        return cls(
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            similarity_top_k=similarity_top_k,
            temperature=temp_float
        )


class RAGSettings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    image_generation: ImageGenerationSettings = Field(default_factory=ImageGenerationSettings)
    contextual_retrieval: ContextualRetrievalSettings = Field(default_factory=ContextualRetrievalSettings)


# Singleton pattern for settings - use this instead of creating new instances
_settings_instance: RAGSettings | None = None


@lru_cache(maxsize=1)
def get_settings() -> RAGSettings:
    """Get singleton RAGSettings instance. Use this instead of RAGSettings()."""
    return RAGSettings()


def get_settings_instance() -> RAGSettings:
    """Alternative singleton accessor without lru_cache."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = RAGSettings()
    return _settings_instance
