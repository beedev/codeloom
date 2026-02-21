import logging
import os
from pathlib import Path
from typing import Optional, Set

import backoff
import requests
import yaml
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

from ...setting import get_settings, RAGSettings

load_dotenv()

logger = logging.getLogger(__name__)


def _load_models_from_yaml() -> dict[str, Set[str]]:
    """Load model sets from config/models.yaml.

    Returns:
        Dictionary mapping provider names to sets of model names.
    """
    # Find config/models.yaml relative to project root
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "models.yaml"

    model_sets = {
        "openai": set(),
        "claude": set(),
        "gemini": set(),
        "groq": set(),
    }

    if not config_path.exists():
        logger.warning(f"models.yaml not found at {config_path}, using empty model sets")
        return model_sets

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        providers = config.get("providers", {})

        # Map YAML provider names to internal names
        provider_mapping = {
            "openai": "openai",
            "google": "gemini",  # Google/Gemini models
            "anthropic": "claude",
            "groq": "groq",
        }

        for yaml_provider, internal_name in provider_mapping.items():
            provider_config = providers.get(yaml_provider, {})
            if provider_config.get("enabled", False):
                models = provider_config.get("models", [])
                for model in models:
                    model_name = model.get("name") if isinstance(model, dict) else model
                    if model_name:
                        model_sets[internal_name].add(model_name)

        logger.debug(f"Loaded models from YAML: {model_sets}")
        return model_sets

    except Exception as e:
        logger.error(f"Error loading models.yaml: {e}")
        return model_sets


# Load model sets from models.yaml at module initialization
_MODEL_SETS = _load_models_from_yaml()


def _log_groq_backoff(details: dict) -> None:
    """Log Groq rate limit backoff events."""
    logger.warning(
        f"Groq rate limited, retry {details['tries']}/5 in {details['wait']:.1f}s"
    )

# Cache for LLM models to avoid re-initialization
_llm_cache: dict = {}


def _get_rate_limit_exceptions():
    """Get rate limit exception classes lazily."""
    from groq import RateLimitError as GroqRateLimitError
    from openai import RateLimitError as OpenAIRateLimitError
    return (GroqRateLimitError, OpenAIRateLimitError)


class GroqWithBackoff:
    """LlamaIndex Groq LLM wrapper with rate limit backoff.

    Wraps the Groq LLM and intercepts complete/stream_complete calls
    to apply exponential backoff on rate limit errors.
    """

    def __init__(self, model_name: str, api_key: str, temperature: float):
        from llama_index.llms.groq import Groq
        self._llm = Groq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )
        self._rate_limit_exceptions = _get_rate_limit_exceptions()
        logger.info(f"Created Groq LLM with rate limit backoff: {model_name}")

    def get_raw_llm(self):
        """Return the underlying LlamaIndex LLM for Settings.llm compatibility."""
        return self._llm

    def __getattr__(self, name):
        """Delegate all attribute access to wrapped LLM."""
        return getattr(self._llm, name)

    @property
    def model(self):
        """Expose model name for compatibility."""
        return self._llm.model

    @property
    def metadata(self):
        """Expose metadata for LlamaIndex compatibility."""
        return self._llm.metadata

    def complete(self, prompt, **kwargs):
        """Complete with exponential backoff on rate limits."""
        @backoff.on_exception(
            backoff.expo,
            self._rate_limit_exceptions,
            max_tries=5,
            max_time=60,
            on_backoff=_log_groq_backoff,
        )
        def _do_complete():
            return self._llm.complete(prompt, **kwargs)
        return _do_complete()

    def stream_complete(self, prompt, **kwargs):
        """Stream complete with exponential backoff on rate limits."""
        @backoff.on_exception(
            backoff.expo,
            self._rate_limit_exceptions,
            max_tries=5,
            max_time=60,
            on_backoff=_log_groq_backoff,
        )
        def _do_stream():
            return self._llm.stream_complete(prompt, **kwargs)
        return _do_stream()

    def chat(self, messages, **kwargs):
        """Chat with exponential backoff on rate limits."""
        @backoff.on_exception(
            backoff.expo,
            self._rate_limit_exceptions,
            max_tries=5,
            max_time=60,
            on_backoff=_log_groq_backoff,
        )
        def _do_chat():
            return self._llm.chat(messages, **kwargs)
        return _do_chat()

    def stream_chat(self, messages, **kwargs):
        """Stream chat with exponential backoff on rate limits."""
        @backoff.on_exception(
            backoff.expo,
            self._rate_limit_exceptions,
            max_tries=5,
            max_time=60,
            on_backoff=_log_groq_backoff,
        )
        def _do_stream_chat():
            return self._llm.stream_chat(messages, **kwargs)
        return _do_stream_chat()


class LocalRAGModel:
    """Manages LLM model initialization and caching.

    Model sets are loaded dynamically from config/models.yaml.
    """

    # Class-level properties that access the dynamically loaded model sets
    @classmethod
    def _get_openai_models(cls) -> Set[str]:
        return _MODEL_SETS.get("openai", set())

    @classmethod
    def _get_claude_models(cls) -> Set[str]:
        return _MODEL_SETS.get("claude", set())

    @classmethod
    def _get_gemini_models(cls) -> Set[str]:
        return _MODEL_SETS.get("gemini", set())

    @classmethod
    def _get_groq_models(cls) -> Set[str]:
        return _MODEL_SETS.get("groq", set())

    # Keep class attributes for backward compatibility (property-like access)
    OPENAI_MODELS = property(lambda self: _MODEL_SETS.get("openai", set()))
    CLAUDE_MODELS = property(lambda self: _MODEL_SETS.get("claude", set()))
    GEMINI_MODELS = property(lambda self: _MODEL_SETS.get("gemini", set()))
    GROQ_MODELS = property(lambda self: _MODEL_SETS.get("groq", set()))

    @classmethod
    def set(
        cls,
        model_name: str = "",
        system_prompt: Optional[str] = None,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None
    ):
        """
        Get or create LLM model with caching.

        Args:
            model_name: Model name (uses settings default if empty)
            system_prompt: System prompt for the model
            host: Ollama host
            setting: RAGSettings instance

        Returns:
            LLM model instance
        """
        setting = setting or get_settings()
        model_name = model_name or setting.ollama.llm

        # Detect provider using dynamically loaded model sets
        if model_name in cls._get_openai_models():
            provider = "openai"
        elif model_name in cls._get_claude_models():
            provider = "claude"
        elif model_name in cls._get_gemini_models():
            provider = "gemini"
        elif model_name in cls._get_groq_models():
            provider = "groq"
        else:
            provider = "ollama"

        # Cache key includes provider and system prompt hash
        prompt_hash = hash(system_prompt) if system_prompt else "none"
        cache_key = f"{provider}_{model_name}_{prompt_hash}"

        # Check cache (but skip if system prompt changes)
        if cache_key in _llm_cache and system_prompt is None:
            logger.debug(f"Using cached LLM model: {model_name}")
            return _llm_cache[cache_key]

        # Create model based on provider
        if provider == "openai":
            model = OpenAI(
                model=model_name,
                temperature=setting.ollama.temperature
            )
        elif provider == "claude":
            from llama_index.llms.anthropic import Anthropic
            model = Anthropic(
                model=model_name,
                api_key=setting.anthropic.api_key,
                temperature=setting.anthropic.temperature,
                max_tokens=setting.anthropic.max_tokens
            )
        elif provider == "gemini":
            from llama_index.llms.gemini import Gemini
            # Gemini API requires model names to be prefixed with "models/"
            gemini_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            model = Gemini(
                model=gemini_model_name,
                api_key=setting.gemini.api_key,
                temperature=setting.gemini.temperature,
                max_tokens=setting.gemini.max_output_tokens
            )
        elif provider == "groq":
            # Create Groq LLM with rate limit backoff wrapper
            model = GroqWithBackoff(
                model_name=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=setting.ollama.temperature,
            )
        else:
            # Create Ollama model
            additional_kwargs = {
                "tfs_z": setting.ollama.tfs_z,
                "top_k": setting.ollama.top_k,
                "top_p": setting.ollama.top_p,
                "repeat_last_n": setting.ollama.repeat_last_n,
                "repeat_penalty": setting.ollama.repeat_penalty,
            }

            model = Ollama(
                model=model_name,
                system_prompt=system_prompt,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=setting.ollama.request_timeout,
                additional_kwargs=additional_kwargs
            )

        # Cache the model
        _llm_cache[cache_key] = model
        logger.debug(f"Created and cached {provider.upper()} model: {model_name}")

        return model

    @staticmethod
    def pull(host: str, model_name: str):
        """
        Pull LLM model from Ollama.

        Args:
            host: Ollama host
            model_name: Model to pull

        Returns:
            Response object from Ollama API
        """
        setting = get_settings()
        payload = {"name": model_name}
        url = f"http://{host}:{setting.ollama.port}/api/pull"

        try:
            return requests.post(url, json=payload, stream=True, timeout=30)
        except requests.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        """
        Check if LLM model exists on Ollama server.

        Args:
            host: Ollama host
            model_name: Model to check

        Returns:
            True if model exists, False otherwise
        """
        setting = get_settings()
        url = f"http://{host}:{setting.ollama.port}/api/tags"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            models = data.get("models")
            if not models:
                return False

            model_names = [m.get("name", "") for m in models]
            return model_name in model_names

        except requests.RequestException as e:
            logger.warning(f"Error checking model existence: {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing Ollama response: {e}")
            return False

    @staticmethod
    def clear_cache() -> None:
        """Clear the LLM model cache."""
        global _llm_cache
        _llm_cache.clear()
        logger.info("LLM model cache cleared")

    @staticmethod
    def list_available_models(host: str) -> list:
        """
        List available models on Ollama server.

        Args:
            host: Ollama host

        Returns:
            List of model names
        """
        setting = get_settings()
        url = f"http://{host}:{setting.ollama.port}/api/tags"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]

        except requests.RequestException as e:
            logger.warning(f"Error listing models: {e}")
            return []

    @classmethod
    def reload_models(cls) -> None:
        """Reload model sets from models.yaml.

        Call this if models.yaml has been modified and you want to
        pick up the changes without restarting the server.
        """
        global _MODEL_SETS
        _MODEL_SETS = _load_models_from_yaml()
        logger.info(f"Reloaded models from YAML: {list(_MODEL_SETS.keys())}")

    @classmethod
    def get_all_known_models(cls) -> dict[str, Set[str]]:
        """Get all known models by provider.

        Returns:
            Dictionary mapping provider names to sets of model names.
        """
        return {
            "openai": cls._get_openai_models(),
            "claude": cls._get_claude_models(),
            "gemini": cls._get_gemini_models(),
            "groq": cls._get_groq_models(),
        }
