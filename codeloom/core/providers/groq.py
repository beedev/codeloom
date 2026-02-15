"""Groq LLM provider implementation.

Groq provides ultra-fast inference using their LPU (Language Processing Unit).
Supports Llama 4, Llama 3.x, Mixtral, GPT-OSS, and other models.

Speed: 300-1000+ tokens/second (vs ~50 for OpenAI)
Cost: ~10x cheaper than OpenAI

GPT-OSS Models (Apache 2.0 licensed):
- openai/gpt-oss-120b: 500+ t/s, $0.15 in / $0.75 out per 1M tokens
- openai/gpt-oss-20b: 1000+ t/s, $0.05 in / $0.15 out per 1M tokens

Rate Limiting: Implements exponential backoff for handling HTTP 429 errors.
"""

import logging
import os
from typing import Generator, Dict, Any, Optional

import backoff
from groq import RateLimitError as GroqRateLimitError
from openai import RateLimitError as OpenAIRateLimitError
from llama_index.llms.groq import Groq

from ..interfaces import LLMProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)

# Rate limit exceptions to catch (LlamaIndex Groq uses OpenAI-like interface)
RATE_LIMIT_EXCEPTIONS = (GroqRateLimitError, OpenAIRateLimitError)


def _log_backoff(details: dict) -> None:
    """Log rate limit backoff events."""
    logger.warning(
        f"Groq rate limited, retry {details['tries']}/5 in {details['wait']:.1f}s"
    )


class GroqLLMProvider(LLMProvider):
    """
    Groq LLM provider for ultra-fast inference.

    Supports:
    - Llama 4 series (Scout, Maverick)
    - Llama 3.x series (70B, 8B)
    - Mixtral 8x7B
    - Qwen, Gemma, and others
    """

    SUPPORTED_MODELS = {
        # Llama 4 series
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        # Llama 3.x series
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        # OpenAI GPT-OSS models (Apache 2.0 licensed)
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # Mixtral
        "mixtral-8x7b-32768",
        # Gemma
        "gemma2-9b-it",
        # Qwen
        "qwen-qwq-32b",
    }

    # Model aliases for convenience
    MODEL_ALIASES = {
        "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama3-70b": "llama-3.3-70b-versatile",
        "llama3-8b": "llama-3.1-8b-instant",
        "mixtral": "mixtral-8x7b-32768",
        # GPT-OSS models (Apache 2.0 licensed - open source)
        "gpt-oss": "openai/gpt-oss-120b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        setting: Optional[RAGSettings] = None,
    ):
        self._setting = setting or get_settings()

        # Get model from param, env, or default
        model = model or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        # Resolve aliases
        self._model = self.MODEL_ALIASES.get(model, model)

        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens or 8192  # Groq default max

        if not self._api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")

        self._llm: Optional[Groq] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Groq client."""
        kwargs = {
            "model": self._model,
            "api_key": self._api_key,
            "temperature": self._temperature,
        }

        if self._max_tokens:
            kwargs["max_tokens"] = self._max_tokens

        self._llm = Groq(**kwargs)

        logger.info(f"Initialized Groq provider with model: {self._model}")

    @backoff.on_exception(
        backoff.expo,
        RATE_LIMIT_EXCEPTIONS,
        max_tries=5,
        max_time=60,
        on_backoff=_log_backoff,
    )
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt.

        Implements exponential backoff for rate limit (HTTP 429) errors.
        Max 5 retries over 60 seconds with exponential delays.
        """
        if self._llm is None:
            raise RuntimeError("Groq client not initialized")

        response = self._llm.complete(prompt, **kwargs)
        return str(response)

    @backoff.on_exception(
        backoff.expo,
        RATE_LIMIT_EXCEPTIONS,
        max_tries=5,
        max_time=60,
        on_backoff=_log_backoff,
    )
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream completion tokens.

        Implements exponential backoff for rate limit (HTTP 429) errors.
        Note: Backoff applies to stream initialization, not mid-stream.
        """
        if self._llm is None:
            raise RuntimeError("Groq client not initialized")

        for token in self._llm.stream_complete(prompt, **kwargs):
            yield token.delta

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        context_windows = {
            "meta-llama/llama-4-scout-17b-16e-instruct": 131072,
            "meta-llama/llama-4-maverick-17b-128e-instruct": 131072,
            "llama-3.3-70b-versatile": 131072,
            "llama-3.3-70b-specdec": 131072,
            "llama-3.1-8b-instant": 131072,
            "openai/gpt-oss-120b": 131072,
            "openai/gpt-oss-20b": 131072,
            "mixtral-8x7b-32768": 32768,
            "gemma2-9b-it": 8192,
        }

        return {
            "name": self._model,
            "provider": "groq",
            "context_window": context_windows.get(self._model, 131072),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "capabilities": ["completion", "streaming", "chat", "function_calling"],
            "pricing": self._get_pricing(),
            "speed": self._get_speed(),
            "rate_limits": {
                "free_tier": "30 RPM",
                "paid_tier": "varies by model",
            },
            "retry_config": {
                "max_tries": 5,
                "max_time_seconds": 60,
                "strategy": "exponential_backoff",
            },
        }

    def _get_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens."""
        pricing = {
            "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
            "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
            "openai/gpt-oss-20b": {"input": 0.05, "output": 0.15},
            "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        }
        return pricing.get(self._model, {"input": 0.20, "output": 0.60})

    def _get_speed(self) -> str:
        """Get approximate speed in tokens/second."""
        speeds = {
            "meta-llama/llama-4-scout-17b-16e-instruct": "~800 tok/s",
            "meta-llama/llama-4-maverick-17b-128e-instruct": "~600 tok/s",
            "llama-3.3-70b-versatile": "~250 tok/s",
            "llama-3.1-8b-instant": "~1000 tok/s",
            "openai/gpt-oss-120b": "~500 tok/s",
            "openai/gpt-oss-20b": "~1000 tok/s",
            "mixtral-8x7b-32768": "~400 tok/s",
        }
        return speeds.get(self._model, "~500 tok/s")

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def get_llama_index_llm(self) -> Any:
        """Return LlamaIndex-compatible LLM instance."""
        return self._llm

    def validate_connection(self) -> bool:
        """Check if Groq API is reachable."""
        try:
            # Simple test completion
            self._llm.complete("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"Groq connection check failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported Groq models."""
        return sorted(cls.SUPPORTED_MODELS)

    @classmethod
    def get_model_aliases(cls) -> Dict[str, str]:
        """Return model aliases for convenience."""
        return cls.MODEL_ALIASES.copy()
