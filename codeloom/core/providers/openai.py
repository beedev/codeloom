"""OpenAI LLM provider implementation."""

import logging
import os
from typing import Generator, Dict, Any, Optional

from llama_index.llms.openai import OpenAI

from ..interfaces import LLMProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider for GPT models.

    Supports:
    - GPT-4 series (gpt-4, gpt-4-turbo, gpt-4o, gpt-4.1, gpt-4.1-nano)
    - GPT-3.5 series
    - O-series reasoning models (o1, o3, o4 - temperature must be 1)
    """

    SUPPORTED_MODELS = {
        # GPT-4 series
        "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-4o-mini", "gpt-4-turbo-preview",
        "gpt-4-0125-preview", "gpt-4-1106-preview",
        "gpt-4o-2024-11-20", "gpt-4o-2024-08-06",
        # O-series reasoning models (temperature=1 only)
        "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-pro",
        "o4-mini",
    }

    # O-series reasoning models only support temperature=1
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"}

    # Context window sizes for each model (used by LlamaIndex for token management)
    CONTEXT_WINDOWS = {
        "gpt-3.5-turbo": 16384,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-1106-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-2024-11-20": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4.1": 1000000,       # GPT-4.1 supports 1M context
        "gpt-4.1-mini": 1000000,
        "gpt-4.1-nano": 1000000,
        "o1": 200000,
        "o1-mini": 128000,
        "o1-preview": 128000,
        "o3": 200000,
        "o3-mini": 200000,
        "o3-pro": 200000,
        "o4-mini": 200000,
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

        self._model = model or os.getenv("LLM_MODEL", "gpt-4.1")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self._llm: Optional[OpenAI] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the OpenAI client."""
        # Get context window for this model (critical for LlamaIndex token management)
        context_window = self.CONTEXT_WINDOWS.get(self._model, 128000)

        kwargs = {
            "model": self._model,
            "api_key": self._api_key,
            "temperature": self._temperature,
            "context_window": context_window,
        }

        if self._max_tokens:
            kwargs["max_tokens"] = self._max_tokens

        self._llm = OpenAI(**kwargs)

        logger.info(f"Initialized OpenAI: {self._model} (context_window: {context_window:,})")

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt."""
        if self._llm is None:
            raise RuntimeError("OpenAI client not initialized")

        response = self._llm.complete(prompt, **kwargs)
        return str(response)

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream completion tokens."""
        if self._llm is None:
            raise RuntimeError("OpenAI client not initialized")

        for token in self._llm.stream_complete(prompt, **kwargs):
            yield token.delta

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self._model,
            "provider": "openai",
            "context_window": self.CONTEXT_WINDOWS.get(self._model, 128000),
            "temperature": self._temperature,
            "capabilities": ["completion", "streaming", "chat", "function_calling"],
            "pricing": self._get_pricing()
        }

    def _get_pricing(self) -> Dict[str, float]:
        """Get approximate pricing per 1K tokens."""
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4.1": {"input": 0.005, "output": 0.015},
            "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4.1-nano": {"input": 0.00015, "output": 0.0006},
            "o1": {"input": 0.015, "output": 0.06},
            "o1-mini": {"input": 0.003, "output": 0.012},
        }
        return pricing.get(self._model, {"input": 0.01, "output": 0.03})

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def get_llama_index_llm(self) -> Any:
        """Return LlamaIndex-compatible LLM instance."""
        return self._llm

    def validate_connection(self) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            # Simple test completion
            self._llm.complete("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"OpenAI connection check failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported OpenAI models."""
        return sorted(cls.SUPPORTED_MODELS)
