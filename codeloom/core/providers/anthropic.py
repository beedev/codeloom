"""Anthropic Claude LLM provider implementation."""

import logging
import os
from typing import Generator, Dict, Any, Optional

from ..interfaces import LLMProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class AnthropicLLMProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.

    Supports:
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
    - claude-3-opus-20240229
    """

    SUPPORTED_MODELS = {
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        setting: Optional[RAGSettings] = None,
    ):
        self._setting = setting or get_settings()

        self._model = model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or self._setting.anthropic.api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self._llm = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from llama_index.llms.anthropic import Anthropic

            self._llm = Anthropic(
                model=self._model,
                api_key=self._api_key,
                temperature=self._temperature,
                max_tokens=self._max_tokens
            )

            logger.debug(f"Initialized Anthropic provider with model: {self._model}")
        except ImportError:
            raise ImportError(
                "llama-index-llms-anthropic not installed. "
                "Run: pip install llama-index-llms-anthropic"
            )

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt."""
        if self._llm is None:
            raise RuntimeError("Anthropic client not initialized")

        response = self._llm.complete(prompt, **kwargs)
        return str(response)

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream completion tokens."""
        if self._llm is None:
            raise RuntimeError("Anthropic client not initialized")

        for token in self._llm.stream_complete(prompt, **kwargs):
            yield token.delta

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        context_windows = {
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }

        return {
            "name": self._model,
            "provider": "anthropic",
            "context_window": context_windows.get(self._model, 200000),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "capabilities": ["completion", "streaming", "chat", "vision"],
            "pricing": self._get_pricing()
        }

    def _get_pricing(self) -> Dict[str, float]:
        """Get approximate pricing per 1K tokens."""
        pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
        return pricing.get(self._model, {"input": 0.003, "output": 0.015})

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def get_llama_index_llm(self) -> Any:
        """Return LlamaIndex-compatible LLM instance."""
        return self._llm

    def validate_connection(self) -> bool:
        """Check if Anthropic API is reachable."""
        try:
            self._llm.complete("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"Anthropic connection check failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported Anthropic models."""
        return sorted(cls.SUPPORTED_MODELS)
