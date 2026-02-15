"""Ollama LLM provider implementation."""

import logging
import os
from typing import Generator, Dict, Any, Optional

import requests
from llama_index.llms.ollama import Ollama

from ..interfaces import LLMProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class OllamaLLMProvider(LLMProvider):
    """
    Ollama LLM provider for local model inference.

    Supports any model available through Ollama:
    - llama3.1, llama2, codellama
    - mistral, mixtral
    - phi, gemma, qwen
    - And many more
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = "localhost",
        port: Optional[int] = None,
        temperature: float = 0.7,
        context_window: int = 8000,
        request_timeout: float = 120.0,
        setting: Optional[RAGSettings] = None,
        system_prompt: Optional[str] = None,
    ):
        self._setting = setting or get_settings()

        # Use environment or settings for defaults
        self._model = model or os.getenv("LLM_MODEL") or self._setting.ollama.llm
        self._host = host or os.getenv("OLLAMA_HOST", "localhost")
        self._port = port or int(os.getenv("OLLAMA_PORT", str(self._setting.ollama.port)))
        self._temperature = temperature
        self._context_window = context_window
        self._request_timeout = request_timeout
        self._system_prompt = system_prompt

        self._llm: Optional[Ollama] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Ollama client."""
        additional_kwargs = {
            "tfs_z": self._setting.ollama.tfs_z,
            "top_k": self._setting.ollama.top_k,
            "top_p": self._setting.ollama.top_p,
            "repeat_last_n": self._setting.ollama.repeat_last_n,
            "repeat_penalty": self._setting.ollama.repeat_penalty,
        }

        self._llm = Ollama(
            model=self._model,
            system_prompt=self._system_prompt,
            base_url=f"http://{self._host}:{self._port}",
            temperature=self._temperature,
            context_window=self._context_window,
            request_timeout=self._request_timeout,
            additional_kwargs=additional_kwargs
        )

        logger.debug(f"Initialized Ollama provider with model: {self._model}")

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt."""
        if self._llm is None:
            raise RuntimeError("Ollama client not initialized")

        response = self._llm.complete(prompt, **kwargs)
        return str(response)

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream completion tokens."""
        if self._llm is None:
            raise RuntimeError("Ollama client not initialized")

        for token in self._llm.stream_complete(prompt, **kwargs):
            yield token.delta

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self._model,
            "provider": "ollama",
            "context_window": self._context_window,
            "temperature": self._temperature,
            "host": f"{self._host}:{self._port}",
            "capabilities": ["completion", "streaming", "chat"]
        }

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def get_llama_index_llm(self) -> Any:
        """Return LlamaIndex-compatible LLM instance."""
        return self._llm

    def validate_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            url = f"http://{self._host}:{self._port}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    def list_models(self) -> list:
        """List available models on Ollama server."""
        try:
            url = f"http://{self._host}:{self._port}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception as e:
            logger.warning(f"Error listing Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama server."""
        try:
            url = f"http://{self._host}:{self._port}/api/pull"
            response = requests.post(url, json={"name": model_name}, timeout=600)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists on Ollama server."""
        available = self.list_models()
        return model_name in available
