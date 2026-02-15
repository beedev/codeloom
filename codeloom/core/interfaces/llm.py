"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations can include:
    - Ollama (local models)
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Google (Gemini)
    - Custom providers
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            The generated completion text
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Stream completion tokens for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Provider-specific parameters

        Yields:
            Individual tokens or chunks of the response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the current model.

        Returns:
            Dictionary containing:
            - name: Model name/identifier
            - context_window: Maximum context length
            - pricing: Optional pricing information
            - capabilities: Optional list of capabilities
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the current model name."""
        pass

    @abstractmethod
    def get_llama_index_llm(self) -> Any:
        """
        Return a LlamaIndex-compatible LLM instance.

        This allows the provider to be used with LlamaIndex's
        chat engines and query engines.

        Returns:
            A LlamaIndex LLM instance
        """
        pass

    def validate_connection(self) -> bool:
        """
        Validate that the provider connection is working.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            self.complete("Hello", max_tokens=5)
            return True
        except Exception:
            return False
