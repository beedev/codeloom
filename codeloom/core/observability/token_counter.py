"""Token counting utility for query logging."""

import logging
from typing import Tuple
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Utility class for estimating token counts in text.

    Uses GPT-2 tokenizer as a general-purpose token estimator.
    Token counts are estimates and may vary from actual LLM token usage.
    """

    def __init__(self):
        """Initialize the token counter with a general-purpose tokenizer."""
        try:
            # Use GPT-2 tokenizer as a universal approximation
            # It's lightweight and provides reasonably accurate estimates for most models
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logger.info("TokenCounter initialized with gpt2 tokenizer")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not self._tokenizer or not text:
            # Fallback: rough estimate (1 token ≈ 0.75 words)
            return max(1, int(len(text.split()) * 1.3))

        try:
            tokens = self._tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback to word-based estimate
            return max(1, int(len(text.split()) * 1.3))

    def count_query_tokens(self, query: str, response: str) -> Tuple[int, int]:
        """
        Count tokens for both query and response.

        Args:
            query: User query text
            response: LLM response text

        Returns:
            Tuple of (prompt_tokens, completion_tokens)
        """
        prompt_tokens = self.count_tokens(query)
        completion_tokens = self.count_tokens(response)

        return prompt_tokens, completion_tokens


# Global singleton instance
_token_counter = None


def get_token_counter() -> TokenCounter:
    """
    Get the global TokenCounter instance.

    Returns:
        TokenCounter singleton instance
    """
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter
