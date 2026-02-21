"""Token counting via tiktoken.

Uses cl100k_base encoding (matches OpenAI embedding tokenizer).
"""

import tiktoken


class TokenCounter:
    """Count tokens using tiktoken encoding."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        return len(self._encoder.encode(text))
