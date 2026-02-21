"""LLM utility functions for consistent LLM handling across the codebase.

This module provides shared utilities for working with LLM instances,
preventing code duplication and ensuring consistent behavior.
"""

import logging

logger = logging.getLogger(__name__)


def unwrap_llm(llm):
    """Extract raw LlamaIndex LLM from wrapper classes.

    LlamaIndex's resolve_llm() and Settings.llm setter require instances
    of the LLM base class. Wrapper classes (e.g., GroqWithBackoff for
    rate limiting) must be unwrapped before passing to LlamaIndex components.

    Args:
        llm: LLM instance or wrapper with get_raw_llm() method

    Returns:
        Raw LlamaIndex LLM instance

    Example:
        >>> from codeloom.core.utils import unwrap_llm
        >>> raw_llm = unwrap_llm(Settings.llm)
        >>> retriever = QueryFusionRetriever(..., llm=raw_llm)
    """
    if hasattr(llm, 'get_raw_llm'):
        raw_llm = llm.get_raw_llm()
        logger.debug(f"Unwrapped LLM: {type(llm).__name__} → {type(raw_llm).__name__}")
        return raw_llm
    return llm
