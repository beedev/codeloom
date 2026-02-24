"""LLM utility functions for consistent LLM handling across the codebase.

This module provides shared utilities for working with LLM instances,
preventing code duplication and ensuring consistent behavior.
"""

import logging

logger = logging.getLogger(__name__)


def unwrap_llm(llm):
    """Extract raw LlamaIndex LLM from wrapper classes.

    Handles both LLMGateway (CustomLLM subclass with _llm attr) and legacy
    wrappers with get_raw_llm(). LlamaIndex components that need the raw
    provider LLM (e.g., QueryFusionRetriever) should call this.

    Args:
        llm: LLM instance, LLMGateway, or legacy wrapper

    Returns:
        Raw LlamaIndex LLM instance (unwrapped from any gateway/wrapper)

    Example:
        >>> from codeloom.core.utils import unwrap_llm
        >>> raw_llm = unwrap_llm(Settings.llm)
        >>> retriever = QueryFusionRetriever(..., llm=raw_llm)
    """
    # LLMGateway stores wrapped LLM as _llm
    if hasattr(llm, '_llm') and not isinstance(getattr(llm, '_llm', None), type(None)):
        raw_llm = llm._llm
        logger.debug(f"Unwrapped LLM: {type(llm).__name__} → {type(raw_llm).__name__}")
        return raw_llm
    # Legacy wrappers (e.g., GroqWithBackoff)
    if hasattr(llm, 'get_raw_llm'):
        raw_llm = llm.get_raw_llm()
        logger.debug(f"Unwrapped LLM: {type(llm).__name__} → {type(raw_llm).__name__}")
        return raw_llm
    return llm
