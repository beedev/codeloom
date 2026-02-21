"""Stateless LLM completion functions.

These functions execute LLM completions without shared state,
making them safe for multi-user concurrent access.
"""

import logging
from typing import Optional, Generator, Any

from llama_index.core import Settings

from ..prompt import get_system_prompt, get_context_prompt

logger = logging.getLogger(__name__)


def execute_query(
    query: str,
    context: str,
    llm: Optional[Any] = None,
    language: str = "eng",
    is_rag_prompt: bool = True,
) -> str:
    """Execute a stateless LLM query with context.

    This is thread-safe because it uses explicit parameters
    rather than shared global state.

    Args:
        query: User's query string
        context: Pre-built context from build_hierarchical_context
        llm: LLM instance (defaults to Settings.llm)
        language: Language code for prompts
        is_rag_prompt: Whether to use RAG-optimized system prompt

    Returns:
        LLM response text

    Example:
        response = execute_query(
            query="What are the key findings?",
            context=context,
            llm=Settings.llm,
        )
    """
    if llm is None:
        llm = Settings.llm

    # Build prompt using same pattern as /api/query
    system_prompt = get_system_prompt(language, is_rag_prompt=is_rag_prompt)
    context_prompt_template = get_context_prompt(language)
    context_prompt = context_prompt_template.format(context_str=context)

    prompt = f"""{system_prompt}

{context_prompt}

User question: {query}"""

    response = llm.complete(prompt)
    return response.text


def execute_query_streaming(
    query: str,
    context: str,
    llm: Optional[Any] = None,
    language: str = "eng",
    is_rag_prompt: bool = True,
) -> Generator[str, None, None]:
    """Execute a stateless LLM query with streaming response.

    Args:
        query: User's query string
        context: Pre-built context from build_hierarchical_context
        llm: LLM instance (defaults to Settings.llm)
        language: Language code for prompts
        is_rag_prompt: Whether to use RAG-optimized system prompt

    Yields:
        Response text chunks as they are generated

    Example:
        for chunk in execute_query_streaming(query, context):
            print(chunk, end="", flush=True)
    """
    if llm is None:
        llm = Settings.llm

    # Build prompt using same pattern as /api/query
    system_prompt = get_system_prompt(language, is_rag_prompt=is_rag_prompt)
    context_prompt_template = get_context_prompt(language)
    context_prompt = context_prompt_template.format(context_str=context)

    prompt = f"""{system_prompt}

{context_prompt}

User question: {query}"""

    # Use streaming completion
    for token in llm.stream_complete(prompt):
        yield token.delta


def build_prompt(
    query: str,
    context: str,
    language: str = "eng",
    is_rag_prompt: bool = True,
) -> str:
    """Build the full prompt without executing it.

    Useful for debugging or when you need to customize execution.

    Args:
        query: User's query string
        context: Pre-built context
        language: Language code for prompts
        is_rag_prompt: Whether to use RAG-optimized system prompt

    Returns:
        Complete prompt string
    """
    system_prompt = get_system_prompt(language, is_rag_prompt=is_rag_prompt)
    context_prompt_template = get_context_prompt(language)
    context_prompt = context_prompt_template.format(context_str=context)

    return f"""{system_prompt}

{context_prompt}

User question: {query}"""
