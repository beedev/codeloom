"""Query utility functions for the stateless retrieval pattern.

This module provides shared utilities for query processing,
including query expansion with conversation history.
"""

import logging
from typing import List, Dict, Any, Optional

from ..prompt import get_condense_prompt

logger = logging.getLogger(__name__)


def expand_query_with_history(
    query: str,
    conversation_history: List[Dict[str, Any]],
    llm: Any,
    min_history_length: int = 2,
    max_history_turns: int = 4,
    max_content_length: int = 500,
) -> str:
    """Expand follow-up queries using conversation history.

    Transforms ambiguous follow-up queries (e.g., "what about X?") into
    standalone queries by incorporating context from recent conversation.

    Args:
        query: Original user query
        conversation_history: List of {"role": "user"|"assistant", "content": str}
        llm: LLM instance for query expansion
        min_history_length: Minimum history entries before expansion (default: 2)
        max_history_turns: Maximum recent turns to include (default: 4)
        max_content_length: Max characters per message (default: 500)

    Returns:
        Expanded query string, or original query if expansion fails/not needed

    Example:
        >>> history = [
        ...     {"role": "user", "content": "Tell me about Python"},
        ...     {"role": "assistant", "content": "Python is a programming language..."},
        ...     {"role": "user", "content": "What about its typing?"}
        ... ]
        >>> expanded = expand_query_with_history("What about its typing?", history, llm)
        >>> print(expanded)  # "What is Python's typing system?"
    """
    # Skip if insufficient history
    if not conversation_history or len(conversation_history) < min_history_length:
        return query

    try:
        # Format recent history
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:max_content_length]}"
            for msg in conversation_history[-max_history_turns:]
        ])

        # Generate expanded query
        condense_prompt = get_condense_prompt().format(
            chat_history=history_text,
            question=query
        )
        expanded = llm.complete(condense_prompt).text.strip()

        # Validate expansion
        if expanded and len(expanded) > 5 and expanded != query:
            logger.info(f"Query expanded: '{query}' → '{expanded}'")
            return expanded

        return query

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query


def expand_query_with_history_timed(
    query: str,
    conversation_history: List[Dict[str, Any]],
    llm: Any,
    timings: Optional[Dict[str, int]] = None,
    timing_key: str = "query_expansion_ms",
) -> str:
    """Expand query with timing tracking for performance monitoring.

    Wrapper around expand_query_with_history that records execution time.

    Args:
        query: Original user query
        conversation_history: Conversation history
        llm: LLM instance
        timings: Dict to store timing (mutated in place)
        timing_key: Key name for the timing entry

    Returns:
        Expanded query string
    """
    import time

    if timings is None:
        return expand_query_with_history(query, conversation_history, llm)

    start = time.time()
    result = expand_query_with_history(query, conversation_history, llm)
    timings[timing_key] = int((time.time() - start) * 1000)

    return result
