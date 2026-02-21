"""Stateless RAG operations for multi-user, thread-safe queries.

This module extracts the fast API pattern from /api/query into reusable
functions that can be used by Project Chat, SQL Chat, and the API.

Key Design Principles:
- No global state mutations
- Per-request retrievers (thread-safe)
- Explicit parameter passing (no hidden dependencies)
- Database-backed memory (not process state)

Usage:
    from codeloom.core.stateless import (
        fast_retrieve,
        get_raptor_summaries,
        build_hierarchical_context,
        execute_query,
    )

    # Multi-user safe query flow
    nodes = pipeline._get_cached_nodes(project_id)
    retrieval_results = fast_retrieve(nodes, query, project_id, pipeline)
    raptor_summaries = get_raptor_summaries(query, project_id, pipeline)
    context = build_hierarchical_context(retrieval_results, raptor_summaries)
    response = execute_query(query, context, llm)
"""

from .retrieval import (
    fast_retrieve,
    enhanced_retrieve,
    get_raptor_summaries,
    create_retriever,
)
from .context import (
    build_hierarchical_context,
    build_context_with_history,
    format_sources,
)
from .completion import (
    execute_query,
    execute_query_streaming,
)
from .memory import (
    load_conversation_history,
    save_conversation_turn,
    generate_session_id,
    format_history_for_context,
)
from .query_utils import (
    expand_query_with_history,
    expand_query_with_history_timed,
)

__all__ = [
    # Retrieval
    "fast_retrieve",
    "enhanced_retrieve",
    "get_raptor_summaries",
    "create_retriever",
    # Context
    "build_hierarchical_context",
    "build_context_with_history",
    "format_sources",
    # Completion
    "execute_query",
    "execute_query_streaming",
    # Memory
    "load_conversation_history",
    "save_conversation_turn",
    "generate_session_id",
    "format_history_for_context",
    # Query Utils
    "expand_query_with_history",
    "expand_query_with_history_timed",
]
