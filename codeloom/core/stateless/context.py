"""Context building functions for hierarchical RAG responses.

Builds context from RAPTOR summaries (high-level) and
retrieved chunks (detailed evidence).
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

from llama_index.core.schema import TextNode, NodeWithScore

logger = logging.getLogger(__name__)


def build_hierarchical_context(
    retrieval_results: List[NodeWithScore],
    raptor_summaries: Optional[List[Tuple[TextNode, float]]] = None,
    max_summaries: int = 3,
    max_chunks: int = 6,
) -> str:
    """Build hierarchical context from RAPTOR summaries and chunks.

    The context is structured with:
    1. HIGH-LEVEL CONTEXT: RAPTOR summaries for document-level understanding
    2. DETAILED EVIDENCE: Retrieved chunks for specific information

    This pattern (from /api/query) provides both breadth and depth.

    Args:
        retrieval_results: List of NodeWithScore from retrieval
        raptor_summaries: Optional list of (TextNode, score) RAPTOR summaries
        max_summaries: Maximum number of summaries to include
        max_chunks: Maximum number of chunks to include

    Returns:
        Formatted context string for LLM prompt

    Example:
        context = build_hierarchical_context(
            retrieval_results=chunks,
            raptor_summaries=summaries,
        )
    """
    context_parts = []

    # Add RAPTOR summaries first (high-level framing)
    if raptor_summaries:
        summary_texts = []
        for node, score in raptor_summaries[:max_summaries]:
            summary_texts.append(node.text)

        if summary_texts:
            context_parts.append(
                "## HIGH-LEVEL CONTEXT (Document Summaries)\n" +
                "\n\n".join(summary_texts)
            )

    # Add retrieved chunks (detailed evidence)
    if retrieval_results:
        chunk_texts = []
        for node_with_score in retrieval_results[:max_chunks]:
            node = node_with_score.node
            metadata = node.metadata or {}
            doc_name = metadata.get("file_name", "Unknown")
            chunk_texts.append(f"[Source: {doc_name}]\n{node.text}")

        if chunk_texts:
            context_parts.append(
                "## DETAILED EVIDENCE (Relevant Passages)\n" +
                "\n\n---\n\n".join(chunk_texts)
            )

    if not context_parts:
        return "No relevant context found."

    return "\n\n".join(context_parts)


def format_sources(
    retrieval_results: List[NodeWithScore],
    max_sources: int = 6,
    excerpt_length: int = 200,
) -> List[Dict[str, Any]]:
    """Format retrieval results as source citations.

    Args:
        retrieval_results: List of NodeWithScore from retrieval
        max_sources: Maximum number of sources to return
        excerpt_length: Length of text excerpt

    Returns:
        List of source dictionaries with document, excerpt, score

    Example:
        sources = format_sources(retrieval_results)
        # Returns: [{"document": "paper.pdf", "excerpt": "...", "score": 0.92}]
    """
    sources = []

    for node_with_score in retrieval_results[:max_sources]:
        node = node_with_score.node
        metadata = node.metadata or {}

        text = node.text
        if len(text) > excerpt_length:
            text = text[:excerpt_length] + "..."

        sources.append({
            "filename": metadata.get("file_name", "Unknown"),
            "snippet": text,
            "score": float(round(node_with_score.score or 0.0, 3)),
        })

    return sources


def build_context_with_history(
    retrieval_results: List[NodeWithScore],
    raptor_summaries: Optional[List[Tuple[TextNode, float]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_history: int = 10,
    max_summaries: int = 3,
    max_chunks: int = 6,
) -> str:
    """Build context including conversation history for chat continuity.

    This extends build_hierarchical_context with conversation memory
    for the v2 chat endpoint.

    Args:
        retrieval_results: List of NodeWithScore from retrieval
        raptor_summaries: Optional list of (TextNode, score) RAPTOR summaries
        conversation_history: List of {"role": "user"|"assistant", "content": str}
        max_history: Maximum number of history turns to include
        max_summaries: Maximum number of summaries to include
        max_chunks: Maximum number of chunks to include

    Returns:
        Formatted context string with history and retrieval

    Example:
        context = build_context_with_history(
            retrieval_results=chunks,
            raptor_summaries=summaries,
            conversation_history=[
                {"role": "user", "content": "What are the key findings?"},
                {"role": "assistant", "content": "The key findings are..."},
            ],
        )
    """
    context_parts = []

    # Add conversation history (for continuity)
    if conversation_history:
        history_turns = conversation_history[-max_history:]
        if history_turns:
            history_text = []
            for turn in history_turns:
                role = turn.get("role", "unknown").capitalize()
                content = turn.get("content", "")
                history_text.append(f"{role}: {content}")

            context_parts.append(
                "## CONVERSATION HISTORY\n" +
                "\n\n".join(history_text)
            )

    # Add RAPTOR summaries (high-level framing)
    if raptor_summaries:
        summary_texts = []
        for node, score in raptor_summaries[:max_summaries]:
            summary_texts.append(node.text)

        if summary_texts:
            context_parts.append(
                "## HIGH-LEVEL CONTEXT (Document Summaries)\n" +
                "\n\n".join(summary_texts)
            )

    # Add retrieved chunks (detailed evidence)
    if retrieval_results:
        chunk_texts = []
        for node_with_score in retrieval_results[:max_chunks]:
            node = node_with_score.node
            metadata = node.metadata or {}
            doc_name = metadata.get("file_name", "Unknown")
            chunk_texts.append(f"[Source: {doc_name}]\n{node.text}")

        if chunk_texts:
            context_parts.append(
                "## DETAILED EVIDENCE (Relevant Passages)\n" +
                "\n\n---\n\n".join(chunk_texts)
            )

    if not context_parts:
        return "No relevant context found."

    return "\n\n".join(context_parts)
