"""Prompts for two-stage document routing.

This module contains LLM prompts for:
1. Stage 1: Document routing decision (which docs to query)
2. Direct synthesis from summaries (when retrieval not needed)
"""

from typing import List
from ..interfaces.routing import DocumentSummary, IRoutingPrompts


# =============================================================================
# Stage 1: Document Routing Prompt
# =============================================================================

DOCUMENT_ROUTING_PROMPT = """You are an intelligent document routing assistant. Given a user query and summaries of all documents in a project, determine the best strategy to answer the query.

## Available Documents ({document_count} total):

{document_summaries}

## User Query:
{query}

## Routing Strategies:

1. **DIRECT_SYNTHESIS**: Use when the query can be answered by synthesizing information from the document summaries above. No need to look deeper into the documents.
   - Best for: "Summarize all documents", "What are the main themes?", "Give me an overview", "Key takeaways"
   - Returns: A synthesized answer combining insights from all summaries

2. **DEEP_DIVE**: Use when the query requires detailed information from specific document(s). Select 1-3 most relevant documents.
   - Best for: "What does [specific doc] say about X?", "Details about [specific topic]", "Explain the [specific concept] mentioned in..."
   - Returns: List of document IDs to search deeply

3. **MULTI_DOC_ANALYSIS**: Use when the query requires analyzing or comparing information across multiple documents.
   - Best for: "Compare the approaches in...", "How do the documents differ on...", "Find contradictions between..."
   - Returns: All relevant document IDs for cross-analysis

## Instructions:
1. Analyze the query intent
2. Consider which documents are most relevant
3. Choose the most efficient strategy
4. Respond with valid JSON only

## Response Format:
{{
  "strategy": "DIRECT_SYNTHESIS" | "DEEP_DIVE" | "MULTI_DOC_ANALYSIS",
  "selected_documents": ["source_id_1", "source_id_2"],
  "reasoning": "Brief explanation of why this strategy was chosen",
  "confidence": 0.0 to 1.0
}}

Notes:
- For DIRECT_SYNTHESIS, leave selected_documents as empty array []
- For DEEP_DIVE, include 1-3 most relevant document source_ids
- For MULTI_DOC_ANALYSIS, include all relevant document source_ids
- If unsure, prefer DEEP_DIVE over DIRECT_SYNTHESIS for accuracy

Respond with JSON only:"""


# =============================================================================
# Direct Synthesis Prompt (for DIRECT_SYNTHESIS strategy)
# =============================================================================

SYNTHESIS_PROMPT = """Based on the following document summaries, provide a comprehensive response to the user's query.

## Document Summaries:

{summaries}

## User Query:
{query}

## Instructions:
- Synthesize information from ALL relevant documents
- Highlight key themes, patterns, and commonalities
- Note any contrasting or complementary points between documents
- Be comprehensive but concise
- Structure your response clearly
- Reference specific documents when making claims

Your response:"""


# =============================================================================
# Prompt Formatter Implementation
# =============================================================================

class RoutingPrompts(IRoutingPrompts):
    """Implementation of routing prompts interface."""

    def get_routing_prompt(
        self,
        query: str,
        summaries_text: str,
        document_count: int
    ) -> str:
        """Generate the Stage 1 routing prompt."""
        return DOCUMENT_ROUTING_PROMPT.format(
            query=query,
            document_summaries=summaries_text,
            document_count=document_count
        )

    def get_synthesis_prompt(
        self,
        query: str,
        summaries_text: str
    ) -> str:
        """Generate the synthesis prompt for DIRECT_SYNTHESIS."""
        return SYNTHESIS_PROMPT.format(
            query=query,
            summaries=summaries_text
        )

    def format_summaries_for_llm(
        self,
        summaries: List[DocumentSummary]
    ) -> str:
        """Format document summaries for LLM consumption."""
        formatted_parts = []

        for i, summary in enumerate(summaries, 1):
            parts = [
                f"### Document {i}: {summary.file_name}",
                f"**ID**: {summary.source_id}",
                f"**Chunks**: {summary.chunk_count}",
            ]

            if summary.dense_summary:
                parts.append(f"\n**Summary**:\n{summary.dense_summary}")

            if summary.key_insights:
                insights_text = "\n".join(f"  - {insight}" for insight in summary.key_insights[:5])
                parts.append(f"\n**Key Insights**:\n{insights_text}")

            formatted_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(formatted_parts)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_routing_prompt(query: str, summaries_text: str, document_count: int) -> str:
    """Get the document routing prompt.

    Args:
        query: User query text
        summaries_text: Pre-formatted document summaries
        document_count: Number of documents

    Returns:
        Complete routing prompt
    """
    return DOCUMENT_ROUTING_PROMPT.format(
        query=query,
        document_summaries=summaries_text,
        document_count=document_count
    )


def get_synthesis_prompt(query: str, summaries_text: str) -> str:
    """Get the synthesis prompt for direct synthesis.

    Args:
        query: User query text
        summaries_text: Pre-formatted document summaries

    Returns:
        Complete synthesis prompt
    """
    return SYNTHESIS_PROMPT.format(
        query=query,
        summaries=summaries_text
    )


def format_summaries(
    summaries: List[DocumentSummary],
    max_total_chars: int = 8000
) -> str:
    """Format document summaries for LLM consumption, prioritizing key insights.

    Key insights are prioritized over dense summaries because:
    - They're already condensed bullet points
    - Higher information density per token
    - Better signals for routing decisions

    Args:
        summaries: List of DocumentSummary objects
        max_total_chars: Maximum total characters (default 8000, ~2000 tokens)

    Returns:
        Formatted string with all summaries
    """
    formatted_parts = []
    total_chars = 0

    for i, summary in enumerate(summaries, 1):
        parts = [
            f"### Document {i}: {summary.file_name}",
            f"**ID**: {summary.source_id}",
        ]

        # PRIORITY 1: Key insights (full, they're already condensed)
        if summary.key_insights:
            insights_text = " | ".join(summary.key_insights)
            parts.append(f"**Key Insights**: {insights_text}")

        # PRIORITY 2: Brief summary context (200 chars max)
        if summary.dense_summary:
            brief_summary = summary.dense_summary[:200]
            if len(summary.dense_summary) > 200:
                brief_summary += "..."
            parts.append(f"**Overview**: {brief_summary}")

        section = "\n".join(parts)

        # Check if adding this section would exceed limit
        if total_chars + len(section) > max_total_chars:
            # Add minimal version with just insights
            minimal_parts = [
                f"### Document {i}: {summary.file_name}",
                f"**ID**: {summary.source_id}",
            ]
            if summary.key_insights:
                minimal_parts.append(f"**Key Insights**: {' | '.join(summary.key_insights[:3])}")
            section = "\n".join(minimal_parts)

            # If still too long, use bare minimum
            if total_chars + len(section) > max_total_chars:
                section = f"### Document {i}: {summary.file_name} (ID: {summary.source_id})"

        formatted_parts.append(section)
        total_chars += len(section)

    return "\n\n".join(formatted_parts)
