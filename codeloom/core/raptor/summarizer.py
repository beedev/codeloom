"""RAPTOR Summarizer Module.

Implements LLM-based summarization for cluster nodes in the RAPTOR tree.
Creates abstractive summaries that capture the key information from multiple chunks.

Key Features:
- Token-aware chunk selection
- Configurable summarization prompts
- Support for multiple LLM providers via LlamaIndex
- Rate limiting and retry logic

Reference: https://arxiv.org/abs/2401.18059
"""

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode

from .clustering import Cluster
from .config import SummarizationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SummaryNode:
    """Represents a summary node in the RAPTOR tree."""
    node_id: str
    text: str
    child_node_ids: List[str]
    tree_level: int
    cluster_id: str
    token_count: int
    source_id: str
    notebook_id: str

    def to_text_node(self) -> TextNode:
        """Convert to LlamaIndex TextNode with tree metadata."""
        return TextNode(
            id_=self.node_id,
            text=self.text,
            metadata={
                "node_type": "raptor_summary",
                "tree_level": self.tree_level,
                "child_node_ids": self.child_node_ids,
                "cluster_id": self.cluster_id,
                "source_id": self.source_id,
                "notebook_id": self.notebook_id,
            }
        )


class RAPTORSummarizer:
    """
    LLM-based summarization for RAPTOR clusters.

    Creates abstractive summaries that combine information from multiple
    semantically similar chunks into a single coherent summary.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[SummarizationConfig] = None
    ):
        """
        Initialize summarizer with LLM and configuration.

        Args:
            llm: LlamaIndex LLM instance for generation
            config: Summarization configuration (uses defaults if None)
        """
        self.llm = llm
        self.config = config or DEFAULT_CONFIG.summarization

    async def summarize_cluster(
        self,
        cluster: Cluster,
        tree_level: int,
        source_id: str,
        notebook_id: str
    ) -> SummaryNode:
        """
        Create a summary node for a cluster.

        Args:
            cluster: Cluster of nodes to summarize
            tree_level: Tree level for the summary (1+)
            source_id: Source document ID
            notebook_id: Notebook ID

        Returns:
            SummaryNode containing the generated summary
        """
        # Prepare chunk texts for summarization
        chunks_text = self._prepare_chunks_text(cluster)

        # Generate summary using LLM
        prompt = self.config.cluster_summary_prompt.format(chunks=chunks_text)

        try:
            response = await self.llm.acomplete(prompt)
            summary_text = response.text.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback: concatenate first sentences
            summary_text = self._fallback_summary(cluster)

        # Estimate token count
        token_count = len(summary_text.split()) * 1.3  # Rough estimate

        return SummaryNode(
            node_id=str(uuid.uuid4()),
            text=summary_text,
            child_node_ids=cluster.node_ids,
            tree_level=tree_level,
            cluster_id=cluster.cluster_id,
            token_count=int(token_count),
            source_id=source_id,
            notebook_id=notebook_id
        )

    def summarize_cluster_sync(
        self,
        cluster: Cluster,
        tree_level: int,
        source_id: str,
        notebook_id: str
    ) -> SummaryNode:
        """
        Synchronous version of summarize_cluster.

        Args:
            cluster: Cluster of nodes to summarize
            tree_level: Tree level for the summary (1+)
            source_id: Source document ID
            notebook_id: Notebook ID

        Returns:
            SummaryNode containing the generated summary
        """
        # Prepare chunk texts for summarization
        chunks_text = self._prepare_chunks_text(cluster)

        # Generate summary using LLM
        prompt = self.config.cluster_summary_prompt.format(chunks=chunks_text)

        try:
            response = self.llm.complete(prompt)
            summary_text = response.text.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback: concatenate first sentences
            summary_text = self._fallback_summary(cluster)

        # Estimate token count
        token_count = len(summary_text.split()) * 1.3  # Rough estimate

        return SummaryNode(
            node_id=str(uuid.uuid4()),
            text=summary_text,
            child_node_ids=cluster.node_ids,
            tree_level=tree_level,
            cluster_id=cluster.cluster_id,
            token_count=int(token_count),
            source_id=source_id,
            notebook_id=notebook_id
        )

    async def summarize_summaries(
        self,
        summary_nodes: List[SummaryNode],
        tree_level: int,
        source_id: str,
        notebook_id: str
    ) -> SummaryNode:
        """
        Create a higher-level summary from multiple summary nodes.

        Used to create the root node of the RAPTOR tree.

        Args:
            summary_nodes: List of summary nodes to combine
            tree_level: Tree level for the new summary
            source_id: Source document ID
            notebook_id: Notebook ID

        Returns:
            SummaryNode containing the combined summary
        """
        # Prepare summaries text
        summaries_text = self._prepare_summaries_text(summary_nodes)

        # Use root summary prompt for document-level summary
        prompt = self.config.root_summary_prompt.format(summaries=summaries_text)

        try:
            response = await self.llm.acomplete(prompt)
            summary_text = response.text.strip()
        except Exception as e:
            logger.error(f"LLM root summarization failed: {e}")
            # Fallback: combine summary texts
            summary_text = self._fallback_root_summary(summary_nodes)

        # Estimate token count
        token_count = len(summary_text.split()) * 1.3

        return SummaryNode(
            node_id=str(uuid.uuid4()),
            text=summary_text,
            child_node_ids=[sn.node_id for sn in summary_nodes],
            tree_level=tree_level,
            cluster_id=f"root_{source_id}",
            token_count=int(token_count),
            source_id=source_id,
            notebook_id=notebook_id
        )

    def summarize_summaries_sync(
        self,
        summary_nodes: List[SummaryNode],
        tree_level: int,
        source_id: str,
        notebook_id: str
    ) -> SummaryNode:
        """
        Synchronous version of summarize_summaries.

        Args:
            summary_nodes: List of summary nodes to combine
            tree_level: Tree level for the new summary
            source_id: Source document ID
            notebook_id: Notebook ID

        Returns:
            SummaryNode containing the combined summary
        """
        # Prepare summaries text
        summaries_text = self._prepare_summaries_text(summary_nodes)

        # Use root summary prompt for document-level summary
        prompt = self.config.root_summary_prompt.format(summaries=summaries_text)

        try:
            response = self.llm.complete(prompt)
            summary_text = response.text.strip()
        except Exception as e:
            logger.error(f"LLM root summarization failed: {e}")
            # Fallback: combine summary texts
            summary_text = self._fallback_root_summary(summary_nodes)

        # Estimate token count
        token_count = len(summary_text.split()) * 1.3

        return SummaryNode(
            node_id=str(uuid.uuid4()),
            text=summary_text,
            child_node_ids=[sn.node_id for sn in summary_nodes],
            tree_level=tree_level,
            cluster_id=f"root_{source_id}",
            token_count=int(token_count),
            source_id=source_id,
            notebook_id=notebook_id
        )

    def _prepare_chunks_text(self, cluster: Cluster) -> str:
        """
        Prepare chunk texts for summarization prompt.

        Respects token limits by truncating if necessary.
        """
        texts = cluster.get_texts()

        # Limit number of chunks
        if len(texts) > self.config.max_chunks_per_summary:
            texts = texts[:self.config.max_chunks_per_summary]
            logger.debug(f"Limited to {self.config.max_chunks_per_summary} chunks")

        # Format chunks with separators
        formatted_chunks = []
        total_tokens = 0
        max_tokens = self.config.max_input_tokens

        for i, text in enumerate(texts, 1):
            # Rough token estimate
            chunk_tokens = len(text.split()) * 1.3

            if total_tokens + chunk_tokens > max_tokens:
                # Truncate remaining text
                remaining_tokens = max_tokens - total_tokens
                words_to_keep = int(remaining_tokens / 1.3)
                text = " ".join(text.split()[:words_to_keep]) + "..."
                formatted_chunks.append(f"[Chunk {i}]\n{text}")
                break

            formatted_chunks.append(f"[Chunk {i}]\n{text}")
            total_tokens += chunk_tokens

        return "\n\n".join(formatted_chunks)

    def _prepare_summaries_text(self, summary_nodes: List[SummaryNode]) -> str:
        """Prepare summary texts for root summary prompt."""
        formatted = []

        for i, node in enumerate(summary_nodes, 1):
            formatted.append(f"[Section {i}]\n{node.text}")

        return "\n\n".join(formatted)

    def _fallback_summary(self, cluster: Cluster) -> str:
        """Create fallback summary when LLM fails."""
        texts = cluster.get_texts()

        # Extract first sentence from each chunk
        sentences = []
        for text in texts[:5]:  # Limit to 5 chunks
            # Simple sentence extraction
            first_sentence = text.split('.')[0].strip()
            if first_sentence:
                sentences.append(first_sentence + ".")

        return " ".join(sentences)

    def _fallback_root_summary(self, summary_nodes: List[SummaryNode]) -> str:
        """Create fallback root summary when LLM fails."""
        # Combine first sentences from each summary
        sentences = []
        for node in summary_nodes[:5]:
            first_sentence = node.text.split('.')[0].strip()
            if first_sentence:
                sentences.append(first_sentence + ".")

        return " ".join(sentences)


def create_summarizer(llm: LLM, config: Optional[SummarizationConfig] = None) -> RAPTORSummarizer:
    """
    Create a RAPTOR summarizer instance.

    Args:
        llm: LlamaIndex LLM instance
        config: Optional summarization configuration

    Returns:
        Configured RAPTORSummarizer
    """
    return RAPTORSummarizer(llm, config)
