"""RAPTOR Tree Builder Module.

Orchestrates the construction of hierarchical summary trees from document chunks.
Coordinates clustering and summarization to build multi-level abstraction trees.

Key Features:
- Recursive tree building from leaf chunks to root summary
- Embedding generation for summary nodes
- Progress tracking and status updates
- Integration with vector store

Reference: https://arxiv.org/abs/2401.18059
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode, TextNode

from .config import RAPTORConfig, DEFAULT_CONFIG
from .clustering import RAPTORClustering, Cluster
from .summarizer import RAPTORSummarizer, SummaryNode

logger = logging.getLogger(__name__)


@dataclass
class TreeBuildResult:
    """Result of RAPTOR tree building."""
    success: bool
    source_id: str
    notebook_id: str

    # Tree statistics
    total_nodes: int = 0
    levels: Dict[int, int] = field(default_factory=dict)
    max_level: int = 0

    # Nodes to store
    summary_nodes: List[TextNode] = field(default_factory=list)

    # Timing
    build_time_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "success": self.success,
            "source_id": self.source_id,
            "notebook_id": self.notebook_id,
            "total_nodes": self.total_nodes,
            "levels": self.levels,
            "max_level": self.max_level,
            "build_time_seconds": self.build_time_seconds,
            "error": self.error,
        }


class RAPTORTreeBuilder:
    """
    Builds hierarchical RAPTOR trees from document chunks.

    The tree is built bottom-up:
    1. Start with document chunks (level 0)
    2. Cluster chunks by semantic similarity
    3. Summarize each cluster (level 1)
    4. Repeat clustering/summarization until root node
    """

    def __init__(
        self,
        llm: LLM,
        embed_model: BaseEmbedding,
        config: Optional[RAPTORConfig] = None
    ):
        """
        Initialize tree builder.

        Args:
            llm: LlamaIndex LLM for summarization
            embed_model: Embedding model for summary embeddings
            config: RAPTOR configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.llm = llm
        self.embed_model = embed_model

        # Initialize components
        self.clustering = RAPTORClustering(self.config.clustering)
        self.summarizer = RAPTORSummarizer(llm, self.config.summarization)

    def build_tree(
        self,
        chunks: List[BaseNode],
        source_id: str,
        notebook_id: str,
        progress_callback: Optional[callable] = None
    ) -> TreeBuildResult:
        """
        Build RAPTOR tree from document chunks.

        Args:
            chunks: List of document chunks (level 0 nodes)
            source_id: Source document ID
            notebook_id: Notebook ID
            progress_callback: Optional callback(stage, progress, message)

        Returns:
            TreeBuildResult with summary nodes and statistics
        """
        started_at = datetime.utcnow()
        start_time = time.time()

        result = TreeBuildResult(
            success=False,
            source_id=source_id,
            notebook_id=notebook_id,
            started_at=started_at
        )

        try:
            if len(chunks) < self.config.tree_building.min_nodes_to_cluster:
                logger.info(
                    f"Too few chunks ({len(chunks)}) for RAPTOR tree, "
                    f"minimum is {self.config.tree_building.min_nodes_to_cluster}"
                )
                # Still mark as success - just no tree needed
                result.success = True
                result.total_nodes = len(chunks)
                result.levels = {0: len(chunks)}
                return result

            if progress_callback:
                progress_callback("init", 0.0, f"Starting RAPTOR tree build for {len(chunks)} chunks")

            # Mark chunks as level 0
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    chunk.metadata["tree_level"] = 0

            # Build tree recursively
            all_summary_nodes = []
            current_level_nodes = chunks
            current_level = 0
            levels = {0: len(chunks)}

            while (
                len(current_level_nodes) >= self.config.tree_building.min_nodes_to_cluster and
                current_level < self.config.tree_building.max_tree_depth
            ):
                current_level += 1

                if progress_callback:
                    progress = current_level / self.config.tree_building.max_tree_depth
                    progress_callback(
                        "clustering",
                        progress * 0.4,
                        f"Building level {current_level}: clustering {len(current_level_nodes)} nodes"
                    )

                # Cluster current level nodes
                clusters = self.clustering.cluster_nodes(current_level_nodes)

                if len(clusters) <= 1 and current_level > 1:
                    # Only one cluster - create root and stop
                    logger.info(f"Single cluster at level {current_level}, creating root node")
                    break

                if progress_callback:
                    progress_callback(
                        "summarizing",
                        0.4 + (current_level / self.config.tree_building.max_tree_depth) * 0.5,
                        f"Level {current_level}: summarizing {len(clusters)} clusters"
                    )

                # Summarize each cluster
                level_summary_nodes = []
                for cluster in clusters:
                    summary_node = self.summarizer.summarize_cluster_sync(
                        cluster=cluster,
                        tree_level=current_level,
                        source_id=source_id,
                        notebook_id=notebook_id
                    )
                    level_summary_nodes.append(summary_node)

                # Generate embeddings for summary nodes
                summary_texts = [sn.text for sn in level_summary_nodes]
                embeddings = self.embed_model.get_text_embedding_batch(
                    summary_texts,
                    show_progress=False
                )

                # Convert to TextNodes with embeddings
                for i, summary_node in enumerate(level_summary_nodes):
                    text_node = summary_node.to_text_node()
                    text_node.embedding = embeddings[i]
                    all_summary_nodes.append(text_node)

                levels[current_level] = len(level_summary_nodes)
                logger.info(
                    f"Level {current_level}: created {len(level_summary_nodes)} summary nodes "
                    f"from {len(clusters)} clusters"
                )

                # Prepare for next level
                # Convert SummaryNodes to TextNodes for clustering
                current_level_nodes = [sn.to_text_node() for sn in level_summary_nodes]
                for i, node in enumerate(current_level_nodes):
                    node.embedding = embeddings[i]

            # Create root node if we have multiple summaries at the top level
            if len(current_level_nodes) > 1:
                current_level += 1

                if progress_callback:
                    progress_callback("root", 0.9, "Creating root summary node")

                # Get the SummaryNodes we just created (convert back)
                top_level_summaries = []
                for node in current_level_nodes:
                    # Reconstruct SummaryNode for root summarization
                    sn = SummaryNode(
                        node_id=node.node_id,
                        text=node.get_content(),
                        child_node_ids=node.metadata.get("child_node_ids", []),
                        tree_level=node.metadata.get("tree_level", current_level - 1),
                        cluster_id=node.metadata.get("cluster_id", ""),
                        token_count=len(node.get_content().split()),
                        source_id=source_id,
                        notebook_id=notebook_id
                    )
                    top_level_summaries.append(sn)

                root_summary = self.summarizer.summarize_summaries_sync(
                    summary_nodes=top_level_summaries,
                    tree_level=current_level,
                    source_id=source_id,
                    notebook_id=notebook_id
                )

                # Generate embedding for root
                root_embedding = self.embed_model.get_text_embedding(root_summary.text)
                root_text_node = root_summary.to_text_node()
                root_text_node.embedding = root_embedding
                root_text_node.metadata["tree_root_id"] = root_text_node.node_id

                all_summary_nodes.append(root_text_node)
                levels[current_level] = 1

                # Update all nodes with root ID
                root_id = root_text_node.node_id
                for node in all_summary_nodes:
                    node.metadata["tree_root_id"] = root_id

                logger.info(f"Created root node at level {current_level}")

            # Finalize result
            result.success = True
            result.summary_nodes = all_summary_nodes
            result.total_nodes = len(chunks) + len(all_summary_nodes)
            result.levels = levels
            result.max_level = max(levels.keys()) if levels else 0
            result.build_time_seconds = time.time() - start_time
            result.completed_at = datetime.utcnow()

            if progress_callback:
                progress_callback(
                    "complete",
                    1.0,
                    f"RAPTOR tree complete: {result.total_nodes} nodes across {len(levels)} levels"
                )

            logger.info(
                f"RAPTOR tree built: {result.total_nodes} total nodes, "
                f"{len(all_summary_nodes)} summaries, "
                f"max level {result.max_level}, "
                f"time {result.build_time_seconds:.2f}s"
            )

        except Exception as e:
            logger.error(f"RAPTOR tree build failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.error = str(e)
            result.build_time_seconds = time.time() - start_time
            result.completed_at = datetime.utcnow()

        return result

    async def build_tree_async(
        self,
        chunks: List[BaseNode],
        source_id: str,
        notebook_id: str,
        progress_callback: Optional[callable] = None
    ) -> TreeBuildResult:
        """
        Async version of build_tree.

        Uses async LLM calls for better performance with remote LLMs.
        """
        started_at = datetime.utcnow()
        start_time = time.time()

        result = TreeBuildResult(
            success=False,
            source_id=source_id,
            notebook_id=notebook_id,
            started_at=started_at
        )

        try:
            if len(chunks) < self.config.tree_building.min_nodes_to_cluster:
                result.success = True
                result.total_nodes = len(chunks)
                result.levels = {0: len(chunks)}
                return result

            # Mark chunks as level 0
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    chunk.metadata["tree_level"] = 0

            # Build tree recursively
            all_summary_nodes = []
            current_level_nodes = chunks
            current_level = 0
            levels = {0: len(chunks)}

            while (
                len(current_level_nodes) >= self.config.tree_building.min_nodes_to_cluster and
                current_level < self.config.tree_building.max_tree_depth
            ):
                current_level += 1

                # Cluster current level nodes
                clusters = self.clustering.cluster_nodes(current_level_nodes)

                if len(clusters) <= 1 and current_level > 1:
                    break

                # Summarize each cluster (async)
                import asyncio
                summary_tasks = [
                    self.summarizer.summarize_cluster(
                        cluster=cluster,
                        tree_level=current_level,
                        source_id=source_id,
                        notebook_id=notebook_id
                    )
                    for cluster in clusters
                ]
                level_summary_nodes = await asyncio.gather(*summary_tasks)

                # Generate embeddings for summary nodes
                summary_texts = [sn.text for sn in level_summary_nodes]
                embeddings = self.embed_model.get_text_embedding_batch(
                    summary_texts,
                    show_progress=False
                )

                # Convert to TextNodes with embeddings
                for i, summary_node in enumerate(level_summary_nodes):
                    text_node = summary_node.to_text_node()
                    text_node.embedding = embeddings[i]
                    all_summary_nodes.append(text_node)

                levels[current_level] = len(level_summary_nodes)

                # Prepare for next level
                current_level_nodes = [sn.to_text_node() for sn in level_summary_nodes]
                for i, node in enumerate(current_level_nodes):
                    node.embedding = embeddings[i]

            # Create root node if needed
            if len(current_level_nodes) > 1:
                current_level += 1

                top_level_summaries = []
                for node in current_level_nodes:
                    sn = SummaryNode(
                        node_id=node.node_id,
                        text=node.get_content(),
                        child_node_ids=node.metadata.get("child_node_ids", []),
                        tree_level=node.metadata.get("tree_level", current_level - 1),
                        cluster_id=node.metadata.get("cluster_id", ""),
                        token_count=len(node.get_content().split()),
                        source_id=source_id,
                        notebook_id=notebook_id
                    )
                    top_level_summaries.append(sn)

                root_summary = await self.summarizer.summarize_summaries(
                    summary_nodes=top_level_summaries,
                    tree_level=current_level,
                    source_id=source_id,
                    notebook_id=notebook_id
                )

                root_embedding = self.embed_model.get_text_embedding(root_summary.text)
                root_text_node = root_summary.to_text_node()
                root_text_node.embedding = root_embedding
                root_text_node.metadata["tree_root_id"] = root_text_node.node_id

                all_summary_nodes.append(root_text_node)
                levels[current_level] = 1

                root_id = root_text_node.node_id
                for node in all_summary_nodes:
                    node.metadata["tree_root_id"] = root_id

            # Finalize result
            result.success = True
            result.summary_nodes = all_summary_nodes
            result.total_nodes = len(chunks) + len(all_summary_nodes)
            result.levels = levels
            result.max_level = max(levels.keys()) if levels else 0
            result.build_time_seconds = time.time() - start_time
            result.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"RAPTOR tree build failed: {e}")
            result.error = str(e)
            result.build_time_seconds = time.time() - start_time
            result.completed_at = datetime.utcnow()

        return result


def create_tree_builder(
    llm: LLM,
    embed_model: BaseEmbedding,
    config: Optional[RAPTORConfig] = None
) -> RAPTORTreeBuilder:
    """
    Create a RAPTOR tree builder instance.

    Args:
        llm: LlamaIndex LLM for summarization
        embed_model: Embedding model for summary embeddings
        config: Optional RAPTOR configuration

    Returns:
        Configured RAPTORTreeBuilder
    """
    return RAPTORTreeBuilder(llm, embed_model, config)
