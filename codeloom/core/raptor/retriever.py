"""RAPTOR retriever for level-aware hierarchical retrieval.

Implements multi-level retrieval that selects appropriate tree levels
based on query intent using LLM classification:
- Summary queries → higher levels (summaries)
- Detail queries → lower levels (chunks)
- Mixed queries → all levels
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever

from .config import RetrievalConfig, RAPTORConfig, DEFAULT_CONFIG

if TYPE_CHECKING:
    from ..vector_store import PGVectorStore

logger = logging.getLogger(__name__)


class RAPTORQueryType(Enum):
    """Types of queries for RAPTOR retrieval."""
    SUMMARY = "summary"      # High-level overview queries
    DETAIL = "detail"        # Specific, detailed queries
    MIXED = "mixed"          # Queries needing both levels


# LLM prompt for intent classification
INTENT_CLASSIFICATION_PROMPT = """Classify this query into ONE category based on what type of information the user needs:

SUMMARY - User wants high-level overview, main themes, key points, general understanding
Examples: "summarize this", "what are the key insights?", "give me an overview", "main takeaways"

DETAIL - User wants specific facts, exact information, particular sections, quotes
Examples: "what does it say about pricing?", "find the exact numbers", "what's on page 5?"

Query: "{query}"

Respond with ONLY one word: SUMMARY or DETAIL"""


@dataclass
class RAPTORRetrievalResult:
    """Result from RAPTOR retrieval."""
    nodes: List[NodeWithScore]
    query_type: RAPTORQueryType
    levels_searched: List[int]
    total_nodes_by_level: dict


class RAPTORRetriever(BaseRetriever):
    """Level-aware retriever for RAPTOR trees.

    Retrieves from different tree levels based on query intent:
    - Summary queries search higher levels (tree_level >= 1)
    - Detail queries search all levels, emphasizing level 0
    - Mixed queries search all levels with balanced weights
    """

    def __init__(
        self,
        vector_store: "PGVectorStore",
        notebook_id: str,
        source_ids: Optional[List[str]] = None,
        config: Optional[RAPTORConfig] = None,
        similarity_top_k: int = 10,
    ):
        """Initialize the RAPTOR retriever.

        Args:
            vector_store: Vector store containing RAPTOR nodes
            notebook_id: Notebook ID to retrieve from
            source_ids: Optional list of source IDs to filter by
            config: RAPTOR configuration
            similarity_top_k: Number of results to return
        """
        super().__init__()
        self.vector_store = vector_store
        self.notebook_id = notebook_id
        self.source_ids = source_ids
        self.config = config or DEFAULT_CONFIG
        self.retrieval_config = self.config.retrieval
        self.similarity_top_k = similarity_top_k

    def detect_query_type(self, query: str) -> Tuple[RAPTORQueryType, float]:
        """Detect whether query needs summary or detail retrieval using LLM.

        Uses the configured LLM to classify query intent, which is more
        accurate than keyword matching for understanding user intent.

        Args:
            query: The user's query string

        Returns:
            Tuple of (query_type, confidence)
        """
        try:
            # Get LLM from Settings
            llm = Settings.llm
            if not llm:
                logger.warning("No LLM available for intent detection, defaulting to MIXED")
                return RAPTORQueryType.MIXED, 0.5

            # Format the classification prompt
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)

            # Get LLM classification (simple completion)
            response = llm.complete(prompt)
            result = response.text.strip().upper()

            # Parse response
            if "SUMMARY" in result:
                logger.info(f"LLM classified query as SUMMARY: '{query[:50]}...'")
                return RAPTORQueryType.SUMMARY, 0.9
            elif "DETAIL" in result:
                logger.info(f"LLM classified query as DETAIL: '{query[:50]}...'")
                return RAPTORQueryType.DETAIL, 0.9
            else:
                # Unexpected response - default to DETAIL (safer, includes all levels)
                logger.warning(f"Unexpected LLM response '{result}', defaulting to DETAIL")
                return RAPTORQueryType.DETAIL, 0.5

        except Exception as e:
            logger.error(f"Error in LLM intent detection: {e}, defaulting to DETAIL")
            return RAPTORQueryType.DETAIL, 0.5

    def get_levels_for_query_type(
        self,
        query_type: RAPTORQueryType
    ) -> List[int]:
        """Get tree levels to search based on query type.

        Args:
            query_type: Detected query type

        Returns:
            List of tree levels to search
        """
        if query_type == RAPTORQueryType.SUMMARY:
            # Search higher levels (summaries)
            return self.retrieval_config.summary_query_levels
        elif query_type == RAPTORQueryType.DETAIL:
            # Search lower levels (chunks and low-level summaries)
            return self.retrieval_config.detail_query_levels
        else:
            # Mixed: search all levels
            return list(range(max(
                max(self.retrieval_config.summary_query_levels) + 1,
                max(self.retrieval_config.detail_query_levels) + 1
            )))

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes based on query intent and RAPTOR tree structure.

        Args:
            query_bundle: Query to retrieve for

        Returns:
            List of scored nodes
        """
        query = query_bundle.query_str

        # Detect query type
        query_type, confidence = self.detect_query_type(query)
        logger.info(
            f"RAPTOR query type: {query_type.value} (confidence: {confidence:.2f})"
        )

        # Get levels to search
        levels = self.get_levels_for_query_type(query_type)
        logger.debug(f"Searching RAPTOR levels: {levels}")

        # Retrieve nodes from appropriate levels
        all_nodes = []
        nodes_by_level = {}

        for level in levels:
            try:
                level_nodes = self.vector_store.get_nodes_by_tree_level(
                    notebook_id=self.notebook_id,
                    tree_level=level,
                    source_ids=self.source_ids
                )
                nodes_by_level[level] = len(level_nodes)
                all_nodes.extend(level_nodes)
            except Exception as e:
                logger.warning(f"Error retrieving level {level}: {e}")

        if not all_nodes:
            logger.warning("No RAPTOR nodes found, falling back to level 0")
            # Fallback to level 0 (original chunks)
            all_nodes = self.vector_store.get_nodes_by_tree_level(
                notebook_id=self.notebook_id,
                tree_level=0,
                source_ids=self.source_ids
            )
            nodes_by_level = {0: len(all_nodes)}

        logger.info(
            f"Retrieved {len(all_nodes)} nodes from levels {list(nodes_by_level.keys())}"
        )

        # Create index and retrieve
        if not all_nodes:
            return []

        # Use consistent top_k since we now always include L0 chunks
        # The score boosting will prioritize summaries for summary queries
        effective_top_k = self.similarity_top_k

        # Build index from collected nodes
        index = VectorStoreIndex(nodes=all_nodes)
        vector_retriever = index.as_retriever(
            similarity_top_k=effective_top_k
        )

        # Use hybrid retrieval if enabled (BM25 + Vector fusion)
        if self.retrieval_config.use_hybrid_retrieval:
            try:
                # BM25 for keyword matching - crucial for structured content
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    similarity_top_k=effective_top_k
                )

                # Get number of query variations (1 = original only, 3+ = expansion)
                num_queries = getattr(
                    self.retrieval_config, 'num_query_variations', 3
                )

                # Fuse BM25 and Vector retrievers with query expansion
                fusion_retriever = QueryFusionRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    retriever_weights=[
                        self.retrieval_config.bm25_weight,
                        self.retrieval_config.vector_weight
                    ],
                    similarity_top_k=effective_top_k,
                    num_queries=num_queries,  # Generate query variations for better coverage
                    mode="dist_based_score"
                )

                results = fusion_retriever.retrieve(query_bundle)
                logger.info(
                    f"Hybrid retrieval (BM25={self.retrieval_config.bm25_weight:.1f}, "
                    f"Vector={self.retrieval_config.vector_weight:.1f}, queries={num_queries}) "
                    f"returned {len(results)} nodes"
                )
            except Exception as e:
                # Fallback to vector-only if hybrid fails
                logger.warning(f"Hybrid retrieval failed, falling back to vector: {e}")
                results = vector_retriever.retrieve(query_bundle)
                logger.info(f"Vector-only fallback returned {len(results)} nodes")
        else:
            # Vector-only retrieval
            results = vector_retriever.retrieve(query_bundle)
            logger.info(f"Vector search returned {len(results)} nodes (top_k={effective_top_k})")

        # Boost scores for higher-level nodes in summary queries
        if query_type == RAPTORQueryType.SUMMARY:
            results = self._boost_summary_nodes(results)
        elif query_type == RAPTORQueryType.DETAIL:
            results = self._boost_detail_nodes(results)

        return results

    def _boost_summary_nodes(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Boost scores for summary nodes (higher tree levels).

        Args:
            nodes: Retrieved nodes with scores

        Returns:
            Nodes with boosted scores for summaries
        """
        boost_factor = self.retrieval_config.summary_level_boost

        for node_with_score in nodes:
            tree_level = node_with_score.node.metadata.get("tree_level", 0)
            if tree_level > 0:
                # Boost based on level
                level_boost = 1.0 + (tree_level * (boost_factor - 1.0) / 3.0)
                node_with_score.score = (node_with_score.score or 0.0) * level_boost

        # Re-sort by score
        return sorted(nodes, key=lambda x: x.score or 0, reverse=True)

    def _boost_detail_nodes(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Boost scores for detail nodes (lower tree levels).

        Args:
            nodes: Retrieved nodes with scores

        Returns:
            Nodes with boosted scores for details
        """
        boost_factor = self.retrieval_config.detail_level_boost

        for node_with_score in nodes:
            tree_level = node_with_score.node.metadata.get("tree_level", 0)
            if tree_level == 0:
                # Boost level 0 nodes
                node_with_score.score = (node_with_score.score or 0.0) * boost_factor

        # Re-sort by score
        return sorted(nodes, key=lambda x: x.score or 0, reverse=True)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieval (delegates to sync for now)."""
        return self._retrieve(query_bundle)


def create_raptor_retriever(
    vector_store: "PGVectorStore",
    notebook_id: str,
    source_ids: Optional[List[str]] = None,
    config: Optional[RAPTORConfig] = None,
    similarity_top_k: int = 10,
) -> RAPTORRetriever:
    """Create a RAPTOR retriever.

    Convenience function for creating a configured RAPTOR retriever.

    Args:
        vector_store: Vector store containing RAPTOR nodes
        notebook_id: Notebook ID to retrieve from
        source_ids: Optional list of source IDs to filter by
        config: RAPTOR configuration
        similarity_top_k: Number of results to return

    Returns:
        Configured RAPTORRetriever instance
    """
    return RAPTORRetriever(
        vector_store=vector_store,
        notebook_id=notebook_id,
        source_ids=source_ids,
        config=config,
        similarity_top_k=similarity_top_k,
    )


def has_raptor_tree(
    vector_store: "PGVectorStore",
    source_id: str,
    notebook_id: str,
) -> bool:
    """Check if a source has a RAPTOR tree built.

    Args:
        vector_store: Vector store to check
        source_id: Source ID to check
        notebook_id: Notebook containing the source

    Returns:
        True if RAPTOR tree exists (has nodes at level > 0)
    """
    try:
        # Check for level 1+ nodes
        stats = vector_store.get_tree_stats(source_id)
        return stats.get("max_level", 0) > 0
    except Exception as e:
        logger.debug(f"Error checking RAPTOR tree: {e}")
        return False
