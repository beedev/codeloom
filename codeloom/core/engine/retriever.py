"""
Retrieval engine for DBNotebook RAG pipeline.

This module provides the core retrieval logic, including:
- TwoStageRetriever: Hybrid BM25 + vector retrieval with optional reranking
- LocalRetriever: Smart retriever selection based on index size and query type
- QueryIntent detection: Routes queries to appropriate retrieval strategy
- RAPTOR integration: Hierarchical retrieval for summary-type queries

The retrieval pipeline adapts based on:
- Index size: Small indexes (≤6 nodes) use simple vector search
- Query intent: Summary queries leverage RAPTOR trees when available
- Reranker availability: Falls back to fusion scoring when reranker disabled
"""

import logging
import os
import re
import time
from enum import Enum
from typing import List, Optional, Tuple, Dict

from dotenv import load_dotenv
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from dbnotebook.core.providers.reranker_provider import get_shared_reranker, is_reranker_enabled
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, IndexNode
from llama_index.core.llms.llm import LLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex

from ..prompt import get_query_gen_prompt
from ...setting import get_settings, RAGSettings, QueryTimeSettings
from ..raptor import RAPTORRetriever, has_raptor_tree, RAPTORConfig
from ..utils import unwrap_llm

load_dotenv()

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intent for transformation-aware retrieval."""
    SUMMARY = "summary"       # User wants document overview/summary
    INSIGHTS = "insights"     # User wants key takeaways/insights
    QUESTIONS = "questions"   # User wants reflection questions
    SEARCH = "search"         # Default: semantic search across all content


# Intent detection patterns
INTENT_PATTERNS = {
    QueryIntent.SUMMARY: [
        r'\bsummar(y|ize|ise)\b',
        r'\boverview\b',
        r'\bwhat\s+is\s+(this|the\s+document)\s+about\b',
        r'\bgive\s+me\s+(a\s+)?summary\b',
        r'\btl;?dr\b',
        r'\bin\s+brief\b',
        r'\bhigh[- ]?level\b',
        r'\bmain\s+(points?|idea)\b',
    ],
    QueryIntent.INSIGHTS: [
        r'\bkey\s+(points?|takeaways?|insights?)\b',
        r'\bimportant\s+(points?|takeaways?|things?)\b',
        r'\bwhat\s+(are|were)\s+the\s+(key|main|important)\b',
        r'\binsights?\b',
        r'\btakeaways?\b',
        r'\blessons?\s+learned\b',
        r'\bactionable\b',
        r'\bwhat\s+should\s+i\s+(know|remember)\b',
    ],
    QueryIntent.QUESTIONS: [
        r'\bquestions?\s+(to|for)\s+(ask|explore|consider)\b',
        r'\breflection\s+questions?\b',
        r'\bwhat\s+questions?\s+(should|can|could)\b',
        r'\bhelp\s+me\s+(think|explore)\b',
        r'\bthink\s+(about|deeper)\b',
        r'\bexplore\s+further\b',
        r'\bdiscussion\s+questions?\b',
    ],
}


class TwoStageRetriever(QueryFusionRetriever):
    """
    Two-stage retriever that combines fusion retrieval with reranking.

    Stage 1: BM25 + Vector retrieval with query fusion
    Stage 2: Rerank results using cross-encoder model
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        setting: RAGSettings | None = None,
        llm: Optional[LLM] = None,
        query_gen_prompt: Optional[str] = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = 20,
        num_queries: int = 4,
        use_async: bool = False,  # Disabled to avoid nested async issues with Flask
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        retriever_weights: Optional[List[float]] = None
    ) -> None:
        super().__init__(
            retrievers, llm, query_gen_prompt, mode, similarity_top_k, num_queries,
            use_async, verbose, callback_manager, objects, object_map, retriever_weights
        )
        self._setting = setting or get_settings()
        # Store default from env var, but allow per-request override via is_reranker_enabled()
        env_disable_reranker = os.getenv("DISABLE_RERANKER", "").lower() == "true"
        if env_disable_reranker:
            logger.info("DISABLE_RERANKER env var is set - reranker will be disabled unless overridden by API")
        logger.debug(f"TwoStageRetriever initialized (default rerank model: {self._setting.retriever.rerank_llm})")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        results = self._simple_fusion(results)

        # Check reranker status at query time (respects per-request config from set_reranker_config)
        if not is_reranker_enabled():
            logger.debug("Reranker disabled (per-request or env var)")
            return results[:self._setting.retriever.top_k_rerank]

        # Get reranker at query time (may have changed via set_reranker_config)
        rerank_model = get_shared_reranker(
            model=self._setting.retriever.rerank_llm,
            top_n=self._setting.retriever.top_k_rerank
        )
        if rerank_model is None:
            logger.debug("Reranker model not available, returning fusion results")
            return results[:self._setting.retriever.top_k_rerank]

        return rerank_model.postprocess_nodes(results, query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)

        # Check reranker status at query time (respects per-request config from set_reranker_config)
        if not is_reranker_enabled():
            logger.debug("Reranker disabled (per-request or env var)")
            return results[:self._setting.retriever.top_k_rerank]

        # Get reranker at query time (may have changed via set_reranker_config)
        rerank_model = get_shared_reranker(
            model=self._setting.retriever.rerank_llm,
            top_n=self._setting.retriever.top_k_rerank
        )
        if rerank_model is None:
            logger.debug("Reranker model not available, returning fusion results")
            return results[:self._setting.retriever.top_k_rerank]

        return rerank_model.postprocess_nodes(results, query_bundle)


class LocalRetriever:
    """
    Factory for creating optimized retrievers based on document count.

    Strategies:
    - Small collections (≤ top_k_rerank): Simple vector retriever
    - Large collections: Router with fusion and two-stage options

    Query-Time Settings:
    - Call set_query_settings() before each query to apply per-request overrides
    - Settings affect similarity_top_k, retriever_weights, etc.
    """

    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        self._setting = setting or get_settings()
        self._host = host
        self._index_cache: Optional[VectorStoreIndex] = None
        self._cached_node_count: int = 0
        self._cached_notebook_id: Optional[str] = None
        # Query-time settings (updated per-request, cleared after use)
        self._query_settings: Optional[QueryTimeSettings] = None

        # Retriever cache for performance optimization
        # Cache format: {cache_key: (retriever, timestamp)}
        # Cache key: f"{notebook_id}:{node_count}"
        self._retriever_cache: Dict[str, Tuple[BaseRetriever, float]] = {}
        self._retriever_cache_ttl = 300  # 5 minutes TTL

        logger.debug("LocalRetriever initialized")

    def set_query_settings(self, settings: Optional[QueryTimeSettings]) -> None:
        """Set query-time settings for the next retrieval operation.

        These settings override defaults from config files for a single query.
        Settings are automatically cleared after retrieval.

        Args:
            settings: QueryTimeSettings instance, or None to use defaults
        """
        self._query_settings = settings
        if settings:
            logger.debug(
                f"Query settings applied: bm25={settings.bm25_weight:.2f}, "
                f"vector={settings.vector_weight:.2f}, top_k={settings.similarity_top_k}, "
                f"temp={settings.temperature:.2f}"
            )

    def clear_query_settings(self) -> None:
        """Clear query-time settings, reverting to defaults."""
        self._query_settings = None

    def _get_similarity_top_k(self) -> int:
        """Get similarity_top_k, preferring query-time settings."""
        if self._query_settings:
            return self._query_settings.similarity_top_k
        return self._setting.retriever.similarity_top_k

    def _get_retriever_weights(self) -> List[float]:
        """Get retriever weights [bm25, vector], preferring query-time settings."""
        if self._query_settings:
            return [self._query_settings.bm25_weight, self._query_settings.vector_weight]
        return self._setting.retriever.retriever_weights

    def detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Detect query intent based on keyword patterns.

        Returns:
            Tuple of (QueryIntent, confidence_score)
            confidence_score: 0.0-1.0 indicating pattern match strength
        """
        query_lower = query.lower().strip()

        # Check each intent type
        intent_scores = {}

        for intent, patterns in INTENT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matches += 1

            if matches > 0:
                # Score based on number of pattern matches
                intent_scores[intent] = min(matches / 2.0, 1.0)

        if not intent_scores:
            # No patterns matched - default to search
            return QueryIntent.SEARCH, 0.0

        # Return highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]

        logger.debug(f"Detected intent: {best_intent.value} (confidence: {confidence:.2f}) for query: {query[:50]}...")
        return best_intent, confidence

    def get_node_types_for_intent(self, intent: QueryIntent) -> List[str]:
        """
        Get the node types to prioritize based on detected intent.

        Returns:
            List of node_type values to filter by
        """
        if intent == QueryIntent.SUMMARY:
            # Summary queries: prioritize summary nodes, fall back to chunks
            return ["summary", "chunk"]
        elif intent == QueryIntent.INSIGHTS:
            # Insight queries: prioritize insight nodes, include summary and chunks
            return ["insight", "summary", "chunk"]
        elif intent == QueryIntent.QUESTIONS:
            # Question queries: prioritize question nodes
            return ["question", "insight", "chunk"]
        else:
            # Default search: all node types (chunks first for semantic search)
            return ["chunk", "summary", "insight", "question"]

    def get_intent_aware_nodes(
        self,
        query: str,
        nodes: List[BaseNode],
        vector_store=None,
        notebook_id: Optional[str] = None
    ) -> Tuple[List[BaseNode], QueryIntent]:
        """
        Get nodes filtered/prioritized by detected query intent.

        For transformation-aware retrieval:
        - Summary queries → prioritize summary nodes
        - Insight queries → prioritize insight nodes
        - Question queries → prioritize question nodes
        - Search queries → use all nodes with standard retrieval

        Args:
            query: The user's query string
            nodes: Base nodes (chunks) available for retrieval
            vector_store: Optional PGVectorStore for fetching transformation nodes
            notebook_id: Notebook ID for filtering transformation nodes

        Returns:
            Tuple of (filtered_nodes, detected_intent)
        """
        intent, confidence = self.detect_intent(query)

        # If no strong intent detected or no vector store, use standard nodes
        if intent == QueryIntent.SEARCH or confidence < 0.3 or not vector_store:
            return nodes, intent

        # Get node types to prioritize
        node_types = self.get_node_types_for_intent(intent)
        logger.info(f"Intent-aware retrieval: {intent.value} → prioritizing {node_types}")

        # Try to get transformation nodes from vector store
        if hasattr(vector_store, 'get_nodes_by_notebook_and_types') and notebook_id:
            try:
                # Get transformation nodes for this notebook
                transformation_nodes = vector_store.get_nodes_by_notebook_and_types(
                    notebook_id=notebook_id,
                    node_types=node_types[:2]  # Primary types for this intent
                )

                if transformation_nodes:
                    logger.info(
                        f"Found {len(transformation_nodes)} transformation nodes "
                        f"for intent {intent.value}"
                    )
                    # Combine transformation nodes with regular chunks
                    # Transformation nodes first (higher priority)
                    combined = transformation_nodes + nodes
                    return combined, intent

            except Exception as e:
                logger.warning(f"Error fetching transformation nodes: {e}")

        # Fall back to regular nodes
        return nodes, intent

    def _get_or_create_index(
        self,
        nodes: List[BaseNode],
        notebook_id: Optional[str] = None,
        force_rebuild: bool = False
    ) -> VectorStoreIndex:
        """Get cached index or create new one.

        Args:
            nodes: Document nodes to index
            notebook_id: Notebook ID for cache isolation
            force_rebuild: Force rebuild even if cached

        Returns:
            VectorStoreIndex for the nodes
        """
        # Check if cache is valid (same notebook and node count)
        cache_valid = (
            not force_rebuild and
            self._index_cache is not None and
            len(nodes) == self._cached_node_count and
            notebook_id == self._cached_notebook_id
        )

        if cache_valid:
            logger.debug(f"Using cached vector index for notebook {notebook_id}")
            return self._index_cache

        # Build new index
        start_time = time.time()
        logger.debug(f"Creating new vector index with {len(nodes)} nodes for notebook {notebook_id}")
        self._index_cache = VectorStoreIndex(nodes=nodes)
        self._cached_node_count = len(nodes)
        self._cached_notebook_id = notebook_id
        build_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Built vector index: {len(nodes)} nodes in {build_time_ms}ms")
        return self._index_cache

    def _get_normal_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: Optional[LLM] = None,
        language: str = "eng",
    ) -> VectorIndexRetriever:
        """Create simple vector retriever for small collections."""
        return VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._get_similarity_top_k(),
            embed_model=Settings.embed_model,
            verbose=False
        )

    def _get_hybrid_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: Optional[LLM] = None,
        language: str = "eng",
        gen_query: bool = True
    ) -> BaseRetriever:
        """Create hybrid BM25 + vector retriever.

        Uses query-time settings if set, otherwise falls back to config defaults.
        """
        llm = llm or Settings.llm
        similarity_top_k = self._get_similarity_top_k()
        weights = self._get_retriever_weights()

        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=False
        )

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=similarity_top_k,
            verbose=False
        )

        retrievers = [bm25_retriever, vector_retriever]

        if gen_query:
            # Fusion retriever with query generation
            # use_async=False to avoid nested async issues with Flask
            return QueryFusionRetriever(
                retrievers=retrievers,
                retriever_weights=weights,
                llm=llm,
                query_gen_prompt=get_query_gen_prompt(language),
                similarity_top_k=self._setting.retriever.top_k_rerank,
                num_queries=self._setting.retriever.num_queries,
                mode=self._setting.retriever.fusion_mode,
                use_async=False,
                verbose=False
            )
        else:
            # Two-stage retriever with reranking
            # use_async=False to avoid nested async issues with Flask
            return TwoStageRetriever(
                retrievers=retrievers,
                retriever_weights=weights,
                setting=self._setting,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                use_async=False,
                verbose=False
            )

    def get_retrievers(
        self,
        llm: LLM,
        language: str,
        nodes: List[BaseNode],
        offering_filter: Optional[List[str]] = None,
        practice_filter: Optional[List[str]] = None,
        vector_store=None,
        notebook_id: Optional[str] = None
    ) -> BaseRetriever:
        """
        Get appropriate retriever based on collection size with optional filtering.

        Uses caching to avoid rebuilding retrievers for repeat queries.
        Cache is invalidated after TTL (5 minutes) or when notebook/nodes change.

        Args:
            llm: Language model for query generation
            language: Language code for prompts
            nodes: Document nodes to index
            offering_filter: List of offering IDs to filter by (OR operation)
            practice_filter: List of practice names to filter by (OR operation)
            vector_store: Optional LocalVectorStore instance for metadata filtering
            notebook_id: Notebook ID for cache isolation

        Returns:
            Configured retriever instance
        """
        # Unwrap LLM wrappers (e.g., GroqWithBackoff) for LlamaIndex compatibility
        llm = unwrap_llm(llm)

        # Check retriever cache first (only if notebook_id is valid)
        cache_key = f"{notebook_id}:{len(nodes)}" if notebook_id else None
        current_time = time.time()

        if cache_key and cache_key in self._retriever_cache:
            retriever, timestamp = self._retriever_cache[cache_key]
            if current_time - timestamp < self._retriever_cache_ttl:
                logger.debug(f"Retriever cache hit for {cache_key}")
                return retriever
            else:
                logger.debug(f"Retriever cache expired for {cache_key}")

        # Cache miss - build retriever
        start_time = time.time()

        # Apply metadata filtering if requested
        filtered_nodes = nodes

        if (offering_filter or practice_filter) and vector_store:
            # Use vector store's filtering capability
            filtered_index = vector_store.get_index_with_filter(
                nodes=nodes,
                offering_ids=offering_filter,
                practice_names=practice_filter
            )

            if filtered_index is None:
                logger.warning("No nodes matched the filters, using unfiltered retriever")
                vector_index = self._get_or_create_index(nodes, notebook_id=notebook_id)
            else:
                vector_index = filtered_index
                # Update filtered_nodes count for retriever selection
                if offering_filter or practice_filter:
                    # Estimate filtered node count
                    # Note: offering_filter contains notebook_id (legacy naming)
                    if offering_filter:
                        # Use notebook_id - the current architecture
                        filtered_nodes = vector_store.get_nodes_by_metadata(
                            nodes, {"notebook_id": offering_filter[0]}
                        )
                    elif practice_filter:
                        filtered_nodes = vector_store.get_nodes_by_metadata(
                            nodes, {"it_practice": practice_filter[0]}
                        )
                    # Fall back to all nodes if no matches
                    if not filtered_nodes:
                        filtered_nodes = nodes
        elif offering_filter or practice_filter:
            # Manual filtering without vector store
            logger.debug("Filtering nodes manually without vector store")
            logger.debug(f"Offering filter: {offering_filter}, Practice filter: {practice_filter}")
            logger.debug(f"Total nodes to filter: {len(nodes)}")

            filtered_nodes = []

            for node in nodes:
                metadata = node.metadata or {}

                # Debug: Log metadata for each node (only at debug level to reduce noise)
                logger.debug(f"Node: {metadata.get('file_name', 'unknown')}")

                # Check offering filter (by name, id, or notebook_id)
                if offering_filter:
                    node_offering_id = metadata.get("offering_id")
                    node_offering_name = metadata.get("offering_name")
                    node_notebook_id = metadata.get("notebook_id")

                    # Match against offering_id, offering_name, OR notebook_id
                    if (node_offering_id and node_offering_id in offering_filter) or \
                       (node_offering_name and node_offering_name in offering_filter) or \
                       (node_notebook_id and node_notebook_id in offering_filter):
                        logger.debug(f"✓ Node MATCHED: {metadata.get('file_name')}")
                        filtered_nodes.append(node)
                        continue
                    else:
                        logger.debug(f"✗ Node REJECTED: {metadata.get('file_name')}")

                # Check practice filter
                if practice_filter:
                    node_practice = metadata.get("it_practice")
                    if node_practice and node_practice in practice_filter:
                        filtered_nodes.append(node)
                        continue

            if not filtered_nodes:
                logger.warning("No nodes matched the filters, using all nodes")
                filtered_nodes = nodes

            logger.info(
                f"Filtered {len(filtered_nodes)} nodes from {len(nodes)} "
                f"(offerings={offering_filter}, practices={practice_filter})"
            )

            vector_index = self._get_or_create_index(filtered_nodes, notebook_id=notebook_id)
        else:
            # No filtering
            vector_index = self._get_or_create_index(nodes, notebook_id=notebook_id)

        # Select retriever strategy based on node count
        node_count = len(filtered_nodes)

        if node_count > self._setting.retriever.top_k_rerank:
            logger.debug(f"Using two-stage retriever for {node_count} nodes")
            retriever = self._get_hybrid_retriever(vector_index, llm, language, gen_query=False)
        else:
            logger.debug(f"Using simple retriever for {node_count} nodes")
            retriever = self._get_normal_retriever(vector_index, llm, language)

        # Cache the retriever (only if notebook_id is valid)
        build_time_ms = int((time.time() - start_time) * 1000)
        if cache_key:
            self._retriever_cache[cache_key] = (retriever, current_time)
            logger.info(f"Cached retriever for {cache_key} (built in {build_time_ms}ms)")
        else:
            logger.debug(f"Skipping retriever cache (no notebook_id) - built in {build_time_ms}ms")

        return retriever

    def get_all_nodes_for_offering(
        self,
        nodes: List[BaseNode],
        offering_filter: List[str]
    ) -> List[BaseNode]:
        """
        Retrieve ALL nodes for specific offerings (no top-k limit).

        Useful for comprehensive queries like "summarize Nexus offering"
        where we need full context, not just top-k similar chunks.

        Args:
            nodes: All available nodes
            offering_filter: List of offering names/IDs to retrieve

        Returns:
            All nodes matching the offering filter
        """
        if not offering_filter:
            logger.warning("No offering filter provided for comprehensive retrieval")
            return []

        filtered_nodes = []

        for node in nodes:
            metadata = node.metadata or {}
            node_offering_id = metadata.get("offering_id")
            node_offering_name = metadata.get("offering_name")

            if (node_offering_id and node_offering_id in offering_filter) or \
               (node_offering_name and node_offering_name in offering_filter):
                filtered_nodes.append(node)

        logger.info(
            f"Comprehensive retrieval: {len(filtered_nodes)} nodes for offerings {offering_filter}"
        )

        return filtered_nodes

    def clear_cache(self) -> None:
        """Clear all caches (index cache and retriever cache)."""
        self._index_cache = None
        self._cached_node_count = 0
        self._cached_notebook_id = None
        self._retriever_cache.clear()
        logger.debug("Retriever caches cleared (index + retriever)")

    def get_raptor_aware_retriever(
        self,
        llm: LLM,
        language: str,
        nodes: List[BaseNode],
        vector_store=None,
        notebook_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        raptor_config: Optional[RAPTORConfig] = None,
    ) -> BaseRetriever:
        """Get a retriever that uses RAPTOR trees when available.

        This method checks if RAPTOR trees exist for the sources and uses
        level-aware retrieval for better summary/detail query handling.

        Args:
            llm: Language model for query generation
            language: Language code for prompts
            nodes: Document nodes to index
            vector_store: Vector store with RAPTOR nodes
            notebook_id: Notebook ID for RAPTOR retrieval
            source_ids: Source IDs to check for RAPTOR trees
            raptor_config: Optional RAPTOR configuration

        Returns:
            RAPTOR-aware retriever if available, otherwise standard retriever
        """
        # Check if RAPTOR retrieval is possible
        if not vector_store or not notebook_id:
            logger.debug("RAPTOR not available: missing vector_store or notebook_id")
            return self.get_retrievers(llm, language, nodes, vector_store=vector_store)

        # Check if any sources have RAPTOR trees
        sources_with_raptor = []
        if source_ids:
            for source_id in source_ids:
                if has_raptor_tree(vector_store, source_id, notebook_id):
                    sources_with_raptor.append(source_id)

        if not sources_with_raptor:
            logger.debug("No RAPTOR trees found, using standard retrieval")
            return self.get_retrievers(llm, language, nodes, vector_store=vector_store)

        logger.info(
            f"Using RAPTOR retrieval for {len(sources_with_raptor)} sources "
            f"out of {len(source_ids) if source_ids else 'all'}"
        )

        # Create RAPTOR retriever
        # Use query-time similarity_top_k for initial retrieval
        return RAPTORRetriever(
            vector_store=vector_store,
            notebook_id=notebook_id,
            source_ids=sources_with_raptor,
            config=raptor_config,
            similarity_top_k=self._get_similarity_top_k(),
        )

    def get_combined_raptor_retriever(
        self,
        llm: LLM,
        language: str,
        nodes: List[BaseNode],
        vector_store=None,
        notebook_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
    ) -> BaseRetriever:
        """Get a retriever that combines RAPTOR and standard retrieval.

        For sources with RAPTOR trees, uses hierarchical retrieval.
        For sources without RAPTOR trees, uses standard retrieval.
        Results are merged and re-ranked.

        Args:
            llm: Language model for query generation
            language: Language code for prompts
            nodes: Document nodes to index
            vector_store: Vector store with RAPTOR nodes
            notebook_id: Notebook ID for RAPTOR retrieval
            source_ids: Source IDs to retrieve from

        Returns:
            Combined retriever with RAPTOR and standard retrieval
        """
        # Unwrap LLM wrappers (e.g., GroqWithBackoff) for LlamaIndex compatibility
        llm = unwrap_llm(llm)

        if not vector_store or not notebook_id:
            return self.get_retrievers(llm, language, nodes, vector_store=vector_store)

        # Split sources by RAPTOR availability
        raptor_sources = []
        standard_sources = []

        if source_ids:
            for source_id in source_ids:
                if has_raptor_tree(vector_store, source_id, notebook_id):
                    raptor_sources.append(source_id)
                else:
                    standard_sources.append(source_id)
        else:
            # Check all nodes' source IDs
            seen_sources = set()
            for node in nodes:
                source_id = node.metadata.get("source_id")
                if source_id and source_id not in seen_sources:
                    seen_sources.add(source_id)
                    if has_raptor_tree(vector_store, source_id, notebook_id):
                        raptor_sources.append(source_id)
                    else:
                        standard_sources.append(source_id)

        logger.info(
            f"RAPTOR sources: {len(raptor_sources)}, Standard sources: {len(standard_sources)}"
        )

        # If all sources have RAPTOR, use pure RAPTOR retrieval
        if raptor_sources and not standard_sources:
            return RAPTORRetriever(
                vector_store=vector_store,
                notebook_id=notebook_id,
                source_ids=raptor_sources,
                similarity_top_k=self._get_similarity_top_k(),
            )

        # If no sources have RAPTOR, use standard retrieval
        if not raptor_sources:
            return self.get_retrievers(llm, language, nodes, vector_store=vector_store)

        # Mixed: use standard retrieval for now
        # TODO: Implement proper fusion of RAPTOR and standard retrievers
        logger.debug("Mixed RAPTOR/standard sources, falling back to standard retrieval")
        return self.get_retrievers(llm, language, nodes, vector_store=vector_store)

    def get_unified_retriever(
        self,
        llm: LLM,
        language: str,
        nodes: List[BaseNode],
        vector_store=None,
        notebook_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        use_reranker: bool = True,
        reranker_top_k: Optional[int] = None,
    ) -> BaseRetriever:
        """Get a retriever with explicit reranker control.

        This method creates a TwoStageRetriever that:
        1. Uses hybrid BM25 + vector retrieval
        2. Applies reranking with the specified top_k

        Use this when you need explicit control over reranker behavior
        without relying on global configuration.

        Args:
            llm: Language model for query generation
            language: Language code for prompts
            nodes: Document nodes to index
            vector_store: Optional vector store for metadata filtering
            notebook_id: Notebook ID for cache isolation
            source_ids: Optional source IDs to filter by
            use_reranker: Whether to apply reranking (default: True)
            reranker_top_k: Override for reranker top_k (default: from config)

        Returns:
            Configured retriever with explicit reranker control
        """
        # Unwrap LLM wrappers
        llm = unwrap_llm(llm)

        # Get or create index
        vector_index = self._get_or_create_index(nodes, notebook_id=notebook_id)

        # Get settings
        similarity_top_k = self._get_similarity_top_k()
        weights = self._get_retriever_weights()
        effective_rerank_top_k = reranker_top_k or self._setting.retriever.top_k_rerank

        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=False
        )

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=similarity_top_k,
            verbose=False
        )

        retrievers = [bm25_retriever, vector_retriever]

        if use_reranker:
            # TwoStageRetriever with reranking
            logger.debug(f"Creating unified retriever with reranker (top_k={effective_rerank_top_k})")
            return TwoStageRetriever(
                retrievers=retrievers,
                retriever_weights=weights,
                setting=self._setting,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=similarity_top_k,
                num_queries=1,  # Single query for speed
                mode=self._setting.retriever.fusion_mode,
                use_async=False,
                verbose=False
            )
        else:
            # Simple fusion without reranking
            logger.debug("Creating unified retriever without reranker")
            return QueryFusionRetriever(
                retrievers=retrievers,
                retriever_weights=weights,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=effective_rerank_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                use_async=False,
                verbose=False
            )
