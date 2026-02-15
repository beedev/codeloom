"""
PGVectorStore - PostgreSQL + pgvector based vector store.

Replaces ChromaDB with PostgreSQL pgvector for unified database architecture.
Benefits:
- O(log n) metadata filtering via SQL indexes
- Native hybrid search (BM25 + vector)
- ACID transactions across metadata + vectors
- Incremental vector updates (no index rebuild)
- Shared connection pool with DatabaseManager (no duplicate pools)
"""

import os
import logging
from typing import List, Optional, Dict, Any, Callable, Tuple

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, MetadataFilter
from llama_index.vector_stores.postgres import PGVectorStore as LlamaPGVectorStore
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ...setting import get_settings, RAGSettings
from .base import IVectorStore

load_dotenv()

logger = logging.getLogger(__name__)


class PGVectorStore(IVectorStore):
    """
    PostgreSQL + pgvector based vector store implementing IVectorStore interface.

    Drop-in replacement for LocalVectorStore (ChromaDB) with enhanced features:
    - SQL-native metadata filtering (O(log n) vs O(n) client-side)
    - Incremental add/delete without full index rebuild
    - Unified PostgreSQL backend for all data
    - Shared connection pool with DatabaseManager (dependency injection support)
    """

    def __init__(
        self,
        host: str = "localhost",
        setting: Optional[RAGSettings] = None,
        persist: bool = True,
        database_url: Optional[str] = None,
        session_factory: Optional[Callable] = None,
    ) -> None:
        """
        Initialize PGVectorStore with optional dependency injection.

        Args:
            host: Host identifier (for legacy compatibility)
            setting: RAG settings configuration
            persist: Enable persistence (default: True)
            database_url: Optional database URL to override environment
            session_factory: Optional injected session factory from DatabaseManager.
                           If provided, this vector store will use the shared connection pool
                           instead of creating its own pool.
        """
        self._setting = setting or get_settings()
        self._host = host
        self._persist = persist

        # Determine if we own the connection pool or are using an injected one
        self._owns_pool = session_factory is None

        if session_factory:
            # Use injected session factory (shared pool)
            self._session_factory = session_factory
            logger.info("PGVectorStore using injected session factory (shared pool)")

            # We still need connection params for LlamaIndex PGVectorStore
            # Extract from database_url if provided, otherwise from environment
            if database_url:
                self._connection_string = database_url
            else:
                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    self._connection_string = database_url
                else:
                    # Fallback: build from individual env vars
                    db_host = os.getenv("POSTGRES_HOST", "localhost")
                    db_port = int(os.getenv("POSTGRES_PORT", "5433"))
                    db_name = os.getenv("POSTGRES_DB", "dbnotebook_dev")
                    db_user = os.getenv("POSTGRES_USER", "postgres")
                    db_password = os.getenv("POSTGRES_PASSWORD", "root")
                    self._connection_string = (
                        f"postgresql://{db_user}:{db_password}@"
                        f"{db_host}:{db_port}/{db_name}"
                    )

            # Parse connection string for individual params
            from urllib.parse import urlparse
            parsed = urlparse(self._connection_string)
            self._db_host = parsed.hostname or "localhost"
            self._db_port = parsed.port or 5432
            self._db_name = parsed.path.lstrip('/') if parsed.path else "dbnotebook_dev"
            self._db_user = parsed.username or "postgres"
            self._db_password = parsed.password or ""

        else:
            # Create our own connection pool (legacy behavior)
            # Use DATABASE_URL directly (same as DatabaseManager) for consistency
            # Falls back to building from individual vars for backwards compatibility
            if database_url:
                self._connection_string = database_url
            else:
                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    self._connection_string = database_url
                else:
                    # Fallback: use individual environment variables
                    self._db_host = os.getenv("POSTGRES_HOST", "localhost")
                    self._db_port = int(os.getenv("POSTGRES_PORT", "5433"))
                    self._db_name = os.getenv("POSTGRES_DB", "dbnotebook_dev")
                    self._db_user = os.getenv("POSTGRES_USER", "postgres")
                    self._db_password = os.getenv("POSTGRES_PASSWORD", "root")
                    self._connection_string = (
                        f"postgresql://{self._db_user}:{self._db_password}@"
                        f"{self._db_host}:{self._db_port}/{self._db_name}"
                    )

            # Parse connection string for individual params
            from urllib.parse import urlparse
            parsed = urlparse(self._connection_string)
            self._db_host = parsed.hostname or "localhost"
            self._db_port = parsed.port or 5432
            self._db_name = parsed.path.lstrip('/') if parsed.path else "dbnotebook_dev"
            self._db_user = parsed.username or "postgres"
            self._db_password = parsed.password or ""

            # Create own SQLAlchemy engine and session factory
            self._engine = create_engine(self._connection_string)
            self._session_factory = sessionmaker(bind=self._engine)
            logger.info("PGVectorStore created own connection pool")

        # pgvector settings
        self._table_name = os.getenv("PGVECTOR_TABLE_NAME", "embeddings")
        self._embed_dim = int(os.getenv("PGVECTOR_EMBED_DIM", "768"))

        # Initialize LlamaIndex PGVectorStore
        self._vector_store = self._create_vector_store()

        # LlamaIndex prefixes table names with "data_"
        self._actual_table_name = f"data_{self._table_name}"

        # Index cache
        self._index_cache: Optional[VectorStoreIndex] = None
        self._cached_node_count: int = 0

        logger.info(
            f"PGVectorStore initialized: {self._db_host}:{self._db_port}/{self._db_name}, "
            f"table={self._actual_table_name}, embed_dim={self._embed_dim}"
        )

        # Ensure indexes exist for fast metadata filtering
        self._ensure_metadata_indexes()

    @classmethod
    def from_session_factory(
        cls,
        session_factory: Callable,
        database_url: str,
        **kwargs
    ) -> "PGVectorStore":
        """
        Create PGVectorStore using injected session factory (dependency injection pattern).

        This factory method allows sharing a connection pool with DatabaseManager,
        preventing duplicate connection pools and resource waste.

        Args:
            session_factory: SQLAlchemy sessionmaker from DatabaseManager
            database_url: Database connection URL
            **kwargs: Additional arguments passed to PGVectorStore constructor

        Returns:
            PGVectorStore instance using the shared connection pool

        Example:
            # Share connection pool with DatabaseManager
            db_manager = DatabaseManager(database_url)
            vector_store = PGVectorStore.from_session_factory(
                session_factory=db_manager.SessionLocal,
                database_url=database_url
            )
        """
        return cls(
            session_factory=session_factory,
            database_url=database_url,
            **kwargs
        )

    def _ensure_metadata_indexes(self) -> None:
        """Create indexes on metadata JSONB for fast filtering and uniqueness."""
        try:
            session = self._session_factory()
            try:
                # Create index on notebook_id for O(log n) lookups
                session.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_notebook_id
                    ON {self._actual_table_name} ((metadata_->>'notebook_id'))
                """))
                # Create index on source_id for document lookups
                session.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_source_id
                    ON {self._actual_table_name} ((metadata_->>'source_id'))
                """))
                # Create index on node_type for transformation filtering
                # Supports values: 'chunk', 'summary', 'insight', 'question'
                session.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_node_type
                    ON {self._actual_table_name} ((metadata_->>'node_type'))
                """))
                # Create unique index on text + notebook_id to prevent duplicates
                # This prevents the same text chunk from being added multiple times
                session.execute(text(f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_{self._table_name}_unique_text_notebook
                    ON {self._actual_table_name} (md5(text), (metadata_->>'notebook_id'))
                """))
                session.commit()
                logger.debug("Metadata indexes ensured")
            finally:
                session.close()
        except Exception as e:
            # Table might not exist yet - that's OK
            logger.debug(f"Could not create metadata indexes (table may not exist yet): {e}")

    def _create_vector_store(self) -> LlamaPGVectorStore:
        """Create LlamaIndex PGVectorStore instance."""
        return LlamaPGVectorStore.from_params(
            database=self._db_name,
            host=self._db_host,
            port=str(self._db_port),
            user=self._db_user,
            password=self._db_password,
            table_name=self._table_name,
            embed_dim=self._embed_dim,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

    # =========================================================================
    # IVectorStore Interface Implementation
    # =========================================================================

    def add_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: Optional[str] = None
    ) -> int:
        """
        Add nodes to the vector store incrementally with duplicate detection.

        This is O(n) for n new nodes, NOT O(total) like ChromaDB rebuild.
        Duplicates are detected using md5 hash of text + notebook_id.

        Args:
            nodes: List of nodes to add
            notebook_id: Optional notebook ID to set on all nodes

        Returns:
            Number of nodes actually added (after deduplication)
        """
        if not nodes:
            return 0

        try:
            # Add notebook_id to metadata if provided
            if notebook_id:
                for node in nodes:
                    if hasattr(node, 'metadata'):
                        node.metadata["notebook_id"] = notebook_id

            # Filter out duplicates by checking existing text hashes
            unique_nodes = self._filter_duplicate_nodes(nodes, notebook_id)

            if not unique_nodes:
                logger.info(f"All {len(nodes)} nodes already exist, skipping add")
                return 0

            if len(unique_nodes) < len(nodes):
                logger.info(f"Filtered {len(nodes) - len(unique_nodes)} duplicate nodes")

            # Add only unique nodes to pgvector store
            self._vector_store.add(unique_nodes)

            # Invalidate cache
            self._index_cache = None
            self._cached_node_count = 0

            logger.info(f"Added {len(unique_nodes)} nodes to pgvector store")
            return len(unique_nodes)

        except Exception as e:
            # Sanitize error - don't log full SQL with embeddings
            error_msg = str(e)
            if len(error_msg) > 300:
                if "duplicate key" in error_msg.lower():
                    error_msg = "Duplicate key violation - nodes already exist"
                else:
                    error_msg = f"{type(e).__name__}: {error_msg[:150]}... [truncated]"
            logger.error(f"Error adding nodes to pgvector: {error_msg}")
            return 0

    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Query for similar nodes using vector similarity.

        Args:
            query_embedding: Query vector embedding
            similarity_top_k: Maximum number of results to return
            filters: Optional metadata filters to apply (e.g., {"notebook_id": "..."})

        Returns:
            List of (node, similarity_score) tuples, sorted by relevance
        """
        try:
            # Convert dict filters to LlamaIndex MetadataFilters if provided
            metadata_filters = None
            if filters:
                filter_list = [
                    MetadataFilter(key=k, value=v)
                    for k, v in filters.items()
                ]
                metadata_filters = MetadataFilters(filters=filter_list)

            # Create VectorStoreQuery object (required by LlamaIndex PGVectorStore)
            vs_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=similarity_top_k,
                filters=metadata_filters,
            )

            # Use LlamaIndex's built-in query method
            results = self._vector_store.query(vs_query)

            # Convert to expected format
            return [(node, score) for node, score in zip(results.nodes, results.similarities)]
        except Exception as e:
            logger.error(f"Error querying pgvector: {e}")
            return []

    def delete(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[dict] = None
    ) -> bool:
        """
        Delete nodes by ID or metadata filter.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if node_ids:
                # Delete by IDs
                self._vector_store.delete_nodes(node_ids)
            elif filters:
                # Delete by filter - need to query first then delete
                # This is a limitation of the current LlamaIndex interface
                logger.warning("Deletion by filters not directly supported, use specific delete methods")
                return False
            else:
                logger.warning("Either node_ids or filters must be provided for deletion")
                return False

            # Invalidate cache
            self._index_cache = None
            self._cached_node_count = 0

            return True
        except Exception as e:
            logger.error(f"Error deleting nodes from pgvector: {e}")
            return False

    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """
        Get specific nodes by their IDs.

        Args:
            node_ids: List of node IDs to retrieve

        Returns:
            List of nodes matching the given IDs
        """
        if not node_ids:
            return []

        try:
            session = self._session_factory()
            try:
                # Query nodes by IDs
                placeholders = ", ".join([f":id_{i}" for i in range(len(node_ids))])
                params = {f"id_{i}": node_id for i, node_id in enumerate(node_ids)}

                result = session.execute(
                    text(f"""
                        SELECT id, text, metadata_, embedding
                        FROM {self._actual_table_name}
                        WHERE id IN ({placeholders})
                    """),
                    params
                )
                rows = result.fetchall()

                nodes = []
                for row in rows:
                    node_id, text_content, metadata, embedding = row

                    if isinstance(metadata, str):
                        import json
                        metadata = json.loads(metadata)

                    # Parse embedding
                    parsed_embedding = None
                    if embedding is not None:
                        if isinstance(embedding, str):
                            import json
                            parsed_embedding = json.loads(embedding)
                        elif hasattr(embedding, 'tolist'):
                            parsed_embedding = embedding.tolist()
                        else:
                            parsed_embedding = list(embedding)

                    node = TextNode(
                        id_=str(node_id),
                        text=text_content or "",
                        metadata=metadata or {},
                        embedding=parsed_embedding
                    )
                    nodes.append(node)

                return nodes
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error retrieving nodes by ID: {e}")
            return []

    @property
    def is_connected(self) -> bool:
        """
        Check if the vector store is connected and ready to use.

        Returns:
            True if connected and operational, False otherwise
        """
        try:
            session = self._session_factory()
            try:
                # Test connection with simple query
                session.execute(text("SELECT 1"))
                return True
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Vector store connection check failed: {e}")
            return False

    def reset(self) -> None:
        """Reset the vector store (clear cache and collection)."""
        self.clear_collection()
        self._index_cache = None
        self._cached_node_count = 0

    def get_collection_stats(self) -> dict:
        """Get statistics about the embeddings table."""
        try:
            session = self._session_factory()
            try:
                result = session.execute(
                    text(f"SELECT COUNT(*) FROM {self._actual_table_name}")
                )
                count = result.scalar() or 0

                return {
                    "name": self._table_name,
                    "count": count,
                    "cached": self._index_cache is not None,
                    "cached_node_count": self._cached_node_count,
                    "database": self._db_name,
                    "host": self._db_host,
                    "port": self._db_port,
                    "pool_shared": not self._owns_pool
                }
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Legacy Methods (preserved for backward compatibility)
    # =========================================================================

    def get_index(
        self,
        nodes: List[BaseNode],
        force_rebuild: bool = False
    ) -> Optional[VectorStoreIndex]:
        """
        Get or create vector index with caching.

        Args:
            nodes: List of nodes to index
            force_rebuild: Force rebuild even if cached

        Returns:
            VectorStoreIndex or None if no nodes
        """
        if not nodes:
            return None

        # Return cached index if node count matches and not forcing rebuild
        if (
            not force_rebuild and
            self._index_cache is not None and
            len(nodes) == self._cached_node_count
        ):
            logger.debug("Using cached vector index")
            return self._index_cache

        try:
            # Create storage context with pgvector store
            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )

            # Create index with nodes
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context
            )

            # Cache the index
            self._index_cache = index
            self._cached_node_count = len(nodes)

            logger.debug(f"Created pgvector index with {len(nodes)} nodes")
            return index

        except Exception as e:
            # Sanitize error message - don't log full SQL with embeddings
            error_type = type(e).__name__
            error_msg = str(e)
            # Truncate long error messages (like SQL with embedding arrays)
            if len(error_msg) > 500:
                # Check for common patterns
                if "duplicate key" in error_msg.lower():
                    error_msg = "Duplicate key violation - some nodes already exist in database"
                elif "embedding" in error_msg.lower() or "vector" in error_msg.lower():
                    error_msg = f"{error_type}: {error_msg[:200]}... [truncated - contains embedding data]"
                else:
                    error_msg = f"{error_type}: {error_msg[:200]}... [truncated]"
            logger.error(f"Error creating pgvector index: {error_msg}")
            # Fallback to in-memory index
            index = VectorStoreIndex(nodes=nodes)
            self._index_cache = index
            self._cached_node_count = len(nodes)
            return index

    def _filter_duplicate_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: Optional[str] = None
    ) -> List[BaseNode]:
        """
        Filter out nodes that already exist in the database.

        Uses md5 hash of text to efficiently check for duplicates.

        Args:
            nodes: List of nodes to check
            notebook_id: Notebook ID to check within

        Returns:
            List of nodes that don't exist in the database
        """
        import hashlib

        if not nodes:
            return []

        try:
            # Calculate md5 hashes for all nodes
            node_hashes = {}
            for node in nodes:
                node_text = node.get_content() if hasattr(node, 'get_content') else str(node.text)
                text_hash = hashlib.md5(node_text.encode()).hexdigest()
                node_hashes[text_hash] = node

            # Query existing hashes in one batch
            session = self._session_factory()
            try:
                if notebook_id:
                    result = session.execute(text(f"""
                        SELECT md5(text) as text_hash
                        FROM {self._actual_table_name}
                        WHERE metadata_->>'notebook_id' = :notebook_id
                        AND md5(text) = ANY(:hashes)
                    """), {"notebook_id": notebook_id, "hashes": list(node_hashes.keys())})
                else:
                    result = session.execute(text(f"""
                        SELECT md5(text) as text_hash
                        FROM {self._actual_table_name}
                        WHERE md5(text) = ANY(:hashes)
                    """), {"hashes": list(node_hashes.keys())})

                existing_hashes = {row[0] for row in result}
            finally:
                session.close()

            # Return nodes that don't exist
            unique_nodes = [
                node for text_hash, node in node_hashes.items()
                if text_hash not in existing_hashes
            ]

            return unique_nodes

        except Exception as e:
            logger.warning(f"Error checking for duplicates, adding all nodes: {e}")
            return nodes

    def get_index_with_filter(
        self,
        nodes: List[BaseNode],
        offering_ids: Optional[List[str]] = None,
        practice_names: Optional[List[str]] = None,
        force_rebuild: bool = False
    ) -> Optional[VectorStoreIndex]:
        """
        Get or create filtered vector index based on offerings or practices.

        Args:
            nodes: List of all available nodes
            offering_ids: List of offering IDs to filter by (OR operation)
            practice_names: List of practice names to filter by (OR operation)
            force_rebuild: Force rebuild even if cached

        Returns:
            VectorStoreIndex with filtered nodes, or None if no matching nodes
        """
        if not nodes:
            return None

        # Filter nodes based on metadata
        filtered_nodes = nodes

        if offering_ids or practice_names:
            filtered_nodes = []

            for node in nodes:
                metadata = node.metadata or {}

                # Check notebook_id filter (for notebook-based architecture)
                if offering_ids:
                    node_notebook_id = metadata.get("notebook_id")
                    if node_notebook_id and node_notebook_id in offering_ids:
                        filtered_nodes.append(node)
                        continue

                # Check offering filter (by name or id) - legacy support
                if offering_ids:
                    node_offering_id = metadata.get("offering_id")
                    node_offering_name = metadata.get("offering_name")
                    if (node_offering_id and node_offering_id in offering_ids) or \
                       (node_offering_name and node_offering_name in offering_ids):
                        filtered_nodes.append(node)
                        continue

                # Check practice filter
                if practice_names:
                    node_practice = metadata.get("it_practice")
                    if node_practice and node_practice in practice_names:
                        filtered_nodes.append(node)
                        continue

            if not filtered_nodes:
                logger.warning(
                    f"No nodes found matching filters: "
                    f"offerings/notebooks={offering_ids}, practices={practice_names}"
                )
                return None

            logger.info(
                f"Filtered {len(filtered_nodes)} nodes from {len(nodes)} total "
                f"(offerings/notebooks={offering_ids}, practices={practice_names})"
            )

        return self.get_index(filtered_nodes, force_rebuild=force_rebuild)

    def get_nodes_by_metadata(
        self,
        nodes: List[BaseNode],
        metadata_filters: Dict[str, Any]
    ) -> List[BaseNode]:
        """
        Filter nodes by arbitrary metadata key-value pairs.

        Args:
            nodes: List of nodes to filter
            metadata_filters: Dictionary of metadata key-value pairs to match

        Returns:
            List of nodes matching all specified metadata filters
        """
        if not metadata_filters:
            return nodes

        filtered_nodes = []

        for node in nodes:
            metadata = node.metadata or {}

            # Check if all filter conditions are met (AND operation)
            matches_all = True
            for key, value in metadata_filters.items():
                if metadata.get(key) != value:
                    matches_all = False
                    break

            if matches_all:
                filtered_nodes.append(node)

        logger.debug(
            f"Filtered {len(filtered_nodes)} nodes from {len(nodes)} "
            f"using metadata filters: {metadata_filters}"
        )

        return filtered_nodes

    def clear_collection(self) -> None:
        """Clear all embeddings from the table."""
        try:
            session = self._session_factory()
            try:
                session.execute(text(f"TRUNCATE TABLE {self._actual_table_name}"))
                session.commit()
            finally:
                session.close()

            self._index_cache = None
            self._cached_node_count = 0
            logger.info(f"Cleared table: {self._table_name}")
        except Exception as e:
            logger.warning(f"Error clearing table: {e}")

    # =========================================================================
    # Notebook-Specific Methods (NotebookLM Architecture)
    # =========================================================================

    def get_nodes_by_notebook(
        self,
        nodes: List[BaseNode],
        notebook_id: str
    ) -> List[BaseNode]:
        """
        Filter nodes by notebook_id for NotebookLM-style isolation.

        Args:
            nodes: List of nodes to filter
            notebook_id: Notebook UUID to filter by

        Returns:
            List of nodes belonging to the specified notebook
        """
        return self.get_nodes_by_metadata(nodes, {"notebook_id": notebook_id})

    def get_index_by_notebook(
        self,
        nodes: List[BaseNode],
        notebook_id: str,
        force_rebuild: bool = False
    ) -> Optional[VectorStoreIndex]:
        """
        Get or create filtered vector index for a specific notebook.

        Args:
            nodes: List of all available nodes
            notebook_id: Notebook UUID to filter by
            force_rebuild: Force rebuild even if cached

        Returns:
            VectorStoreIndex with notebook-filtered nodes, or None if no matching nodes
        """
        filtered_nodes = self.get_nodes_by_notebook(nodes, notebook_id)

        if not filtered_nodes:
            logger.warning(f"No nodes found for notebook: {notebook_id}")
            return None

        logger.info(
            f"Filtered {len(filtered_nodes)} nodes for notebook {notebook_id} "
            f"from {len(nodes)} total"
        )

        return self.get_index(filtered_nodes, force_rebuild=force_rebuild)

    def delete_notebook_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: str
    ) -> List[BaseNode]:
        """
        Filter out nodes belonging to a specific notebook.

        Args:
            nodes: List of all nodes
            notebook_id: Notebook UUID to remove

        Returns:
            List of nodes NOT belonging to the specified notebook
        """
        remaining_nodes = [
            node for node in nodes
            if node.metadata.get("notebook_id") != notebook_id
        ]

        removed_count = len(nodes) - len(remaining_nodes)
        logger.info(
            f"Removed {removed_count} nodes for notebook {notebook_id}, "
            f"{len(remaining_nodes)} nodes remaining"
        )

        return remaining_nodes

    def get_notebook_document_count(
        self,
        nodes: List[BaseNode],
        notebook_id: str
    ) -> int:
        """
        Get count of unique documents in a notebook.

        Args:
            nodes: List of all nodes
            notebook_id: Notebook UUID

        Returns:
            Number of unique documents (source_ids) in the notebook
        """
        notebook_nodes = self.get_nodes_by_notebook(nodes, notebook_id)

        source_ids = set()
        for node in notebook_nodes:
            source_id = node.metadata.get("source_id")
            if source_id:
                source_ids.add(source_id)

        return len(source_ids)

    def load_all_nodes(self) -> List[BaseNode]:
        """
        Load all persisted nodes from PostgreSQL pgvector.

        Returns:
            List of all BaseNode objects stored in pgvector
        """
        try:
            session = self._session_factory()
            try:
                # Query all nodes from the embeddings table
                # The LlamaIndex PGVectorStore uses 'text' column for content
                result = session.execute(
                    text(f"""
                        SELECT id, text, metadata_, embedding
                        FROM {self._actual_table_name}
                    """)
                )
                rows = result.fetchall()

                if not rows:
                    logger.debug("No nodes found in pgvector table")
                    return []

                nodes = []
                for row in rows:
                    node_id, text_content, metadata, embedding = row

                    # Parse metadata if it's a string (JSON)
                    if isinstance(metadata, str):
                        import json
                        metadata = json.loads(metadata)

                    # Parse embedding - pgvector returns as string representation
                    parsed_embedding = None
                    if embedding is not None:
                        if isinstance(embedding, str):
                            # Parse string representation "[0.1, 0.2, ...]"
                            import json
                            parsed_embedding = json.loads(embedding)
                        elif hasattr(embedding, 'tolist'):
                            # numpy array
                            parsed_embedding = embedding.tolist()
                        else:
                            # Already a list or compatible type
                            parsed_embedding = list(embedding)

                    node = TextNode(
                        id_=str(node_id),
                        text=text_content or "",
                        metadata=metadata or {},
                        embedding=parsed_embedding
                    )
                    nodes.append(node)

                logger.info(f"Loaded {len(nodes)} nodes from pgvector")
                return nodes
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error loading nodes from pgvector: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def delete_document_nodes(self, source_id: str) -> bool:
        """
        Delete all nodes for a specific document using SQL.

        O(1) operation using SQL DELETE with WHERE clause.

        Args:
            source_id: UUID of the document whose nodes should be deleted

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            logger.info(f"Deleting nodes for document {source_id} from pgvector")

            session = self._session_factory()
            try:
                # Use SQL to delete nodes by source_id in metadata
                # LlamaIndex stores metadata in 'metadata_' JSONB column
                result = session.execute(
                    text(f"""
                        DELETE FROM {self._actual_table_name}
                        WHERE metadata_->>'source_id' = :source_id
                    """),
                    {"source_id": source_id}
                )
                session.commit()
                deleted_count = result.rowcount
            finally:
                session.close()

            # Invalidate cache
            self._index_cache = None
            self._cached_node_count = 0

            logger.info(
                f"Deleted {deleted_count} nodes for source_id={source_id} "
                f"using SQL DELETE"
            )
            return True

        except Exception as e:
            logger.error(f"Error deleting document nodes from pgvector: {e}")
            return False

    def delete_notebook_nodes_sql(self, notebook_id: str) -> bool:
        """
        Delete all nodes for a specific notebook using SQL.

        O(1) operation using SQL DELETE with WHERE clause.

        Args:
            notebook_id: UUID of the notebook whose nodes should be deleted

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            logger.info(f"Deleting nodes for notebook {notebook_id} from pgvector")

            session = self._session_factory()
            try:
                result = session.execute(
                    text(f"""
                        DELETE FROM {self._actual_table_name}
                        WHERE metadata_->>'notebook_id' = :notebook_id
                    """),
                    {"notebook_id": notebook_id}
                )
                session.commit()
                deleted_count = result.rowcount
            finally:
                session.close()

            # Invalidate cache
            self._index_cache = None
            self._cached_node_count = 0

            logger.info(
                f"Deleted {deleted_count} nodes for notebook_id={notebook_id} "
                f"using SQL DELETE"
            )
            return True

        except Exception as e:
            logger.error(f"Error deleting notebook nodes from pgvector: {e}")
            return False

    def get_nodes_by_notebook_sql(self, notebook_id: str) -> List[BaseNode]:
        """
        Load nodes for a specific notebook using SQL filtering.

        O(log n) operation using SQL WHERE with indexed column.
        Only returns nodes from active sources (respects document toggle).

        Args:
            notebook_id: Notebook UUID to filter by

        Returns:
            List of nodes belonging to the specified notebook (active sources only)
        """
        try:
            session = self._session_factory()
            try:
                # Join with notebook_sources to filter by active status
                # This respects the eye icon toggle in the UI
                result = session.execute(
                    text(f"""
                        SELECT e.id, e.text, e.metadata_, e.embedding
                        FROM {self._actual_table_name} e
                        LEFT JOIN notebook_sources ns
                            ON e.metadata_->>'source_id' = ns.source_id::text
                        WHERE e.metadata_->>'notebook_id' = :notebook_id
                        AND (ns.active = true OR ns.active IS NULL)
                    """),
                    {"notebook_id": notebook_id}
                )
                rows = result.fetchall()

                nodes = []
                for row in rows:
                    node_id, text_content, metadata, embedding = row

                    if isinstance(metadata, str):
                        import json
                        metadata = json.loads(metadata)

                    # Parse embedding - pgvector returns as string representation
                    parsed_embedding = None
                    if embedding is not None:
                        if isinstance(embedding, str):
                            import json
                            parsed_embedding = json.loads(embedding)
                        elif hasattr(embedding, 'tolist'):
                            parsed_embedding = embedding.tolist()
                        else:
                            parsed_embedding = list(embedding)

                    node = TextNode(
                        id_=str(node_id),
                        text=text_content or "",
                        metadata=metadata or {},
                        embedding=parsed_embedding
                    )
                    nodes.append(node)

                logger.debug(f"Loaded {len(nodes)} nodes for notebook {notebook_id}")
                return nodes
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error loading notebook nodes from pgvector: {e}")
            return []

    def get_nodes_by_notebook_and_types(
        self,
        notebook_id: str,
        node_types: Optional[List[str]] = None
    ) -> List[BaseNode]:
        """
        Load nodes for a specific notebook, optionally filtered by node type.

        O(log n) operation using SQL WHERE with indexed columns.
        Only returns nodes from active sources (respects document toggle).
        For transformation nodes, checks parent_source_id for active status.

        Args:
            notebook_id: Notebook UUID to filter by
            node_types: Optional list of node types to include
                       Valid types: 'chunk', 'summary', 'insight', 'question'
                       If None, returns all node types

        Returns:
            List of nodes matching the criteria (active sources only)
        """
        try:
            session = self._session_factory()
            try:
                if node_types:
                    # Filter by both notebook_id and node_type
                    # Join with notebook_sources to filter by active status
                    # Use COALESCE to check parent_source_id for transformation nodes
                    placeholders = ", ".join([f":type_{i}" for i in range(len(node_types))])
                    params = {"notebook_id": notebook_id}
                    for i, t in enumerate(node_types):
                        params[f"type_{i}"] = t

                    query = text(f"""
                        SELECT e.id, e.text, e.metadata_, e.embedding
                        FROM {self._actual_table_name} e
                        LEFT JOIN notebook_sources ns
                            ON COALESCE(
                                e.metadata_->>'parent_source_id',
                                e.metadata_->>'source_id'
                            ) = ns.source_id::text
                        WHERE e.metadata_->>'notebook_id' = :notebook_id
                        AND (
                            e.metadata_->>'node_type' IN ({placeholders})
                            OR e.metadata_->>'node_type' IS NULL
                        )
                        AND (ns.active = true OR ns.active IS NULL)
                    """)
                else:
                    # No type filter - get all nodes
                    # Join with notebook_sources to filter by active status
                    params = {"notebook_id": notebook_id}
                    query = text(f"""
                        SELECT e.id, e.text, e.metadata_, e.embedding
                        FROM {self._actual_table_name} e
                        LEFT JOIN notebook_sources ns
                            ON COALESCE(
                                e.metadata_->>'parent_source_id',
                                e.metadata_->>'source_id'
                            ) = ns.source_id::text
                        WHERE e.metadata_->>'notebook_id' = :notebook_id
                        AND (ns.active = true OR ns.active IS NULL)
                    """)

                result = session.execute(query, params)
                rows = result.fetchall()

                nodes = []
                for row in rows:
                    node_id, text_content, metadata, embedding = row

                    if isinstance(metadata, str):
                        import json
                        metadata = json.loads(metadata)

                    # Parse embedding - pgvector returns as string representation
                    parsed_embedding = None
                    if embedding is not None:
                        if isinstance(embedding, str):
                            import json
                            parsed_embedding = json.loads(embedding)
                        elif hasattr(embedding, 'tolist'):
                            parsed_embedding = embedding.tolist()
                        else:
                            parsed_embedding = list(embedding)

                    node = TextNode(
                        id_=str(node_id),
                        text=text_content or "",
                        metadata=metadata or {},
                        embedding=parsed_embedding
                    )
                    nodes.append(node)

                type_str = str(node_types) if node_types else "all"
                logger.debug(f"Loaded {len(nodes)} nodes for notebook {notebook_id} (types: {type_str})")
                return nodes
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error loading notebook nodes by type from pgvector: {e}")
            return []

    def add_transformation_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: str,
        source_id: str
    ) -> int:
        """
        Add transformation nodes (summary, insights, questions) to the vector store.

        Handles proper node_type metadata assignment.

        Args:
            nodes: List of nodes with node_type in metadata
            notebook_id: Notebook UUID for the nodes
            source_id: Source document UUID

        Returns:
            Number of nodes successfully added
        """
        if not nodes:
            return 0

        # Ensure metadata is set correctly
        for node in nodes:
            if hasattr(node, 'metadata'):
                node.metadata["notebook_id"] = notebook_id
                node.metadata["source_id"] = source_id
                # node_type should already be set by caller

        return self.add_nodes(nodes, notebook_id=notebook_id)

    # =========================================================================
    # RAPTOR Tree Methods (Hierarchical Retrieval Support)
    # =========================================================================

    def get_nodes_by_tree_level(
        self,
        notebook_id: str,
        tree_level: int,
        source_ids: Optional[List[str]] = None
    ) -> List[BaseNode]:
        """
        Get nodes at a specific tree level for RAPTOR retrieval.

        Tree levels:
        - 0: Original document chunks (leaf nodes)
        - 1+: Summary nodes at increasing abstraction levels

        Args:
            notebook_id: Notebook UUID to filter by
            tree_level: Tree level to retrieve (0=chunks, 1+=summaries)
            source_ids: Optional list of source IDs to filter by

        Returns:
            List of nodes at the specified tree level
        """
        try:
            session = self._session_factory()
            try:
                # For level 0, also match NULL tree_level (backward compatibility with existing chunks)
                level_condition = (
                    "(e.metadata_->>'tree_level' = :tree_level OR e.metadata_->>'tree_level' IS NULL)"
                    if tree_level == 0
                    else "e.metadata_->>'tree_level' = :tree_level"
                )

                if source_ids:
                    # Filter by notebook, level, and specific sources
                    placeholders = ", ".join([f":src_{i}" for i in range(len(source_ids))])
                    params = {"notebook_id": notebook_id, "tree_level": str(tree_level)}
                    for i, src in enumerate(source_ids):
                        params[f"src_{i}"] = src

                    result = session.execute(
                        text(f"""
                            SELECT e.id, e.text, e.metadata_, e.embedding
                            FROM {self._actual_table_name} e
                            WHERE e.metadata_->>'notebook_id' = :notebook_id
                            AND {level_condition}
                            AND e.metadata_->>'source_id' IN ({placeholders})
                        """),
                        params
                    )
                else:
                    # Filter by notebook and level only
                    result = session.execute(
                        text(f"""
                            SELECT e.id, e.text, e.metadata_, e.embedding
                            FROM {self._actual_table_name} e
                            WHERE e.metadata_->>'notebook_id' = :notebook_id
                            AND {level_condition}
                        """),
                        {"notebook_id": notebook_id, "tree_level": str(tree_level)}
                    )

                rows = result.fetchall()
                nodes = self._rows_to_nodes(rows)

                logger.debug(
                    f"Retrieved {len(nodes)} nodes at tree_level={tree_level} "
                    f"for notebook {notebook_id}"
                )
                return nodes

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting nodes by tree level: {e}")
            return []

    def get_nodes_by_tree_levels(
        self,
        notebook_id: str,
        tree_levels: List[int],
        source_ids: Optional[List[str]] = None
    ) -> List[BaseNode]:
        """
        Get nodes at multiple tree levels for RAPTOR retrieval.

        Args:
            notebook_id: Notebook UUID to filter by
            tree_levels: List of tree levels to retrieve
            source_ids: Optional list of source IDs to filter by

        Returns:
            List of nodes at the specified tree levels
        """
        try:
            session = self._session_factory()
            try:
                # Build level filter - include NULL for level 0 (backward compatibility)
                level_placeholders = ", ".join([f":lvl_{i}" for i in range(len(tree_levels))])
                params = {"notebook_id": notebook_id}
                for i, lvl in enumerate(tree_levels):
                    params[f"lvl_{i}"] = str(lvl)

                # Add OR condition for NULL tree_level if level 0 is requested
                include_null = 0 in tree_levels
                level_condition = (
                    f"(e.metadata_->>'tree_level' IN ({level_placeholders}) OR e.metadata_->>'tree_level' IS NULL)"
                    if include_null
                    else f"e.metadata_->>'tree_level' IN ({level_placeholders})"
                )

                if source_ids:
                    # Filter by notebook, levels, and specific sources
                    src_placeholders = ", ".join([f":src_{i}" for i in range(len(source_ids))])
                    for i, src in enumerate(source_ids):
                        params[f"src_{i}"] = src

                    result = session.execute(
                        text(f"""
                            SELECT e.id, e.text, e.metadata_, e.embedding
                            FROM {self._actual_table_name} e
                            WHERE e.metadata_->>'notebook_id' = :notebook_id
                            AND {level_condition}
                            AND e.metadata_->>'source_id' IN ({src_placeholders})
                        """),
                        params
                    )
                else:
                    # Filter by notebook and levels only
                    result = session.execute(
                        text(f"""
                            SELECT e.id, e.text, e.metadata_, e.embedding
                            FROM {self._actual_table_name} e
                            WHERE e.metadata_->>'notebook_id' = :notebook_id
                            AND {level_condition}
                        """),
                        params
                    )

                rows = result.fetchall()
                nodes = self._rows_to_nodes(rows)

                logger.debug(
                    f"Retrieved {len(nodes)} nodes at levels {tree_levels} "
                    f"for notebook {notebook_id}"
                )
                return nodes

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting nodes by tree levels: {e}")
            return []

    def add_tree_nodes(
        self,
        nodes: List[BaseNode],
        notebook_id: str,
        source_id: str,
        tree_level: int,
        tree_root_id: Optional[str] = None
    ) -> int:
        """
        Add RAPTOR tree nodes with proper metadata.

        Args:
            nodes: List of nodes to add
            notebook_id: Notebook UUID
            source_id: Source document UUID
            tree_level: Tree level (0=chunk, 1+=summary)
            tree_root_id: Optional root node ID for tree traversal

        Returns:
            Number of nodes successfully added
        """
        if not nodes:
            return 0

        # Set tree metadata on all nodes
        for node in nodes:
            if hasattr(node, 'metadata'):
                node.metadata["notebook_id"] = notebook_id
                node.metadata["source_id"] = source_id
                node.metadata["tree_level"] = tree_level
                node.metadata["node_type"] = "raptor_summary" if tree_level > 0 else "chunk"
                if tree_root_id:
                    node.metadata["tree_root_id"] = tree_root_id

        return self.add_nodes(nodes, notebook_id=notebook_id)

    def delete_tree_nodes(
        self,
        source_id: str,
        min_level: int = 1
    ) -> int:
        """
        Delete RAPTOR tree nodes for a source, preserving original chunks.

        Args:
            source_id: Source document UUID
            min_level: Minimum tree level to delete (default 1 = keep chunks)

        Returns:
            Number of nodes deleted
        """
        try:
            session = self._session_factory()
            try:
                result = session.execute(
                    text(f"""
                        DELETE FROM {self._actual_table_name}
                        WHERE metadata_->>'source_id' = :source_id
                        AND (metadata_->>'tree_level')::int >= :min_level
                    """),
                    {"source_id": source_id, "min_level": min_level}
                )
                session.commit()
                deleted_count = result.rowcount

                # Invalidate cache
                self._index_cache = None
                self._cached_node_count = 0

                logger.info(
                    f"Deleted {deleted_count} tree nodes (level>={min_level}) "
                    f"for source {source_id}"
                )
                return deleted_count

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error deleting tree nodes: {e}")
            return 0

    def get_tree_stats(
        self,
        source_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about the RAPTOR tree for a source.

        Args:
            source_id: Source document UUID

        Returns:
            Dictionary with tree statistics:
            - total_nodes: Total nodes in tree
            - levels: Dict mapping level to node count
            - max_level: Maximum tree level
            - has_tree: Whether RAPTOR tree exists
        """
        try:
            session = self._session_factory()
            try:
                result = session.execute(
                    text(f"""
                        SELECT
                            COALESCE(metadata_->>'tree_level', '0') as level,
                            COUNT(*) as node_count
                        FROM {self._actual_table_name}
                        WHERE metadata_->>'source_id' = :source_id
                        GROUP BY COALESCE(metadata_->>'tree_level', '0')
                        ORDER BY level
                    """),
                    {"source_id": source_id}
                )
                rows = result.fetchall()

                if not rows:
                    return {
                        "total_nodes": 0,
                        "levels": {},
                        "max_level": 0,
                        "has_tree": False
                    }

                levels = {}
                total = 0
                max_level = 0

                for row in rows:
                    level = int(row[0])
                    count = row[1]
                    levels[level] = count
                    total += count
                    max_level = max(max_level, level)

                return {
                    "total_nodes": total,
                    "levels": levels,
                    "max_level": max_level,
                    "has_tree": max_level > 0
                }

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting tree stats: {e}")
            return {
                "total_nodes": 0,
                "levels": {},
                "max_level": 0,
                "has_tree": False,
                "error": str(e)
            }

    def _rows_to_nodes(self, rows: List[Tuple]) -> List[BaseNode]:
        """
        Convert database rows to BaseNode objects.

        Args:
            rows: List of (id, text, metadata, embedding) tuples

        Returns:
            List of TextNode objects
        """
        import json

        nodes = []
        for row in rows:
            node_id, text_content, metadata, embedding = row

            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # Parse embedding
            parsed_embedding = None
            if embedding is not None:
                if isinstance(embedding, str):
                    parsed_embedding = json.loads(embedding)
                elif hasattr(embedding, 'tolist'):
                    parsed_embedding = embedding.tolist()
                else:
                    parsed_embedding = list(embedding)

            node = TextNode(
                id_=str(node_id),
                text=text_content or "",
                metadata=metadata or {},
                embedding=parsed_embedding
            )
            nodes.append(node)

        return nodes

    def get_top_raptor_summaries(
        self,
        notebook_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Single ANN lookup for RAPTOR cluster summaries (tree_level >= 1).

        O(log n) using pgvector HNSW index with metadata filtering.
        Bounded by top_k (default 5) to prevent context bloat.

        Args:
            notebook_id: Notebook UUID to filter by
            query_embedding: Query vector for similarity search
            top_k: Maximum summaries to return (default: 5)

        Returns:
            List of (node, similarity_score) tuples, sorted by relevance
        """
        try:
            session = self._session_factory()
            try:
                # Format embedding as pgvector-compatible array string
                # pgvector expects format: '[0.1, 0.2, ...]'
                embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

                # Use pgvector's <=> cosine distance operator
                # Filter for tree_level >= 1 (summaries only)
                result = session.execute(
                    text(f"""
                        SELECT id, text, metadata_, embedding,
                               1 - (embedding <=> '{embedding_str}'::vector) as similarity
                        FROM {self._actual_table_name}
                        WHERE metadata_->>'notebook_id' = :notebook_id
                        AND (metadata_->>'tree_level')::int >= 1
                        ORDER BY embedding <=> '{embedding_str}'::vector
                        LIMIT :top_k
                    """),
                    {
                        "notebook_id": notebook_id,
                        "top_k": top_k
                    }
                )
                rows = result.fetchall()

                if not rows:
                    return []

                # Convert to (node, score) tuples
                results = []
                for row in rows:
                    node_id, text_content, metadata, embedding, similarity = row

                    if isinstance(metadata, str):
                        import json
                        metadata = json.loads(metadata)

                    node = TextNode(
                        id_=str(node_id),
                        text=text_content or "",
                        metadata=metadata or {},
                    )
                    results.append((node, float(similarity)))

                logger.debug(f"Retrieved {len(results)} RAPTOR summaries for notebook {notebook_id}")
                return results

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"RAPTOR summary retrieval failed: {e}")
            return []  # Graceful fallback - continue without summaries

    def __del__(self):
        """Clean up resources on deletion."""
        if self._owns_pool and hasattr(self, '_engine'):
            try:
                self._engine.dispose()
                logger.debug("PGVectorStore disposed of owned connection pool")
            except Exception as e:
                logger.warning(f"Error disposing engine: {e}")
