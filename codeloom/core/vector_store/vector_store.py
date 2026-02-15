import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import chromadb

from ...setting import get_settings, RAGSettings

load_dotenv()

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """Persistent vector store using ChromaDB.

    NOTE: This is legacy code - PGVectorStore is the primary vector store.
    Kept for backwards compatibility only.
    """

    # Legacy defaults (previously in config/ingestion.yaml storage section)
    DEFAULT_PERSIST_DIR = "data/chroma"
    DEFAULT_COLLECTION_NAME = "collection"

    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
        persist: bool = True,
        persist_dir: str | None = None,
        collection_name: str | None = None
    ) -> None:
        self._setting = setting or get_settings()
        self._host = host
        self._persist = persist

        # Use provided values or legacy defaults
        self._persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR
        self._collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        # Initialize ChromaDB
        if persist:
            persist_path = Path(self._persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_path)
            )
        else:
            self._chroma_client = chromadb.EphemeralClient()

        # Per-notebook cache for vector indices (MVP 3 optimization)
        # Cache key format: f"index_{notebook_id}" or "index_all" for global
        self._index_cache: Dict[str, VectorStoreIndex] = {}
        self._cached_node_counts: Dict[str, int] = {}

        logger.debug("LocalVectorStore initialized with ChromaDB")

    def _get_or_create_collection(self):
        """Get or create ChromaDB collection."""
        return self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

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

        # Create new index
        try:
            if self._persist:
                persist_path = Path(self._persist_dir)

                # Use from_params classmethod which properly initializes PrivateAttr
                vector_store = ChromaVectorStore.from_params(
                    collection_name=self._collection_name,
                    persist_dir=str(persist_path),
                    collection_kwargs={"metadata": {"hnsw:space": "cosine"}}
                )

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                # Create index with nodes - this will automatically add nodes to vector store
                index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context
                )
                logger.debug(f"Created index and persisted {len(nodes)} nodes to ChromaDB")
            else:
                # In-memory index
                index = VectorStoreIndex(nodes=nodes)

            # Cache the index
            self._index_cache = index
            self._cached_node_count = len(nodes)

            logger.debug(f"Created vector index with {len(nodes)} nodes")
            return index

        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to in-memory index
            index = VectorStoreIndex(nodes=nodes)
            self._index_cache = index
            self._cached_node_count = len(nodes)
            return index

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

        # Create index with filtered nodes
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
        """Clear the current collection."""
        try:
            self._chroma_client.delete_collection(self._collection_name)
            self._index_cache = None
            self._cached_node_count = 0
            logger.info(f"Cleared collection: {self._collection_name}")
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")

    def reset(self) -> None:
        """Reset the vector store (clear cache and collection)."""
        self.clear_collection()
        self._index_cache = None
        self._cached_node_count = 0

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        try:
            collection = self._get_or_create_collection()
            return {
                "name": self._collection_name,
                "count": collection.count(),
                "cached": self._index_cache is not None,
                "cached_node_count": self._cached_node_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

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

        Use this to remove notebook nodes from the in-memory node list
        after deleting a notebook from the database.

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

        # Count unique source_ids
        source_ids = set()
        for node in notebook_nodes:
            source_id = node.metadata.get("source_id")
            if source_id:
                source_ids.add(source_id)

        return len(source_ids)

    def load_all_nodes(self) -> List[BaseNode]:
        """
        Load all persisted nodes from ChromaDB.

        This method retrieves all nodes that have been persisted to ChromaDB,
        allowing notebooks to persist across sessions.

        Returns:
            List of all BaseNode objects stored in ChromaDB
        """
        if not self._persist:
            logger.warning("Vector store is not in persist mode, no nodes to load")
            return []

        try:
            collection = self._get_or_create_collection()

            # Get all items from the collection
            results = collection.get(include=["documents", "metadatas", "embeddings"])

            if not results or len(results.get("ids", [])) == 0:
                logger.debug("No nodes found in ChromaDB collection")
                return []

            # Convert ChromaDB results back to BaseNode objects
            from llama_index.core.schema import TextNode

            nodes = []
            for i, node_id in enumerate(results["ids"]):
                # Create TextNode with stored data
                node = TextNode(
                    id_=node_id,
                    text=results["documents"][i] if results.get("documents") is not None and len(results["documents"]) > i else "",
                    metadata=results["metadatas"][i] if results.get("metadatas") is not None and len(results["metadatas"]) > i else {},
                    embedding=results["embeddings"][i] if results.get("embeddings") is not None and len(results["embeddings"]) > i else None
                )
                nodes.append(node)

            logger.info(f"Loaded {len(nodes)} nodes from ChromaDB")
            return nodes

        except Exception as e:
            logger.error(f"Error loading nodes from ChromaDB: {e}")
            return []

    def delete_document_nodes(self, source_id: str) -> bool:
        """
        Delete all nodes for a specific document from ChromaDB using native filtering.

        MVP 3 Optimization: Uses ChromaDB's native delete() with where clause
        instead of loading all nodes and rebuilding the entire index.
        This is O(1) instead of O(n), completing in <100ms regardless of collection size.

        Args:
            source_id: UUID of the document whose nodes should be deleted

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            logger.info(f"Deleting nodes for document {source_id} from ChromaDB")

            # Get collection
            collection = self._get_or_create_collection()

            # Efficient deletion using ChromaDB native filtering (MVP 3)
            collection.delete(where={"source_id": source_id})

            # Invalidate all caches since we don't know which notebook this document belonged to
            self._index_cache = {}
            self._cached_node_counts = {}

            logger.info(f"Successfully deleted nodes for source_id={source_id} using efficient native filtering")
            return True

        except Exception as e:
            logger.error(f"Error deleting document nodes from ChromaDB: {e}")
            return False
