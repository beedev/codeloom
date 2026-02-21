"""Thread-safe node caching with TTL for multi-user RAG pipelines.

Provides efficient caching of document nodes per project to avoid
repeated database queries. Cache entries expire after TTL.

Usage:
    cache = NodeCache(vector_store, ttl=300)
    nodes = cache.get(project_id)  # Thread-safe
    cache.invalidate(project_id)   # Invalidate specific project
    cache.clear()                   # Clear all
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class NodeCache:
    """Thread-safe node cache with TTL for multi-user access.

    Caches document nodes per project to avoid repeated DB queries.
    Cache is invalidated when:
    - TTL expires (default: 5 minutes)
    - Explicitly invalidated (e.g., after document upload)

    Attributes:
        vector_store: PGVectorStore instance for loading nodes
        ttl: Time-to-live in seconds (default: 300)
    """

    def __init__(
        self,
        vector_store,
        ttl: int = 300,
    ):
        """Initialize the node cache.

        Args:
            vector_store: PGVectorStore instance with get_nodes_by_project_sql method
            ttl: Cache TTL in seconds (default: 300 = 5 minutes)
        """
        self._vector_store = vector_store
        self._ttl = ttl
        self._cache: Dict[str, Tuple[List[TextNode], float, int]] = {}
        self._lock = threading.Lock()

    def get(self, project_id: str) -> List[TextNode]:
        """Get nodes for a project with caching.

        Thread-safe for multi-user concurrent access.

        Args:
            project_id: UUID of the project

        Returns:
            List of TextNode objects for the project
        """
        with self._lock:
            current_time = time.time()

            # Check cache
            if project_id in self._cache:
                nodes, timestamp, cached_count = self._cache[project_id]

                # Check TTL
                if current_time - timestamp < self._ttl:
                    logger.debug(f"Cache hit for project {project_id}: {len(nodes)} nodes")
                    return nodes
                else:
                    logger.debug(f"Cache expired for project {project_id}")

            # Cache miss - load from DB
            start_time = time.time()
            nodes = self._vector_store.get_nodes_by_project_sql(project_id)
            load_time_ms = int((time.time() - start_time) * 1000)

            # Store in cache
            self._cache[project_id] = (nodes, current_time, len(nodes))
            logger.info(
                f"Cached {len(nodes)} nodes for project {project_id} "
                f"(loaded in {load_time_ms}ms)"
            )

            return nodes

    def invalidate(self, project_id: Optional[str] = None) -> None:
        """Invalidate cache for a project or all projects.

        Call this after document upload/delete to ensure fresh nodes.
        Thread-safe for multi-user concurrent access.

        Args:
            project_id: Specific project to invalidate, or None for all
        """
        with self._lock:
            if project_id:
                if project_id in self._cache:
                    del self._cache[project_id]
                    logger.debug(f"Invalidated node cache for project {project_id}")
            else:
                self._cache.clear()
                logger.debug("Invalidated all node caches")

    def clear(self) -> None:
        """Clear all cached nodes."""
        self.invalidate()

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats (project_count, total_nodes, oldest_entry)
        """
        with self._lock:
            if not self._cache:
                return {
                    "project_count": 0,
                    "total_nodes": 0,
                    "oldest_entry_age_sec": 0,
                }

            current_time = time.time()
            oldest_age = 0
            total_nodes = 0

            for project_id, (nodes, timestamp, count) in self._cache.items():
                age = current_time - timestamp
                if age > oldest_age:
                    oldest_age = age
                total_nodes += len(nodes)

            return {
                "project_count": len(self._cache),
                "total_nodes": total_nodes,
                "oldest_entry_age_sec": int(oldest_age),
                "ttl_sec": self._ttl,
            }

    @property
    def ttl(self) -> int:
        """Get the cache TTL in seconds."""
        return self._ttl

    @ttl.setter
    def ttl(self, value: int) -> None:
        """Set the cache TTL in seconds."""
        self._ttl = value
