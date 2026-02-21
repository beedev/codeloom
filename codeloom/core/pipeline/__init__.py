"""Pipeline utilities package.

Provides modular components for the LocalRAGPipeline:
- NodeCache: Thread-safe node caching with TTL
- Worker utilities: RAPTOR and Transformation worker management
- Stateless query utilities: Multi-user safe query functions

The main LocalRAGPipeline class remains in codeloom.pipeline for
backward compatibility.

Usage:
    from codeloom.core.pipeline import NodeCache, init_transformation_worker

    # Node caching
    cache = NodeCache(vector_store, ttl=300)
    nodes = cache.get(project_id)

    # Worker management
    transformation_worker = init_transformation_worker(db_manager, embed_callback)
    raptor_worker = init_raptor_worker(db_manager, vector_store)
    shutdown_workers(transformation_worker, raptor_worker)
"""

from .node_cache import NodeCache
from .workers import (
    should_skip_background_workers,
    init_transformation_worker,
    init_raptor_worker,
    shutdown_workers,
    create_transformation_callback,
)

__all__ = [
    "NodeCache",
    "should_skip_background_workers",
    "init_transformation_worker",
    "init_raptor_worker",
    "shutdown_workers",
    "create_transformation_callback",
]
