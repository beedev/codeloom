"""Background worker management for RAG pipeline.

Provides utilities for initializing and managing background workers:
- TransformationWorker: AI transformations (summaries, insights, questions)
- RAPTORWorker: Hierarchical tree building

Usage:
    # Initialize workers
    transformation_worker = init_transformation_worker(
        db_manager=db_manager,
        embed_callback=embed_fn,
    )

    raptor_worker = init_raptor_worker(
        db_manager=db_manager,
        vector_store=vector_store,
    )

    # Shutdown gracefully
    shutdown_workers(transformation_worker, raptor_worker)
"""

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def should_skip_background_workers() -> bool:
    """Check if background workers should be disabled.

    Workers are skipped when:
    - DISABLE_BACKGROUND_WORKERS=true (explicit disable)
    - Multi-worker mode (Gunicorn) - asyncio doesn't fork well

    Returns:
        True if workers should be skipped
    """
    env_val = os.getenv("DISABLE_BACKGROUND_WORKERS", "").lower()
    return env_val in ("true", "1", "yes")


def init_transformation_worker(
    db_manager,
    embed_callback: Callable[[str, str, str, str], None],
    poll_interval: float = 10.0,
    max_concurrent: int = 2,
):
    """Initialize TransformationWorker for AI transformations.

    Args:
        db_manager: DatabaseManager instance
        embed_callback: Callback to embed transformation content
            Signature: (text, node_type, source_id, project_id) -> None
        poll_interval: Seconds between polling for jobs (default: 10)
        max_concurrent: Maximum concurrent transformations (default: 2)

    Returns:
        TransformationWorker instance, or None if skipped
    """
    if not db_manager:
        logger.info("TransformationWorker skipped (no database)")
        return None

    if should_skip_background_workers():
        logger.info("TransformationWorker disabled (DISABLE_BACKGROUND_WORKERS=true)")
        return None

    try:
        from codeloom.core.transformations import TransformationWorker

        worker = TransformationWorker(
            db_manager=db_manager,
            embed_callback=embed_callback,
            poll_interval=poll_interval,
            max_concurrent=max_concurrent,
        )
        worker.start()
        logger.info("TransformationWorker started for AI transformations")
        return worker

    except Exception as e:
        logger.error(f"Failed to initialize TransformationWorker: {e}")
        return None


def init_raptor_worker(
    db_manager,
    vector_store,
    poll_interval: float = 15.0,
    max_concurrent: int = 1,
):
    """Initialize RAPTORWorker for hierarchical tree building.

    Args:
        db_manager: DatabaseManager instance
        vector_store: PGVectorStore instance
        poll_interval: Seconds between polling for jobs (default: 15)
        max_concurrent: Maximum concurrent builds (default: 1)
            Note: Tree building is resource-intensive

    Returns:
        RAPTORWorker instance, or None if skipped
    """
    if not db_manager:
        logger.info("RAPTORWorker skipped (no database)")
        return None

    if should_skip_background_workers():
        logger.info("RAPTORWorker disabled (DISABLE_BACKGROUND_WORKERS=true)")
        return None

    try:
        from codeloom.core.raptor import RAPTORWorker

        worker = RAPTORWorker(
            db_manager=db_manager,
            vector_store=vector_store,
            poll_interval=poll_interval,
            max_concurrent=max_concurrent,
        )
        worker.start()
        logger.info("RAPTORWorker started for hierarchical tree building")
        return worker

    except Exception as e:
        logger.error(f"Failed to initialize RAPTORWorker: {e}")
        return None


def shutdown_workers(
    transformation_worker=None,
    raptor_worker=None,
) -> None:
    """Gracefully shutdown background workers.

    Args:
        transformation_worker: TransformationWorker instance (optional)
        raptor_worker: RAPTORWorker instance (optional)
    """
    logger.info("Shutting down background workers...")

    if transformation_worker:
        try:
            transformation_worker.stop()
            logger.info("TransformationWorker stopped")
        except Exception as e:
            logger.error(f"Error stopping TransformationWorker: {e}")

    if raptor_worker:
        try:
            raptor_worker.stop()
            logger.info("RAPTORWorker stopped")
        except Exception as e:
            logger.error(f"Error stopping RAPTORWorker: {e}")

    logger.info("Worker shutdown complete")


def create_transformation_callback(transformation_worker):
    """Create a callback function for queuing transformation jobs.

    This callback is passed to LocalDataIngestion to queue transformation
    jobs when documents are ingested.

    Args:
        transformation_worker: TransformationWorker instance

    Returns:
        Callback function, or None if worker not available
    """
    if not transformation_worker:
        return None

    from codeloom.core.transformations import TransformationJob

    def callback(
        source_id: str,
        document_text: str,
        project_id: str,
        file_name: str
    ):
        """Queue a transformation job when a document is ingested."""
        job = TransformationJob(
            source_id=source_id,
            document_text=document_text,
            project_id=project_id,
            file_name=file_name,
        )
        transformation_worker.queue_job(job)
        logger.debug(f"Queued transformation job for source: {source_id}")

    return callback
