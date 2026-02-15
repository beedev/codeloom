"""Background worker for RAPTOR tree building.

Processes RAPTOR jobs asynchronously:
1. Polls for sources with pending RAPTOR status
2. Retrieves chunks from vector store
3. Builds hierarchical tree using clustering and summarization
4. Stores summary nodes back to vector store
5. Updates source records with RAPTOR status
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Callable, TYPE_CHECKING
from uuid import UUID

from llama_index.core import Settings
from llama_index.core.schema import BaseNode
from sqlalchemy.orm import Session

from ..db import DatabaseManager
from ..db.models import NotebookSource
from .config import RAPTORConfig, DEFAULT_CONFIG
from .tree_builder import RAPTORTreeBuilder, TreeBuildResult

if TYPE_CHECKING:
    from ..vector_store import PGVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAPTORJob:
    """A job to build RAPTOR tree for a source."""
    source_id: str
    notebook_id: str
    file_name: str


class RAPTORWorker:
    """Background worker for building RAPTOR trees.

    Runs in a background thread, processing RAPTOR jobs from a queue.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: "PGVectorStore",
        config: Optional[RAPTORConfig] = None,
        poll_interval: float = 10.0,
        max_concurrent: int = 1,  # Tree building is resource-intensive
    ):
        """Initialize the RAPTOR worker.

        Args:
            db_manager: Database manager for accessing sources
            vector_store: Vector store for retrieving chunks and storing summaries
            config: RAPTOR configuration
            poll_interval: Seconds between polling for new jobs
            max_concurrent: Maximum concurrent tree builds
        """
        self.db = db_manager
        self.vector_store = vector_store
        self.config = config or DEFAULT_CONFIG
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent

        self._queue: asyncio.Queue = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # LLM and embed model from Settings (set by pipeline)
        self._llm = None
        self._embed_model = None
        self._tree_builder: Optional[RAPTORTreeBuilder] = None

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("RAPTOR worker already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("RAPTOR worker started")

    def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("RAPTOR worker stopped")

    def queue_job(self, job: RAPTORJob):
        """Queue a RAPTOR job for processing."""
        if self._queue and self._loop:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, job
            )
            logger.info(f"Queued RAPTOR job for source: {job.source_id}")

    def _run_loop(self):
        """Run the async event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"RAPTOR worker loop error: {e}")
        finally:
            self._loop.close()

    async def _main_loop(self):
        """Main processing loop."""
        # Start polling task
        poll_task = asyncio.create_task(self._poll_pending())

        while self._running:
            try:
                # Wait for job from queue with timeout
                job = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.poll_interval
                )
                # Process job with semaphore for concurrency control
                asyncio.create_task(self._process_with_semaphore(job))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in RAPTOR main loop: {e}")

        poll_task.cancel()

    async def _poll_pending(self):
        """Poll database for pending RAPTOR builds."""
        while self._running:
            try:
                pending = self._get_pending_sources()
                for source_data in pending:
                    job = RAPTORJob(
                        source_id=source_data["source_id"],
                        notebook_id=source_data["notebook_id"],
                        file_name=source_data["file_name"],
                    )
                    await self._queue.put(job)

                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error polling pending RAPTOR jobs: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _process_with_semaphore(self, job: RAPTORJob):
        """Process job with semaphore for concurrency control."""
        async with self._semaphore:
            await self._process_job(job)

    async def _process_job(self, job: RAPTORJob):
        """Process a single RAPTOR job."""
        source_id = job.source_id
        logger.info(f"Building RAPTOR tree for source: {source_id} ({job.file_name})")

        try:
            # Update status to building
            self._update_status(source_id, "building")

            # Initialize tree builder if needed
            if not self._tree_builder:
                self._init_tree_builder()

            if not self._tree_builder:
                logger.error("Failed to initialize tree builder")
                self._update_status(source_id, "failed", "Tree builder not available")
                return

            # Get chunks from vector store
            chunks = self._get_source_chunks(source_id, job.notebook_id)

            if not chunks:
                logger.warning(f"No chunks found for source: {source_id}")
                self._update_status(source_id, "failed", "No chunks found")
                return

            logger.info(f"Found {len(chunks)} chunks for {job.file_name}")

            # Build the tree
            result = await asyncio.to_thread(
                self._tree_builder.build_tree,
                chunks=chunks,
                source_id=source_id,
                notebook_id=job.notebook_id,
                progress_callback=lambda stage, progress, msg:
                    logger.debug(f"RAPTOR [{source_id[:8]}] {stage}: {progress:.0%} - {msg}")
            )

            if not result.success:
                self._update_status(source_id, "failed", result.error)
                return

            # Store summary nodes in vector store
            if result.summary_nodes:
                stored = self._store_summary_nodes(
                    result.summary_nodes,
                    job.notebook_id,
                    source_id
                )
                logger.info(f"Stored {stored} summary nodes for {job.file_name}")

            # Update status to completed
            self._update_status(source_id, "completed")
            logger.info(
                f"Completed RAPTOR tree for {job.file_name}: "
                f"{result.total_nodes} total nodes, {result.max_level} levels, "
                f"{result.build_time_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to build RAPTOR tree for {source_id}: {e}", exc_info=True)
            self._update_status(source_id, "failed", str(e))

    def _init_tree_builder(self):
        """Initialize the tree builder with current LLM and embed model."""
        try:
            # Get LLM and embed model from Settings
            llm = Settings.llm
            embed_model = Settings.embed_model

            if not llm:
                logger.error("No LLM available in Settings")
                return

            if not embed_model:
                logger.error("No embed model available in Settings")
                return

            self._tree_builder = RAPTORTreeBuilder(
                llm=llm,
                embed_model=embed_model,
                config=self.config
            )
            logger.info("RAPTOR tree builder initialized")

        except Exception as e:
            logger.error(f"Failed to initialize tree builder: {e}")

    def _get_source_chunks(
        self,
        source_id: str,
        notebook_id: str
    ) -> List[BaseNode]:
        """Get chunks for a source from the vector store."""
        try:
            # Get level 0 nodes (original chunks) for this source
            nodes = self.vector_store.get_nodes_by_tree_level(
                notebook_id=notebook_id,
                tree_level=0,
                source_ids=[source_id]
            )
            return nodes
        except Exception as e:
            logger.error(f"Error getting chunks for {source_id}: {e}")
            return []

    def _store_summary_nodes(
        self,
        summary_nodes: List,
        notebook_id: str,
        source_id: str
    ) -> int:
        """Store summary nodes in the vector store."""
        try:
            # Summary nodes are already TextNode objects from tree_builder
            text_nodes = summary_nodes

            if not text_nodes:
                return 0

            # Group by tree level for efficient storage
            nodes_by_level = {}
            for node in text_nodes:
                level = node.metadata.get("tree_level", 1)
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)

            total_stored = 0
            tree_root_id = text_nodes[0].metadata.get("tree_root_id") if text_nodes else None

            for level, nodes in sorted(nodes_by_level.items()):
                stored = self.vector_store.add_tree_nodes(
                    nodes=nodes,
                    notebook_id=notebook_id,
                    source_id=source_id,
                    tree_level=level,
                    tree_root_id=tree_root_id
                )
                total_stored += stored
                logger.debug(f"Stored {stored} nodes at level {level}")

            return total_stored

        except Exception as e:
            logger.error(f"Error storing summary nodes: {e}")
            return 0

    def _get_pending_sources(self, limit: int = 5) -> List[dict]:
        """Get sources with pending RAPTOR status."""
        try:
            with self.db.get_session() as session:
                sources = session.query(NotebookSource).filter(
                    NotebookSource.raptor_status == "pending"
                ).limit(limit).all()

                return [
                    {
                        "source_id": str(s.source_id),
                        "notebook_id": str(s.notebook_id),
                        "file_name": s.file_name,
                    }
                    for s in sources
                ]
        except Exception as e:
            logger.error(f"Error getting pending RAPTOR sources: {e}")
            return []

    def _update_status(
        self,
        source_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Update RAPTOR status for a source."""
        try:
            with self.db.get_session() as session:
                source = session.query(NotebookSource).filter(
                    NotebookSource.source_id == UUID(source_id)
                ).first()

                if source:
                    source.raptor_status = status
                    source.raptor_error = error
                    if status == "completed":
                        source.raptor_built_at = datetime.utcnow()

                    logger.debug(f"Updated RAPTOR status for {source_id}: {status}")

        except Exception as e:
            logger.error(f"Error updating RAPTOR status for {source_id}: {e}")


def build_raptor_tree_sync(
    db_manager: DatabaseManager,
    vector_store: "PGVectorStore",
    source_id: str,
    notebook_id: str,
    config: Optional[RAPTORConfig] = None,
) -> TreeBuildResult:
    """Build RAPTOR tree synchronously for a single source.

    Convenience function for immediate processing (not background).

    Args:
        db_manager: Database manager
        vector_store: Vector store for chunks
        source_id: Source ID to process
        notebook_id: Notebook ID containing the source
        config: Optional RAPTOR configuration

    Returns:
        TreeBuildResult with build results
    """
    config = config or DEFAULT_CONFIG

    # Get LLM and embed model from Settings
    llm = Settings.llm
    embed_model = Settings.embed_model

    if not llm or not embed_model:
        return TreeBuildResult(
            success=False,
            source_id=source_id,
            notebook_id=notebook_id,
            total_nodes=0,
            levels={},
            max_level=0,
            error="LLM or embed model not available"
        )

    # Build tree
    builder = RAPTORTreeBuilder(llm, embed_model, config)

    # Get chunks
    chunks = vector_store.get_nodes_by_tree_level(
        notebook_id=notebook_id,
        tree_level=0,
        source_ids=[source_id]
    )

    if not chunks:
        return TreeBuildResult(
            success=False,
            source_id=source_id,
            notebook_id=notebook_id,
            total_nodes=0,
            levels={},
            max_level=0,
            error="No chunks found for source"
        )

    result = builder.build_tree(
        chunks=chunks,
        source_id=source_id,
        notebook_id=notebook_id
    )

    # Store summary nodes if successful
    if result.success and result.summary_nodes:
        # Summary nodes are already TextNode objects from tree_builder
        text_nodes = result.summary_nodes
        if text_nodes:
            nodes_by_level = {}
            for node in text_nodes:
                level = node.metadata.get("tree_level", 1)
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)

            tree_root_id = text_nodes[0].metadata.get("tree_root_id") if text_nodes else None

            for level, nodes in sorted(nodes_by_level.items()):
                vector_store.add_tree_nodes(
                    nodes=nodes,
                    notebook_id=notebook_id,
                    source_id=source_id,
                    tree_level=level,
                    tree_root_id=tree_root_id
                )

    # Update database status
    status = "completed" if result.success else "failed"
    try:
        with db_manager.get_session() as session:
            source = session.query(NotebookSource).filter(
                NotebookSource.source_id == UUID(source_id)
            ).first()

            if source:
                source.raptor_status = status
                source.raptor_error = result.error
                if status == "completed":
                    source.raptor_built_at = datetime.utcnow()

    except Exception as e:
        logger.error(f"Error updating status after sync build: {e}")

    return result
