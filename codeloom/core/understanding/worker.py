"""Background worker for deep understanding analysis.

Mirrors the RAPTORWorker pattern from core/raptor/worker.py:
- Daemon thread with its own asyncio event loop
- Queue-based job processing with Semaphore concurrency control
- Polls database for pending jobs

Adds distributed lease protocol:
- Claim jobs via FOR UPDATE SKIP LOCKED
- Heartbeat every 30s
- Stale reclaim at 120s
- Retry policy: configurable max retries with exponential backoff
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4

from llama_index.core import Settings
from sqlalchemy import text

from ..db import DatabaseManager
from .chain_tracer import ChainTracer
from .analyzer import ChainAnalyzer
from .models import DeepContextBundle

if TYPE_CHECKING:
    from .frameworks.base import FrameworkContext

logger = logging.getLogger(__name__)


@dataclass
class UnderstandingJob:
    """A job to analyze a project's entry points."""
    job_id: str          # UUID of the DeepAnalysisJob row
    project_id: str
    worker_id: str


class UnderstandingWorker:
    """Background worker for deep understanding analysis.

    Lifecycle:
    1. start() spawns daemon thread with asyncio loop
    2. _poll_pending() checks for claimable jobs every poll_interval
    3. _process_job() runs the full pipeline per job:
       a. Detect framework context
       b. Detect entry points
       c. Trace call trees
       d. Analyze each chain
       e. Store results + populate analysis_units
       f. Embed narratives for retrieval
    4. stop() signals shutdown
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        poll_interval: float = 15.0,
        max_concurrent: int = 2,
        heartbeat_interval: float = 30.0,
        stale_threshold: float = 120.0,
        max_retries: int = 2,
    ):
        self._db = db_manager
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.heartbeat_interval = heartbeat_interval
        self.stale_threshold = stale_threshold
        self.max_retries = max_retries
        self.worker_id = f"understanding-{uuid4()}"

        self._queue: Optional[asyncio.Queue] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Lazily initialized
        self._tracer: Optional[ChainTracer] = None
        self._analyzer: Optional[ChainAnalyzer] = None

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("Understanding worker already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="understanding-worker"
        )
        self._thread.start()
        logger.info("Understanding worker started")

    def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Understanding worker stopped")

    def _run_loop(self):
        """Run the async event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"Understanding worker loop error: {e}")
        finally:
            self._loop.close()

    async def _main_loop(self):
        """Main processing loop — mirrors RAPTORWorker._main_loop()."""
        poll_task = asyncio.create_task(self._poll_pending())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        while self._running:
            try:
                job = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.poll_interval,
                )
                asyncio.create_task(self._process_with_semaphore(job))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in understanding main loop: {e}")

        poll_task.cancel()
        heartbeat_task.cancel()

    async def _poll_pending(self):
        """Poll for claimable jobs using FOR UPDATE SKIP LOCKED."""
        while self._running:
            try:
                jobs = self._claim_pending_jobs()
                for job in jobs:
                    await self._queue.put(job)
            except Exception as e:
                logger.error(f"Error polling understanding jobs: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _heartbeat_loop(self):
        """Update heartbeat_at for all running jobs owned by this worker."""
        while self._running:
            try:
                self._update_heartbeats()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    async def _process_with_semaphore(self, job: UnderstandingJob):
        """Process job with semaphore for concurrency control."""
        async with self._semaphore:
            await asyncio.to_thread(self._process_job_sync, job)

    def _process_job_sync(self, job: UnderstandingJob):
        """Process a single understanding job (runs in thread pool).

        Pipeline:
        1. Initialize tracer/analyzer if needed
        2. Run framework detection
        3. Detect entry points
        4. For each entry point:
           a. Trace call tree
           b. Analyze chain
           c. Store DeepAnalysis row + analysis_units
        5. Mark job completed
        """
        logger.info(f"Processing understanding job {job.job_id} for project {job.project_id}")

        try:
            # Initialize
            if not self._tracer:
                self._tracer = ChainTracer(self._db)
            if not self._analyzer:
                self._analyzer = ChainAnalyzer()

            # Framework detection
            from .frameworks import detect_and_analyze
            framework_contexts = detect_and_analyze(self._db, job.project_id)

            # Detect entry points
            entry_points = self._tracer.detect_entry_points(job.project_id)
            logger.info(f"Found {len(entry_points)} entry points for project {job.project_id}")

            # Update job progress
            self._update_job_progress(job.job_id, len(entry_points), 0)

            # Process each entry point
            completed = 0
            errors = []

            for ep in entry_points:
                try:
                    # Trace
                    tree = self._tracer.trace_call_tree(
                        job.project_id, ep.unit_id, max_depth=10
                    )

                    # Analyze
                    bundle = self._analyzer.analyze_chain(
                        entry_point=ep,
                        call_tree=tree,
                        framework_contexts=framework_contexts,
                    )

                    # Store
                    self._store_analysis(job.job_id, job.project_id, bundle, tree)

                    completed += 1
                    self._update_job_progress(job.job_id, len(entry_points), completed)

                except Exception as e:
                    logger.error(
                        f"Error analyzing entry point {ep.qualified_name}: {e}",
                        exc_info=True,
                    )
                    errors.append({"entry_point": ep.qualified_name, "error": str(e)})

            # Mark completed
            status = "completed" if not errors or completed > 0 else "failed"
            self._complete_job(job.job_id, status, errors)

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            self._handle_job_failure(job.job_id, str(e))

    # ── DB Operations ───────────────────────────────────────────────────

    def _claim_pending_jobs(self, limit: int = 2) -> List[UnderstandingJob]:
        """Claim pending jobs using FOR UPDATE SKIP LOCKED.

        Also reclaims stale running jobs (heartbeat older than threshold).
        """
        now = datetime.utcnow()
        stale_cutoff = now - timedelta(seconds=self.stale_threshold)

        with self._db.get_session() as session:
            # Reclaim stale jobs first
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = 'pending',
                        worker_id = NULL
                    WHERE status = 'running'
                      AND heartbeat_at < :stale_cutoff
                      AND retry_count < :max_retries
                """),
                {"stale_cutoff": stale_cutoff, "max_retries": self.max_retries},
            )

            # Claim pending jobs
            result = session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = 'running',
                        worker_id = :worker_id,
                        started_at = :now,
                        heartbeat_at = :now,
                        retry_count = retry_count + 1,
                        next_attempt_at = NULL
                    WHERE job_id IN (
                        SELECT job_id FROM deep_analysis_jobs
                        WHERE status = 'pending'
                          AND (next_attempt_at IS NULL OR next_attempt_at <= :now)
                        ORDER BY created_at
                        LIMIT :limit
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING job_id, project_id, worker_id
                """),
                {"now": now, "limit": limit, "worker_id": self.worker_id},
            )
            rows = result.fetchall()

        return [
            UnderstandingJob(
                job_id=str(row.job_id),
                project_id=str(row.project_id),
                worker_id=str(row.worker_id),
            )
            for row in rows
        ]

    def _update_heartbeats(self):
        """Update heartbeat_at for all jobs this worker is processing."""
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET heartbeat_at = :now
                    WHERE status = 'running'
                      AND worker_id = :worker_id
                """),
                {"now": datetime.utcnow(), "worker_id": self.worker_id},
            )

    def _update_job_progress(self, job_id: str, total: int, completed: int):
        """Update job progress counters."""
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET total_entry_points = :total,
                        completed_entry_points = :completed,
                        heartbeat_at = :now
                    WHERE job_id = :job_id
                      AND worker_id = :worker_id
                """),
                {
                    "job_id": UUID(job_id),
                    "total": total,
                    "completed": completed,
                    "now": datetime.utcnow(),
                    "worker_id": self.worker_id,
                },
            )

    def _store_analysis(
        self,
        job_id: str,
        project_id: str,
        bundle: DeepContextBundle,
        tree,
    ):
        """Store a DeepAnalysis row and populate analysis_units.

        Uses upsert (ON CONFLICT on project_id + entry_unit_id + schema_version).
        """
        import json as json_mod
        from .chain_tracer import ChainTracer  # for get_flat_unit_membership

        pid = UUID(project_id)
        entry_uid = UUID(bundle.entry_point.unit_id)

        # Serialize full bundle to JSON for replay/audit compatibility
        result_json = json_mod.dumps({
            "entry_point": {
                "unit_id": bundle.entry_point.unit_id,
                "name": bundle.entry_point.name,
                "qualified_name": bundle.entry_point.qualified_name,
                "file_path": bundle.entry_point.file_path,
                "entry_type": bundle.entry_point.entry_type.value,
                "language": bundle.entry_point.language,
                "metadata": bundle.entry_point.metadata,
                "detected_by": bundle.entry_point.detected_by,
            },
            "tier": bundle.tier.value,
            "total_units": bundle.total_units,
            "total_tokens": bundle.total_tokens,
            "business_rules": bundle.business_rules,
            "data_entities": bundle.data_entities,
            "integrations": bundle.integrations,
            "side_effects": bundle.side_effects,
            "cross_cutting_concerns": bundle.cross_cutting_concerns,
            "narrative": bundle.narrative,
            "confidence": bundle.confidence,
            "coverage": bundle.coverage,
            "chain_truncated": bundle.chain_truncated,
            "schema_version": bundle.schema_version,
            "prompt_version": bundle.prompt_version,
            "analyzed_at": bundle.analyzed_at,
        })

        with self._db.get_session() as session:
            # Upsert deep_analyses
            result = session.execute(
                text("""
                    INSERT INTO deep_analyses (
                        analysis_id, job_id, project_id, entry_unit_id,
                        entry_type, tier, total_units, total_tokens,
                        confidence_score, coverage_pct,
                        result_json, narrative, schema_version, prompt_version
                    ) VALUES (
                        gen_random_uuid(), :job_id, :pid, :entry_uid,
                        :entry_type, :tier, :total_units, :total_tokens,
                        :confidence_score, :coverage_pct,
                        :result_json, :narrative, :schema_version, :prompt_version
                    )
                    ON CONFLICT (project_id, entry_unit_id, schema_version)
                    DO UPDATE SET
                        job_id = EXCLUDED.job_id,
                        tier = EXCLUDED.tier,
                        total_units = EXCLUDED.total_units,
                        total_tokens = EXCLUDED.total_tokens,
                        confidence_score = EXCLUDED.confidence_score,
                        coverage_pct = EXCLUDED.coverage_pct,
                        result_json = EXCLUDED.result_json,
                        narrative = EXCLUDED.narrative,
                        prompt_version = EXCLUDED.prompt_version,
                        analyzed_at = NOW()
                    RETURNING analysis_id
                """),
                {
                    "job_id": UUID(job_id),
                    "pid": pid,
                    "entry_uid": entry_uid,
                    "entry_type": bundle.entry_point.entry_type.value,
                    "tier": bundle.tier.value,
                    "total_units": bundle.total_units,
                    "total_tokens": bundle.total_tokens,
                    "confidence_score": bundle.confidence,
                    "coverage_pct": bundle.coverage * 100.0,
                    "result_json": result_json,
                    "narrative": bundle.narrative,
                    "schema_version": bundle.schema_version,
                    "prompt_version": bundle.prompt_version,
                },
            )
            analysis_id = result.fetchone().analysis_id

            # Populate analysis_units
            tracer = self._tracer or ChainTracer(self._db)
            flat_units = tracer.get_flat_unit_membership(tree)

            for unit in flat_units:
                session.execute(
                    text("""
                        INSERT INTO analysis_units (
                            analysis_id, project_id, unit_id, min_depth, path_count
                        ) VALUES (
                            :aid, :pid, :uid, :min_depth, :path_count
                        )
                        ON CONFLICT (analysis_id, unit_id)
                        DO UPDATE SET
                            min_depth = LEAST(analysis_units.min_depth, EXCLUDED.min_depth),
                            path_count = GREATEST(analysis_units.path_count, EXCLUDED.path_count)
                    """),
                    {
                        "aid": analysis_id,
                        "pid": pid,
                        "uid": UUID(unit["unit_id"]),
                        "min_depth": unit["min_depth"],
                        "path_count": unit["path_count"],
                    },
                )

    def _complete_job(self, job_id: str, status: str, errors: list):
        """Mark job as completed or failed."""
        import json as json_mod

        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE deep_analysis_jobs
                    SET status = :status,
                        completed_at = :now,
                        error_details = :errors
                    WHERE job_id = :job_id
                      AND worker_id = :worker_id
                """),
                {
                    "job_id": UUID(job_id),
                    "status": status,
                    "now": datetime.utcnow(),
                    "errors": json_mod.dumps(errors) if errors else None,
                    "worker_id": self.worker_id,
                },
            )

        # Update project status
        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE projects
                    SET deep_analysis_status = :status
                    WHERE project_id = (
                        SELECT project_id FROM deep_analysis_jobs WHERE job_id = :job_id
                    )
                """),
                {"job_id": UUID(job_id), "status": status},
            )

    def _handle_job_failure(self, job_id: str, error: str):
        """Handle job failure with retry logic."""
        base_backoff_seconds = 15
        with self._db.get_session() as session:
            result = session.execute(
                text("SELECT retry_count FROM deep_analysis_jobs WHERE job_id = :jid"),
                {"jid": UUID(job_id)},
            )
            row = result.fetchone()
            retry_count = row.retry_count if row else 0

            if retry_count >= self.max_retries:
                # Terminal failure
                session.execute(
                    text("""
                        UPDATE deep_analysis_jobs
                        SET status = 'failed',
                            completed_at = :now,
                            error_details = :error
                        WHERE job_id = :jid
                          AND worker_id = :worker_id
                    """),
                    {
                        "jid": UUID(job_id),
                        "now": datetime.utcnow(),
                        "error": error,
                        "worker_id": self.worker_id,
                    },
                )
            else:
                # Back to pending for retry with exponential backoff
                session.execute(
                    text("""
                        UPDATE deep_analysis_jobs
                        SET status = 'pending',
                            worker_id = NULL,
                            next_attempt_at = :next_attempt_at
                        WHERE job_id = :jid
                          AND worker_id = :worker_id
                    """),
                    {
                        "jid": UUID(job_id),
                        "worker_id": self.worker_id,
                        "next_attempt_at": datetime.utcnow() + timedelta(
                            seconds=base_backoff_seconds * (2 ** retry_count)
                        ),
                    },
                )
