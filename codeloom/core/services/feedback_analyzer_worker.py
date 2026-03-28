"""Background worker that periodically runs FeedbackAnalyzerService and ParameterOptimizerService.

Runs in a daemon thread so it does not block application startup or shutdown.
Controlled by the FEEDBACK_ANALYZER_ENABLED and FEEDBACK_ANALYZER_INTERVAL env vars.
"""

import logging
import os
import threading
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from codeloom.core.db import DatabaseManager

logger = logging.getLogger(__name__)

# Environment-variable defaults
_DEFAULT_INTERVAL = 3600      # 1 hour between analysis runs
_DEFAULT_MIN_SAMPLES = 10     # Minimum feedback records before analysis runs


class FeedbackAnalyzerWorker:
    """Daemon thread that periodically analyses RAG feedback and adjusts parameters.

    Lifecycle:
        worker = FeedbackAnalyzerWorker(db_manager)
        worker.start()   # Called by __main__.py when FEEDBACK_ANALYZER_ENABLED=true
        worker.stop()    # Graceful shutdown (called by pipeline teardown)
    """

    def __init__(
        self,
        db_manager: "DatabaseManager",
        interval: float = _DEFAULT_INTERVAL,
    ):
        """Initialise the worker.

        Args:
            db_manager: DatabaseManager for accessing feedback tables.
            interval: Seconds between analysis passes (default: 3600).
        """
        self._db = db_manager
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background analysis thread."""
        if self._running:
            logger.warning("FeedbackAnalyzerWorker already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="FeedbackAnalyzerWorker",
        )
        self._thread.start()
        logger.info(
            f"FeedbackAnalyzerWorker started (interval={self._interval}s)"
        )

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        logger.info("FeedbackAnalyzerWorker stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop — runs analysis, then sleeps until next interval."""
        logger.info("FeedbackAnalyzerWorker: first pass in 60s")
        # Wait 60 seconds before the first pass so the application can fully start up
        if self._stop_event.wait(timeout=60.0):
            return  # Stopped before first run

        while self._running:
            try:
                self._run_once()
            except Exception as exc:
                logger.error(
                    f"FeedbackAnalyzerWorker pass failed: {exc}", exc_info=True
                )

            # Sleep in small increments so we can respond to stop() quickly
            elapsed = 0.0
            step = 30.0
            while self._running and elapsed < self._interval:
                if self._stop_event.wait(timeout=step):
                    return
                elapsed += step

    def _run_once(self) -> None:
        """Run one full analysis + optimisation pass across all projects."""
        logger.debug("FeedbackAnalyzerWorker: starting analysis pass")

        # Lazy import to avoid circular imports at module level
        from codeloom.core.services.feedback_analyzer_service import FeedbackAnalyzerService
        from codeloom.core.services.parameter_optimizer_service import ParameterOptimizerService

        # FeedbackAnalyzerService and ParameterOptimizerService follow the BaseService
        # pattern: they receive (pipeline, db_manager, project_manager). The worker
        # only has db_manager, so we pass None for the others — both services only
        # call _validate_database_available() which checks db_manager alone.
        analyzer = FeedbackAnalyzerService(
            pipeline=None,  # type: ignore[arg-type]
            db_manager=self._db,
            project_manager=None,
        )
        optimizer = ParameterOptimizerService(
            pipeline=None,  # type: ignore[arg-type]
            db_manager=self._db,
            project_manager=None,
        )

        # Run analysis across all projects (project_id=None -> all)
        summary = analyzer.run_full_analysis(project_id=None)

        # Translate insights into adaptive settings for each analysed project
        for proj_id in summary:
            try:
                optimizer.apply_recommendations(project_id=proj_id)
            except Exception as opt_exc:
                logger.warning(
                    f"ParameterOptimizer failed for project {proj_id}: {opt_exc}"
                )

        if summary:
            logger.info(
                f"FeedbackAnalyzerWorker: pass complete | projects={list(summary.keys())}"
            )
        else:
            logger.debug("FeedbackAnalyzerWorker: no projects had sufficient feedback")
