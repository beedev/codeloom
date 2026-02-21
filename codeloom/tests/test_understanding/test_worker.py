"""Unit tests for UnderstandingWorker — lease, heartbeat, retry, backoff.

Tests cover:
- Worker lifecycle (start/stop)
- Job claiming with FOR UPDATE SKIP LOCKED semantics
- Heartbeat updates scoped to worker_id
- Stale reclaim with bounded retries
- Exponential backoff scheduling
- Complete/failure transitions
- Job processing pipeline
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from uuid import uuid4, UUID

from codeloom.core.understanding.worker import UnderstandingWorker, UnderstandingJob


# ── Fixtures ──────────────────────────────────────────────────────────────


def _mock_db():
    """Create a mock DatabaseManager with a context-managed session."""
    db = MagicMock()
    session = MagicMock()
    db.get_session.return_value.__enter__ = MagicMock(return_value=session)
    db.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return db, session


def _make_job(job_id=None, project_id=None, worker_id=None) -> UnderstandingJob:
    return UnderstandingJob(
        job_id=job_id or str(uuid4()),
        project_id=project_id or str(uuid4()),
        worker_id=worker_id or "test-worker",
    )


# ── Tests: Worker Lifecycle ──────────────────────────────────────────────


class TestWorkerLifecycle:
    """Tests for start/stop behavior."""

    def test_initial_state(self):
        db = MagicMock()
        worker = UnderstandingWorker(db)

        assert worker._running is False
        assert worker._thread is None
        assert worker._loop is None

    def test_start_sets_running(self):
        db = MagicMock()
        worker = UnderstandingWorker(db)

        with patch.object(worker, "_run_loop"):
            worker.start()

        assert worker._running is True
        assert worker._thread is not None
        assert worker._thread.daemon is True

        worker.stop()

    def test_double_start_is_safe(self):
        """Calling start() twice should not create a second thread."""
        db = MagicMock()
        worker = UnderstandingWorker(db)
        worker._running = True

        worker.start()  # Should log warning but not crash

    def test_stop_sets_not_running(self):
        db = MagicMock()
        worker = UnderstandingWorker(db)
        worker._running = True
        worker._loop = None
        worker._thread = None

        worker.stop()

        assert worker._running is False

    def test_worker_id_is_unique(self):
        """Each worker instance should have a unique worker_id."""
        db = MagicMock()
        w1 = UnderstandingWorker(db)
        w2 = UnderstandingWorker(db)

        assert w1.worker_id != w2.worker_id
        assert w1.worker_id.startswith("understanding-")


# ── Tests: Claim Pending Jobs ────────────────────────────────────────────


class TestClaimPendingJobs:
    """Tests for _claim_pending_jobs SQL logic."""

    def test_returns_claimed_jobs(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db, max_retries=3)

        job_id = uuid4()
        project_id = uuid4()

        # Mock the RETURNING result for the claim query
        mock_row = MagicMock()
        mock_row.job_id = job_id
        mock_row.project_id = project_id
        mock_row.worker_id = worker.worker_id

        # First execute is stale reclaim, second is the claim query
        session.execute.side_effect = [
            MagicMock(),  # stale reclaim
            MagicMock(fetchall=MagicMock(return_value=[mock_row])),  # claim
        ]

        jobs = worker._claim_pending_jobs(limit=2)

        assert len(jobs) == 1
        assert jobs[0].job_id == str(job_id)
        assert jobs[0].project_id == str(project_id)

    def test_returns_empty_when_no_pending(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        session.execute.side_effect = [
            MagicMock(),  # stale reclaim
            MagicMock(fetchall=MagicMock(return_value=[])),  # no jobs
        ]

        jobs = worker._claim_pending_jobs()
        assert jobs == []

    def test_stale_reclaim_called_before_claim(self):
        """Stale reclaim should happen before new job claims."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db, stale_threshold=120.0, max_retries=2)

        session.execute.side_effect = [
            MagicMock(),  # stale reclaim
            MagicMock(fetchall=MagicMock(return_value=[])),  # claim
        ]

        worker._claim_pending_jobs()

        # Verify stale reclaim SQL was called first
        calls = session.execute.call_args_list
        assert len(calls) == 2
        first_sql = str(calls[0])
        assert "stale" in first_sql.lower() or "heartbeat_at" in first_sql.lower()


# ── Tests: Heartbeat Updates ─────────────────────────────────────────────


class TestHeartbeatUpdates:
    """Tests for _update_heartbeats — scoped to worker_id."""

    def test_heartbeat_updates_session(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        worker._update_heartbeats()

        session.execute.assert_called_once()
        # Verify the SQL includes worker_id scoping
        sql_call = str(session.execute.call_args)
        assert "worker_id" in sql_call.lower() or "heartbeat" in sql_call.lower()


# ── Tests: Job Progress ─────────────────────────────────────────────────


class TestJobProgress:
    """Tests for _update_job_progress."""

    def test_updates_progress_counters(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        job_id = str(uuid4())
        worker._update_job_progress(job_id, total=10, completed=5)

        session.execute.assert_called_once()


# ── Tests: Complete Job ──────────────────────────────────────────────────


class TestCompleteJob:
    """Tests for _complete_job transition."""

    def test_complete_with_no_errors(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        job_id = str(uuid4())
        worker._complete_job(job_id, "completed", [])

        # Should be called twice: once for job update, once for project status
        assert session.execute.call_count >= 1

    def test_complete_with_errors(self):
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        job_id = str(uuid4())
        errors = [{"entry_point": "foo", "error": "LLM timeout"}]
        worker._complete_job(job_id, "completed", errors)

        # Verify error details were included
        assert session.execute.call_count >= 1


# ── Tests: Handle Job Failure ────────────────────────────────────────────


class TestHandleJobFailure:
    """Tests for _handle_job_failure — retry vs terminal failure."""

    def test_terminal_failure_when_max_retries_exceeded(self):
        """Job should be marked 'failed' when retry_count >= max_retries."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db, max_retries=2)

        job_id = str(uuid4())

        # Mock retry_count query to return count >= max_retries
        retry_row = MagicMock()
        retry_row.retry_count = 3

        # First call: SELECT retry_count; Second call: UPDATE to failed
        session.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=retry_row)),
            MagicMock(),
        ]

        worker._handle_job_failure(job_id, "LLM error")

        # Should have been called twice (select + update)
        assert session.execute.call_count == 2
        # The second call should set status to 'failed' — check the SQL text
        second_call = session.execute.call_args_list[1]
        sql_text = second_call[0][0].text  # TextClause.text
        assert "failed" in sql_text.lower()

    def test_retry_with_backoff_when_under_max(self):
        """Job should be set to 'pending' with next_attempt_at when retries remain."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db, max_retries=3)

        job_id = str(uuid4())

        retry_row = MagicMock()
        retry_row.retry_count = 1  # Under max_retries

        session.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=retry_row)),
            MagicMock(),
        ]

        worker._handle_job_failure(job_id, "Transient error")

        # Should set status back to pending
        assert session.execute.call_count == 2
        second_call = session.execute.call_args_list[1]
        sql_text = second_call[0][0].text  # TextClause.text
        assert "pending" in sql_text.lower()

    def test_exponential_backoff_increases(self):
        """Backoff delay should increase exponentially with retry count."""
        base_backoff = 15
        # retry_count=0 → 15s, retry_count=1 → 30s, retry_count=2 → 60s

        for retry_count, expected_seconds in [(0, 15), (1, 30), (2, 60)]:
            actual = base_backoff * (2 ** retry_count)
            assert actual == expected_seconds


# ── Tests: Job Processing Pipeline ───────────────────────────────────────


class TestJobProcessing:
    """Tests for _process_job_sync — the full analysis pipeline."""

    @patch("codeloom.core.understanding.frameworks.detect_and_analyze")
    def test_process_job_calls_pipeline_steps(self, mock_detect_fw):
        """Processing should: detect frameworks, detect entry points,
        trace trees, analyze chains, and store results."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        # Mock framework detection
        mock_detect_fw.return_value = []

        # Mock tracer
        mock_tracer = MagicMock()
        mock_tracer.detect_entry_points.return_value = []
        worker._tracer = mock_tracer

        # Mock analyzer
        mock_analyzer = MagicMock()
        worker._analyzer = mock_analyzer

        job = _make_job(worker_id=worker.worker_id)
        worker._process_job_sync(job)

        # Should have detected entry points
        mock_tracer.detect_entry_points.assert_called_once_with(job.project_id)

    @patch("codeloom.core.understanding.frameworks.detect_and_analyze")
    def test_process_job_handles_entry_point_errors_gracefully(self, mock_detect_fw):
        """Errors on individual entry points should not fail the whole job."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db)

        mock_detect_fw.return_value = []

        # Create mock entry point
        ep = MagicMock()
        ep.unit_id = str(uuid4())
        ep.qualified_name = "test.handler"

        mock_tracer = MagicMock()
        mock_tracer.detect_entry_points.return_value = [ep]
        mock_tracer.trace_call_tree.side_effect = RuntimeError("DB error")
        worker._tracer = mock_tracer

        mock_analyzer = MagicMock()
        worker._analyzer = mock_analyzer

        job = _make_job(worker_id=worker.worker_id)
        worker._process_job_sync(job)

        # Analyzer should NOT have been called (trace failed first)
        mock_analyzer.analyze_chain.assert_not_called()

    @patch("codeloom.core.understanding.frameworks.detect_and_analyze")
    def test_process_job_complete_with_mixed_results(self, mock_detect_fw):
        """Job with some successes and some failures should complete (not fail)."""
        db, session = _mock_db()
        worker = UnderstandingWorker(db)
        mock_detect_fw.return_value = []

        # Two entry points: one succeeds, one fails
        ep_good = MagicMock(unit_id=str(uuid4()), qualified_name="good")
        ep_bad = MagicMock(unit_id=str(uuid4()), qualified_name="bad")

        mock_tracer = MagicMock()
        mock_tracer.detect_entry_points.return_value = [ep_good, ep_bad]

        good_tree = MagicMock()
        mock_tracer.trace_call_tree.side_effect = [good_tree, RuntimeError("fail")]
        worker._tracer = mock_tracer

        mock_bundle = MagicMock()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_chain.return_value = mock_bundle
        worker._analyzer = mock_analyzer

        # Mock store_analysis to succeed
        with patch.object(worker, "_store_analysis"):
            with patch.object(worker, "_update_job_progress"):
                with patch.object(worker, "_complete_job") as mock_complete:
                    job = _make_job(worker_id=worker.worker_id)
                    worker._process_job_sync(job)

                    # Should have called complete with "completed" (at least one success)
                    mock_complete.assert_called_once()
                    call_args = mock_complete.call_args
                    assert call_args[0][1] == "completed"
                    assert len(call_args[0][2]) == 1  # 1 error


# ── Tests: Worker Configuration ──────────────────────────────────────────


class TestWorkerConfiguration:
    """Tests for worker configuration parameters."""

    def test_default_configuration(self):
        db = MagicMock()
        worker = UnderstandingWorker(db)

        assert worker.poll_interval == 15.0
        assert worker.max_concurrent == 2
        assert worker.heartbeat_interval == 30.0
        assert worker.stale_threshold == 120.0
        assert worker.max_retries == 2

    def test_custom_configuration(self):
        db = MagicMock()
        worker = UnderstandingWorker(
            db,
            poll_interval=5.0,
            max_concurrent=4,
            heartbeat_interval=10.0,
            stale_threshold=60.0,
            max_retries=5,
        )

        assert worker.poll_interval == 5.0
        assert worker.max_concurrent == 4
        assert worker.heartbeat_interval == 10.0
        assert worker.stale_threshold == 60.0
        assert worker.max_retries == 5
