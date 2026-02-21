"""Understanding Engine — orchestrator for deep code analysis.

Follows the MigrationEngine pattern from core/migration/engine.py.
Provides the public API consumed by API routes.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from ..db import DatabaseManager
from ..db.models import Project
from .chain_tracer import ChainTracer
from .worker import UnderstandingWorker

logger = logging.getLogger(__name__)


class UnderstandingEngine:
    """Orchestrate deep understanding analysis for a project.

    Public API:
        start_analysis(project_id, user_id) -> job_id
        get_job_status(project_id, job_id) -> status dict
        get_entry_points(project_id) -> list of entry points
        get_analysis_results(project_id) -> list of analysis summaries
        get_chain_detail(project_id, analysis_id) -> full detail
    """

    def __init__(self, db_manager: DatabaseManager, pipeline: Any = None):
        self._db = db_manager
        self._pipeline = pipeline
        self._worker: Optional[UnderstandingWorker] = None
        self._tracer: Optional[ChainTracer] = None

    def _ensure_worker(self):
        """Lazily initialize and start the background worker."""
        if self._worker is None:
            self._worker = UnderstandingWorker(self._db)
        if not self._worker._running:
            self._worker.start()

    def _ensure_tracer(self):
        """Lazily initialize the chain tracer."""
        if self._tracer is None:
            self._tracer = ChainTracer(self._db)

    # ── Public API ──────────────────────────────────────────────────────

    def start_analysis(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Start a deep understanding analysis job for a project.

        Creates a DeepAnalysisJob row and ensures the worker is running.

        Returns:
            Dict with job_id and status
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        uid = UUID(user_id) if isinstance(user_id, str) else user_id
        job_id = uuid4()

        with self._db.get_session() as session:
            session.execute(
                text("""
                    INSERT INTO deep_analysis_jobs (job_id, project_id, user_id, status)
                    VALUES (:jid, :pid, :uid, 'pending')
                """),
                {"jid": job_id, "pid": pid, "uid": uid},
            )

            # Update project status
            session.execute(
                text("UPDATE projects SET deep_analysis_status = 'pending' WHERE project_id = :pid"),
                {"pid": pid},
            )

        self._ensure_worker()

        return {"job_id": str(job_id), "status": "pending", "project_id": str(pid)}

    def get_job_status(self, project_id: str, job_id: str) -> Dict[str, Any]:
        """Get the status of an analysis job.

        Returns:
            Dict with status, progress, timestamps, errors
        """
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT job_id, project_id, status,
                           total_entry_points, completed_entry_points,
                           created_at, started_at, completed_at,
                           error_details, retry_count
                    FROM deep_analysis_jobs
                    WHERE job_id = :jid AND project_id = :pid
                """),
                {"jid": UUID(job_id), "pid": UUID(project_id)},
            )
            row = result.fetchone()

        if not row:
            return {"error": "Job not found"}

        return {
            "job_id": str(row.job_id),
            "project_id": str(row.project_id),
            "status": row.status,
            "progress": {
                "total": row.total_entry_points or 0,
                "completed": row.completed_entry_points or 0,
            },
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "retry_count": row.retry_count,
            "errors": row.error_details,
        }

    def get_entry_points(self, project_id: str) -> List[Dict[str, Any]]:
        """Synchronously detect entry points for a project.

        Does NOT require a running analysis job — useful for previewing
        what the analysis will cover.

        Returns:
            List of entry point dicts
        """
        self._ensure_tracer()
        eps = self._tracer.detect_entry_points(project_id)
        return [
            {
                "unit_id": ep.unit_id,
                "name": ep.name,
                "qualified_name": ep.qualified_name,
                "file_path": ep.file_path,
                "entry_type": ep.entry_type.value,
                "language": ep.language,
                "detected_by": ep.detected_by,
            }
            for ep in eps
        ]

    def get_analysis_results(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all analysis results for a project (summaries only).

        Returns:
            List of analysis summary dicts (no full result_json)
        """
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT a.analysis_id, a.entry_unit_id, a.entry_type,
                           a.tier, a.total_units, a.total_tokens,
                           a.confidence_score, a.coverage_pct,
                           a.narrative, a.schema_version, a.analyzed_at,
                           u.name AS entry_name, u.qualified_name AS entry_qualified,
                           f.file_path AS entry_file
                    FROM deep_analyses a
                    JOIN code_units u ON a.entry_unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE a.project_id = :pid
                    ORDER BY f.file_path, u.name
                """),
                {"pid": UUID(project_id)},
            )
            rows = result.fetchall()

        return [
            {
                "analysis_id": str(row.analysis_id),
                "entry_unit_id": str(row.entry_unit_id),
                "entry_name": row.entry_name,
                "entry_qualified_name": row.entry_qualified,
                "entry_file": row.entry_file,
                "entry_type": row.entry_type,
                "tier": row.tier,
                "total_units": row.total_units,
                "total_tokens": row.total_tokens,
                "confidence_score": row.confidence_score,
                "coverage_pct": row.coverage_pct,
                "narrative": row.narrative,
                "schema_version": row.schema_version,
                "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
            }
            for row in rows
        ]

    def get_chain_detail(
        self,
        project_id: str,
        analysis_id: str,
    ) -> Dict[str, Any]:
        """Get full detail for a single analysis.

        Returns:
            Full analysis dict including result_json with evidence refs
        """
        import json

        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT a.*, u.name AS entry_name,
                           u.qualified_name AS entry_qualified,
                           f.file_path AS entry_file
                    FROM deep_analyses a
                    JOIN code_units u ON a.entry_unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE a.analysis_id = :aid AND a.project_id = :pid
                """),
                {"aid": UUID(analysis_id), "pid": UUID(project_id)},
            )
            row = result.fetchone()

        if not row:
            return {"error": "Analysis not found"}

        # Parse result_json
        result_data = {}
        if row.result_json:
            try:
                result_data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
            except (json.JSONDecodeError, TypeError):
                result_data = {"parse_error": "Could not parse result_json"}

        # Get analysis_units
        with self._db.get_session() as session:
            units_result = session.execute(
                text("""
                    SELECT au.unit_id, au.min_depth, au.path_count,
                           u.name, u.qualified_name, u.unit_type,
                           f.file_path
                    FROM analysis_units au
                    JOIN code_units u ON au.unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE au.analysis_id = :aid
                    ORDER BY au.min_depth, u.name
                """),
                {"aid": UUID(analysis_id)},
            )
            units = [
                {
                    "unit_id": str(r.unit_id),
                    "name": r.name,
                    "qualified_name": r.qualified_name,
                    "unit_type": r.unit_type,
                    "file_path": r.file_path,
                    "min_depth": r.min_depth,
                    "path_count": r.path_count,
                }
                for r in units_result.fetchall()
            ]

        return {
            "analysis_id": str(row.analysis_id),
            "entry_point": {
                "unit_id": str(row.entry_unit_id),
                "name": row.entry_name,
                "qualified_name": row.entry_qualified,
                "file_path": row.entry_file,
                "entry_type": row.entry_type,
            },
            "tier": row.tier,
            "total_units": row.total_units,
            "total_tokens": row.total_tokens,
            "confidence_score": row.confidence_score,
            "coverage_pct": row.coverage_pct,
            "narrative": row.narrative,
            "schema_version": row.schema_version,
            "prompt_version": row.prompt_version,
            "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
            "result": result_data,
            "units": units,
        }
