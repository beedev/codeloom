"""Analytics API — unified project stats + LLM metrics.

Single endpoint that aggregates code breakdown, migration progress,
understanding coverage, and LLM gateway metrics into one response.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func

from ..deps import get_current_user, get_db_manager, get_pipeline
from ...core.db.models import (
    Project, CodeFile, CodeUnit, CodeEdge,
    MigrationPlan, MigrationPhase, FunctionalMVP,
    DeepAnalysis, QueryLog,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/analytics", tags=["analytics"])


@router.get("")
async def get_project_analytics(
    project_id: str,
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
    pipeline=Depends(get_pipeline),
):
    """Aggregated project analytics for dashboard consumption.

    Combines code structure stats, migration progress, understanding
    coverage, query history, and real-time LLM gateway metrics.
    """
    with db_manager.get_session() as session:
        # ── Project basics ────────────────────────────────────────
        project = session.query(Project).filter(
            Project.project_id == project_id
        ).first()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        project_info = {
            "name": project.name,
            "file_count": project.file_count or 0,
            "total_lines": project.total_lines or 0,
            "primary_language": project.primary_language,
            "languages": project.languages or [],
            "ast_status": project.ast_status,
            "asg_status": project.asg_status,
            "deep_analysis_status": project.deep_analysis_status,
        }

        # ── Code breakdown ────────────────────────────────────────
        units_by_type = dict(
            session.query(CodeUnit.unit_type, func.count())
            .filter(CodeUnit.project_id == project_id)
            .group_by(CodeUnit.unit_type)
            .all()
        )

        edges_by_type = dict(
            session.query(CodeEdge.edge_type, func.count())
            .filter(CodeEdge.project_id == project_id)
            .group_by(CodeEdge.edge_type)
            .all()
        )

        files_by_language = dict(
            session.query(CodeFile.language, func.count())
            .filter(CodeFile.project_id == project_id)
            .group_by(CodeFile.language)
            .all()
        )

        code_breakdown = {
            "units_by_type": units_by_type,
            "edges_by_type": edges_by_type,
            "files_by_language": files_by_language,
        }

        # ── Migration progress ────────────────────────────────────
        migration = _get_migration_stats(session, project_id)

        # ── Understanding coverage ────────────────────────────────
        analyses_count = (
            session.query(func.count(DeepAnalysis.analysis_id))
            .filter(DeepAnalysis.project_id == project_id)
            .scalar()
        ) or 0

        entry_points_detected = (
            session.query(func.count(func.distinct(DeepAnalysis.entry_unit_id)))
            .filter(DeepAnalysis.project_id == project_id)
            .scalar()
        ) or 0

        understanding = {
            "analyses_count": analyses_count,
            "entry_points_detected": entry_points_detected,
        }

        # ── Query history ─────────────────────────────────────────
        query_count = (
            session.query(func.count(QueryLog.log_id))
            .filter(QueryLog.project_id == project_id)
            .scalar()
        ) or 0

    # ── LLM metrics (from gateway, outside DB session) ────────
    llm_metrics = {}
    if hasattr(pipeline, 'get_llm_metrics'):
        llm_metrics = pipeline.get_llm_metrics()

    return {
        "project": project_info,
        "code_breakdown": code_breakdown,
        "migration": migration,
        "understanding": understanding,
        "queries": {"total": query_count},
        "llm": llm_metrics,
    }


def _get_migration_stats(session, project_id: str) -> dict:
    """Aggregate migration plan/phase/MVP stats for a project."""
    plans = (
        session.query(MigrationPlan)
        .filter(MigrationPlan.source_project_id == project_id)
        .all()
    )

    if not plans:
        return {"plan_count": 0, "active_plan": None}

    # Pick the most recent active plan (or fallback to latest)
    active = next(
        (p for p in plans if p.status == "in_progress"),
        plans[-1],
    )

    # MVP breakdown
    mvp_rows = (
        session.query(FunctionalMVP.status, func.count())
        .filter(FunctionalMVP.plan_id == active.plan_id)
        .group_by(FunctionalMVP.status)
        .all()
    )
    mvp_counts = dict(mvp_rows)
    mvp_total = sum(mvp_counts.values())

    # Phase breakdown
    phase_rows = (
        session.query(MigrationPhase.status, func.count())
        .filter(MigrationPhase.plan_id == active.plan_id)
        .group_by(MigrationPhase.status)
        .all()
    )
    phase_counts = dict(phase_rows)
    phase_total = sum(phase_counts.values())

    # Confidence average from phase metadata
    confidence_values = (
        session.query(MigrationPhase.phase_metadata)
        .filter(
            MigrationPhase.plan_id == active.plan_id,
            MigrationPhase.phase_metadata.isnot(None),
        )
        .all()
    )
    confidences = []
    gates_passed = 0
    gates_total = 0
    for (meta,) in confidence_values:
        if isinstance(meta, dict):
            conf = meta.get("phase_confidence")
            if conf is not None:
                confidences.append(float(conf))
            if "gates_all_passed" in meta:
                gates_total += 1
                if meta["gates_all_passed"]:
                    gates_passed += 1

    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
    gates_pass_rate = round(gates_passed / gates_total, 2) if gates_total > 0 else None

    return {
        "plan_count": len(plans),
        "active_plan": {
            "plan_id": str(active.plan_id),
            "status": active.status,
            "pipeline_version": f"v{active.pipeline_version}",
            "migration_lane": active.migration_lane_id,
            "mvps": {"total": mvp_total, **mvp_counts},
            "phases": {"total": phase_total, **phase_counts},
            "avg_confidence": avg_confidence,
            "gates_pass_rate": gates_pass_rate,
        },
    }
