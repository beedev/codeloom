"""Understanding API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user, get_db_manager, get_understanding_engine, get_project_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["understanding"])


@router.post("/understanding/{project_id}/analyze", status_code=202)
async def start_analysis(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("asg_status") != "complete":
        raise HTTPException(status_code=400, detail="Project ASG not complete")
    if project.get("deep_analysis_status") == "running":
        raise HTTPException(status_code=409, detail="Analysis already running")

    result = engine.start_analysis(project_id, user["user_id"])
    return result


@router.get("/understanding/{project_id}/status/{job_id}")
async def get_status(
    project_id: str,
    job_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = engine.get_job_status(project_id, job_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/understanding/{project_id}/entry-points")
async def get_entry_points(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    eps = engine.get_entry_points(project_id)
    return {"entry_points": eps, "count": len(eps)}


@router.get("/understanding/{project_id}/results")
async def get_results(
    project_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    analyses = engine.get_analysis_results(project_id)

    # Calculate coverage
    from sqlalchemy import text
    from uuid import UUID
    db = engine._db
    with db.get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    (SELECT COUNT(DISTINCT unit_id)
                     FROM analysis_units WHERE project_id = :pid) AS analyzed,
                    (SELECT COUNT(*) FROM code_units
                     WHERE project_id = :pid
                       AND unit_type IN ('function', 'method', 'class')) AS total
            """),
            {"pid": UUID(project_id)},
        )
        row = result.fetchone()
        coverage = (row.analyzed / row.total * 100) if row.total > 0 else 0.0

    return {"analyses": analyses, "count": len(analyses), "coverage_pct": round(coverage, 1)}


@router.get("/understanding/{project_id}/chain/{analysis_id}")
async def get_chain_detail(
    project_id: str,
    analysis_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_understanding_engine),
    pm=Depends(get_project_manager),
):
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = engine.get_chain_detail(project_id, analysis_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
