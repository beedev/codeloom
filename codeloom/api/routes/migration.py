"""Migration API routes — MVP-centric migration pipeline endpoints.

Plan-level endpoints:
  POST /plan — create plan
  GET  /plans — list plans
  GET  /{plan_id} — plan status with MVPs
  DELETE /{plan_id} — delete plan

Asset inventory (between plan creation and discovery):
  GET  /{plan_id}/asset-inventory — file-type breakdown with strategy suggestions
  POST /{plan_id}/asset-inventory/refine — LLM-refine strategy suggestions
  POST /{plan_id}/asset-strategies — save user-confirmed strategies

Discovery + MVP management:
  POST /{plan_id}/discover — run clustering + Discovery LLM
  GET  /{plan_id}/mvps — list MVPs
  GET  /{plan_id}/mvps/{mvp_id} — MVP detail with phases + architecture mapping
  PUT  /{plan_id}/mvps/{mvp_id} — update MVP (name, units, priority)
  POST /{plan_id}/mvps/merge — merge multiple MVPs
  POST /{plan_id}/mvps/{mvp_id}/split — split units into new MVP
  POST /{plan_id}/mvps/{mvp_id}/analyze — on-demand deep analysis (V2)
  POST /{plan_id}/mvps/create-phases — create per-MVP phases

Phase execution (plan-level and per-MVP):
  POST /{plan_id}/phase/{N}/execute?mvp_id=X — execute phase
  GET  /{plan_id}/phase/{N}?mvp_id=X — get phase output
  POST /{plan_id}/phase/{N}/approve?mvp_id=X — approve phase
  POST /{plan_id}/phase/{N}/reject?mvp_id=X — reject phase

Diff + Download:
  GET  /{plan_id}/phase/{N}/diff-context?mvp_id=X — paired source + migrated files
  GET  /{plan_id}/phase/{N}/download?mvp_id=X&format=zip|single&file_path=... — download files
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from ..deps import get_current_user, require_admin, require_editor, get_migration_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/migration", tags=["migration"])


# ── Request/Response models ──────────────────────────────────────────────

class CreatePlanRequest(BaseModel):
    source_project_id: str
    target_brief: str
    target_stack: dict  # {"languages": [...], "frameworks": [...]}
    constraints: dict | None = None
    migration_type: str = "framework_migration"  # version_upgrade, framework_migration, rewrite


class DiscoveryRequest(BaseModel):
    clustering_params: dict | None = None  # Optional override for clustering


class UpdateMvpRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    unit_ids: list[str] | None = None
    file_ids: list[str] | None = None
    priority: int | None = None


class MergeMvpsRequest(BaseModel):
    mvp_ids: list[int]
    new_name: str | None = None


class SplitMvpRequest(BaseModel):
    unit_ids: list[str]
    new_name: str


class RejectPhaseRequest(BaseModel):
    feedback: str | None = None


class SaveAssetStrategiesRequest(BaseModel):
    strategies: dict  # {lang: {strategy, target}}


# ── Plan Endpoints ──────────────────────────────────────────────────────

@router.post("/plan")
async def create_plan(
    data: CreatePlanRequest,
    user: dict = Depends(require_admin),
    engine=Depends(get_migration_engine),
):
    """Create a new migration plan with 2 plan-level phases."""
    try:
        plan = engine.create_plan(
            user_id=user["user_id"],
            source_project_id=data.source_project_id,
            target_brief=data.target_brief,
            target_stack=data.target_stack,
            constraints=data.constraints,
            migration_type=data.migration_type,
        )
        return plan
    except Exception as e:
        logger.error(f"Failed to create migration plan: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plans")
async def list_plans(
    project_id: str | None = None,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """List migration plans, optionally filtered by project."""
    return engine.list_plans(user_id=user["user_id"], project_id=project_id)


@router.get("/{plan_id}")
async def get_plan(
    plan_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Get full plan status with phase summaries and MVPs."""
    try:
        return engine.get_plan_status(plan_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{plan_id}")
async def delete_plan(
    plan_id: str,
    user: dict = Depends(require_admin),
    engine=Depends(get_migration_engine),
):
    """Delete/abandon a migration plan."""
    try:
        engine.delete_plan(plan_id)
        return {"status": "deleted", "plan_id": plan_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Asset Inventory ────────────────────────────────────────────────────

@router.get("/{plan_id}/asset-inventory")
async def get_asset_inventory(
    plan_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Return file-type breakdown with rule-based strategy suggestions."""
    try:
        return engine.get_asset_inventory(plan_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{plan_id}/asset-inventory/refine")
async def refine_asset_inventory(
    plan_id: str,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """LLM-refine strategy suggestions (async enrichment)."""
    try:
        return engine.refine_asset_strategies(plan_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Asset refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plan_id}/asset-strategies")
async def save_asset_strategies(
    plan_id: str,
    data: SaveAssetStrategiesRequest,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Save user-confirmed per-file-type migration strategies."""
    try:
        return engine.save_asset_strategies(plan_id, data.strategies)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Discovery + MVP Management ──────────────────────────────────────────

@router.post("/{plan_id}/discover")
async def run_discovery(
    plan_id: str,
    data: DiscoveryRequest | None = None,
    user: dict = Depends(require_admin),
    engine=Depends(get_migration_engine),
):
    """Run MVP clustering + SP analysis + Phase 1 LLM.

    This is the main entry point after creating a plan. It:
    1. Runs the MvpClusterer to identify functional MVPs
    2. Persists FunctionalMVP rows
    3. Executes the Phase 1 (Discovery) LLM call
    """
    try:
        params = data.clustering_params if data else None
        return engine.run_discovery(plan_id, clustering_params=params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get("/{plan_id}/mvps")
async def list_mvps(
    plan_id: str,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """List all MVPs for a plan."""
    try:
        status = engine.get_plan_status(plan_id)
        return status.get("mvps", [])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{plan_id}/mvps/{mvp_id}")
async def get_mvp_detail(
    plan_id: str,
    mvp_id: int,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Get detailed MVP info with all phases."""
    try:
        return engine.get_mvp_detail(plan_id, mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{plan_id}/mvps/{mvp_id}")
async def update_mvp(
    plan_id: str,
    mvp_id: int,
    data: UpdateMvpRequest,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Update an MVP's name, description, units, or priority.

    Use after discovery to refine auto-detected clusters before
    creating per-MVP phases.
    """
    try:
        updates = data.model_dump(exclude_none=True)
        if not updates:
            raise ValueError("No updates provided")
        return engine.update_mvp(plan_id, mvp_id, updates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{plan_id}/mvps/merge")
async def merge_mvps(
    plan_id: str,
    data: MergeMvpsRequest,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Merge multiple MVPs into one.

    The first MVP (by priority) absorbs the others.
    """
    try:
        return engine.merge_mvps(plan_id, data.mvp_ids, new_name=data.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{plan_id}/mvps/{mvp_id}/split")
async def split_mvp(
    plan_id: str,
    mvp_id: int,
    data: SplitMvpRequest,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Split units from one MVP into a new MVP."""
    try:
        return engine.split_mvp(plan_id, mvp_id, data.unit_ids, data.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{plan_id}/mvps/{mvp_id}/analyze")
async def analyze_mvp(
    plan_id: str,
    mvp_id: int,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Run on-demand deep analysis for an MVP (non-blocking).

    V2 pipeline. Launches analysis in background via FastAPI BackgroundTasks.
    Result is stored on FunctionalMVP.analysis_output.
    Poll GET /{plan_id}/mvps/{mvp_id}/analysis-status for progress.
    """
    try:
        # Set status to "analyzing" immediately
        pid = __import__("uuid").UUID(plan_id)
        from codeloom.core.db.models import FunctionalMVP
        with engine._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")
            mvp.analysis_status = "analyzing"
            mvp.analysis_error = None

        # Launch background task
        background_tasks.add_task(engine.analyze_mvp, plan_id, mvp_id)

        return {"status": "analyzing", "mvp_id": mvp_id, "plan_id": plan_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"MVP analysis launch failed: {e}")
        raise HTTPException(status_code=500, detail=f"MVP analysis launch failed: {str(e)}")


@router.post("/{plan_id}/mvps/analyze-all")
async def analyze_all_mvps(
    plan_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Run deep analysis for all MVPs that lack analysis_output (non-blocking).

    Launches batch analysis in background. Each MVP's progress is tracked
    via FunctionalMVP.analysis_status. Poll individual MVP status endpoints.
    """
    try:
        pid = __import__("uuid").UUID(plan_id)
        with engine._db.get_session() as session:
            from codeloom.core.db.models import FunctionalMVP
            from sqlalchemy import or_, cast, String
            mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid,
                or_(
                    FunctionalMVP.analysis_output.is_(None),
                    cast(FunctionalMVP.analysis_output, String) == 'null',
                ),
            ).order_by(FunctionalMVP.priority).all()
            mvp_rows = [
                {"mvp_id": m.mvp_id, "name": m.name}
                for m in mvps
            ]

            # Set all target MVPs to "analyzing" immediately
            for m in mvps:
                m.analysis_status = "analyzing"
                m.analysis_error = None

        if not mvp_rows:
            return {"status": "no_work", "analyzed": 0, "total": 0}

        # Launch background task
        background_tasks.add_task(engine._analyze_all_mvps, plan_id, mvp_rows)

        return {
            "status": "analyzing",
            "total": len(mvp_rows),
            "mvp_ids": [r["mvp_id"] for r in mvp_rows],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch MVP analysis launch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch MVP analysis launch failed: {str(e)}")


@router.get("/{plan_id}/mvps/{mvp_id}/analysis-status")
async def get_analysis_status(
    plan_id: str,
    mvp_id: int,
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Poll for background analysis completion status.

    Returns the current analysis_status and analysis_at timestamp
    for a specific MVP. Use this to check if a background analysis
    launched by POST .../analyze has completed.
    """
    try:
        pid = __import__("uuid").UUID(plan_id)
        from codeloom.core.db.models import FunctionalMVP
        with engine._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")

            return {
                "mvp_id": mvp_id,
                "plan_id": plan_id,
                "analysis_status": mvp.analysis_status or "pending",
                "analysis_at": mvp.analysis_at.isoformat() if mvp.analysis_at else None,
                "analysis_error": mvp.analysis_error,
                "has_output": mvp.analysis_output is not None,
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{plan_id}/mvps/create-phases")
async def create_mvp_phases(
    plan_id: str,
    user: dict = Depends(require_admin),
    engine=Depends(get_migration_engine),
):
    """Create per-MVP phases (3-6) for all discovered MVPs.

    Call after Phase 1 is approved and MVPs are finalized.
    """
    try:
        return engine.create_mvp_phases(plan_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Phase Execution (Plan-Level + Per-MVP) ──────────────────────────────

@router.post("/{plan_id}/phase/{phase_number}/execute")
async def execute_phase(
    plan_id: str,
    phase_number: int,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases (3-6)"),
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Execute a migration phase.

    For plan-level phases (1-2): omit mvp_id.
    For per-MVP phases (3-6): provide mvp_id.
    Previous phase must be approved first.
    """
    try:
        return engine.execute_phase(plan_id, phase_number, mvp_id=mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Phase execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Phase execution failed: {str(e)}")


@router.get("/{plan_id}/phase/{phase_number}")
async def get_phase_output(
    plan_id: str,
    phase_number: int,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases (3-6)"),
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Get detailed phase output."""
    try:
        return engine.get_phase_output(plan_id, phase_number, mvp_id=mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{plan_id}/phase/{phase_number}/approve")
async def approve_phase(
    plan_id: str,
    phase_number: int,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases (3-6)"),
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Approve a completed phase. Unlocks the next phase."""
    try:
        return engine.approve_phase(plan_id, phase_number, mvp_id=mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{plan_id}/phase/{phase_number}/reject")
async def reject_phase(
    plan_id: str,
    phase_number: int,
    data: RejectPhaseRequest | None = None,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases (3-6)"),
    user: dict = Depends(require_editor),
    engine=Depends(get_migration_engine),
):
    """Reject a completed phase with optional feedback.

    Resets the phase to 'pending' so it can be re-executed.
    """
    try:
        feedback = data.feedback if data else None
        return engine.reject_phase(plan_id, phase_number, mvp_id=mvp_id, feedback=feedback)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Framework Doc Enrichment ──────────────────────────────────────────

@router.post("/{plan_id}/enrich-docs")
async def enrich_framework_docs(
    plan_id: str,
    user: dict = Depends(require_admin),
    engine=Depends(get_migration_engine),
):
    """Re-fetch framework docs for the plan's target stack.

    Uses Tavily search to retrieve best-practice documentation for each
    target framework, informed by detected source patterns.
    Results are cached on the plan for use by all subsequent phases.
    """
    try:
        return engine.enrich_framework_docs(plan_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Doc enrichment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Doc enrichment failed: {str(e)}")


# ── Diff + Download ──────────────────────────────────────────────────

@router.get("/{plan_id}/phase/{phase_number}/diff-context")
async def get_diff_context(
    plan_id: str,
    phase_number: int,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases"),
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Get diff context: original source paired with migrated output files.

    Returns source_files (reconstructed from CodeUnits), migrated_files
    (from phase output), and a best-effort file_mapping.
    """
    try:
        return engine.get_diff_context(plan_id, phase_number, mvp_id=mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{plan_id}/phase/{phase_number}/download")
async def download_phase_files(
    plan_id: str,
    phase_number: int,
    mvp_id: Optional[int] = Query(None, description="MVP ID for per-MVP phases"),
    format: str = Query("zip", description="'zip' for all files, 'single' for one file"),
    file_path: Optional[str] = Query(None, description="Required when format=single"),
    user: dict = Depends(get_current_user),
    engine=Depends(get_migration_engine),
):
    """Download phase output files as ZIP or individual file."""
    try:
        content, content_type, filename = engine.get_phase_files_download(
            plan_id, phase_number, mvp_id=mvp_id, fmt=format, file_path=file_path,
        )
        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
