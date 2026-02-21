"""Diagram API routes — per-MVP UML diagram generation endpoints.

  GET  /{plan_id}/mvps/{mvp_id}/diagrams           → list available diagrams
  GET  /{plan_id}/mvps/{mvp_id}/diagrams/{type}     → get/generate diagram
  POST /{plan_id}/mvps/{mvp_id}/diagrams/{type}/refresh → force regenerate
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user, get_diagram_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/migration", tags=["diagrams"])


@router.get("/{plan_id}/mvps/{mvp_id}/diagrams")
async def list_diagrams(
    plan_id: str,
    mvp_id: int,
    user: dict = Depends(get_current_user),
    diagram_service=Depends(get_diagram_service),
):
    """List available diagram types and their cache status for an MVP."""
    try:
        return diagram_service.list_available(plan_id, mvp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{plan_id}/mvps/{mvp_id}/diagrams/{diagram_type}")
async def get_diagram(
    plan_id: str,
    mvp_id: int,
    diagram_type: str,
    user: dict = Depends(get_current_user),
    diagram_service=Depends(get_diagram_service),
):
    """Get or generate a specific diagram for an MVP.

    Structural diagrams (class, package, component) are generated on-demand.
    Behavioral diagrams (sequence, usecase, activity, deployment) are
    LLM-generated and cached.
    """
    try:
        return diagram_service.get_diagram(plan_id, mvp_id, diagram_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/{plan_id}/mvps/{mvp_id}/diagrams/{diagram_type}/refresh")
async def refresh_diagram(
    plan_id: str,
    mvp_id: int,
    diagram_type: str,
    user: dict = Depends(get_current_user),
    diagram_service=Depends(get_diagram_service),
):
    """Force regenerate a diagram, bypassing cache."""
    try:
        return diagram_service.get_diagram(plan_id, mvp_id, diagram_type, force_refresh=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
