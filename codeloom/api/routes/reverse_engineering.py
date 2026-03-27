"""Reverse Engineering Documentation API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_current_user, get_project_manager, get_reverse_engineering_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reverse-engineering"])


@router.post("/reverse-engineer/{project_id}/generate", status_code=202)
async def generate_doc(
    project_id: str,
    chapters: Optional[list[int]] = Query(None, description="Chapter numbers to generate (1-9). Omit for all."),
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Generate reverse engineering documentation for a project."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.generate(project_id, chapters=chapters)
    if "error" in result and result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/reverse-engineer/{project_id}/docs")
async def list_docs(
    project_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """List all reverse engineering documents for a project."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    docs = svc.list_docs(project_id)
    return {"docs": docs, "count": len(docs)}


@router.get("/reverse-engineer/{project_id}/doc/latest")
async def get_latest_doc(
    project_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Get the latest reverse engineering document for a project."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.get_document(project_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/reverse-engineer/{project_id}/doc/{doc_id}")
async def get_doc(
    project_id: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Get a specific reverse engineering document."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.get_doc_by_id(doc_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/reverse-engineer/{project_id}/doc/{doc_id}/status")
async def get_doc_status(
    project_id: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Get the generation status of a document."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.get_status(doc_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/reverse-engineer/{project_id}/doc/{doc_id}/chapter/{chapter_num}")
async def get_chapter(
    project_id: str,
    doc_id: str,
    chapter_num: int,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Get a single chapter from a document."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not 1 <= chapter_num <= 9:
        raise HTTPException(status_code=400, detail="Chapter number must be between 1 and 9")

    result = svc.get_chapter(doc_id, chapter_num)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/reverse-engineer/{project_id}/validate")
async def validate_doc_latest(
    project_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Validate the latest reverse engineering document against source code ground truth."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.validate_document(project_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/reverse-engineer/{project_id}/validate/{doc_id}")
async def validate_doc(
    project_id: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
    svc=Depends(get_reverse_engineering_service),
    pm=Depends(get_project_manager),
):
    """Validate a specific reverse engineering document against source code ground truth."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = svc.validate_document(project_id, doc_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result
