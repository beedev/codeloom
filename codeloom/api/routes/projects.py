"""Project management API routes (FastAPI).

Provides CRUD operations, zip upload with ingestion,
and file/unit browsing for code projects.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from ..deps import (
    get_current_user,
    require_admin,
    get_project_manager,
    get_code_ingestion,
    get_db_manager,
    get_understanding_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

MAX_UPLOAD_SIZE_MB = 50


# ── Request/Response models ──────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ProjectResponse(BaseModel):
    project_id: str
    user_id: str
    name: str
    description: str
    primary_language: str | None = None
    languages: list[str] = []
    file_count: int = 0
    total_lines: int = 0
    ast_status: str = "pending"
    asg_status: str = "pending"
    deep_analysis_status: str = "none"
    source_type: str = "zip"
    source_url: str | None = None
    repo_branch: str | None = None
    last_synced_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class GitIngestRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class LocalIngestRequest(BaseModel):
    dir_path: str


class IngestionResponse(BaseModel):
    project_id: str
    files_processed: int
    files_skipped: int
    units_extracted: int
    chunks_created: int
    embeddings_stored: int
    errors: list[str]
    elapsed_seconds: float


# ── Background analysis auto-trigger ─────────────────────────────────────

def _auto_trigger_analysis(engine, project_id: str, user_id: str):
    """Background task: auto-trigger deep understanding after ingestion.

    Creates a DeepAnalysisJob row which the UnderstandingWorker daemon
    picks up automatically. Non-blocking — runs after upload response.
    """
    try:
        engine.start_analysis(project_id, user_id)
        logger.info(f"Auto-triggered deep analysis for project {project_id}")
    except Exception as e:
        logger.warning(f"Auto-trigger analysis failed for {project_id}: {e}")


# ── Routes ───────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectResponse)
async def create_project(
    data: ProjectCreate,
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
):
    """Create a new project."""
    try:
        project = pm.create_project(
            user_id=user["user_id"],
            name=data.name,
            description=data.description,
        )
        return ProjectResponse(**project)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """List all projects for the current user."""
    projects = pm.list_projects(user["user_id"])
    return [ProjectResponse(**p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """Get project details by ID."""
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse(**project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdate,
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """Update project name or description."""
    try:
        success = pm.update_project(
            project_id=project_id,
            name=data.name,
            description=data.description,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectResponse(**pm.get_project(project_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
):
    """Delete a project and all associated data."""
    success = pm.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True, "message": f"Project {project_id} deleted"}


@router.post("/{project_id}/upload", response_model=IngestionResponse)
async def upload_codebase(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
    ingestion=Depends(get_code_ingestion),
    understanding_engine=Depends(get_understanding_engine),
):
    """Upload a zip file and ingest the codebase.

    Validates file type and size, extracts, parses with AST,
    chunks, embeds, and stores in pgvector.
    Auto-triggers deep understanding analysis after successful ingestion.
    """
    # Validate project exists
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    # Save to temp file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".zip", prefix="codeloom_upload_", delete=False
        ) as tmp:
            content = await file.read()

            # Validate size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_UPLOAD_SIZE_MB}MB.",
                )

            tmp.write(content)
            temp_path = tmp.name

        # Run ingestion pipeline (synchronous for MVP)
        result = ingestion.ingest_zip(
            zip_path=temp_path,
            project_id=project_id,
            user_id=user["user_id"],
        )

        # Auto-trigger deep understanding analysis after successful ingestion
        if result.files_processed > 0 and not result.errors:
            background_tasks.add_task(
                _auto_trigger_analysis, understanding_engine, project_id, user["user_id"]
            )

        return IngestionResponse(
            project_id=result.project_id,
            files_processed=result.files_processed,
            files_skipped=result.files_skipped,
            units_extracted=result.units_extracted,
            chunks_created=result.chunks_created,
            embeddings_stored=result.embeddings_stored,
            errors=result.errors,
            elapsed_seconds=result.elapsed_seconds,
        )

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _to_ingestion_response(result) -> IngestionResponse:
    """Convert IngestionResult dataclass to IngestionResponse model."""
    return IngestionResponse(
        project_id=result.project_id,
        files_processed=result.files_processed,
        files_skipped=result.files_skipped,
        units_extracted=result.units_extracted,
        chunks_created=result.chunks_created,
        embeddings_stored=result.embeddings_stored,
        errors=result.errors,
        elapsed_seconds=result.elapsed_seconds,
    )


@router.post("/{project_id}/ingest/git", response_model=IngestionResponse)
async def ingest_from_git(
    project_id: str,
    data: GitIngestRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
    ingestion=Depends(get_code_ingestion),
    understanding_engine=Depends(get_understanding_engine),
):
    """Clone a git repository and ingest the codebase.

    Performs a shallow clone (--depth 1) of the specified branch,
    then runs the full ingestion pipeline.
    Auto-triggers deep understanding analysis after successful ingestion.
    """
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = ingestion.ingest_git(
        repo_url=data.repo_url,
        branch=data.branch,
        project_id=project_id,
        user_id=user["user_id"],
    )

    if result.errors and result.files_processed == 0:
        raise HTTPException(status_code=400, detail=result.errors[0])

    # Auto-trigger deep understanding analysis after successful ingestion
    if result.files_processed > 0:
        background_tasks.add_task(
            _auto_trigger_analysis, understanding_engine, project_id, user["user_id"]
        )

    return _to_ingestion_response(result)


@router.post("/{project_id}/ingest/local", response_model=IngestionResponse)
async def ingest_from_local(
    project_id: str,
    data: LocalIngestRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
    ingestion=Depends(get_code_ingestion),
    understanding_engine=Depends(get_understanding_engine),
):
    """Ingest a codebase from a local directory path.

    Reads files directly from the filesystem (no copy).
    The directory must exist on the server.
    Auto-triggers deep understanding analysis after successful ingestion.
    """
    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not os.path.isdir(data.dir_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {data.dir_path}")

    result = ingestion.ingest_local(
        dir_path=data.dir_path,
        project_id=project_id,
        user_id=user["user_id"],
    )

    if result.errors and result.files_processed == 0:
        raise HTTPException(status_code=400, detail=result.errors[0])

    # Auto-trigger deep understanding analysis after successful ingestion
    if result.files_processed > 0:
        background_tasks.add_task(
            _auto_trigger_analysis, understanding_engine, project_id, user["user_id"]
        )

    return _to_ingestion_response(result)


@router.post("/{project_id}/build-asg")
async def build_asg(
    project_id: str,
    user: dict = Depends(require_admin),
    pm=Depends(get_project_manager),
    db_manager=Depends(get_db_manager),
):
    """Build (or rebuild) the ASG for an existing project.

    Runs the ASG builder on code_units already in the database.
    No re-ingestion needed — just edge extraction.
    """
    import time

    project = pm.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("ast_status") != "complete":
        raise HTTPException(
            status_code=400,
            detail="Project must be fully parsed (ast_status=complete) before building ASG.",
        )

    from uuid import UUID
    from codeloom.core.db.models import Project, CodeEdge

    start = time.time()

    # Clear existing edges so rebuild is idempotent
    with db_manager.get_session() as session:
        session.query(CodeEdge).filter(
            CodeEdge.project_id == UUID(project_id)
        ).delete()
        proj = session.query(Project).filter(
            Project.project_id == UUID(project_id)
        ).first()
        if proj:
            proj.asg_status = "building"

    # Re-enrich class units with field metadata (for type_dep edges)
    from codeloom.core.asg_builder import ASGBuilder
    fields_updated = ASGBuilder.enrich_class_fields_from_db(db_manager, project_id)

    # Build edges
    asg = ASGBuilder(db_manager)
    try:
        edge_count = asg.build_edges(project_id)
        with db_manager.get_session() as session:
            proj = session.query(Project).filter(
                Project.project_id == UUID(project_id)
            ).first()
            if proj:
                proj.asg_status = "complete"
    except Exception as e:
        with db_manager.get_session() as session:
            proj = session.query(Project).filter(
                Project.project_id == UUID(project_id)
            ).first()
            if proj:
                proj.asg_status = "error"
        raise HTTPException(status_code=500, detail=f"ASG build failed: {e}")

    elapsed = time.time() - start
    return {
        "success": True,
        "project_id": project_id,
        "edges_created": edge_count,
        "fields_enriched": fields_updated,
        "elapsed_seconds": round(elapsed, 2),
    }


@router.get("/{project_id}/files")
async def list_files(
    project_id: str,
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """List all code files in a project."""
    files = pm.get_project_files(project_id)
    return {"files": files}


@router.get("/{project_id}/tree")
async def get_file_tree(
    project_id: str,
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """Get nested file tree for UI display."""
    tree = pm.get_file_tree(project_id)
    return tree


@router.get("/{project_id}/units")
async def list_units(
    project_id: str,
    file_id: str | None = None,
    user: dict = Depends(get_current_user),
    pm=Depends(get_project_manager),
):
    """List code units in a project, optionally filtered by file."""
    units = pm.get_project_units(project_id, file_id=file_id)
    return {"units": units}


@router.get("/{project_id}/file/{file_path:path}")
async def get_file_content(
    project_id: str,
    file_path: str,
    user: dict = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
):
    """Get source code of a specific file.

    Returns the file content from code_units (reconstructed from stored source).
    """
    from codeloom.core.db.models import CodeFile, CodeUnit
    from uuid import UUID

    with db_manager.get_session() as session:
        code_file = session.query(CodeFile).filter(
            CodeFile.project_id == UUID(project_id),
            CodeFile.file_path == file_path,
        ).first()

        if not code_file:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Get all units for this file, ordered by line number
        units = session.query(CodeUnit).filter(
            CodeUnit.file_id == code_file.file_id,
        ).order_by(CodeUnit.start_line).all()

        return {
            "file_id": str(code_file.file_id),
            "file_path": code_file.file_path,
            "language": code_file.language,
            "line_count": code_file.line_count,
            "units": [
                {
                    "unit_id": str(u.unit_id),
                    "unit_type": u.unit_type,
                    "name": u.name,
                    "start_line": u.start_line,
                    "end_line": u.end_line,
                    "signature": u.signature,
                    "source": u.source,
                }
                for u in units
            ],
        }
