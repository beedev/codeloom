"""FastAPI dependencies for CodeLoom.

Provides shared dependencies (auth, database, services) via FastAPI's
Depends() injection system.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)


async def get_db_manager(request: Request):
    """Get DatabaseManager from app state."""
    return request.app.state.db_manager


async def get_project_manager(request: Request):
    """Get ProjectManager from app state."""
    return request.app.state.project_manager


async def get_pipeline(request: Request):
    """Get pipeline from app state."""
    return request.app.state.pipeline


async def get_code_ingestion(request: Request):
    """Get CodeIngestionService from app state."""
    svc = request.app.state.code_ingestion
    if svc is None:
        raise HTTPException(status_code=503, detail="Code ingestion service not available")
    return svc


async def get_conversation_store(request: Request):
    """Get ConversationStore from app state."""
    return request.app.state.conversation_store


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency for authentication.

    Checks session for logged-in user. Returns user dict with roles or raises 401.
    """
    session = request.session
    user_id = session.get("user_id")
    username = session.get("username")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    roles = session.get("roles", [])
    return {"user_id": user_id, "username": username, "roles": roles}


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role. Returns user dict or raises 403."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def require_editor(user: dict = Depends(get_current_user)) -> dict:
    """Require editor or admin role. Returns user dict or raises 403."""
    roles = user.get("roles", [])
    if not any(r in roles for r in ("admin", "editor")):
        raise HTTPException(status_code=403, detail="Editor access required")
    return user


async def get_optional_user(request: Request) -> Optional[dict]:
    """Like get_current_user but returns None instead of 401."""
    session = request.session
    user_id = session.get("user_id")
    if not user_id:
        return None
    return {"user_id": user_id, "username": session.get("username"), "roles": session.get("roles", [])}


async def get_migration_engine(request: Request):
    """Get or create MigrationEngine from app state."""
    if not hasattr(request.app.state, 'migration_engine') or request.app.state.migration_engine is None:
        from codeloom.core.migration.engine import MigrationEngine
        request.app.state.migration_engine = MigrationEngine(
            request.app.state.db_manager,
            request.app.state.pipeline,
        )
    return request.app.state.migration_engine


async def get_diagram_service(request: Request):
    """Get or create DiagramService from app state."""
    if not hasattr(request.app.state, 'diagram_service') or request.app.state.diagram_service is None:
        from codeloom.core.diagrams import DiagramService
        request.app.state.diagram_service = DiagramService(
            request.app.state.db_manager,
            request.app.state.pipeline,
        )
    return request.app.state.diagram_service


async def get_understanding_engine(request: Request):
    """Get or create UnderstandingEngine from app state."""
    if not hasattr(request.app.state, 'understanding_engine') or \
       request.app.state.understanding_engine is None:
        from codeloom.core.understanding.engine import UnderstandingEngine
        request.app.state.understanding_engine = UnderstandingEngine(
            request.app.state.db_manager,
            request.app.state.pipeline,
        )
    return request.app.state.understanding_engine
