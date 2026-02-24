"""FastAPI application factory for CodeLoom.

Creates and configures the FastAPI app with CORS, auth,
and all route modules registered.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

logger = logging.getLogger(__name__)


def create_app(
    pipeline,
    db_manager,
    project_manager,
    code_ingestion=None,
    conversation_store=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        pipeline: LocalRAGPipeline instance
        db_manager: DatabaseManager instance
        project_manager: ProjectManager instance
        code_ingestion: CodeIngestionService instance (optional)
        conversation_store: ConversationStore instance (optional)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="CodeLoom API",
        description="Code intelligence and migration platform",
        version="0.1.0",
    )

    # Session middleware (required for auth sessions)
    secret_key = os.getenv("FLASK_SECRET_KEY", "codeloom-dev-secret-change-me")
    app.add_middleware(SessionMiddleware, secret_key=secret_key)

    # CORS for React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store shared dependencies on app state
    app.state.pipeline = pipeline
    app.state.db_manager = db_manager
    app.state.project_manager = project_manager
    app.state.code_ingestion = code_ingestion
    app.state.conversation_store = conversation_store

    # Register routers
    from .routes.fastapi_auth import router as auth_router
    from .routes.projects import router as projects_router
    from .routes.code_chat import router as code_chat_router
    from .routes.fastapi_settings import router as settings_router
    from .routes.graph import router as graph_router
    from .routes.migration import router as migration_router
    from .routes.understanding import router as understanding_router
    from .routes.diagrams import router as diagrams_router
    from .routes.analytics import router as analytics_router

    app.include_router(auth_router, prefix="/api")
    app.include_router(projects_router, prefix="/api")
    app.include_router(code_chat_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")
    app.include_router(graph_router, prefix="/api")
    app.include_router(migration_router, prefix="/api")
    app.include_router(understanding_router, prefix="/api")
    app.include_router(diagrams_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "service": "codeloom"}

    logger.info("FastAPI app created with all routes registered")
    return app
