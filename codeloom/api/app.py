"""FastAPI application factory for CodeLoom.

Creates and configures the FastAPI app with CORS, auth,
and all route modules registered.
"""

import logging
import os
from pathlib import Path

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
    mcp_server=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        pipeline: LocalRAGPipeline instance
        db_manager: DatabaseManager instance
        project_manager: ProjectManager instance
        code_ingestion: CodeIngestionService instance (optional)
        conversation_store: ConversationStore instance (optional)
        mcp_server: CodeLoomMCPServer instance (optional) — mounted at /mcp for
            streamable HTTP transport when provided.

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

    # CORS for React dev server and production reverse proxy
    cors_env = os.getenv("CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins != ["*"] else [],
        allow_origin_regex=r".*" if cors_origins == ["*"] else None,
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
    app.state.mcp_server = mcp_server

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
    from .routes.reverse_engineering import router as reveng_router

    app.include_router(auth_router, prefix="/api")
    app.include_router(projects_router, prefix="/api")
    app.include_router(code_chat_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")
    app.include_router(graph_router, prefix="/api")
    app.include_router(migration_router, prefix="/api")
    app.include_router(understanding_router, prefix="/api")
    app.include_router(diagrams_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")
    app.include_router(reveng_router, prefix="/api")

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "service": "codeloom"}

    # Mount MCP streamable HTTP transport for remote Claude Code access
    if mcp_server is not None:
        try:
            from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
            from starlette.requests import Request
            from starlette.responses import Response

            session_manager = StreamableHTTPSessionManager(
                app=mcp_server.server,
                stateless=True,
            )
            app.state.mcp_session_manager = session_manager

            @app.on_event("startup")
            async def start_mcp_session_manager():
                """Start the MCP session manager task group."""
                import asyncio
                asyncio.create_task(_run_mcp_manager(session_manager))

            async def _run_mcp_manager(mgr):
                import asyncio as _aio
                async with mgr.run():
                    logger.info("MCP session manager running")
                    while True:
                        await _aio.sleep(3600)

            @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
            @app.api_route("/mcp/{path:path}", methods=["GET", "POST", "DELETE"])
            async def mcp_endpoint(request: Request):
                """Forward requests to MCP StreamableHTTPSessionManager."""
                import asyncio
                for _ in range(10):
                    try:
                        # handle_request sends its own response via ASGI send
                        await session_manager.handle_request(
                            request.scope, request.receive, request._send
                        )
                        # Return None — response already sent by handle_request
                        return
                    except RuntimeError:
                        await asyncio.sleep(0.5)
                return Response("MCP session manager not ready", status_code=503)

            logger.info("MCP streamable HTTP transport mounted at /mcp")
        except ImportError:
            logger.warning(
                "mcp.server.streamable_http_manager not available — "
                "MCP HTTP transport not mounted. Install mcp[http] for HTTP support."
            )
        except Exception as exc:
            logger.warning("Failed to mount MCP HTTP transport: %s", exc)

    logger.info("FastAPI app created with all routes registered")

    # ── Serve built frontend (production / Docker) ───────────────────
    # When frontend/dist exists (built by Docker multi-stage build),
    # serve the static assets and provide SPA fallback for client-side
    # routing.  In local dev, Vite dev server handles this instead.
    _dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if _dist.is_dir():
        from starlette.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        # Mount hashed assets at /assets (JS, CSS, images from Vite build)
        _assets = _dist / "assets"
        if _assets.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_assets)),
                name="frontend-assets",
            )

        # SPA catch-all: any path not matched by API routes → index.html
        _index = _dist / "index.html"

        @app.get("/{full_path:path}")
        async def _spa_fallback(full_path: str):
            """Serve index.html for all non-API routes (SPA client-side routing)."""
            # Check if a static file exists at the requested path
            candidate = _dist / full_path
            if candidate.is_file() and full_path:
                return FileResponse(str(candidate))
            return FileResponse(str(_index))

        logger.info("Serving frontend from %s", _dist)

    return app

