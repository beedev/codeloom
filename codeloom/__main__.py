import argparse
import logging
import os
import sys
from pathlib import Path

# Set threading env vars BEFORE importing libraries that use them
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow nested event loops (fixes LlamaIndex asyncio + threading conflicts)
import nest_asyncio
nest_asyncio.apply()

# Limit PyTorch to single thread per operation (prevents segfaults in multi-threaded server)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Limit ONNX Runtime threads (prevents segfaults when reranker called concurrently)
import onnxruntime as ort
ort.set_default_logger_severity(3)  # Reduce logging

import llama_index.core

from .pipeline import LocalRAGPipeline
from .ollama import run_ollama_server, is_port_open
from .core.db.db import DatabaseManager
from .core.project.project_manager import ProjectManager


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("psycopg2").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data/data"
UPLOAD_DIR = "uploads"


def _seed_default_roles(db_manager):
    """Seed default RBAC roles and assign admin role to the default admin user."""
    from .core.db.models import Role, UserRole, User

    _DEFAULT_ROLES = {
        "admin": ["manage_users", "manage_roles", "manage_projects", "create_project", "view_all", "edit_all"],
        "editor": ["view_assigned", "edit_assigned"],
        "viewer": ["view_assigned"],
    }

    with db_manager.get_session() as session:
        for role_name, permissions in _DEFAULT_ROLES.items():
            existing = session.query(Role).filter(Role.name == role_name).first()
            if not existing:
                session.add(Role(name=role_name, permissions=permissions))
                logger.info(f"Created default role: {role_name}")

        session.flush()

        # Assign admin role to the default 'admin' user if no roles assigned
        admin_user = session.query(User).filter(User.username == "admin").first()
        if admin_user:
            has_roles = session.query(UserRole).filter(
                UserRole.user_id == admin_user.user_id
            ).first()
            if not has_roles:
                admin_role = session.query(Role).filter(Role.name == "admin").first()
                if admin_role:
                    session.add(UserRole(
                        user_id=admin_user.user_id,
                        role_id=admin_role.role_id,
                    ))
                    logger.info("Assigned admin role to default admin user")


def main():
    """Main entry point for CodeLoom."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="CodeLoom - Code Intelligence Platform")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for Ollama server (localhost or host.docker.internal)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9005,
        help="Port for the API server"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (auto-reload)"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info(f"Starting CodeLoom - Host: {args.host}")

    # Ensure directories exist
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    upload_path = Path(UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)

    # Start Ollama server if running locally (not in Docker)
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    if "host.docker.internal" not in ollama_host:
        port_number = 11434
        if not is_port_open(port_number):
            logger.info("Starting Ollama server...")
            run_ollama_server()
    else:
        logger.info(f"Running in Docker - using external Ollama at {ollama_host}")

    # Initialize settings
    from .setting import get_settings
    settings = get_settings()

    logger.info("LlamaIndex verbose logging disabled")

    # Initialize pipeline with database support
    logger.info("Initializing RAG pipeline...")
    database_url = os.getenv("DATABASE_URL")
    pipeline = LocalRAGPipeline(host=args.host, database_url=database_url)

    # Use the pipeline's database managers
    db_manager = pipeline._db_manager
    project_manager = pipeline._project_manager

    # Ensure default user exists
    if project_manager:
        try:
            project_manager.ensure_default_user()
            logger.info("Project feature initialized successfully")
        except Exception as e:
            logger.error(f"Failed to ensure default user: {e}")
    else:
        logger.warning("DATABASE_URL not set. Project feature will be unavailable.")

    # Seed default RBAC roles
    if db_manager:
        try:
            _seed_default_roles(db_manager)
        except Exception as e:
            logger.error(f"Failed to seed default roles: {e}")

    # Initialize code ingestion service
    code_ingestion = None
    if db_manager and pipeline._vector_store:
        from .core.ingestion.code_ingestion import CodeIngestionService
        code_ingestion = CodeIngestionService(db_manager, pipeline._vector_store)
        logger.info("CodeIngestionService initialized")

    # Initialize conversation store
    conversation_store = None
    if db_manager:
        from .core.conversation.conversation_store import ConversationStore
        conversation_store = ConversationStore(db_manager)
        logger.info("ConversationStore initialized")

    # Build FastAPI app
    from .api.app import create_app
    app = create_app(
        pipeline=pipeline,
        db_manager=db_manager,
        project_manager=project_manager,
        code_ingestion=code_ingestion,
        conversation_store=conversation_store,
    )

    # Launch with uvicorn
    import uvicorn

    logger.info(f"Starting FastAPI server on http://0.0.0.0:{args.port}")
    print(f"\n  CodeLoom is running at: http://localhost:{args.port}")
    print(f"  API docs at: http://localhost:{args.port}/docs\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.debug,
    )


if __name__ == "__main__":
    main()
