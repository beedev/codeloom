"""Centralized configuration defaults.

All environment variable defaults are defined HERE and only here.
Import from this module instead of using os.getenv with inline defaults.
This prevents config conflicts where different files have different defaults
for the same environment variable.

Usage:
    from codeloom.core.defaults import LLM_PROVIDER, LLM_MODEL

    # Or for lazy access in functions:
    from codeloom.core import defaults
    provider = defaults.LLM_PROVIDER
"""

import os

from dotenv import load_dotenv

# Load .env FIRST so all os.getenv calls below pick up user configuration
# override=False ensures that variables explicitly set by Docker/system are NOT overwritten by .env
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "")  # Empty = auto-detect from provider

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Image & Vision
# ---------------------------------------------------------------------------
IMAGE_GENERATION_PROVIDER: str = os.getenv("IMAGE_GENERATION_PROVIDER", "gemini")
VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "gemini")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://codeloom:codeloom@localhost:5432/codeloom_dev",
)
PGVECTOR_EMBED_DIM: int = int(os.getenv("PGVECTOR_EMBED_DIM", "1536"))

# ---------------------------------------------------------------------------
# Server / Session
# ---------------------------------------------------------------------------
FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------
DISABLE_BACKGROUND_WORKERS: bool = (
    os.getenv("DISABLE_BACKGROUND_WORKERS", "false").lower() == "true"
)
RBAC_STRICT_MODE: bool = (
    os.getenv("RBAC_STRICT_MODE", "false").lower() == "true"
)
