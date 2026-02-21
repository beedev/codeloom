"""FastAPI settings routes for CodeLoom.

Provides runtime configuration for LLM models and reranker.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..deps import get_current_user, require_admin, get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


# ---- Request/Response models ------------------------------------------------


class ModelUpdateRequest(BaseModel):
    provider: str
    model: str


class RerankerUpdateRequest(BaseModel):
    enabled: bool = True
    model: str = "base"
    top_n: int = 10


class LLMOverrideConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


class MigrationLLMRequest(BaseModel):
    understanding_llm: Optional[LLMOverrideConfig] = None
    generation_llm: Optional[LLMOverrideConfig] = None


# ---- Models endpoints --------------------------------------------------------


@router.get("/models")
async def get_models(
    user: dict = Depends(require_admin),
    pipeline=Depends(get_pipeline),
):
    """Return available LLM providers/models and current selection."""
    from codeloom.setting.setting import get_models_settings

    settings = get_models_settings()

    providers = []
    for name, cfg in settings.providers.items():
        if not cfg.enabled:
            continue
        has_key = not cfg.requires_api_key or settings.has_api_key(name)
        if not has_key:
            continue
        providers.append({
            "name": name,
            "type": cfg.type,
            "models": [
                {
                    "name": m.name,
                    "display_name": m.display_name or m.name,
                }
                for m in cfg.models
                if m.enabled
            ],
        })

    current_model = pipeline.get_model_name() or settings.default_model
    current_provider = os.getenv("LLM_PROVIDER", settings.default_provider)

    return {
        "providers": providers,
        "current": {
            "provider": current_provider,
            "model": current_model,
        },
        "default_provider": settings.default_provider,
        "default_model": settings.default_model,
    }


@router.post("/models")
async def set_model(
    data: ModelUpdateRequest,
    user: dict = Depends(require_admin),
    pipeline=Depends(get_pipeline),
):
    """Switch the active LLM provider and model at runtime."""
    from codeloom.setting.setting import get_models_settings

    settings = get_models_settings()

    # Validate provider exists and is enabled
    if data.provider not in settings.providers:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {data.provider}")
    if not settings.is_provider_enabled(data.provider):
        raise HTTPException(status_code=400, detail=f"Provider not enabled: {data.provider}")

    # Set the provider env var so set_model() picks the right backend
    os.environ["LLM_PROVIDER"] = data.provider

    # Switch model on the pipeline
    pipeline.set_model_name(data.model)
    pipeline.set_model()

    logger.info(f"LLM switched to {data.provider}/{data.model}")

    return {
        "success": True,
        "provider": data.provider,
        "model": data.model,
    }


# ---- Reranker endpoints -----------------------------------------------------


@router.get("/reranker")
async def get_reranker(
    user: dict = Depends(require_admin),
):
    """Return current reranker config and available models."""
    from codeloom.core.providers.reranker_provider import (
        get_reranker_config,
        list_available_models,
    )

    return {
        "config": get_reranker_config(),
        "available_models": list_available_models(),
    }


@router.post("/reranker")
async def set_reranker(
    data: RerankerUpdateRequest,
    user: dict = Depends(require_admin),
):
    """Update reranker configuration at runtime."""
    from codeloom.core.providers.reranker_provider import set_reranker_config

    result = set_reranker_config(
        model=data.model,
        enabled=data.enabled,
        top_n=data.top_n,
    )

    logger.info(f"Reranker updated: enabled={data.enabled}, model={data.model}")

    return {
        "success": True,
        "config": result,
    }


# ---- Migration LLM Override endpoints ----------------------------------------


@router.get("/migration-llm")
async def get_migration_llm(
    user: dict = Depends(require_admin),
):
    """Return current LLM override configuration for migration phases."""
    from codeloom.core.config.config_loader import get_llm_overrides_config

    overrides = get_llm_overrides_config()

    return {
        "understanding_llm": overrides.get("understanding_llm"),
        "generation_llm": overrides.get("generation_llm"),
    }


@router.post("/migration-llm")
async def set_migration_llm(
    data: MigrationLLMRequest,
    user: dict = Depends(require_admin),
):
    """Update LLM override configuration for migration phases.

    Changes are applied in-memory and written back to config/codeloom.yaml.
    Set a field to null to clear the override (use default LLM).
    """
    import yaml
    from codeloom.core.config.config_loader import (
        get_config_path,
        load_unified_config,
        reload_configs,
    )

    config_file = get_config_path() / "codeloom.yaml"
    if not config_file.exists():
        raise HTTPException(status_code=500, detail="Config file not found")

    # Read current config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}

    # Ensure migration.llm_overrides section exists
    if "migration" not in config:
        config["migration"] = {}
    if "llm_overrides" not in config["migration"]:
        config["migration"]["llm_overrides"] = {}

    overrides = config["migration"]["llm_overrides"]

    # Apply updates
    if data.understanding_llm is not None:
        overrides["understanding_llm"] = {
            "provider": data.understanding_llm.provider,
            "model": data.understanding_llm.model,
            "temperature": data.understanding_llm.temperature,
        }
    else:
        overrides["understanding_llm"] = None

    if data.generation_llm is not None:
        overrides["generation_llm"] = {
            "provider": data.generation_llm.provider,
            "model": data.generation_llm.model,
            "temperature": data.generation_llm.temperature,
        }
    else:
        overrides["generation_llm"] = None

    # Write back
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Clear caches so next read picks up changes
    reload_configs()

    logger.info("Migration LLM overrides updated: %s", data.model_dump())

    return {
        "success": True,
        "understanding_llm": overrides.get("understanding_llm"),
        "generation_llm": overrides.get("generation_llm"),
    }
