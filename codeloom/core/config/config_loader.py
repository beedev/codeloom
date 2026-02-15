"""Unified configuration loader for YAML config files.

Loads configuration from config/dbnotebook.yaml (unified) with fallback to
legacy separate files (raptor.yaml, ingestion.yaml, sql_chat.yaml).

Config files are loaded once at startup and cached.
"""

import logging
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """Get path to config directory.

    Searches in order:
    1. Relative to this file's project root
    2. Current working directory
    """
    # Try relative to this file (project structure)
    current_file = Path(__file__)
    # Go up: config_loader.py -> config -> core -> dbnotebook -> project_root
    project_root = current_file.parent.parent.parent.parent
    config_path = project_root / "config"

    if config_path.exists():
        return config_path

    # Fallback to current working directory
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    # Return project path even if doesn't exist (for error messages)
    return config_path


def _load_yaml_file(filename: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        filename: Name of the YAML file (e.g., 'dbnotebook.yaml')

    Returns:
        Parsed YAML as dictionary, empty dict if file not found
    """
    config_path = get_config_path() / filename

    if not config_path.exists():
        logger.debug(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            logger.debug(f"Loaded config from {config_path}")
            return data or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading {config_path}: {e}")
        return {}


# =============================================================================
# UNIFIED CONFIG (Primary)
# =============================================================================

@lru_cache(maxsize=1)
def load_unified_config() -> Dict[str, Any]:
    """Load unified configuration from config/dbnotebook.yaml.

    Returns:
        Dictionary with all configuration sections
    """
    config = _load_yaml_file("dbnotebook.yaml")
    if config:
        logger.info("Loaded unified config from dbnotebook.yaml")
    return config


def _get_unified_section(section: str) -> Dict[str, Any]:
    """Get a section from unified config with fallback mapping.

    Maps legacy section names to unified config structure:
    - 'raptor' -> unified['raptor']
    - 'ingestion' -> unified['ingestion']
    - 'sql_chat' -> unified['sql_chat']
    - 'retrieval' -> unified['retrieval']
    - 'llm' -> unified['llm']
    """
    unified = load_unified_config()
    return unified.get(section, {})


# =============================================================================
# LEGACY CONFIG LOADERS (Backward Compatibility)
# =============================================================================

@lru_cache(maxsize=1)
def load_raptor_config() -> Dict[str, Any]:
    """Load RAPTOR configuration.

    Tries unified config first, falls back to legacy raptor.yaml.
    """
    # Try unified config first
    unified = load_unified_config()
    if 'raptor' in unified:
        return unified['raptor']

    # Fallback to legacy file
    legacy = _load_yaml_file("raptor.yaml")
    if legacy:
        logger.info("Using legacy raptor.yaml (consider migrating to dbnotebook.yaml)")
    return legacy


@lru_cache(maxsize=1)
def load_ingestion_config() -> Dict[str, Any]:
    """Load ingestion configuration.

    Tries unified config first, falls back to legacy ingestion.yaml.
    """
    # Try unified config first
    unified = load_unified_config()
    if 'ingestion' in unified:
        return unified['ingestion']

    # Fallback to legacy file
    legacy = _load_yaml_file("ingestion.yaml")
    if legacy:
        logger.info("Using legacy ingestion.yaml (consider migrating to dbnotebook.yaml)")
    return legacy


@lru_cache(maxsize=1)
def load_sql_chat_config() -> Dict[str, Any]:
    """Load SQL Chat configuration.

    Tries unified config first, falls back to legacy sql_chat.yaml.
    """
    # Try unified config first
    unified = load_unified_config()
    if 'sql_chat' in unified:
        return unified['sql_chat']

    # Fallback to legacy file
    legacy = _load_yaml_file("sql_chat.yaml")
    if legacy:
        logger.info("Using legacy sql_chat.yaml (consider migrating to dbnotebook.yaml)")
    return legacy


@lru_cache(maxsize=1)
def load_retrieval_config() -> Dict[str, Any]:
    """Load retrieval configuration from unified config."""
    unified = load_unified_config()
    return unified.get('retrieval', {})


@lru_cache(maxsize=1)
def load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from unified config."""
    unified = load_unified_config()
    return unified.get('llm', {})


def reload_configs() -> None:
    """Clear config caches to reload from files.

    Call this if config files are modified at runtime.
    """
    load_unified_config.cache_clear()
    load_raptor_config.cache_clear()
    load_ingestion_config.cache_clear()
    load_sql_chat_config.cache_clear()
    load_retrieval_config.cache_clear()
    load_llm_config.cache_clear()
    logger.info("Config caches cleared - will reload on next access")


# =============================================================================
# CONVENIENCE GETTERS
# =============================================================================

# RAPTOR configuration getters
def get_clustering_config() -> Dict[str, Any]:
    """Get clustering configuration."""
    return load_raptor_config().get("clustering", {})


def get_summarization_config() -> Dict[str, Any]:
    """Get summarization configuration."""
    return load_raptor_config().get("summarization", {})


def get_tree_building_config() -> Dict[str, Any]:
    """Get tree building configuration."""
    return load_raptor_config().get("tree_building", {})


def get_level_retrieval_config() -> Dict[str, Any]:
    """Get RAPTOR level retrieval configuration."""
    return load_raptor_config().get("level_retrieval", {})


def get_raptor_presets() -> Dict[str, Any]:
    """Get RAPTOR quality presets."""
    return load_raptor_config().get("presets", {})


def get_raptor_keywords() -> Dict[str, Any]:
    """Get RAPTOR intent detection keywords."""
    return load_raptor_config().get("keywords", {})


# Ingestion configuration getters
def get_chunking_config() -> Dict[str, Any]:
    """Get chunking configuration."""
    return load_ingestion_config().get("chunking", {})


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration."""
    return load_ingestion_config().get("embedding", {})


def get_contextual_retrieval_config() -> Dict[str, Any]:
    """Get contextual retrieval configuration."""
    return load_ingestion_config().get("contextual_retrieval", {})


# Retrieval configuration getters (unified)
def get_retriever_settings() -> Dict[str, Any]:
    """Get retriever settings.

    Prefers unified retrieval config, falls back to ingestion.yaml retriever section.
    """
    # Try unified retrieval config
    retrieval = load_retrieval_config()
    if retrieval:
        return retrieval

    # Fallback to legacy ingestion.yaml retriever section
    return load_ingestion_config().get("retriever", {})


def get_reranker_config() -> Dict[str, Any]:
    """Get reranker configuration from unified config."""
    return load_retrieval_config().get("reranker", {})


def get_chat_v2_config() -> Dict[str, Any]:
    """Get Chat V2 specific retrieval settings."""
    # Try unified config first
    retrieval = load_retrieval_config()
    if 'chat_v2' in retrieval:
        return retrieval['chat_v2']

    # Fallback to legacy ingestion.yaml chat_v2 section
    return load_ingestion_config().get("chat_v2", {})


# LLM configuration getters
def get_llm_settings() -> Dict[str, Any]:
    """Get LLM settings.

    Prefers unified llm config, falls back to ingestion.yaml llm section.
    """
    llm = load_llm_config()
    if llm:
        return llm

    # Fallback to legacy ingestion.yaml
    return load_ingestion_config().get("llm", {})


# SQL Chat configuration getters
def get_sql_chat_connections_config() -> Dict[str, Any]:
    """Get SQL Chat connection settings."""
    return load_sql_chat_config().get("connections", {})


def get_sql_chat_query_config() -> Dict[str, Any]:
    """Get SQL Chat query execution settings."""
    return load_sql_chat_config().get("query", {})


def get_sql_chat_few_shot_config() -> Dict[str, Any]:
    """Get SQL Chat few-shot learning settings."""
    return load_sql_chat_config().get("few_shot", {})


def get_sql_chat_security_config() -> Dict[str, Any]:
    """Get SQL Chat security settings."""
    return load_sql_chat_config().get("security", {})


# =============================================================================
# UNIVERSAL CONFIG GETTER
# =============================================================================

def get_config_value(
    config_type: str,
    *keys: str,
    default: Any = None
) -> Any:
    """Get a nested config value with fallback default.

    Args:
        config_type: Config section name. Supports:
            - 'dbnotebook': Root of unified config (navigate with keys)
            - 'raptor': RAPTOR section
            - 'ingestion': Ingestion section
            - 'sql_chat': SQL Chat section
            - 'retrieval': Retrieval section (unified only)
            - 'llm': LLM section (unified only)
        *keys: Nested keys to traverse
        default: Default value if key not found

    Returns:
        Config value or default

    Examples:
        # Unified config access
        get_config_value('dbnotebook', 'retrieval', 'reranker', 'model')

        # Legacy-compatible access
        get_config_value('raptor', 'clustering', 'min_cluster_size', default=3)
        get_config_value('sql_chat', 'few_shot', 'rag_integration', 'enabled', default=True)
        get_config_value('ingestion', 'chunking', 'chunk_size', default=512)

        # Direct section access
        get_config_value('retrieval', 'similarity_top_k', default=20)
        get_config_value('llm', 'temperature', default=0.1)
    """
    # Map config type to loader
    if config_type == 'dbnotebook':
        config = load_unified_config()
    elif config_type == 'raptor':
        config = load_raptor_config()
    elif config_type == 'ingestion':
        config = load_ingestion_config()
    elif config_type == 'sql_chat':
        config = load_sql_chat_config()
    elif config_type == 'retrieval':
        config = load_retrieval_config()
    elif config_type == 'llm':
        config = load_llm_config()
    else:
        logger.warning(f"Unknown config type: {config_type}")
        return default

    # Traverse nested keys
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default

    return config


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def check_config_migration_status() -> Dict[str, bool]:
    """Check which config files are being used.

    Returns:
        Dict with 'unified' and 'legacy_*' keys indicating which configs are active
    """
    unified = load_unified_config()
    has_unified = bool(unified)

    legacy_raptor = bool(_load_yaml_file("raptor.yaml"))
    legacy_ingestion = bool(_load_yaml_file("ingestion.yaml"))
    legacy_sql_chat = bool(_load_yaml_file("sql_chat.yaml"))

    return {
        "unified": has_unified,
        "legacy_raptor": legacy_raptor and 'raptor' not in unified,
        "legacy_ingestion": legacy_ingestion and 'ingestion' not in unified,
        "legacy_sql_chat": legacy_sql_chat and 'sql_chat' not in unified,
    }


def get_retrieval_config_value(key: str, default: Any = None) -> Any:
    """Get a retrieval config value (convenience for common access pattern).

    This is a shorthand for getting values from the unified retrieval section.
    """
    return get_config_value('retrieval', key, default=default)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for RAPTOR config module (uses 'level_retrieval' section from raptor config)
get_retrieval_config = get_level_retrieval_config
