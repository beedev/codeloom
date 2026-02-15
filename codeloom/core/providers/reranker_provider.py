"""Thread-safe Reranker Provider with multiple backend support.

This module provides reranker instances supporting both local cross-encoder models
and Groq LLM-based reranking for fast cloud-based inference.

## Backend Types

### Local Cross-Encoder (Default)
- Thread-safe wrapper for SentenceTransformerRerank (ONNX)
- Models: xsmall (~2s), base (~10s), large (~30s) on CPU
- Supports HuggingFace IDs and local paths

### Groq LLM Reranker
- Ultra-fast cloud-based reranking (~300ms)
- Uses Groq's Structured Outputs for reliable JSON scoring
- Models: groq:scout (recommended), groq:maverick, groq:llama70b

## Usage Examples

    from dbnotebook.core.providers.reranker_provider import get_shared_reranker

    # Local cross-encoder (default)
    reranker = get_shared_reranker(model="base", top_n=10)

    # Groq LLM reranker (via prefix)
    reranker = get_shared_reranker(model="groq:scout", top_n=10)

    # Process nodes
    reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle)

## API Parameter Control

API endpoints accept reranker_model parameter:
    {"reranker_model": "base"}        # Local cross-encoder
    {"reranker_model": "groq:scout"}  # Groq LLM reranker
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, List, Union

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

# Thread-safe singleton with reentrant lock (RLock allows same thread to acquire multiple times)
_reranker_lock = threading.RLock()
_shared_reranker: Optional[SentenceTransformerRerank] = None
_reranker_config: dict = {}
_reranker_enabled: bool = True  # Runtime enable/disable flag

# Default local model directory (relative to project root)
DEFAULT_LOCAL_MODEL_DIR = "models/rerankers"

# Local cross-encoder model aliases
MODEL_ALIASES = {
    "base": "mxbai-rerank-base-v1",
    "base-v1": "mxbai-rerank-base-v1",
    "large": "mxbai-rerank-large-v1",
    "large-v1": "mxbai-rerank-large-v1",
    "xsmall": "mxbai-rerank-xsmall-v1",
    "xsmall-v1": "mxbai-rerank-xsmall-v1",
    "disabled": None,  # Special alias to disable reranking
}

# Groq model aliases for LLM-based reranking
# Use with "groq:" prefix, e.g., "groq:scout"
# Note: GPT-OSS models are NOT supported for reranking (incompatible with JSON modes)
GROQ_MODEL_ALIASES = {
    # Llama 4 models (recommended for reranking - fast, cheap)
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    # Llama 3.x models
    "llama70b": "llama-3.3-70b-versatile",
    "llama8b": "llama-3.1-8b-instant",
}

# Models that support Groq's Structured Outputs (json_schema mode)
GROQ_STRUCTURED_OUTPUT_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
}


def _is_valid_local_model(path: Path) -> bool:
    """Check if a local model directory contains valid model files."""
    if not path.exists() or not path.is_dir():
        return False
    # Check for common model files
    model_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    return any((path / f).exists() for f in model_files)


def resolve_model_path(model: str) -> Optional[str]:
    """Resolve model identifier to actual model path or HuggingFace ID.

    Priority:
    1. If model is "disabled", return None
    2. If model is an alias (e.g., "base", "large"), expand it
    3. If model is a local path with valid model files, use it
    4. Otherwise, assume it's a HuggingFace model ID

    Args:
        model: Model identifier (alias, local path, or HuggingFace ID)

    Returns:
        Resolved model path/ID, or None if disabled
    """
    # Handle disabled alias
    if model.lower() == "disabled":
        return None

    # Check for aliases
    if model.lower() in MODEL_ALIASES:
        alias_value = MODEL_ALIASES[model.lower()]
        if alias_value is None:
            return None
        # Try to find local model for alias (must have valid model files)
        local_path = Path(DEFAULT_LOCAL_MODEL_DIR) / alias_value
        if _is_valid_local_model(local_path):
            logger.info(f"Using local model for alias '{model}': {local_path}")
            return str(local_path)
        # Fall back to HuggingFace ID
        return f"mixedbread-ai/{alias_value}"

    # Check if it's a local path with valid model files
    local_path = Path(model)
    if _is_valid_local_model(local_path):
        logger.info(f"Using local model path: {model}")
        return model

    # Check in default local model directory
    local_path = Path(DEFAULT_LOCAL_MODEL_DIR) / model
    if _is_valid_local_model(local_path):
        logger.info(f"Found model in local directory: {local_path}")
        return str(local_path)

    # Assume it's a HuggingFace model ID
    return model


class ThreadSafeReranker:
    """Thread-safe wrapper for SentenceTransformerRerank.

    ONNX Runtime (used internally by SentenceTransformerRerank) is not thread-safe.
    This wrapper serializes access using a lock to prevent concurrent inference
    which would cause SIGSEGV crashes.
    """

    def __init__(
        self,
        reranker: SentenceTransformerRerank,
        lock: threading.Lock,
        top_n: int
    ):
        self._reranker = reranker
        self._lock = lock
        self._top_n = top_n

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None
    ) -> List[NodeWithScore]:
        """Thread-safe reranking with lock.

        Args:
            nodes: List of nodes with scores to rerank
            query_bundle: Query bundle for reranking (optional if query_str provided)
            query_str: Query string (optional if query_bundle provided)

        Returns:
            Reranked list of nodes, limited to top_n
        """
        if not nodes:
            return nodes

        with self._lock:
            # Temporarily set top_n for this call
            original_top_n = self._reranker.top_n
            self._reranker.top_n = self._top_n
            try:
                return self._reranker.postprocess_nodes(
                    nodes,
                    query_bundle=query_bundle,
                    query_str=query_str
                )
            finally:
                self._reranker.top_n = original_top_n

    @property
    def top_n(self) -> int:
        """Get the top_n value."""
        return self._top_n

    @top_n.setter
    def top_n(self, value: int) -> None:
        """Set the top_n value for subsequent calls."""
        self._top_n = value


class GroqReranker:
    """Groq LLM-based reranker with Structured Outputs for reliable scoring.

    Uses Groq's ultra-fast inference (~300ms) for document relevance scoring.
    Implements the same interface as ThreadSafeReranker for drop-in replacement.

    Supports Structured Outputs (json_schema mode) for guaranteed JSON format
    on compatible models (Llama 4, GPT-OSS).

    Performance:
    - groq:scout: ~300ms, $0.0001/query (recommended)
    - groq:maverick: ~400ms, $0.0002/query
    - groq:llama70b: ~600ms, $0.0004/query
    - groq:llama70b: ~600ms (high quality)
    """

    def __init__(self, model: str = "scout", top_n: int = 10):
        """Initialize Groq reranker.

        Args:
            model: Groq model alias (scout, maverick, llama70b) or full ID
            top_n: Number of top results to return after reranking
        """
        self._model_alias = model
        self._model = GROQ_MODEL_ALIASES.get(model, model)
        self._top_n = top_n
        self._client = None

        # Lazy initialization of Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Required for Groq LLM reranking. "
                "Get your key at https://console.groq.com/"
            )

        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"Initialized Groq reranker: {self._model}")
        except ImportError:
            raise ImportError(
                "groq package required for Groq reranking. "
                "Install with: pip install groq"
            )

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None
    ) -> List[NodeWithScore]:
        """Rerank nodes using Groq LLM with Structured Outputs.

        Args:
            nodes: List of nodes with scores to rerank
            query_bundle: Query bundle for reranking (optional if query_str provided)
            query_str: Query string (optional if query_bundle provided)

        Returns:
            Reranked list of nodes, limited to top_n
        """
        if not nodes:
            return nodes

        if len(nodes) == 1:
            return nodes

        # Extract query string
        query = query_str or (query_bundle.query_str if query_bundle else "")
        if not query:
            logger.warning("No query provided for reranking, returning original order")
            return nodes[:self._top_n]

        # Build reranking prompt
        prompt = self._build_rerank_prompt(query, nodes)

        try:
            # Build request parameters
            request_params = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,  # Deterministic scoring
                "max_tokens": 500,  # Enough for score array
            }

            # Use Structured Outputs for supported models (guaranteed JSON format)
            if self._model in GROQ_STRUCTURED_OUTPUT_MODELS:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rerank_scores",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "scores": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Relevance scores 0-10 for each document in order"
                                }
                            },
                            "required": ["scores"],
                            "additionalProperties": False
                        }
                    }
                }
            else:
                # Fallback to json_object mode for other models
                request_params["response_format"] = {"type": "json_object"}

            # Call Groq API
            response = self._client.chat.completions.create(**request_params)
            scores = self._parse_scores(
                response.choices[0].message.content,
                len(nodes)
            )

            logger.debug(
                f"Groq rerank ({self._model_alias}): "
                f"{len(nodes)} nodes → scores: {scores[:5]}..."
            )

        except Exception as e:
            logger.warning(
                f"Groq reranking failed ({type(e).__name__}: {e}), "
                "returning original order"
            )
            # Fallback: return original order
            return nodes[:self._top_n]

        # Reorder nodes by score (descending)
        scored_nodes = list(zip(nodes, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        return [node for node, _ in scored_nodes[:self._top_n]]

    def _build_rerank_prompt(self, query: str, nodes: List[NodeWithScore]) -> str:
        """Build prompt for relevance scoring.

        Args:
            query: User query
            nodes: List of nodes to score

        Returns:
            Formatted prompt for Groq LLM
        """
        docs = []
        for i, node in enumerate(nodes):
            # Truncate long documents to save tokens
            text = node.node.get_content()[:500]
            docs.append(f"[{i}] {text}")

        return f"""Score the relevance of each document to the query on a scale of 0-10.
Return a JSON object with a "scores" array containing one score per document in the same order.
Higher scores mean more relevant. Be strict - only highly relevant documents should score 8+.

Query: {query}

Documents:
{chr(10).join(docs)}

Return only valid JSON: {{"scores": [score1, score2, ...]}}"""

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse scores from Groq response.

        Args:
            response: JSON response from Groq
            expected_count: Expected number of scores

        Returns:
            List of relevance scores (0.0-10.0)
        """
        try:
            data = json.loads(response.strip())

            # Handle structured output format: {"scores": [...]}
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
            elif isinstance(data, list):
                scores = data
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")

            # Validate and normalize scores
            if len(scores) == expected_count:
                # Clamp scores to 0-10 range
                return [max(0.0, min(10.0, float(s))) for s in scores]
            else:
                logger.warning(
                    f"Score count mismatch: got {len(scores)}, expected {expected_count}"
                )

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Score parsing failed ({type(e).__name__}: {e})")

        # Fallback: return uniform scores (preserve original order)
        return [1.0] * expected_count

    @property
    def top_n(self) -> int:
        """Get the top_n value."""
        return self._top_n

    @top_n.setter
    def top_n(self, value: int) -> None:
        """Set the top_n value for subsequent calls."""
        self._top_n = value

    @property
    def model(self) -> str:
        """Get the full model ID."""
        return self._model

    @property
    def model_alias(self) -> str:
        """Get the model alias used at initialization."""
        return self._model_alias


def _resolve_groq_model(model_suffix: str) -> str:
    """Resolve Groq model alias to full model ID.

    Args:
        model_suffix: Model name after "groq:" prefix (e.g., "scout")

    Returns:
        Full Groq model ID
    """
    return GROQ_MODEL_ALIASES.get(model_suffix, model_suffix)


def get_shared_reranker(
    model: str = "base",
    top_n: int = 10
) -> Optional[Union[ThreadSafeReranker, GroqReranker]]:
    """Get reranker instance based on model specification.

    Supports two backend types:
    1. Local cross-encoder (default): Thread-safe ONNX models
    2. Groq LLM: Ultra-fast cloud reranking via "groq:" prefix

    Model specification:
    - Local aliases: "base", "large", "xsmall", "disabled"
    - Local paths: "models/rerankers/mxbai-rerank-base-v1"
    - HuggingFace IDs: "mixedbread-ai/mxbai-rerank-base-v1"
    - Groq models: "groq:scout", "groq:maverick", "groq:llama70b"

    Environment variables:
    - RERANKER_MODEL: Default model (overrides parameter default)
    - GROQ_API_KEY: Required for groq: models

    Args:
        model: Reranker model specification (default: "base")
        top_n: Number of top results to return after reranking

    Returns:
        Reranker instance (ThreadSafeReranker or GroqReranker), or None if disabled
    """
    global _shared_reranker, _reranker_config, _reranker_enabled

    # RERANKER_MODEL env var overrides parameter default
    env_model = os.getenv("RERANKER_MODEL", "").strip()
    if env_model:
        model = env_model

    # Early return if reranker is disabled globally
    if not _reranker_enabled:
        return None

    # Use model from config if set, otherwise use parameter
    config_model = _reranker_config.get("model", model)

    # Check for groq: prefix - route to Groq reranker
    if config_model.lower().startswith("groq:"):
        groq_model = config_model[5:]  # Remove "groq:" prefix

        # Check for API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                f"GROQ_API_KEY not set for model '{config_model}', "
                "falling back to local 'base' reranker"
            )
            config_model = "base"
        else:
            # Return new GroqReranker instance (not singleton - lightweight)
            try:
                return GroqReranker(model=groq_model, top_n=top_n)
            except Exception as e:
                logger.warning(
                    f"Failed to create Groq reranker ({e}), "
                    "falling back to local 'base' reranker"
                )
                config_model = "base"

    with _reranker_lock:
        # Resolve to actual path or HuggingFace ID
        effective_model = resolve_model_path(config_model)

        # Handle "disabled" resolution
        if effective_model is None:
            _reranker_enabled = False
            _shared_reranker = None
            logger.info("Reranker disabled via model resolution")
            return None

        # Check if we need to create or recreate the reranker
        current_resolved = _reranker_config.get("resolved_model")
        if _shared_reranker is None or current_resolved != effective_model:
            logger.info(f"Initializing shared reranker: {effective_model}")
            _shared_reranker = SentenceTransformerRerank(
                model=effective_model,
                top_n=top_n
            )
            _reranker_config = {
                "model": config_model,
                "resolved_model": effective_model,
                "top_n": top_n
            }
            logger.info("Shared reranker initialized successfully")

    return ThreadSafeReranker(_shared_reranker, _reranker_lock, top_n)


def clear_shared_reranker() -> None:
    """Clear the shared reranker instance.

    Useful for testing or when you need to release memory.
    """
    global _shared_reranker, _reranker_config
    with _reranker_lock:
        _shared_reranker = None
        _reranker_config = {}
        logger.debug("Shared reranker cleared")


def set_reranker_config(
    model: Optional[str] = None,
    enabled: bool = True,
    top_n: Optional[int] = None
) -> dict:
    """Configure reranker at runtime.

    Allows switching reranker model or disabling reranker entirely without restart.
    Thread-safe - changes are applied atomically.

    Args:
        model: Reranker model name. If None, keeps current model.
               Local options: "xsmall", "base", "large", "disabled"
               Groq options: "groq:scout", "groq:maverick", "groq:llama70b"
        enabled: Whether reranking is enabled. If False, get_shared_reranker returns None.
        top_n: Default top_n for reranking. If None, keeps current value.

    Returns:
        Current configuration after update
    """
    global _shared_reranker, _reranker_config, _reranker_enabled

    with _reranker_lock:
        _reranker_enabled = enabled

        if not enabled:
            # Disable reranking - clear the instance to free memory
            _shared_reranker = None
            logger.info("Reranker disabled")
            return get_reranker_config()

        # Check if model change requires reload
        needs_reload = False
        if model and model != _reranker_config.get("model"):
            needs_reload = True
            _reranker_config["model"] = model
            # Clear resolved_model so it gets re-resolved
            _reranker_config.pop("resolved_model", None)

        if top_n is not None:
            _reranker_config["top_n"] = top_n

        if needs_reload:
            # Force reload on next get_shared_reranker() call
            _shared_reranker = None
            logger.info(f"Reranker model changed to {model}, will reload on next use")

        return get_reranker_config()


def get_reranker_config() -> dict:
    """Get current reranker configuration.

    Returns:
        Dict with keys:
        - enabled: Whether reranking is enabled
        - model: Original model identifier (alias, path, HuggingFace ID, or groq:*)
        - resolved_model: Actual path/ID used for loading
        - top_n: Default top_n for reranking
        - loaded: Whether model is currently in memory (local models only)
        - is_local: Whether using a local model path
        - is_groq: Whether using Groq LLM reranker
    """
    with _reranker_lock:
        model = _reranker_config.get("model", "base")
        is_groq = model.lower().startswith("groq:")

        resolved = _reranker_config.get("resolved_model")
        if resolved is None and _reranker_enabled and not is_groq:
            resolved = resolve_model_path(model)

        # For Groq models, resolved is the full Groq model ID
        if is_groq:
            groq_alias = model[5:]  # Remove "groq:" prefix
            resolved = GROQ_MODEL_ALIASES.get(groq_alias, groq_alias)

        return {
            "enabled": _reranker_enabled,
            "model": model,
            "resolved_model": resolved,
            "top_n": _reranker_config.get("top_n", 10),
            "loaded": _shared_reranker is not None,
            "is_local": resolved and os.path.exists(resolved) if resolved and not is_groq else False,
            "is_groq": is_groq
        }


def is_reranker_enabled() -> bool:
    """Check if reranker is enabled.

    Returns:
        True if reranker is enabled, False otherwise
    """
    return _reranker_enabled


def list_available_models() -> list[dict]:
    """List available reranker models from models.yaml config.

    Reads reranker configuration from config/models.yaml.
    Groq models are only included if the groq provider is enabled (has API key).

    Returns:
        List of model info dicts with keys:
        - id: Model identifier to use in API calls
        - name: Human-readable name
        - type: "local" or "groq"
        - description: Model description
    """
    from ..config.config_loader import _load_yaml_file

    models = []

    try:
        # Load models.yaml config
        config = _load_yaml_file("models.yaml")
        rerankers_config = config.get("rerankers", {})

        # Check if Groq provider is enabled (has API key)
        groq_enabled = bool(os.getenv("GROQ_API_KEY"))

        # Add local models (always available)
        local_models = rerankers_config.get("local", [])
        for model in local_models:
            models.append({
                "id": model.get("id"),
                "name": model.get("name"),
                "type": "local",
                "description": model.get("description", ""),
            })

        # Add Groq models only if Groq is enabled
        if groq_enabled:
            groq_models = rerankers_config.get("groq", [])
            for model in groq_models:
                models.append({
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "type": "groq",
                    "description": model.get("description", ""),
                })

        # Add disabled option
        models.append({
            "id": "disabled",
            "name": "Disabled",
            "type": "disabled",
            "description": "No reranking",
        })

    except Exception as e:
        logger.warning(f"Failed to load reranker config from models.yaml: {e}")
        # Fallback to hardcoded defaults
        models = [
            {"id": "xsmall", "name": "XSmall", "type": "local", "description": "Fastest"},
            {"id": "base", "name": "Base", "type": "local", "description": "Balanced"},
            {"id": "large", "name": "Large", "type": "local", "description": "Best"},
            {"id": "disabled", "name": "Disabled", "type": "disabled", "description": "No reranking"},
        ]

    return models
