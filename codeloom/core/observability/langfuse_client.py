"""Langfuse observability client for LLM tracing and evaluation.

Provides a singleton LangfuseTracer that wraps the Langfuse Python SDK
with graceful degradation — if Langfuse is disabled or unavailable, all
methods become no-ops and the main query flow is never disrupted.

Uses the decorator-less trace/span/generation API (works in both 3.x and 4.x).

Configuration (env vars):
    LANGFUSE_ENABLED     — "true" to enable (default: false)
    LANGFUSE_PUBLIC_KEY  — Public key from Langfuse project
    LANGFUSE_SECRET_KEY  — Secret key from Langfuse project
    LANGFUSE_HOST        — Langfuse host (default: https://cloud.langfuse.com)
                           Also accepts LANGFUSE_BASE_URL as alias
"""

import logging
import os
import threading
from typing import Any, Dict, Optional
from uuid import uuid4


def _make_trace_id() -> str:
    """Generate a Langfuse-compatible trace ID: 32 lowercase hex chars."""
    return uuid4().hex


logger = logging.getLogger(__name__)

_tracer_instance: Optional["LangfuseTracer"] = None
_tracer_lock = threading.Lock()


class LangfuseTracer:
    """Thread-safe Langfuse tracing wrapper with graceful degradation.

    All public methods are wrapped in try/except so tracing failures
    NEVER propagate into the main RAG pipeline flow.

    Uses the decorator-less API: client.trace() -> trace.span() / trace.generation()

    Usage::

        tracer = get_tracer()
        trace_id = tracer.start_trace("rag_query", user_id=..., notebook_id=..., metadata={})
        tracer.log_span(trace_id, "retrieval", input_data=..., output_data=..., timing_ms=...)
        tracer.log_generation(trace_id, "llm_call", model=..., prompt=..., completion=..., usage=...)
        tracer.end_trace(trace_id, status="success")
        tracer.flush()
    """

    _MAX_ACTIVE_TRACES = 10_000

    def __init__(self) -> None:
        self._enabled: bool = False
        self._client = None
        # trace_id -> Langfuse trace object
        self._traces: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._host: str = ""
        self._project_id: str = ""
        self._initialize()

    def _initialize(self) -> None:
        """Attempt to initialize Langfuse SDK from environment."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        enabled_str = os.getenv("LANGFUSE_ENABLED", "false").lower()
        if enabled_str not in ("true", "1", "yes"):
            logger.info("Langfuse tracing disabled (LANGFUSE_ENABLED not set to true)")
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        )

        if not public_key or not secret_key:
            logger.warning(
                "Langfuse enabled but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY "
                "not set — tracing will be disabled"
            )
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self._host = host
            self._enabled = True
            logger.info(f"Langfuse 3.x tracing initialized | host={host}")
        except ImportError:
            logger.warning(
                "langfuse package not installed — tracing disabled. "
                "Install with: pip install langfuse>=3.0.0"
            )
        except Exception as exc:
            logger.warning(f"Langfuse initialization failed — tracing disabled: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_trace(
        self,
        name: str,
        user_id: str,
        notebook_id: str,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new trace. Returns trace_id (32 lowercase hex chars)."""
        trace_id = _make_trace_id()
        if not self._enabled or self._client is None:
            return trace_id

        try:
            input_data: Dict[str, Any] = {
                "user_id": user_id,
                "project_id": notebook_id,
                **({"query": query} if query else {}),
                **(metadata or {}),
            }

            trace = self._client.trace(
                id=trace_id,
                name=name,
                user_id=user_id,
                input=input_data,
                metadata={"notebook_id": notebook_id, **(metadata or {})},
            )

            with self._lock:
                if len(self._traces) >= self._MAX_ACTIVE_TRACES:
                    oldest_key = next(iter(self._traces))
                    self._traces.pop(oldest_key)
                self._traces[trace_id] = trace

            logger.debug(f"Langfuse trace started: {trace_id} | name={name}")
        except Exception as exc:
            logger.debug(f"Langfuse start_trace failed (non-fatal): {exc}")

        return trace_id

    def log_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timing_ms: Optional[float] = None,
    ) -> None:
        """Log a retrieval/processing span under an existing trace."""
        if not self._enabled or not trace_id:
            return

        try:
            trace = self._traces.get(trace_id)
            if not trace:
                return

            span = trace.span(
                name=name,
                input=input_data,
                output=output_data,
                metadata={
                    **(metadata or {}),
                    **({"timing_ms": timing_ms} if timing_ms is not None else {}),
                },
            )
            span.end()
        except Exception as exc:
            logger.debug(f"Langfuse log_span failed (non-fatal): {exc}")

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: Any,
        completion: Any,
        usage: Optional[Dict[str, int]] = None,
        timing_ms: Optional[float] = None,
    ) -> None:
        """Log an LLM generation (input/output/tokens) under an existing trace."""
        if not self._enabled or not trace_id:
            return

        try:
            trace = self._traces.get(trace_id)
            if not trace:
                return

            usage_details: Optional[Dict[str, int]] = None
            if usage:
                usage_details = {
                    "input": usage.get("prompt_tokens", usage.get("input", 0)),
                    "output": usage.get("completion_tokens", usage.get("output", 0)),
                }

            gen = trace.generation(
                name=name,
                model=model,
                input=prompt,
                output=completion,
                usage=usage_details,
                metadata={"timing_ms": timing_ms} if timing_ms is not None else None,
            )
            gen.end()
        except Exception as exc:
            logger.debug(f"Langfuse log_generation failed (non-fatal): {exc}")

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Attach a numeric score to an existing trace."""
        if not self._enabled or not trace_id:
            return

        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:
            logger.debug(f"Langfuse log_score failed (non-fatal): {exc}")

    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End an active trace and flush to Langfuse."""
        if not self._enabled or not trace_id:
            return

        try:
            with self._lock:
                trace = self._traces.pop(trace_id, None)

            if trace:
                output: Dict[str, Any] = {"status": status}
                if response is not None:
                    output["response"] = response
                if metadata:
                    output.update(metadata)
                trace.update(output=output)
                logger.debug(f"Langfuse trace ended: {trace_id} | status={status}")

            # Flush asynchronously
            threading.Thread(target=self._async_flush, daemon=True).start()
        except Exception as exc:
            logger.debug(f"Langfuse end_trace failed (non-fatal): {exc}")

    def _async_flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            pass

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """Return the Langfuse dashboard URL for a trace."""
        if not self._enabled or not trace_id:
            return None
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        """Block until all buffered events have been sent to Langfuse."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug(f"Langfuse flush failed (non-fatal): {exc}")


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

def get_tracer() -> LangfuseTracer:
    """Return the singleton LangfuseTracer, initializing lazily on first call."""
    global _tracer_instance
    if _tracer_instance is None:
        with _tracer_lock:
            if _tracer_instance is None:
                _tracer_instance = LangfuseTracer()
    return _tracer_instance
