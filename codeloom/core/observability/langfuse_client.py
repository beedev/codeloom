"""Langfuse observability client using direct HTTP ingestion API.

The Langfuse 3.x SDK's OTEL-based start_span/start_generation silently
drops traces due to OpenTelemetry version conflicts.  This client bypasses
the SDK's tracing layer entirely and posts events directly to the
/api/public/ingestion REST endpoint — the same approach that was verified
to work with 207/201 responses.

The SDK is still used for auth_check() and get_trace_url() — only the
trace/span/generation creation uses direct HTTP.

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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_tracer_instance: Optional["LangfuseTracer"] = None
_tracer_lock = threading.Lock()


def _make_trace_id() -> str:
    return uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LangfuseTracer:
    """Thread-safe Langfuse tracing via direct HTTP ingestion.

    All public methods are wrapped in try/except so tracing failures
    NEVER propagate into the main RAG pipeline flow.
    """

    _MAX_BATCH_SIZE = 20
    _FLUSH_INTERVAL_S = 5.0

    def __init__(self) -> None:
        self._enabled: bool = False
        self._host: str = ""
        self._public_key: str = ""
        self._secret_key: str = ""
        self._batch: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        enabled_str = os.getenv("LANGFUSE_ENABLED", "false").lower()
        if enabled_str not in ("true", "1", "yes"):
            logger.info("Langfuse tracing disabled (LANGFUSE_ENABLED not set to true)")
            return

        self._public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self._host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        ).rstrip("/")

        if not self._public_key or not self._secret_key:
            logger.warning("Langfuse enabled but keys not set — tracing disabled")
            return

        # Verify auth
        try:
            resp = requests.get(
                f"{self._host}/api/public/projects",
                auth=(self._public_key, self._secret_key),
                timeout=5,
            )
            if resp.status_code == 200:
                self._enabled = True
                logger.info(f"Langfuse 3.x tracing initialized | host={self._host}")
            else:
                logger.warning(f"Langfuse auth failed (status {resp.status_code}) — tracing disabled")
        except Exception as exc:
            logger.warning(f"Langfuse connection failed — tracing disabled: {exc}")

    def _enqueue(self, event: Dict[str, Any]) -> None:
        """Add an event to the batch queue, flush if full."""
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._MAX_BATCH_SIZE:
                batch = self._batch[:]
                self._batch.clear()
                threading.Thread(target=self._send_batch, args=(batch,), daemon=True).start()
            elif self._flush_timer is None:
                self._flush_timer = threading.Timer(self._FLUSH_INTERVAL_S, self._timed_flush)
                self._flush_timer.daemon = True
                self._flush_timer.start()

    def _timed_flush(self) -> None:
        with self._lock:
            batch = self._batch[:]
            self._batch.clear()
            self._flush_timer = None
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: List[Dict[str, Any]]) -> None:
        try:
            resp = requests.post(
                f"{self._host}/api/public/ingestion",
                auth=(self._public_key, self._secret_key),
                json={"batch": batch},
                timeout=10,
            )
            if resp.status_code not in (200, 207):
                logger.debug(f"Langfuse ingestion returned {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            logger.debug(f"Langfuse batch send failed (non-fatal): {exc}")

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
        trace_id = _make_trace_id()
        if not self._enabled:
            return trace_id

        try:
            self._enqueue({
                "id": uuid4().hex,
                "type": "trace-create",
                "timestamp": _now_iso(),
                "body": {
                    "id": trace_id,
                    "name": name,
                    "userId": user_id,
                    "input": {
                        "query": query or "",
                        "project_id": notebook_id,
                        **(metadata or {}),
                    },
                    "metadata": {"project_id": notebook_id, **(metadata or {})},
                    "timestamp": _now_iso(),
                },
            })
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
        if not self._enabled or not trace_id:
            return

        try:
            now = _now_iso()
            self._enqueue({
                "id": uuid4().hex,
                "type": "span-create",
                "timestamp": now,
                "body": {
                    "traceId": trace_id,
                    "name": name,
                    "input": input_data,
                    "output": output_data,
                    "metadata": {
                        **(metadata or {}),
                        **({"timing_ms": timing_ms} if timing_ms is not None else {}),
                    },
                    "startTime": now,
                    "endTime": now,
                },
            })
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
        if not self._enabled or not trace_id:
            return

        try:
            now = _now_iso()
            usage_body = {}
            if usage:
                usage_body = {
                    "input": usage.get("prompt_tokens", usage.get("input", 0)),
                    "output": usage.get("completion_tokens", usage.get("output", 0)),
                }

            self._enqueue({
                "id": uuid4().hex,
                "type": "generation-create",
                "timestamp": now,
                "body": {
                    "traceId": trace_id,
                    "name": name,
                    "model": model,
                    "input": prompt,
                    "output": completion,
                    "usage": usage_body if usage_body else None,
                    "metadata": {"timing_ms": timing_ms} if timing_ms is not None else None,
                    "startTime": now,
                    "endTime": now,
                },
            })
        except Exception as exc:
            logger.debug(f"Langfuse log_generation failed (non-fatal): {exc}")

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        if not self._enabled or not trace_id:
            return

        try:
            self._enqueue({
                "id": uuid4().hex,
                "type": "score-create",
                "timestamp": _now_iso(),
                "body": {
                    "traceId": trace_id,
                    "name": name,
                    "value": value,
                    "comment": comment,
                },
            })
        except Exception as exc:
            logger.debug(f"Langfuse log_score failed (non-fatal): {exc}")

    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled or not trace_id:
            return

        try:
            output: Dict[str, Any] = {"status": status}
            if response is not None:
                output["response"] = response
            if metadata:
                output.update(metadata)

            self._enqueue({
                "id": uuid4().hex,
                "type": "trace-create",
                "timestamp": _now_iso(),
                "body": {
                    "id": trace_id,
                    "output": output,
                },
            })

            # Flush synchronously on trace end (ensures delivery before process exit)
            with self._lock:
                batch = self._batch[:]
                self._batch.clear()
                if self._flush_timer:
                    self._flush_timer.cancel()
                    self._flush_timer = None
            if batch:
                self._send_batch(batch)
        except Exception as exc:
            logger.debug(f"Langfuse end_trace failed (non-fatal): {exc}")

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        if not self._enabled or not trace_id:
            return None
        try:
            # Get project ID from API
            resp = requests.get(
                f"{self._host}/api/public/projects",
                auth=(self._public_key, self._secret_key),
                timeout=5,
            )
            if resp.status_code == 200:
                projects = resp.json().get("data", [])
                if projects:
                    pid = projects[0]["id"]
                    return f"{self._host}/project/{pid}/traces/{trace_id}"
        except Exception:
            pass
        return None

    def flush(self) -> None:
        """Block until all buffered events have been sent."""
        if not self._enabled:
            return
        with self._lock:
            batch = self._batch[:]
            self._batch.clear()
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
        if batch:
            self._send_batch(batch)


def get_tracer() -> LangfuseTracer:
    global _tracer_instance
    if _tracer_instance is None:
        with _tracer_lock:
            if _tracer_instance is None:
                _tracer_instance = LangfuseTracer()
    return _tracer_instance
