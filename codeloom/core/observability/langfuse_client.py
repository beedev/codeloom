"""Langfuse observability client using direct HTTP ingestion API.

Bypasses the SDK's OTEL transport (which silently drops traces) and posts
events directly to /api/public/ingestion.

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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_tracer_instance: Optional["LangfuseTracer"] = None
_tracer_lock = threading.Lock()


def _make_trace_id() -> str:
    return uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _offset_iso(ms: float) -> str:
    """Return ISO timestamp offset by -ms milliseconds from now."""
    dt = datetime.now(timezone.utc) - timedelta(milliseconds=ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class LangfuseTracer:
    """Thread-safe Langfuse tracing via direct HTTP ingestion."""

    _MAX_BATCH_SIZE = 20
    _FLUSH_INTERVAL_S = 5.0

    def __init__(self) -> None:
        self._enabled: bool = False
        self._host: str = ""
        self._public_key: str = ""
        self._secret_key: str = ""
        self._project_id: str = ""
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

        try:
            resp = requests.get(
                f"{self._host}/api/public/projects",
                auth=(self._public_key, self._secret_key),
                timeout=5,
            )
            if resp.status_code == 200:
                projects = resp.json().get("data", [])
                if projects:
                    self._project_id = projects[0]["id"]
                self._enabled = True
                logger.info(f"Langfuse 3.x tracing initialized | host={self._host}")
            else:
                logger.warning(f"Langfuse auth failed ({resp.status_code}) — tracing disabled")
        except Exception as exc:
            logger.warning(f"Langfuse connection failed — tracing disabled: {exc}")

    def _enqueue(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._MAX_BATCH_SIZE:
                batch = self._batch[:]
                self._batch.clear()
                self._send_batch(batch)
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
                logger.debug(f"Langfuse ingestion {resp.status_code}: {resp.text[:200]}")
            else:
                errors = resp.json().get("errors", [])
                if errors:
                    logger.debug(f"Langfuse ingestion errors: {errors}")
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
            now = _now_iso()
            self._enqueue({
                "id": uuid4().hex,
                "type": "trace-create",
                "timestamp": now,
                "body": {
                    "id": trace_id,
                    "name": name,
                    "userId": user_id,
                    "timestamp": now,
                    "input": {
                        "query": query or "",
                        "project_id": notebook_id,
                    },
                    "metadata": {
                        "project_id": notebook_id,
                        **(metadata or {}),
                    },
                },
            })
        except Exception as exc:
            logger.debug(f"Langfuse start_trace failed: {exc}")

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
            end_time = _now_iso()
            start_time = _offset_iso(timing_ms) if timing_ms else end_time

            self._enqueue({
                "id": uuid4().hex,
                "type": "span-create",
                "timestamp": end_time,
                "body": {
                    "traceId": trace_id,
                    "name": name,
                    "startTime": start_time,
                    "endTime": end_time,
                    "input": input_data,
                    "output": output_data,
                    "metadata": metadata,
                },
            })
        except Exception as exc:
            logger.debug(f"Langfuse log_span failed: {exc}")

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
            end_time = _now_iso()
            start_time = _offset_iso(timing_ms) if timing_ms else end_time

            # Langfuse generation body — model, usage, startTime, endTime
            # are the fields that populate Latency, Cost, Model Name columns
            body: Dict[str, Any] = {
                "traceId": trace_id,
                "name": name,
                "model": model,
                "startTime": start_time,
                "endTime": end_time,
                "input": prompt if isinstance(prompt, (dict, list)) else {"text": str(prompt)},
                "output": completion if isinstance(completion, (dict, list)) else {"text": str(completion)},
            }

            if usage:
                body["usage"] = {
                    "input": usage.get("prompt_tokens", usage.get("input", 0)),
                    "output": usage.get("completion_tokens", usage.get("output", 0)),
                    "unit": "TOKENS",
                }

            if timing_ms:
                body["metadata"] = {"timing_ms": round(timing_ms, 1)}

            self._enqueue({
                "id": uuid4().hex,
                "type": "generation-create",
                "timestamp": end_time,
                "body": body,
            })
        except Exception as exc:
            logger.debug(f"Langfuse log_generation failed: {exc}")

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
            logger.debug(f"Langfuse log_score failed: {exc}")

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
                output["response"] = response[:2000]
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

            # Flush synchronously
            with self._lock:
                batch = self._batch[:]
                self._batch.clear()
                if self._flush_timer:
                    self._flush_timer.cancel()
                    self._flush_timer = None
            if batch:
                self._send_batch(batch)
        except Exception as exc:
            logger.debug(f"Langfuse end_trace failed: {exc}")

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        if not self._enabled or not trace_id:
            return None
        if self._project_id:
            return f"{self._host}/project/{self._project_id}/traces/{trace_id}"
        return None

    def flush(self) -> None:
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
