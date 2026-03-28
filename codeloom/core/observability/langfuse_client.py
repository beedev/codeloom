"""Langfuse observability — direct HTTP POST, no SDK tracing.

Simple: collect events during a request, POST them all at once when the
trace ends. No batching, no timers, no OTEL, no singletons-within-singletons.

Configuration (env vars):
    LANGFUSE_ENABLED     — "true" to enable (default: false)
    LANGFUSE_PUBLIC_KEY  — Public key from Langfuse project
    LANGFUSE_SECRET_KEY  — Secret key from Langfuse project
    LANGFUSE_HOST        — Langfuse host (default: https://cloud.langfuse.com)
                           Also accepts LANGFUSE_BASE_URL as alias
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_tracer_instance: Optional["LangfuseTracer"] = None
_tracer_lock = threading.Lock()


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _iso_offset(ms: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(milliseconds=ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe_json(obj: Any) -> Any:
    """Convert numpy/non-serializable types to native Python."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


class LangfuseTracer:
    """Collect trace events per request, POST all at once on end_trace."""

    def __init__(self) -> None:
        self._enabled = False
        self._host = ""
        self._auth = ("", "")
        self._project_id = ""
        # trace_id -> list of ingestion events
        self._traces: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if os.getenv("LANGFUSE_ENABLED", "false").lower() not in ("true", "1", "yes"):
            logger.info("Langfuse tracing disabled")
            return

        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")
        self._host = (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")).rstrip("/")

        if not pk or not sk:
            logger.warning("Langfuse keys not set — disabled")
            return

        self._auth = (pk, sk)
        try:
            r = requests.get(f"{self._host}/api/public/projects", auth=self._auth, timeout=5)
            if r.status_code == 200:
                projects = r.json().get("data", [])
                self._project_id = projects[0]["id"] if projects else ""
                self._enabled = True
                logger.info(f"Langfuse 3.x tracing initialized | host={self._host}")
            else:
                logger.warning(f"Langfuse auth failed ({r.status_code})")
        except Exception as e:
            logger.warning(f"Langfuse connection failed: {e}")

    def _post(self, events: List[Dict]) -> None:
        """POST events to Langfuse ingestion API. Called once per trace."""
        try:
            safe_events = _safe_json(events)
            r = requests.post(
                f"{self._host}/api/public/ingestion",
                auth=self._auth,
                data=json.dumps({"batch": safe_events}),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            result = r.json()
            ok = len(result.get("successes", []))
            errs = result.get("errors", [])
            types = [e.get("type", "?") for e in events]
            logger.info(f"Langfuse: {types} → {ok} ok, {len(errs)} errors")
            for e in errs:
                logger.warning(f"Langfuse error: {e.get('message','')} {str(e.get('error',''))[:200]}")
        except Exception as e:
            logger.error(f"Langfuse POST failed: {e}")

    # -- Public API (5 methods, that's it) --

    def start_trace(self, name: str, user_id: str, notebook_id: str,
                    query: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        tid = str(uuid4())
        if not self._enabled:
            return tid

        event = {
            "id": str(uuid4()), "type": "trace-create", "timestamp": _iso(),
            "body": {
                "id": tid, "name": name, "userId": user_id, "timestamp": _iso(),
                "input": _safe_json({"query": query or "", "project_id": notebook_id}),
                "metadata": _safe_json({"project_id": notebook_id, **(metadata or {})}),
            },
        }
        with self._lock:
            self._traces[tid] = [event]
        return tid

    def log_span(self, trace_id: str, name: str, input_data: Optional[Dict] = None,
                 output_data: Optional[Dict] = None, metadata: Optional[Dict] = None,
                 timing_ms: Optional[float] = None) -> None:
        if not self._enabled or trace_id not in self._traces:
            return
        end = _iso()
        start = _iso_offset(timing_ms) if timing_ms else end
        event = {
            "id": str(uuid4()), "type": "span-create", "timestamp": end,
            "body": {
                "id": str(uuid4()), "traceId": trace_id, "name": name,
                "startTime": start, "endTime": end,
                "input": _safe_json(input_data), "output": _safe_json(output_data),
                "metadata": _safe_json(metadata),
            },
        }
        with self._lock:
            self._traces[trace_id].append(event)

    def log_generation(self, trace_id: str, name: str, model: str, prompt: Any,
                       completion: Any, usage: Optional[Dict[str, int]] = None,
                       timing_ms: Optional[float] = None) -> None:
        if not self._enabled or trace_id not in self._traces:
            return
        end = _iso()
        start = _iso_offset(timing_ms) if timing_ms else end
        body: Dict[str, Any] = {
            "id": str(uuid4()), "traceId": trace_id, "name": name, "model": model,
            "startTime": start, "endTime": end,
            "input": {"text": str(prompt)} if not isinstance(prompt, dict) else _safe_json(prompt),
            "output": {"text": str(completion)} if not isinstance(completion, dict) else _safe_json(completion),
        }
        if usage:
            body["usage"] = {
                "input": int(usage.get("prompt_tokens", usage.get("input", 0))),
                "output": int(usage.get("completion_tokens", usage.get("output", 0))),
                "unit": "TOKENS",
            }
        event = {"id": str(uuid4()), "type": "generation-create", "timestamp": end, "body": body}
        with self._lock:
            self._traces[trace_id].append(event)

    def log_score(self, trace_id: str, name: str, value: float,
                  comment: Optional[str] = None) -> None:
        """Post score directly — scores come from feedback, after the trace is closed."""
        if not self._enabled or not trace_id:
            return
        event = {
            "id": str(uuid4()), "type": "score-create", "timestamp": _iso(),
            "body": {"id": str(uuid4()), "traceId": trace_id, "name": name,
                     "value": float(value), "comment": comment},
        }
        # POST immediately — score arrives after trace is already closed
        self._post([event])

    def end_trace(self, trace_id: str, status: str = "success",
                  response: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        if not self._enabled or trace_id not in self._traces:
            return

        output = _safe_json({"status": status, **({"response": response[:2000]} if response else {}), **(metadata or {})})
        update_event = {
            "id": str(uuid4()), "type": "trace-create", "timestamp": _iso(),
            "body": {"id": trace_id, "output": output},
        }

        with self._lock:
            events = self._traces.pop(trace_id, [])
            events.append(update_event)

        if events:
            self._post(events)

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        if self._enabled and self._project_id:
            return f"{self._host}/project/{self._project_id}/traces/{trace_id}"
        return None

    def flush(self) -> None:
        """Flush any orphaned traces (safety net)."""
        if not self._enabled:
            return
        with self._lock:
            all_events = []
            for events in self._traces.values():
                all_events.extend(events)
            self._traces.clear()
        if all_events:
            self._post(all_events)


def get_tracer() -> LangfuseTracer:
    global _tracer_instance
    if _tracer_instance is None:
        with _tracer_lock:
            if _tracer_instance is None:
                _tracer_instance = LangfuseTracer()
    return _tracer_instance
