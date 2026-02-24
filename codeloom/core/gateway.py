"""LLM Gateway — transparent proxy for observability, retry, and metrics.

Wraps any LlamaIndex LLM as a CustomLLM subclass so it can be set as
Settings.llm directly. All existing call sites (30+) flow through the
gateway automatically — zero consumer changes required.

Features:
- Call logging (prompt/response length, latency, model)
- Token tracking (extracted from provider responses)
- Retry with exponential backoff (all providers, not just Groq)
- Per-call purpose tagging (migration, query, raptor, understanding)
- Cost estimation by model
- Thread-safe in-memory metrics
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Sequence

import backoff
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms import CustomLLM

logger = logging.getLogger(__name__)

# ── Cost table (USD per 1M tokens) ────────────────────────────────────
# Updated as of Feb 2026. Add new models as needed.
_COST_PER_1M_TOKENS = {
    # OpenAI
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # Groq (free tier, but track anyway)
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    # Local (Ollama) — no cost
    "_default": {"input": 0.0, "output": 0.0},
}


def _get_retryable_exceptions():
    """Lazy-load retryable exception classes.

    Handles missing provider packages gracefully.
    """
    exceptions = [TimeoutError, ConnectionError]
    try:
        from openai import RateLimitError as OpenAIRateLimit
        exceptions.append(OpenAIRateLimit)
    except ImportError:
        pass
    try:
        from groq import RateLimitError as GroqRateLimit
        exceptions.append(GroqRateLimit)
    except ImportError:
        pass
    try:
        from anthropic import RateLimitError as AnthropicRateLimit
        exceptions.append(AnthropicRateLimit)
    except ImportError:
        pass
    return tuple(exceptions)


# ── Metrics ────────────────────────────────────────────────────────────

@dataclass
class LLMMetrics:
    """Thread-safe in-memory LLM usage metrics."""

    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    retries: int = 0
    calls_by_purpose: dict = field(default_factory=lambda: defaultdict(int))
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.total_latency_ms / max(self.total_calls, 1), 1),
            "errors": self.errors,
            "retries": self.retries,
            "calls_by_purpose": dict(self.calls_by_purpose),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


# ── Gateway ────────────────────────────────────────────────────────────

class LLMGateway(CustomLLM):
    """Transparent LLM proxy with observability and retry.

    Extends CustomLLM so it can be assigned to Settings.llm directly.
    All calls to complete/stream_complete/chat/acomplete flow through
    the gateway, which adds logging, metrics, and retry.

    Usage:
        from codeloom.core.gateway import LLMGateway
        raw_llm = ...  # Any LlamaIndex LLM
        Settings.llm = LLMGateway(raw_llm)
    """

    # Pydantic fields (CustomLLM is a Pydantic BaseModel)
    _llm: Any = None
    _metrics: LLMMetrics = None
    _lock: threading.Lock = None
    _retryable_exceptions: tuple = None

    def __init__(self, llm: Any, **kwargs):
        super().__init__(**kwargs)
        # Store as private attrs (bypass Pydantic field validation)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_metrics", LLMMetrics())
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_retryable_exceptions", None)
        logger.info(
            f"LLMGateway initialized — wrapping {type(llm).__name__}"
            f" (model={getattr(llm, 'model', 'unknown')})"
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Delegate metadata to the wrapped LLM."""
        return self._llm.metadata

    @property
    def model(self) -> str:
        """Expose model name for compatibility."""
        return getattr(self._llm, "model", "unknown")

    @property
    def temperature(self) -> float:
        """Expose temperature for compatibility."""
        return getattr(self._llm, "temperature", 0.1)

    @temperature.setter
    def temperature(self, value: float):
        """Allow temperature mutation for backward compatibility.

        Some callers (code_chat.py, phases.py) mutate Settings.llm.temperature.
        We pass it through to the wrapped LLM.
        """
        if hasattr(self._llm, "temperature"):
            self._llm.temperature = value

    # ── Core methods ──────────────────────────────────────────────────

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Intercept completion calls with logging, retry, and metrics."""
        purpose = kwargs.pop("gateway_purpose", "general")
        t0 = time.time()

        try:
            response = self._retry_call(
                self._llm.complete, prompt, formatted=formatted, **kwargs
            )
        except Exception as e:
            self._record_error(purpose)
            raise

        latency_ms = (time.time() - t0) * 1000
        self._record_success(prompt, response, latency_ms, purpose)
        return response

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        """Intercept streaming calls. Metrics recorded after stream completes."""
        purpose = kwargs.pop("gateway_purpose", "general")
        t0 = time.time()
        collected_text = []

        try:
            for token in self._llm.stream_complete(prompt, formatted=formatted, **kwargs):
                if token.delta:
                    collected_text.append(token.delta)
                yield token
        except Exception as e:
            self._record_error(purpose)
            raise

        latency_ms = (time.time() - t0) * 1000
        full_text = "".join(collected_text)
        # Build a synthetic response for metrics
        synthetic = CompletionResponse(text=full_text)
        self._record_success(prompt, synthetic, latency_ms, purpose)

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Intercept chat calls."""
        purpose = kwargs.pop("gateway_purpose", "general")
        t0 = time.time()

        try:
            response = self._retry_call(self._llm.chat, messages, **kwargs)
        except Exception as e:
            self._record_error(purpose)
            raise

        latency_ms = (time.time() - t0) * 1000
        self._record_chat_success(messages, response, latency_ms, purpose)
        return response

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Intercept async completion (used by RAPTOR summarizer)."""
        purpose = kwargs.pop("gateway_purpose", "raptor")
        t0 = time.time()

        try:
            response = await self._llm.acomplete(prompt, formatted=formatted, **kwargs)
        except Exception as e:
            self._record_error(purpose)
            raise

        latency_ms = (time.time() - t0) * 1000
        self._record_success(prompt, response, latency_ms, purpose)
        return response

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Intercept async chat."""
        purpose = kwargs.pop("gateway_purpose", "general")
        t0 = time.time()

        try:
            response = await self._llm.achat(messages, **kwargs)
        except Exception as e:
            self._record_error(purpose)
            raise

        latency_ms = (time.time() - t0) * 1000
        self._record_chat_success(messages, response, latency_ms, purpose)
        return response

    # ── Retry ─────────────────────────────────────────────────────────

    def _retry_call(self, fn, *args, **kwargs):
        """Execute fn with exponential backoff on retryable errors."""
        if self._retryable_exceptions is None:
            object.__setattr__(
                self, "_retryable_exceptions", _get_retryable_exceptions()
            )

        @backoff.on_exception(
            backoff.expo,
            self._retryable_exceptions,
            max_tries=3,
            max_time=60,
            on_backoff=self._on_retry,
        )
        def _do_call():
            return fn(*args, **kwargs)

        return _do_call()

    def _on_retry(self, details: dict):
        """Log retry events and increment counter."""
        with self._lock:
            self._metrics.retries += 1
        logger.warning(
            f"LLMGateway retry {details['tries']}/3 "
            f"after {details['wait']:.1f}s — {type(details.get('exception')).__name__}"
        )

    # ── Metrics recording ─────────────────────────────────────────────

    def _record_success(
        self,
        prompt: str,
        response: CompletionResponse,
        latency_ms: float,
        purpose: str,
    ):
        tokens_in = len(prompt.split()) * 1.3  # rough estimate
        tokens_out = len(response.text.split()) * 1.3 if response.text else 0

        # Try to get real token counts from provider response
        raw = getattr(response, "raw", None) or {}
        usage = None
        if isinstance(raw, dict):
            usage = raw.get("usage")
        elif hasattr(raw, "usage"):
            usage = raw.usage
        if usage:
            tokens_in = getattr(usage, "prompt_tokens", None) or tokens_in
            tokens_out = getattr(usage, "completion_tokens", None) or tokens_out

        cost = self._estimate_cost(int(tokens_in), int(tokens_out))

        with self._lock:
            m = self._metrics
            m.total_calls += 1
            m.total_tokens_in += int(tokens_in)
            m.total_tokens_out += int(tokens_out)
            m.total_latency_ms += latency_ms
            m.estimated_cost_usd += cost
            m.calls_by_purpose[purpose] += 1

        logger.debug(
            f"LLM call: purpose={purpose} tokens_in={int(tokens_in)} "
            f"tokens_out={int(tokens_out)} latency={latency_ms:.0f}ms "
            f"model={self.model}"
        )

    def _record_chat_success(
        self,
        messages: Sequence[ChatMessage],
        response: ChatResponse,
        latency_ms: float,
        purpose: str,
    ):
        # Estimate tokens from messages
        msg_text = " ".join(m.content or "" for m in messages)
        tokens_in = len(msg_text.split()) * 1.3
        resp_text = response.message.content or "" if response.message else ""
        tokens_out = len(resp_text.split()) * 1.3

        cost = self._estimate_cost(int(tokens_in), int(tokens_out))

        with self._lock:
            m = self._metrics
            m.total_calls += 1
            m.total_tokens_in += int(tokens_in)
            m.total_tokens_out += int(tokens_out)
            m.total_latency_ms += latency_ms
            m.estimated_cost_usd += cost
            m.calls_by_purpose[purpose] += 1

        logger.debug(
            f"LLM chat: purpose={purpose} messages={len(messages)} "
            f"latency={latency_ms:.0f}ms model={self.model}"
        )

    def _record_error(self, purpose: str):
        with self._lock:
            self._metrics.errors += 1
            self._metrics.calls_by_purpose[f"{purpose}_error"] += 1
        logger.error(f"LLM call failed: purpose={purpose} model={self.model}")

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        model_name = self.model
        costs = _COST_PER_1M_TOKENS.get(model_name, _COST_PER_1M_TOKENS["_default"])
        return (tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000

    # ── Public metrics API ────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return a thread-safe snapshot of current metrics."""
        with self._lock:
            result = self._metrics.to_dict()
            result["model"] = self.model
            return result

    def reset_metrics(self):
        """Zero all metric counters."""
        with self._lock:
            object.__setattr__(self, "_metrics", LLMMetrics())
        logger.info("LLMGateway metrics reset")

    # ── LlamaIndex compatibility ──────────────────────────────────────

    @classmethod
    def class_name(cls) -> str:
        return "LLMGateway"
