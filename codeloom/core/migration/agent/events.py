"""SSE event types emitted by the migration agent loop.

Each event serializes to a single SSE ``data:`` line via ``to_sse()``.
The frontend consumes these to render the real-time agent execution panel.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentEvent:
    """Base class for all agent events."""

    type: str

    def to_sse(self) -> str:
        """Serialize to Server-Sent Events format."""
        return f"data: {json.dumps(asdict(self), default=str)}\n\n"


@dataclass
class AgentStartEvent(AgentEvent):
    """Emitted once when the agent loop begins."""

    type: str = "agent_start"
    turn: int = 0
    max_turns: int = 10
    phase_type: str = ""
    tool_count: int = 0


@dataclass
class ThinkingEvent(AgentEvent):
    """Emitted when the agent produces reasoning text before/between tool calls."""

    type: str = "thinking"
    content: str = ""
    turn: int = 0


@dataclass
class ToolCallEvent(AgentEvent):
    """Emitted when the agent invokes a tool."""

    type: str = "tool_call"
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    turn: int = 0


@dataclass
class ToolResultEvent(AgentEvent):
    """Emitted after a tool execution completes."""

    type: str = "tool_result"
    call_id: str = ""
    result: str = ""
    duration_ms: int = 0
    truncated: bool = False


@dataclass
class OutputEvent(AgentEvent):
    """Emitted when the agent produces its final answer."""

    type: str = "output"
    content: str = ""


@dataclass
class AgentDoneEvent(AgentEvent):
    """Emitted once when the agent loop finishes successfully."""

    type: str = "agent_done"
    turns_used: int = 0
    tools_called: int = 0
    total_ms: int = 0
    token_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent(AgentEvent):
    """Emitted on agent errors."""

    type: str = "error"
    error: str = ""
    recoverable: bool = False
