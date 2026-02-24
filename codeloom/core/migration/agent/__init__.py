"""Agentic migration engine -- tool-use loop for migration phases.

Replaces single-shot LLM calls with an iterative agent that can
read source code, search the codebase, look up framework docs,
and validate output on demand.
"""

from .loop import MigrationAgent
from .events import (
    AgentEvent,
    AgentStartEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    OutputEvent,
    AgentDoneEvent,
    ErrorEvent,
)
from .tools import ToolDefinition, build_tools_for_phase

__all__ = [
    "MigrationAgent",
    "AgentEvent",
    "AgentStartEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "OutputEvent",
    "AgentDoneEvent",
    "ErrorEvent",
    "ToolDefinition",
    "build_tools_for_phase",
]
