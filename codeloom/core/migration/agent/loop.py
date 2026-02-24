"""Agentic migration loop -- iterative tool-use execution for migration phases.

Replaces single-shot ``llm.complete(prompt)`` with a multi-turn conversation
where the LLM can call tools (read source, search codebase, look up docs,
validate syntax) and iterate until it produces a final answer.

The loop yields ``AgentEvent`` objects that serialize to SSE for real-time
frontend visualization.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from .events import (
    AgentDoneEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    OutputEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .tools import ToolDefinition

logger = logging.getLogger(__name__)

# Maximum characters of tool output to feed back into conversation
_MAX_TOOL_RESULT_CHARS = 90000


class MigrationAgent:
    """Agentic migration executor using an iterative tool-use loop.

    The agent receives a system prompt (migration instructions), a task prompt
    (phase-specific context), and a set of tools. It calls ``llm.chat()`` with
    function-calling schemas, executes any requested tools, feeds results back,
    and repeats until the LLM produces a final text answer or ``max_turns`` is
    reached.

    Usage::

        agent = MigrationAgent(llm=gateway, tools=tool_list, system_prompt="...")
        for event in agent.execute(task_prompt="...", phase_type="transform"):
            yield event.to_sse()
        # agent.result contains the final parsed output
    """

    def __init__(
        self,
        llm,
        tools: List[ToolDefinition],
        system_prompt: str,
        max_turns: int = 10,
    ):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self.result: Optional[str] = None
        self._total_tool_calls = 0

    def execute(
        self,
        task_prompt: str,
        phase_type: str = "transform",
    ) -> Generator[AgentEvent, None, None]:
        """Run the agent loop, yielding events for SSE streaming.

        Args:
            task_prompt: The phase-specific prompt/instructions for the LLM.
            phase_type: Label for UI display (e.g. "transform", "analyze").

        Yields:
            AgentEvent subclasses for real-time frontend rendering.
        """
        start_ms = _now_ms()
        yield AgentStartEvent(
            turn=0,
            max_turns=self._max_turns,
            phase_type=phase_type,
            tool_count=len(self._tools),
        )

        messages: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
            ChatMessage(role=MessageRole.USER, content=task_prompt),
        ]

        tool_schemas = [t.to_openai_schema() for t in self._tools.values()]

        for turn in range(self._max_turns):
            # On the last turn, drop tools to force the LLM to produce a final answer
            force_final = turn >= self._max_turns - 1
            effective_tools = [] if force_final else tool_schemas

            if force_final:
                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "You have used all available tool turns. "
                            "Produce your FINAL answer now using the information you have gathered. "
                            "Follow the output format specified in the task instructions EXACTLY."
                        ),
                    )
                )
                yield ThinkingEvent(
                    content="Reached final turn — producing answer with gathered context.",
                    turn=turn,
                )

            try:
                response = self._llm.chat(
                    messages,
                    tools=effective_tools,
                    gateway_purpose="migration_agent",
                )
            except Exception as e:
                err_str = str(e)

                # tool_use_failed: the model tried to produce output but the
                # API parsed it as a malformed tool call. Retry without tools
                # so the model can emit its answer as plain text.
                if "tool_use_failed" in err_str:
                    logger.warning(
                        "tool_use_failed on turn %d — retrying without tools",
                        turn,
                    )
                    yield ThinkingEvent(
                        content="Tool call format error — retrying without tools to produce final output.",
                        turn=turn,
                    )
                    try:
                        response = self._llm.chat(
                            messages,
                            tools=[],
                            gateway_purpose="migration_agent",
                        )
                    except Exception as retry_err:
                        logger.error(
                            "Retry without tools also failed on turn %d: %s",
                            turn, retry_err,
                        )
                        yield ErrorEvent(error=str(retry_err), recoverable=False)
                        return

                    # The retry response has no tool calls — treat as final answer
                    text_content = _extract_text(response)
                    if text_content:
                        yield OutputEvent(content=text_content)
                    self.result = text_content
                    yield AgentDoneEvent(
                        turns_used=turn + 1,
                        tools_called=self._total_tool_calls,
                        total_ms=_now_ms() - start_ms,
                    )
                    return

                logger.error("Agent LLM call failed on turn %d: %s", turn, e)
                yield ErrorEvent(error=err_str, recoverable=False)
                return

            # Parse response blocks
            tool_calls = _extract_tool_calls(response)
            text_content = _extract_text(response)

            if not tool_calls:
                # No tool calls = final answer
                if text_content:
                    yield OutputEvent(content=text_content)
                self.result = text_content
                yield AgentDoneEvent(
                    turns_used=turn + 1,
                    tools_called=self._total_tool_calls,
                    total_ms=_now_ms() - start_ms,
                )
                return

            # Emit thinking text if present alongside tool calls
            if text_content:
                yield ThinkingEvent(content=text_content, turn=turn)

            # Append the assistant's message (with tool calls) to history
            messages.append(response.message)

            # Execute each tool call
            for tc in tool_calls:
                call_id = tc.tool_call_id or str(uuid.uuid4())[:8]
                tool_name = tc.tool_name
                # LlamaIndex ToolCallBlock may store args in tool_kwargs (dict)
                # or as a JSON string that needs parsing.
                raw_kwargs = tc.tool_kwargs
                if isinstance(raw_kwargs, dict):
                    tool_args = raw_kwargs
                elif isinstance(raw_kwargs, str):
                    try:
                        tool_args = json.loads(raw_kwargs)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                else:
                    tool_args = {}
                logger.debug("Tool call: %s args=%s (raw type=%s)", tool_name, tool_args, type(raw_kwargs).__name__)

                yield ToolCallEvent(
                    tool=tool_name,
                    args=tool_args,
                    call_id=call_id,
                    turn=turn,
                )

                # Execute the tool
                t0 = _now_ms()
                result_text = self._execute_tool(tool_name, tool_args)
                duration = _now_ms() - t0
                truncated = len(result_text) > _MAX_TOOL_RESULT_CHARS
                if truncated:
                    result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n...(truncated)"

                self._total_tool_calls += 1

                yield ToolResultEvent(
                    call_id=call_id,
                    result=result_text,
                    duration_ms=duration,
                    truncated=truncated,
                )

                # Append tool result to conversation history
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=result_text,
                        additional_kwargs={"tool_call_id": call_id, "name": tool_name},
                    )
                )

        # Exhausted max_turns without final answer
        yield ErrorEvent(
            error=f"Agent reached max_turns ({self._max_turns}) without producing a final answer.",
            recoverable=True,
        )

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a single tool by name and return the string result."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self._tools.keys())}"
        try:
            return tool.execute(args)
        except Exception as e:
            logger.warning("Tool '%s' failed: %s", tool_name, e, exc_info=True)
            return f"Tool error: {e}"


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: ChatResponse) -> List[ToolCallBlock]:
    """Extract ToolCallBlock instances from a ChatResponse."""
    blocks = getattr(response.message, "blocks", []) or []
    return [b for b in blocks if isinstance(b, ToolCallBlock)]


def _extract_text(response: ChatResponse) -> str:
    """Extract concatenated text content from a ChatResponse."""
    blocks = getattr(response.message, "blocks", []) or []
    parts = []
    for b in blocks:
        if isinstance(b, TextBlock) and b.text:
            parts.append(b.text)
    return "\n".join(parts).strip()


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)
