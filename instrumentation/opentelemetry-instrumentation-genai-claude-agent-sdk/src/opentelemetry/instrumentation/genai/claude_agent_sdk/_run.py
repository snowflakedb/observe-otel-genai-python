# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuilds a span tree from the Claude Agent SDK's message stream.

The SDK does not run the agent in this process. It spawns the Claude Code CLI
and streams typed messages back, so there is no model call to wrap and nothing
emits telemetry on its own. Every span here is reconstructed from the stream:
``message_id`` delimits API calls, ``parent_tool_use_id`` identifies a
subagent, and a ``tool_result`` block ends the tool span it names.

One agent invocation gets one ``invoke_agent`` span, the common ancestor of
everything else. It opens in the caller's active context, so it is the trace
root only when the caller has no span of their own.

    invoke_agent <agent>
      chat <model>                 one per API call
      execute_tool <name>
      execute_tool Task
        invoke_agent <subagent>    nested, keyed by parent_tool_use_id
          chat <model>

Spans do not open and close in call-stack order — a backgrounded tool stays
open while main-thread model calls come and go — so every span is given its
parent explicitly via ``context=`` rather than inheriting the active one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.instrumentation.genai.claude_agent_sdk._content import (
    blocks_of,
    input_message,
    output_message,
    prompt_to_messages,
    text_output,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
    LocalAgentInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    TextPart,
)

_logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "claude-agent-sdk"

# Map key for the main thread's chat context. A space cannot collide with a
# tool_use_id.
_MAIN = " main"

# Non-semconv. Set when a chat span's token counts are a mid-stream snapshot
# rather than the call's final usage, so a consumer can tell the difference
# instead of silently trusting an undercount.
ATTR_USAGE_IS_PARTIAL = "genai.usage.is_partial"

# Non-semconv. The SDK reports cost and nothing in the convention carries it.
ATTR_USAGE_COST_USD = "gen_ai.usage.cost_usd"

# Routing env vars the CLI reads to pick a hosting platform, mapped to the
# semconv provider value. A hosting platform outranks the vendor it hosts, so a
# Bedrock-served Claude model reports aws.bedrock.
_PROVIDER_ENV: tuple[tuple[str, str], ...] = (
    ("CLAUDE_CODE_USE_BEDROCK", "aws.bedrock"),
    ("CLAUDE_CODE_USE_VERTEX", "gcp.vertex_ai"),
    ("CLAUDE_CODE_USE_FOUNDRY", "azure.ai.foundry"),
)

# Task statuses meaning the task has finished. Mirrors the SDK's own set; the
# two lifecycle vocabularies differ (task_notification reports "stopped", while
# task_updated reports the raw "killed"), so both are listed.
_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "stopped", "killed"}
)


def resolve_provider(sdk_options: Any) -> str:
    """Pick the provider before the run starts, so the value is right from the
    moment a span opens.

    ``options.env`` replaces rather than merges, so when it is set it is exactly
    what the CLI subprocess will see; otherwise our own environment applies.
    """
    env = getattr(sdk_options, "env", None)
    if not isinstance(env, dict):
        env = os.environ
    for key, provider in _PROVIDER_ENV:
        value = env.get(key)
        if value and str(value).lower() not in ("0", "false"):
            return provider
    return "anthropic"


def _system_prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt or None
    if isinstance(prompt, dict):
        parts = [
            f"preset: {prompt['preset']}" if prompt.get("preset") else None,
            prompt.get("append"),
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def _fold_input_tokens(usage: dict[str, Any]) -> int:
    """Anthropic reports input_tokens excluding cache reads and writes; the
    convention requires the total, with the cache counters as subsets."""
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )


@dataclass
class _ChatContext:
    """Chat-span state for one agent context: the main thread, or one subagent."""

    key: str
    parent_context: Context
    invocation: InferenceInvocation | None = None
    message_id: str | None = None
    blocks: list[Any] = field(default_factory=list)
    # Mid-stream snapshot from the assistant message; superseded by final_usage.
    usage: dict[str, Any] | None = None
    # The call's real usage, from the message_delta stream event.
    final_usage: dict[str, Any] | None = None
    thinking_tokens: int | None = None
    stop_reason: str | None = None
    model: str | None = None
    # Running conversation, replayed as each chat span's input.
    history: list[InputMessage] = field(default_factory=list)


@dataclass
class _Subagent:
    invocation: LocalAgentInvocation
    output: list[OutputMessage] = field(default_factory=list)


class AgentRun:
    """Turns one SDK message stream into spans.

    ``multi_turn`` selects the agent-invocation boundary. ``query()`` and
    ``receive_response()`` each read exactly one invocation, so their span ends
    with the stream. ``receive_messages()`` is continuous, so it closes an
    invocation on a result that answers a prompt the caller submitted and opens
    the next one on the following message.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        sdk_options: Any = None,
        agent_name: str | None = None,
        conversation_id: str | None = None,
        multi_turn: bool = False,
    ) -> None:
        self._handler = handler
        self._sdk_options = sdk_options
        self._agent_name = agent_name or DEFAULT_AGENT_NAME
        self._conversation_id_override = conversation_id
        self._multi_turn = multi_turn
        self._provider = resolve_provider(sdk_options)

        self._root: LocalAgentInvocation | None = None
        self._root_context: Context | None = None
        self._caller_context: Context = otel_context.get_current()

        self._chats: dict[str, _ChatContext] = {}
        self._tools: dict[str, ToolInvocation] = {}
        self._subagents: dict[str, _Subagent] = {}
        # tool_use_id -> the Task tool_use block that launched a subagent.
        self._launches: dict[str, Any] = {}
        # tool_use_ids the CLI runs as a task. Their tool_result cannot be
        # trusted to mean the work finished.
        self._task_tools: set[str] = set()
        # task_id -> tool_use_id, because task_updated carries only the task_id.
        self._task_ids: dict[str, str] = {}
        # tool_use_id -> the terminal status a task_updated reported, so the
        # status survives until whichever message actually closes the span.
        self._task_terminal: dict[str, str] = {}

        self._session_id: str | None = None
        self._model: str | None = None
        self._compacted = False
        self._prompt: Any = None
        self._final_error: Error | None = None
        self._saw_result = False

    # --- public surface ---------------------------------------------------

    async def iterate(self, start: Any, prompt: Any = None) -> Any:
        """Wrap the SDK's stream, observing each message on the way past."""
        self._prompt = prompt
        try:
            stream = start()
        except Exception as exc:
            self._open_root()
            self._fail_root(exc)
            raise

        try:
            async for message in stream:
                self._observe(message)
                yield message
        except Exception as exc:
            self._fail_root(exc)
            raise
        finally:
            self._dispose()

    def _observe(self, message: Any) -> None:
        """Handle one message without letting it affect the caller.

        A mapping slip must not interrupt the stream, and the caller's active
        context must look untouched: spans are parented explicitly, but
        util-genai still attaches each span it starts, and these spans do not
        finish in the order they opened.
        """
        saved = otel_context.get_current()
        try:
            self._on_message(message)
        except Exception:  # pylint: disable=broad-except
            _logger.debug(
                "claude-agent-sdk instrumentation failed on a message",
                exc_info=True,
            )
        finally:
            if otel_context.get_current() is not saved:
                otel_context.attach(saved)

    # --- dispatch ---------------------------------------------------------

    def _on_message(self, msg: Any) -> None:
        from claude_agent_sdk.types import (  # pylint: disable=import-outside-toplevel
            AssistantMessage,
            ConversationResetMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            TaskUpdatedMessage,
            UserMessage,
        )

        # The Task* messages subclass SystemMessage, so they are matched first.
        if isinstance(msg, TaskStartedMessage):
            self._on_task_started(msg)
        elif isinstance(msg, TaskNotificationMessage):
            self._on_task_notification(msg)
        elif isinstance(msg, TaskUpdatedMessage):
            self._on_task_updated(msg)
        elif isinstance(msg, SystemMessage):
            self._on_system(msg)
        elif isinstance(msg, AssistantMessage):
            self._on_assistant(msg)
        elif isinstance(msg, UserMessage):
            self._on_user(msg)
        elif isinstance(msg, StreamEvent):
            self._on_stream_event(msg)
        elif isinstance(msg, ResultMessage):
            self._on_result(msg)
        elif isinstance(msg, ConversationResetMessage):
            self._session_id = msg.new_conversation_id

    # --- root agent -------------------------------------------------------

    @property
    def _conversation_id(self) -> str | None:
        return self._conversation_id_override or self._session_id

    def _open_root(self) -> LocalAgentInvocation:
        if self._root is not None:
            return self._root
        root = self._handler.invoke_local_agent(
            agent_name=self._agent_name,
            request_model=getattr(self._sdk_options, "model", None)
            or self._model,
            context=self._caller_context,
        )
        root.conversation_id = self._conversation_id
        # LocalAgentInvocation is INTERNAL and carries no provider of its own,
        # but the Explorer reads the provider off every row.
        root.attributes[GenAI.GEN_AI_PROVIDER_NAME] = self._provider
        if self._handler.should_capture_content():
            root.input_messages = prompt_to_messages(self._prompt)
        self._root = root
        self._root_context = root.context
        return root

    def _root_ctx(self) -> Context:
        self._open_root()
        assert self._root_context is not None
        return self._root_context

    # --- chat spans -------------------------------------------------------

    def _chat_context(self, parent_tool_use_id: str | None) -> _ChatContext:
        key = parent_tool_use_id or _MAIN
        existing = self._chats.get(key)
        if existing is not None:
            return existing

        if parent_tool_use_id:
            parent_context = self._subagent(
                parent_tool_use_id
            ).invocation.context
            history: list[InputMessage] = []
        else:
            parent_context = self._root_ctx()
            history = list(prompt_to_messages(self._prompt))

        ctx = _ChatContext(
            key=key, parent_context=parent_context, history=history
        )
        self._chats[key] = ctx
        return ctx

    def _open_chat(
        self, ctx: _ChatContext, message_id: str | None, model: str | None
    ) -> None:
        resolved = (
            model
            or self._model
            or getattr(self._sdk_options, "model", None)
            or "unknown"
        )
        invocation = self._handler.inference(
            self._provider,
            request_model=resolved,
            context=ctx.parent_context,
        )
        invocation.conversation_id = self._conversation_id
        invocation.response_id = message_id
        if self._compacted:
            invocation.conversation_compacted = True
        effort = getattr(self._sdk_options, "effort", None)
        if isinstance(effort, str):
            invocation.reasoning_level = effort
        if self._handler.should_capture_content():
            invocation.input_messages = list(ctx.history)
            system = _system_prompt_text(
                getattr(self._sdk_options, "system_prompt", None)
            )
            if system:
                invocation.system_instruction = [TextPart(content=system)]

        ctx.invocation = invocation
        ctx.message_id = message_id
        ctx.blocks = []
        ctx.usage = None
        ctx.final_usage = None
        ctx.thinking_tokens = None
        ctx.stop_reason = None
        ctx.model = resolved

    def _close_chat(
        self, ctx: _ChatContext, error: Error | None = None
    ) -> None:
        invocation = ctx.invocation
        if invocation is None:
            return
        ctx.invocation = None

        # message_delta carries the call's real totals; the assistant message's
        # usage is only a mid-stream snapshot, measured far below the truth.
        usage = ctx.final_usage or ctx.usage
        if usage:
            invocation.input_tokens = _fold_input_tokens(usage)
            invocation.output_tokens = int(usage.get("output_tokens") or 0)
            read = int(usage.get("cache_read_input_tokens") or 0)
            write = int(usage.get("cache_creation_input_tokens") or 0)
            if read:
                invocation.cache_read_input_tokens = read
            if write:
                invocation.cache_write_input_tokens = write
        if ctx.thinking_tokens:
            invocation.thinking_tokens = ctx.thinking_tokens
        if ctx.final_usage is None:
            invocation.attributes[ATTR_USAGE_IS_PARTIAL] = True

        if ctx.model:
            invocation.response_model_name = ctx.model
        if ctx.stop_reason:
            invocation.finish_reasons = [ctx.stop_reason]
        if self._handler.should_capture_content():
            message = output_message(ctx.blocks, "assistant", ctx.stop_reason)
            invocation.output_messages = [message] if message else []

        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    # --- subagents --------------------------------------------------------

    def _subagent(self, parent_tool_use_id: str) -> _Subagent:
        """Open a subagent span lazily, reading its identity from the Task call
        that launched it.

        Parented to that Task tool's span when it is still open — which it is,
        because a Task tool's real completion cannot arrive before the subagent
        has spoken.
        """
        existing = self._subagents.get(parent_tool_use_id)
        if existing is not None:
            return existing

        launch = self._launches.get(parent_tool_use_id)
        tool_input = getattr(launch, "input", None) or {}
        name = (
            tool_input.get("subagent_type")
            or tool_input.get("agent_type")
            or getattr(launch, "name", None)
            or "subagent"
        )
        parent_tool = self._tools.get(parent_tool_use_id)
        parent_context = (
            parent_tool.context
            if parent_tool is not None
            else self._root_ctx()
        )

        invocation = self._handler.invoke_local_agent(
            agent_name=str(name),
            request_model=tool_input.get("model") or self._model,
            context=parent_context,
        )
        invocation.conversation_id = self._conversation_id
        invocation.attributes[GenAI.GEN_AI_PROVIDER_NAME] = self._provider
        description = tool_input.get("description")
        if isinstance(description, str) and description:
            invocation.agent_description = description
        if self._handler.should_capture_content():
            prompt = tool_input.get("prompt")
            if isinstance(prompt, str) and prompt:
                invocation.input_messages = [
                    InputMessage(role="user", parts=[TextPart(content=prompt)])
                ]

        entry = _Subagent(invocation=invocation)
        self._subagents[parent_tool_use_id] = entry
        return entry

    def _finish_subagent(self, tool_use_id: str) -> None:
        """End a subagent span once the Task tool that spawned it has returned."""
        entry = self._subagents.pop(tool_use_id, None)
        if entry is None:
            return
        ctx = self._chats.pop(tool_use_id, None)
        if ctx is not None:
            self._close_chat(ctx)
        if entry.output and self._handler.should_capture_content():
            entry.invocation.output_messages = entry.output
        entry.invocation.stop()

    # --- tool spans -------------------------------------------------------

    def _open_tool(self, block: Any, parent_tool_use_id: str | None) -> None:
        tool_call_id = getattr(block, "id", None)
        if not tool_call_id or tool_call_id in self._tools:
            return
        name = getattr(block, "name", None) or "unknown_tool"

        if parent_tool_use_id:
            subagent = self._subagents.get(parent_tool_use_id)
            parent_context = (
                subagent.invocation.context
                if subagent is not None
                else self._root_ctx()
            )
        else:
            parent_context = self._root_ctx()

        invocation = self._handler.tool(
            name=name,
            tool_type="extension" if name.startswith("mcp__") else "function",
            agent_name=self._agent_name,
            context=parent_context,
        )
        invocation.tool_call_id = tool_call_id
        # ToolInvocation has no conversation_id field of its own, and a span
        # without one does not appear in the Conversations view at all.
        if self._conversation_id:
            invocation.attributes[GenAI.GEN_AI_CONVERSATION_ID] = (
                self._conversation_id
            )
        if invocation.should_capture_content:
            invocation.arguments = getattr(block, "input", None)
        self._tools[tool_call_id] = invocation

    def _close_tool(
        self,
        tool_call_id: str | None,
        result: Any = None,
        error_message: str | None = None,
    ) -> None:
        if not tool_call_id:
            return
        # A Task tool owns a nested subagent span. End the child first: a child
        # outliving its parent renders as a bar sticking out past it.
        self._finish_subagent(tool_call_id)
        invocation = self._tools.pop(tool_call_id, None)
        status = self._task_terminal.pop(tool_call_id, None)
        if invocation is None:
            return
        if error_message is None and status not in (None, "completed"):
            error_message = f"task {status}"
        if result is not None and invocation.should_capture_content:
            invocation.tool_result = result
        if error_message is not None:
            invocation.fail(Error(type="tool_error", message=error_message))
        else:
            invocation.stop()

    # --- message handlers -------------------------------------------------

    def _on_system(self, msg: Any) -> None:
        if msg.subtype == "init":
            data = msg.data or {}
            self._session_id = data.get("session_id") or self._session_id
            self._model = data.get("model") or self._model
            self._open_root()
            return
        if msg.subtype == "compact_boundary":
            self._compacted = True
            root = self._open_root()
            root.attributes["gen_ai.conversation.compacted"] = True

    def _on_task_started(self, msg: Any) -> None:
        """The CLI registered this call as a task.

        Always ahead of that call's tool_result, whether it then runs inline or
        in the background.
        """
        if msg.tool_use_id:
            self._task_tools.add(msg.tool_use_id)
            self._task_ids[msg.task_id] = msg.tool_use_id

    def _on_task_notification(self, msg: Any) -> None:
        """The task really finished. ``summary`` is its output — the same text an
        inline tool_result carries, and what a backgrounded one lacks."""
        tool_use_id = msg.tool_use_id or self._task_ids.get(msg.task_id)
        if not tool_use_id:
            return
        self._task_tools.discard(tool_use_id)
        self._task_ids.pop(msg.task_id, None)
        failed = (
            msg.status in _TERMINAL_TASK_STATUSES and msg.status != "completed"
        )
        self._close_tool(
            tool_use_id,
            msg.summary,
            f"task {msg.status}" if failed else None,
        )

    def _on_task_updated(self, msg: Any) -> None:
        """The task has reached a terminal state.

        This does not close the span. Measured on CLI 2.1.220, a completed task
        reports here first and its task_notification — which carries the real
        output — arrives on the same millisecond straight after; closing here
        would drop that output.

        What it does is stop treating the task's tool_result as an untrustworthy
        receipt, so whichever signal arrives next closes the span. That covers
        the case the SDK documents where a task stopped via TaskStop reports
        "killed" here and its notification is suppressed entirely, which would
        otherwise leak the span until teardown.
        """
        status = msg.status or (msg.patch or {}).get("status")
        if status not in _TERMINAL_TASK_STATUSES:
            return
        tool_use_id = self._task_ids.get(msg.task_id)
        if not tool_use_id:
            return
        self._task_tools.discard(tool_use_id)
        self._task_terminal[tool_use_id] = status

    def _on_assistant(self, msg: Any) -> None:
        parent = msg.parent_tool_use_id
        blocks = blocks_of(msg.content)
        if msg.model:
            self._model = msg.model

        # Tool spans open from the stream, not from a tool result: only the
        # stream carries parent_tool_use_id, which is what puts the span in the
        # right place.
        for block in blocks:
            block_id = getattr(block, "id", None)
            if block_id and getattr(block, "name", None) is not None:
                self._launches[block_id] = block

        ctx = self._chat_context(parent)

        # A new message_id means a new API call; the same id means another
        # content block of the call already in flight.
        if ctx.invocation is None or (
            msg.message_id and msg.message_id != ctx.message_id
        ):
            self._close_chat(ctx)
            self._open_chat(ctx, msg.message_id, msg.model)

        ctx.blocks.extend(blocks)
        if msg.usage:
            ctx.usage = msg.usage
        if msg.stop_reason:
            ctx.stop_reason = msg.stop_reason

        for block in blocks:
            if _is_tool_use(block):
                self._open_tool(block, parent)

        message = output_message(blocks, "assistant", msg.stop_reason)
        if message is not None:
            ctx.history.append(
                InputMessage(role="assistant", parts=message.parts)
            )
            if parent and parent in self._subagents:
                self._subagents[parent].output.append(message)

    def _on_user(self, msg: Any) -> None:
        parent = msg.parent_tool_use_id
        ctx = self._chat_context(parent)
        blocks = blocks_of(msg.content)

        # A user message closes the API call in flight — the next assistant
        # message belongs to a fresh request.
        self._close_chat(ctx)

        saw_tool_result = False
        for block in blocks:
            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id:
                continue
            saw_tool_result = True
            # A task waits for its notification; this may only be a receipt.
            if tool_use_id in self._task_tools:
                continue
            self._close_tool(
                tool_use_id,
                getattr(block, "content", None),
                "tool error" if getattr(block, "is_error", None) else None,
            )

        message = input_message(blocks, "tool" if saw_tool_result else "user")
        if message is not None:
            ctx.history.append(message)

    def _on_stream_event(self, msg: Any) -> None:
        key = msg.parent_tool_use_id or _MAIN
        ctx = self._chats.get(key)
        if ctx is None or ctx.invocation is None:
            return
        event = msg.event or {}
        event_type = event.get("type")

        # message_delta closes a call on the wire: it is the one event carrying
        # the call's final usage and stop reason. Without it we are left with
        # the mid-stream snapshot, which understates output tokens badly.
        if event_type == "message_delta":
            if ctx.final_usage is not None:
                return
            usage = event.get("usage")
            if usage:
                ctx.final_usage = usage
                details = usage.get("output_tokens_details") or {}
                thinking = details.get("thinking_tokens")
                if isinstance(thinking, int) and thinking > 0:
                    ctx.thinking_tokens = thinking
            stop = (event.get("delta") or {}).get("stop_reason")
            if stop:
                ctx.stop_reason = stop
            return

        # Let util-genai time the stream from real chunk arrivals, which is what
        # gen_ai.client.time_to_first_chunk and time_per_output_chunk measure.
        if event_type == "content_block_delta":
            ctx.invocation.record_stream_chunk()

    def _on_result(self, msg: Any) -> None:
        """A result is a checkpoint, not necessarily the end.

        It closes the model call in flight and supersedes the previous result,
        but tools are left alone — a backgrounded delegation legitimately
        outlives the result that announced it.
        """
        root = self._open_root()
        self._session_id = msg.session_id or self._session_id
        self._saw_result = True

        error = _result_error(msg)
        # An API failure is the one result error that belongs on the model call:
        # the CLI relays it as a synthetic assistant message, so that call's
        # span is open and would otherwise close as a success.
        api_error = error if msg.subtype == "success" else None
        for ctx in list(self._chats.values()):
            self._close_chat(ctx, api_error if ctx.key == _MAIN else None)
        self._chats.clear()
        self._final_error = error

        # model_usage is cumulative over the whole invocation, so the last
        # result to arrive leaves the right value behind.
        totals = _fold_model_usage(msg.model_usage)
        if totals["input_tokens"]:
            root.input_tokens = totals["input_tokens"]
        if totals["output_tokens"]:
            root.output_tokens = totals["output_tokens"]
        if totals["cache_read"]:
            root.attributes[GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = (
                totals["cache_read"]
            )
        if totals["cache_write"]:
            root.attributes[GenAI.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
                totals["cache_write"]
            )
        cost = totals["cost_usd"] or (msg.total_cost_usd or 0.0)
        if cost:
            root.attributes[ATTR_USAGE_COST_USD] = cost
        if msg.stop_reason:
            root.finish_reasons = [msg.stop_reason]
        if (
            self._handler.should_capture_content()
            and isinstance(msg.result, str)
            and msg.result
            and not msg.is_error
        ):
            root.output_messages = text_output(msg.result)

        if self._multi_turn and _answers_caller(msg):
            self._end_root()

    # --- teardown ---------------------------------------------------------

    def _end_root(self) -> None:
        """Close everything still open and end the agent span."""
        for tool_use_id in list(self._subagents):
            self._finish_subagent(tool_use_id)
        for tool_use_id in list(self._tools):
            status = self._task_terminal.get(tool_use_id)
            if status == "completed":
                reason = None
            elif status is not None:
                reason = f"task {status}"
            else:
                reason = "tool never returned"
            self._close_tool(tool_use_id, None, reason)
        for ctx in list(self._chats.values()):
            self._close_chat(ctx)
        self._chats.clear()
        self._launches.clear()
        self._task_tools.clear()
        self._task_ids.clear()
        self._task_terminal.clear()

        root, self._root = self._root, None
        self._root_context = None
        self._compacted = False
        error, self._final_error = self._final_error, None
        self._saw_result = False
        if root is None:
            return
        if error is not None:
            root.fail(error)
        else:
            root.stop()

    def _fail_root(self, exc: BaseException) -> None:
        error = Error.from_exception(exc)
        for ctx in list(self._chats.values()):
            self._close_chat(ctx, error)
        self._chats.clear()
        self._final_error = error
        self._end_root()

    def _dispose(self) -> None:
        """Close anything the stream left open — a break, or an abandoned
        generator."""
        if self._root is None:
            return
        if not self._saw_result and self._final_error is None:
            self._final_error = Error(
                type="incomplete_stream",
                message="agent stream ended before any result message",
            )
        self._end_root()


def _is_tool_use(block: Any) -> bool:
    """A client-side tool call, which becomes an execute_tool span.

    Server tool calls are executed by the API on the model's behalf and never
    come back for a result, so they are content only.
    """
    return (
        getattr(block, "id", None) is not None
        and getattr(block, "name", None) is not None
        and getattr(block, "input", None) is not None
        and not type(block).__name__.startswith("ServerTool")
    )


def _answers_caller(msg: Any) -> bool:
    """True when this result answers a prompt the caller submitted.

    In streaming-input mode the session also injects its own turns — background
    task notifications, scheduled prompts, messages relayed from peers — and
    each produces a result of its own. Those are internal checkpoints.
    """
    origin = getattr(msg, "origin", None)
    if origin is None:
        return True
    return origin.get("kind") == "human"


def _fold_model_usage(model_usage: Any) -> dict[str, Any]:
    """Sum per-model usage across models.

    No subtraction: model_usage is cumulative over the whole invocation, which
    is exactly the span it lands on. It also covers subagents and compaction,
    which result.usage does not.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cost_usd": 0.0,
    }
    for usage in (model_usage or {}).values():
        totals["input_tokens"] += int(usage.get("inputTokens") or 0)
        totals["output_tokens"] += int(usage.get("outputTokens") or 0)
        totals["cache_read"] += int(usage.get("cacheReadInputTokens") or 0)
        totals["cache_write"] += int(
            usage.get("cacheCreationInputTokens") or 0
        )
        totals["cost_usd"] += float(usage.get("costUSD") or 0.0)
    # The convention wants input_tokens to include the cached counts.
    totals["input_tokens"] += totals["cache_read"] + totals["cache_write"]
    return totals


def _result_error(msg: Any) -> Error | None:
    """The error a result reports, if any.

    Which field holds the text depends on the subtype, following the rule the
    SDK applies to its own thrown error: on an API failure the subtype is still
    "success" and the text sits in ``result``; the early-stop subtypes put it
    in ``errors``.
    """
    if not msg.is_error:
        return None
    if msg.subtype == "success":
        text = msg.result if isinstance(msg.result, str) else ""
    else:
        text = "; ".join(
            e for e in (msg.errors or []) if isinstance(e, str) and e.strip()
        )
    status = getattr(msg, "api_error_status", None)
    if isinstance(status, int):
        text = f"{text} (HTTP {status})" if text else f"HTTP {status}"
    return Error(
        type=msg.subtype if msg.subtype != "success" else "api_error",
        message=text
        or f"claude agent run failed: {getattr(msg, 'terminal_reason', None) or msg.subtype}",
    )
