# Copyright Snowflake Inc.
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
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from observeinc.instrumentation.genai.claude_agent_sdk._content import (
    blocks_of,
    input_messages,
    output_message,
    prompt_to_messages,
    text_output,
)
from observeinc.instrumentation.genai.claude_agent_sdk._types import (
    InitData,
    StreamEventPayload,
    TaskToolInput,
    Usage,
    UsageTotals,
)
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.invocation import (
    InferenceInvocation,
    LocalAgentInvocation,
    ToolInvocation,
)
from observeinc.util.genai.types import (
    Error,
    GenericPart,
    InputMessage,
    OutputMessage,
    SystemInstructionPart,
    TextPart,
)
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

if TYPE_CHECKING:
    # The SDK is an optional dependency: it must stay importable-at-type-time
    # only, so that this module imports without it installed.
    from claude_agent_sdk.types import (
        ClaudeAgentOptions,
        ModelUsage,
        ResultMessage,
        StreamEvent,
        SystemMessage,
        SystemPromptCustom,
        SystemPromptFile,
        SystemPromptPreset,
        TaskUpdatedMessage,
        ToolUseBlock,
        UserMessage,
    )

_logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "claude-agent-sdk"

# Map key for the main thread's chat context. A space cannot collide with a
# tool_use_id.
_MAIN = " main"

# Prefix for the attributes the convention has no key for. One prefix for all of
# them, and a vendor-scoped one: a custom key under gen_ai.* squats on a name a
# future semconv cost attribute will claim, and "genai." — which is_partial used
# to sit under — is one character away from the real namespace.
_VENDOR = "claude_agent_sdk"

# Non-semconv. Set when a chat span's token counts are a mid-stream snapshot
# rather than the call's final usage, so a consumer can tell the difference
# instead of silently trusting an undercount.
ATTR_USAGE_IS_PARTIAL = f"{_VENDOR}.usage.is_partial"

# Non-semconv. The SDK reports cost and nothing in the convention carries it.
ATTR_USAGE_COST_USD = f"{_VENDOR}.usage.cost_usd"

# The name util-genai gives cache-creation tokens on inference spans. Written
# here by hand because LocalAgentInvocation has no field for it, and spelled out
# rather than taken from semconv's GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS so
# that one trace reports this count under one name on every span.
ATTR_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"

# Likewise: LocalAgentInvocation has no compaction field, so the agent span
# writes this key directly while a chat span reaches the same key through
# InferenceInvocation.conversation_compacted. One constant for the one path that
# spells it out, so the two cannot drift silently.
ATTR_CONVERSATION_COMPACTED = "gen_ai.conversation.compacted"

_PROVIDER = GenAI.GenAiProviderNameValues

# Routing env vars the CLI reads to pick a hosting platform, in the order it
# checks them as of CLI 2.1.220 — the order decides the answer when two are set.
# Mantle is a second Bedrock endpoint; the two "Claude Platform on <cloud>" flags
# reach Anthropic's own API.
#
# The CLI can also route through a gateway picked from session auth state rather
# than the environment, which cannot be seen from here.
_PROVIDER_ENV: tuple[tuple[str, str], ...] = (
    ("CLAUDE_CODE_USE_BEDROCK", _PROVIDER.AWS_BEDROCK.value),
    ("CLAUDE_CODE_USE_FOUNDRY", _PROVIDER.AZURE_AI_INFERENCE.value),
    ("CLAUDE_CODE_USE_ANTHROPIC_AWS", _PROVIDER.ANTHROPIC.value),
    ("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", _PROVIDER.ANTHROPIC.value),
    ("CLAUDE_CODE_USE_MANTLE", _PROVIDER.AWS_BEDROCK.value),
    ("CLAUDE_CODE_USE_VERTEX", _PROVIDER.GCP_VERTEX_AI.value),
)

# How the CLI reads these flags: on only if the trimmed, lowercased value is in
# this allowlist. A denylist would read "no" and "off" as on.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Task statuses meaning the task has finished. Mirrors the SDK's own set; the
# two lifecycle vocabularies differ (task_notification reports "stopped", while
# task_updated reports the raw "killed"), so both are listed.
_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "stopped", "killed"}
)

# The tool names that launch a subagent, and so the only ones whose input is a
# TaskToolInput. Both, because CLI 2.1.220 accepts either and resolves a
# subagent_type for "Agent" and "Task" alike; every recorded session uses "Agent".
_DELEGATION_TOOLS = frozenset({"Agent", "Task"})

# Terminal reasons of a turn the caller stopped with interrupt(). Its result is
# an error, but the run did not fail.
_INTERRUPTED = frozenset({"aborted_streaming", "aborted_tools"})

# How many messages of conversation a chat span replays as its input. Bounded so
# one span's attributes stay small enough for an exporter to accept.
_MAX_HISTORY_MESSAGES = 40


def resolve_provider(sdk_options: ClaudeAgentOptions | None) -> str:
    """Pick the provider before the run starts, so the value is right from the
    moment a span opens.

    The CLI subprocess sees ``options.env`` merged over our own environment one
    key at a time, so a routing flag is resolved the same way: the option wins
    where it sets that key, and the ambient value applies everywhere else.
    ``options.env`` defaults to an empty dict rather than None, so its presence
    says nothing about whether the caller meant to override routing. Setting a
    flag to the empty string is how the CLI documents clearing one inherited
    from a shell profile, and an empty value is not in ``_TRUTHY``, so it reads
    as unset here too.

    Flags are checked in the CLI's own resolution order, which is not the order
    they are documented in: two of them set together resolve to whichever comes
    first below, the same one the CLI would pick.
    """
    raw_env: object = getattr(sdk_options, "env", None)
    options_env: dict[str, object] = (
        cast("dict[str, object]", raw_env) if isinstance(raw_env, dict) else {}
    )
    for key, provider in _PROVIDER_ENV:
        value = options_env.get(key, os.environ.get(key))
        if value is not None and str(value).strip().lower() in _TRUTHY:
            return provider
    return _PROVIDER.ANTHROPIC.value


@dataclass
class _SystemPromptSource(GenericPart):
    """Names a system prompt the CLI assembles rather than one the caller wrote.

    The preset and file forms identify a prompt without carrying its text, so
    recording either as a TextPart puts a label where the schema expects an
    instruction. GenericPart is the schema's slot for a provider-specific part,
    and the fields a subclass adds serialize alongside its type.
    """

    type: str = f"{_VENDOR}.system_prompt_source"
    source: str = ""
    name: str = ""


def _system_prompt_parts(
    prompt: str
    | SystemPromptPreset
    | SystemPromptCustom
    | SystemPromptFile
    | None,
) -> list[SystemInstructionPart]:
    """The system prompt as instruction parts.

    Only text the caller actually supplied is an instruction. The custom form
    holds the real, full prompt under "prompt" and was being dropped, while the
    bare string form — the same content through a different API shape — was
    captured; which of the two a caller reached for decided whether it was
    recorded. The preset and file forms name a prompt this process never sees, so
    they say so instead of standing in for instruction text: "preset:
    claude_code" was a label, not something the model was told.
    """
    if isinstance(prompt, str):
        return [TextPart(content=prompt)] if prompt else []
    if not isinstance(prompt, dict):
        return []
    kind = prompt.get("type")
    if kind == "custom":
        text = cast("SystemPromptCustom", prompt).get("prompt")
        return [TextPart(content=text)] if text else []
    if kind == "file":
        path = cast("SystemPromptFile", prompt).get("path")
        return [_SystemPromptSource(source="file", name=path)] if path else []
    if kind != "preset":
        return []
    preset = cast("SystemPromptPreset", prompt)
    parts: list[SystemInstructionPart] = []
    preset_name = preset.get("preset")
    if preset_name:
        parts.append(_SystemPromptSource(source="preset", name=preset_name))
    # append is the caller's own text, bolted onto the preset.
    append = preset.get("append")
    if append:
        parts.append(TextPart(content=append))
    return parts


def _merge_usage(snapshot: Usage | None, final: Usage | None) -> Usage | None:
    """The call's usage, preferring ``message_delta``'s counts field by field.

    Only ``output_tokens`` is guaranteed on a ``message_delta``; the input and
    cache counters are conditional. Taking the delta wholesale therefore throws
    away whatever the assistant message's snapshot knew — a delta carrying just
    ``{"output_tokens": 812}`` reported zero input tokens and no cache counters
    against a snapshot holding 42,500. Anthropic's SDK stream accumulator merges
    these the same way, per field.
    """
    if final is None:
        return snapshot
    if snapshot is None:
        return final
    merged = cast("Usage", dict(snapshot))
    merged.update(final)
    return merged


def _fold_input_tokens(usage: Usage) -> int:
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
    blocks: list[object] = field(default_factory=list[object])
    # Mid-stream snapshot from the assistant message. Kept alongside final_usage
    # rather than replaced by it: see _merge_usage.
    usage: Usage | None = None
    # The call's real usage, from the message_delta stream event.
    final_usage: Usage | None = None
    # A user message ended this call while its final usage was still on the
    # wire; the next message_delta closes the span.
    pending_close: bool = False
    thinking_tokens: int | None = None
    stop_reason: str | None = None
    # The model the API said answered, from message_start or the assistant
    # message. Not the requested model: a fallback taken from the session's
    # configuration or from options is a request, and reporting it as the
    # response would claim the API confirmed something it never said.
    response_model: str | None = None
    # Running conversation, replayed as each chat span's input.
    history: list[InputMessage] = field(default_factory=list[InputMessage])
    # The prompt a subagent was launched with. The CLI forwards it as a user
    # message for a foreground subagent but not for a background one.
    launch_prompt: list[InputMessage] = field(
        default_factory=list[InputMessage]
    )

    def set_final_usage(self, usage: Usage) -> None:
        self.final_usage = usage
        details = usage.get("output_tokens_details") or {}
        thinking = details.get("thinking_tokens")
        if isinstance(thinking, int) and thinking > 0:
            self.thinking_tokens = thinking


@dataclass
class _Subagent:
    invocation: LocalAgentInvocation
    output: list[OutputMessage] = field(default_factory=list[OutputMessage])


@dataclass
class _ToolCall:
    """Everything one ``tool_use_id`` accumulates, from launch to close.

    The stream reports a call's fate across several messages that can arrive in
    any order, so most fields start empty and are filled as it goes. An entry
    exists as soon as a ``tool_use`` block is seen, which is before — and
    sometimes without — a span ever opening for it.
    """

    # The span, while it is open. None once closed, and for a block that never
    # opened one.
    invocation: ToolInvocation | None = None
    # The tool_use block. A delegation tool's subagent takes its name, model and
    # prompt from here, so it outlives the tool's own span.
    launch: object | None = None
    # The subagent this call runs inside, if any. Lets a subtree be closed from
    # the leaves up, so no child outlives its parent.
    owner: str | None = None
    # The CLI runs this call as a task and it has not reported finishing. While
    # true, its tool_result is a receipt rather than a completion.
    task_running: bool = False
    # Its tool_result has already been spent as a receipt, so no later one is
    # coming and whatever arrives next has to close the span.
    receipt_spent: bool = False
    # Close held for exactly one message, waiting for a task_notification that
    # usually follows on the same millisecond. See _flush_task_closes.
    close_queued: bool = False
    # The terminal status a task_updated reported, kept until the span closes.
    terminal_status: str | None = None
    # The subagent this call launched, while its span is open.
    subagent: _Subagent | None = None
    # That subagent's span has ended. A later message naming it must not reopen
    # the subtree.
    subagent_finished: bool = False

    @property
    def outstanding(self) -> bool:
        """The turn has to wait for this call: its task, a subagent that has not
        reported finishing, or a close waiting on the task's output."""
        return (
            self.task_running
            or self.close_queued
            or (self.subagent is not None and self.terminal_status is None)
        )


class AgentRun:
    """Turns one SDK message stream into spans.

    ``multi_turn`` selects the agent-invocation boundary. ``query()`` reads
    exactly one invocation, so its span ends with the stream. A client
    connection is one continuous stream, however many ``receive_response()``
    calls read it, so it opens one invocation per turn. A turn ends at its
    result, or, when background work it started is still running, once that work
    finishes or the next turn's init arrives. A reader that stops early leaves the
    turn open for the next read; ``close()`` ends it.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        sdk_options: Any = None,
        agent_name: str | None = None,
        multi_turn: bool = False,
    ) -> None:
        self._handler = handler
        self._sdk_options = sdk_options
        self._agent_name = agent_name or DEFAULT_AGENT_NAME
        self._conversation_id_override: str | None = None
        self._multi_turn = multi_turn
        self._provider = resolve_provider(sdk_options)
        # Whether the CLI forwards stream events. Off, a call's final usage
        # never arrives and a span must not be held open waiting for it.
        self._partial_messages = bool(
            getattr(sdk_options, "include_partial_messages", False)
        )

        self._root: LocalAgentInvocation | None = None
        self._root_context: Context | None = None
        # The caller's context at the last submit(). A turn the CLI starts by
        # itself opens here.
        self._caller_context: Context = otel_context.get_current()

        # Chat-span state per agent context, keyed by _MAIN or a subagent's
        # parent_tool_use_id. A different keyspace from _calls: one agent context
        # runs many tool calls.
        self._chats: dict[str, _ChatContext] = {}
        # tool_use_id -> everything known about that call. One entry per call, so
        # a handler cannot update half of its state.
        self._calls: dict[str, _ToolCall] = {}
        # task_id -> tool_use_id, because task_updated carries only the task_id.
        self._task_ids: dict[str, str] = {}
        # The turn has its result, and ends once no call is outstanding.
        self._end_pending = False
        # tool_use_ids the CLI may still report on after their turn or subagent
        # ended. Anything naming one is dropped, and one closed before it
        # reported back is not a failure.
        self._detached: set[str] = set()

        # The CLI's cumulative counters as of the last result, and their value
        # when the previous turn ended. See _usage_delta.
        self._usage_totals: UsageTotals = _zero_usage()
        self._usage_baseline: UsageTotals = _zero_usage()
        # A resumed or continued session's counters carry on from the earlier
        # session's, by an amount the stream never states. Until a turn has set
        # the baseline from them, a turn's own share is unknown.
        self._inherits_usage = bool(
            getattr(sdk_options, "resume", None)
            or getattr(sdk_options, "continue_conversation", False)
        )

        self._session_id: str | None = None
        self._model: str | None = None
        self._compacted = False
        self._prompt: Any = None
        # What the caller sent, in order, for the turns about to open.
        self._submitted: deque[tuple[Any, str | None, Context]] = deque()
        self._final_error: Error | None = None
        self._saw_result = False

    # --- public surface ---------------------------------------------------

    def submit(self, prompt: Any, conversation_id: str | None = None) -> None:
        """Record what the caller sent, and the context they sent it in, for the
        next turn to report and open under.

        Only a plain prompt is queued. A streamed one can hold any number of
        turns, including none, so it cannot be matched to one.
        """
        self._caller_context = otel_context.get_current()
        if isinstance(prompt, str):
            self._submitted.append(
                (prompt, conversation_id, self._caller_context)
            )

    def close(self) -> None:
        """End whatever is still open, as when the connection goes away."""
        self._teardown(self._stop)

    async def iterate(self, start: Any) -> Any:
        """Wrap the SDK's stream, observing each message on the way past."""
        try:
            stream = start()
        except Exception as exc:
            self._teardown(self._fail_start, exc)
            raise

        try:
            async for message in stream:
                self._observe(message)
                yield message
        except GeneratorExit:
            # The consumer stopped reading, which is not a failure. A client
            # connection goes on: receive_response() stops at each result and
            # the next call picks up the same stream.
            if not self._multi_turn:
                self._teardown(self._stop)
            raise
        except BaseException as exc:
            # BaseException so a cancelled or interrupted run is recorded as
            # such, rather than as a stream that stopped for no stated reason.
            # A turn that already has its result was only waiting on background
            # work, so being cancelled then is not a failure.
            if self._end_pending and not isinstance(exc, Exception):
                self._teardown(self._end_root)
            else:
                self._teardown(self._fail_root, exc)
            raise
        self._teardown(self._dispose)

    def _observe(self, message: Any) -> None:
        """Handle one message without letting it affect the caller: a mapping
        slip must not interrupt the stream or change the caller's context."""
        with _keep_current_context():
            try:
                self._on_message(message)
            except Exception:  # pylint: disable=broad-except
                _logger.debug(
                    "claude-agent-sdk instrumentation failed on a message",
                    exc_info=True,
                )
            # Checked once the message is fully handled, never part-way through
            # closing something.
            self._guarded(self._maybe_end_root)

    # --- dispatch ---------------------------------------------------------

    def _on_message(self, msg: Any) -> None:
        # Not at module scope: the SDK is an optional dependency, and
        # auto-instrumentation imports this package to ask whether it applies,
        # which has to work before the SDK is known to be installed. The
        # TYPE_CHECKING block covers the annotations; isinstance needs the
        # classes themselves. Reaching here means a message arrived, so the SDK
        # is loaded.
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

        self._flush_task_closes(msg, TaskNotificationMessage)
        if self._is_detached(msg):
            # Detach the calls it launches as well: their task messages name
            # only their own tool_use_id, not the launcher's.
            for block in blocks_of(getattr(msg, "content", None)):
                if _is_tool_use(block):
                    self._detached.add(cast("ToolUseBlock", block).id)
            return

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
            self._on_conversation_reset(msg)

    def _is_detached(self, msg: Any) -> bool:
        """Whether this message reports on work that outlived its turn, or the
        subagent that started it."""
        return (
            getattr(msg, "parent_tool_use_id", None) in self._detached
            or getattr(msg, "tool_use_id", None) in self._detached
        )

    # --- root agent -------------------------------------------------------

    @property
    def _conversation_id(self) -> str | None:
        return self._conversation_id_override or self._session_id

    def _open_root(self) -> LocalAgentInvocation:
        if self._root is not None:
            return self._root
        # A turn the caller did not start, like one reporting a finished task,
        # has no prompt, and stays where the caller last was.
        if self._submitted:
            self._prompt, self._conversation_id_override, context = (
                self._submitted.popleft()
            )
        else:
            self._prompt, context = None, self._caller_context
        root = self._handler.invoke_local_agent(
            agent_name=self._agent_name,
            request_model=getattr(self._sdk_options, "model", None)
            or self._model,
            context=context,
            conversation_id=self._conversation_id,
        )
        # LocalAgentInvocation is INTERNAL and carries no provider of its own,
        # but Observe reads the provider off every row.
        root.attributes[GenAI.GEN_AI_PROVIDER_NAME] = self._provider
        if self._compacted:
            root.attributes[ATTR_CONVERSATION_COMPACTED] = True
        if self._handler.should_capture_content():
            root.input_messages = prompt_to_messages(self._prompt)
            root.system_instruction = self._system_prompt()
        self._root = root
        self._root_context = root.context
        return root

    def _system_prompt(self) -> list[SystemInstructionPart]:
        return _system_prompt_parts(
            getattr(self._sdk_options, "system_prompt", None)
        )

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

        launch_prompt: list[InputMessage] = []
        if parent_tool_use_id:
            subagent = self._live_subagent(parent_tool_use_id)
            parent_context = (
                subagent.invocation.context
                if subagent is not None
                else self._root_ctx()
            )
            history: list[InputMessage] = []
            if subagent is not None:
                launch_prompt = list(subagent.invocation.input_messages)
        else:
            parent_context = self._root_ctx()
            history = list(prompt_to_messages(self._prompt))

        ctx = _ChatContext(
            key=key,
            parent_context=parent_context,
            history=history,
            launch_prompt=launch_prompt,
        )
        self._chats[key] = ctx
        return ctx

    def _open_chat(
        self, ctx: _ChatContext, message_id: str | None, model: str | None
    ) -> None:
        # Left unset when nothing knows the model. A fabricated identifier is
        # worse than an absent one: it reaches the span name, both model
        # attributes and the metric attribute set, and semconv makes the model
        # attributes optional precisely so this case has an answer.
        request_model = (
            model or self._model or getattr(self._sdk_options, "model", None)
        )
        invocation = self._handler.inference(
            self._provider,
            request_model=request_model,
            context=ctx.parent_context,
        )
        invocation.conversation_id = self._conversation_id
        invocation.response_id = message_id
        if self._compacted:
            invocation.conversation_compacted = True
        effort = getattr(self._sdk_options, "effort", None)
        if isinstance(effort, str):
            invocation.reasoning_level = effort
        if not ctx.history:
            ctx.history.extend(ctx.launch_prompt)
        ctx.launch_prompt = []
        if self._handler.should_capture_content():
            invocation.input_messages = list(ctx.history)
            invocation.system_instruction = self._system_prompt()

        ctx.invocation = invocation
        ctx.message_id = message_id
        ctx.blocks = []
        ctx.usage = None
        ctx.final_usage = None
        ctx.pending_close = False
        ctx.thinking_tokens = None
        ctx.stop_reason = None
        ctx.response_model = model

    def _close_chat(
        self, ctx: _ChatContext, error: Error | None = None
    ) -> None:
        invocation = ctx.invocation
        if invocation is None:
            return
        ctx.invocation = None
        ctx.pending_close = False

        usage = _merge_usage(ctx.usage, ctx.final_usage)
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

        if ctx.response_model:
            invocation.response_model_name = ctx.response_model
        if ctx.stop_reason:
            invocation.finish_reasons = [ctx.stop_reason]
        if self._handler.should_capture_content():
            message = output_message(ctx.blocks, "assistant", ctx.stop_reason)
            invocation.output_messages = [message] if message else []

        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    # --- tool call state --------------------------------------------------

    def _call(self, tool_use_id: str) -> _ToolCall:
        """The state for this call, created empty on first mention."""
        call = self._calls.get(tool_use_id)
        if call is None:
            call = _ToolCall()
            self._calls[tool_use_id] = call
        return call

    # --- subagents --------------------------------------------------------

    def _live_subagent(self, parent_tool_use_id: str) -> _Subagent | None:
        """The open subagent for this Task call, or None once it has ended.

        The CLI can emit a message naming a subagent after that subagent's span
        is closed. Re-opening it would start a second ``invoke_agent`` that
        nothing is left to close, so callers hang late messages off the root
        instead.
        """
        if self._call(parent_tool_use_id).subagent_finished:
            return None
        return self._subagent(parent_tool_use_id)

    def _subagent(self, parent_tool_use_id: str) -> _Subagent:
        """Open a subagent span lazily, reading its identity from the Task call
        that launched it.

        Parented to that Task tool's span when it is still open — which it is,
        because a Task tool's real completion cannot arrive before the subagent
        has spoken.
        """
        call = self._call(parent_tool_use_id)
        if call.subagent is not None:
            return call.subagent

        launch = call.launch
        launch_name = getattr(launch, "name", None)
        # Only the delegation tool's input follows TaskToolInput. Every tool_use
        # block gets a launch recorded, so reading an arbitrary tool's arguments
        # as one picks up any unrelated argument named "model" and writes it to
        # gen_ai.request.model, which is always emitted and is a metric dimension
        # besides.
        tool_input = cast(
            "TaskToolInput",
            (getattr(launch, "input", None) or {})
            if launch_name in _DELEGATION_TOOLS
            else {},
        )
        name = (
            tool_input.get("subagent_type")
            or tool_input.get("agent_type")
            or launch_name
            or "subagent"
        )
        parent_tool = call.invocation
        parent_context = (
            parent_tool.context
            if parent_tool is not None
            else self._root_ctx()
        )

        invocation = self._handler.invoke_local_agent(
            agent_name=str(name),
            request_model=tool_input.get("model") or self._model,
            context=parent_context,
            conversation_id=self._conversation_id,
        )
        invocation.attributes[GenAI.GEN_AI_PROVIDER_NAME] = self._provider
        if self._handler.should_capture_content():
            # The description is the model's own one-line summary of the user's
            # request, so it is content, not the static application-provided text
            # semconv has in mind for gen_ai.agent.description.
            description = tool_input.get("description")
            if isinstance(description, str) and description:
                invocation.agent_description = description
            prompt = tool_input.get("prompt")
            if isinstance(prompt, str) and prompt:
                invocation.input_messages = [
                    InputMessage(role="user", parts=[TextPart(content=prompt)])
                ]

        entry = _Subagent(invocation=invocation)
        call.subagent = entry
        return entry

    def _finish_subagent(self, tool_use_id: str) -> None:
        """End a subagent span once the Task tool that spawned it has returned."""
        call = self._calls.get(tool_use_id)
        if call is None or call.subagent is None:
            return
        # Cleared first, so a malformed stream in which the subagent owns its
        # own launcher cannot bring the close back here.
        entry, call.subagent = call.subagent, None
        call.subagent_finished = True
        for owned_id, owned in list(self._calls.items()):
            if owned.owner == tool_use_id and owned.invocation is not None:
                if owned.task_running and owned.receipt_spent:
                    # A background task it started keeps running without it:
                    # detach it, as the end of a turn would.
                    self._detached.add(owned_id)
                self._guarded(
                    self._close_tool,
                    owned_id,
                    None,
                    self._unfinished_reason(owned_id),
                )
        ctx = self._chats.pop(tool_use_id, None)
        if ctx is not None:
            self._guarded(self._close_chat, ctx)
        if entry.output and self._handler.should_capture_content():
            entry.invocation.output_messages = entry.output
        entry.invocation.stop()

    # --- tool spans -------------------------------------------------------

    def _open_tool(self, block: Any, parent_tool_use_id: str | None) -> None:
        tool_call_id = getattr(block, "id", None)
        if not tool_call_id:
            return
        call = self._call(tool_call_id)
        if call.invocation is not None:
            return
        name = getattr(block, "name", None) or "unknown_tool"

        parent_call = (
            self._calls.get(parent_tool_use_id) if parent_tool_use_id else None
        )
        subagent = parent_call.subagent if parent_call is not None else None
        if subagent is not None and parent_tool_use_id:
            parent_context = subagent.invocation.context
            call.owner = parent_tool_use_id
        else:
            parent_context = self._root_ctx()

        invocation = self._handler.tool(
            name=name,
            tool_type="extension" if name.startswith("mcp__") else "function",
            # On execute_tool the convention means the agent doing the executing,
            # which for a delegated call is the subagent, not the run's agent.
            agent_name=subagent.invocation.agent_name
            if subagent is not None
            else self._agent_name,
            context=parent_context,
        )
        invocation.tool_call_id = tool_call_id
        # Known from the call itself, so a receipt that beats task_started is
        # still held as one.
        tool_input: dict[str, Any] = getattr(block, "input", None) or {}
        if name in _DELEGATION_TOOLS or (
            name == "Bash" and tool_input.get("run_in_background") is True
        ):
            call.task_running = True
        invocation.conversation_id = self._conversation_id
        if invocation.should_capture_content:
            invocation.arguments = tool_input
        call.invocation = invocation

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
        self._guarded(self._finish_subagent, tool_call_id)
        call = self._calls.get(tool_call_id)
        if call is None:
            return
        # launch deliberately survives: a subagent still reads its identity from
        # it after this span has gone.
        invocation, call.invocation = call.invocation, None
        status, call.terminal_status = call.terminal_status, None
        call.task_running = False
        call.receipt_spent = False
        call.close_queued = False
        if invocation is None:
            return
        if status not in (None, "completed"):
            error_message = f"task {status}"
        if result is not None and invocation.should_capture_content:
            invocation.tool_result = result
        if error_message is not None:
            invocation.fail(Error(type="tool_error", message=error_message))
        else:
            invocation.stop()

    # --- message handlers -------------------------------------------------

    def _on_system(self, msg: SystemMessage) -> None:
        if msg.subtype == "init":
            # Every turn opens with an init. A turn still waiting on background
            # work ends here, or a task that never finishes would keep every
            # later turn inside it.
            if self._end_pending:
                self._end_root()
            data = cast("InitData", msg.data or {})
            self._session_id = data.get("session_id") or self._session_id
            self._model = data.get("model") or self._model
            self._open_root()
            return
        if msg.subtype == "compact_boundary":
            self._compacted = True
            root = self._open_root()
            root.attributes[ATTR_CONVERSATION_COMPACTED] = True
            # The turns before the boundary are not what the next call sees. The
            # summary the CLI put in their place does not reach this process, so
            # the recorded input goes on understating the real one rather than
            # overstating it.
            for ctx in self._chats.values():
                ctx.history.clear()

    def _on_conversation_reset(self, msg: Any) -> None:
        """The session's conversation was replaced without the connection
        ending — a ``/clear``, or any other flow that discards the transcript.

        The SDK documents this as zeroing the running totals later results
        report, so the baseline they are measured against has to go back to zero
        with them. The transcript it discards is also the one being replayed as
        each chat span's input, so that goes too — and an empty conversation is
        not a compacted one.
        """
        self._session_id = msg.new_conversation_id
        self._usage_totals = _zero_usage()
        self._usage_baseline = _zero_usage()
        self._inherits_usage = False
        self._compacted = False
        for ctx in self._chats.values():
            ctx.history.clear()

    def _on_task_started(self, msg: Any) -> None:
        """The CLI registered this call as a task, whether it then runs in the
        foreground or in the background."""
        if msg.tool_use_id:
            self._call(msg.tool_use_id).task_running = True
            self._task_ids[msg.task_id] = msg.tool_use_id

    def _on_task_notification(self, msg: Any) -> None:
        """The task really finished. For a backgrounded task, whose tool_result
        was only a receipt, ``summary`` is the one output it gets."""
        tool_use_id = msg.tool_use_id or self._task_ids.get(msg.task_id)
        if not tool_use_id:
            return
        call = self._call(tool_use_id)
        call.task_running = False
        self._task_ids.pop(msg.task_id, None)
        if not call.receipt_spent:
            # In the foreground: its tool_result follows, with the full output.
            call.terminal_status = msg.status
            return
        failed = (
            msg.status in _TERMINAL_TASK_STATUSES and msg.status != "completed"
        )
        self._close_tool(
            tool_use_id,
            msg.summary,
            f"task {msg.status}" if failed else None,
        )

    def _on_task_updated(self, msg: TaskUpdatedMessage) -> None:
        """The task reached a terminal state.

        If its tool_result is still to come, that closes the span. If the
        tool_result already arrived as a launch receipt, no further message is
        guaranteed, because the SDK can suppress the notification, so the close
        is queued (see _flush_task_closes).
        """
        if msg.status not in _TERMINAL_TASK_STATUSES:
            return
        tool_use_id = self._task_ids.get(msg.task_id)
        if not tool_use_id:
            return
        call = self._call(tool_use_id)
        call.task_running = False
        call.terminal_status = msg.status
        if call.receipt_spent:
            call.close_queued = True

    def _flush_task_closes(
        self, msg: Any, notification_type: type[Any]
    ) -> None:
        """Close a task whose terminal update arrived with its receipt spent.

        Held for exactly one message so the task_notification that usually
        follows on the same millisecond still gets to supply the real output. If
        the next message is anything else, no notification is coming.
        """
        if isinstance(msg, notification_type):
            return
        for tool_use_id, call in list(self._calls.items()):
            if call.close_queued:
                self._close_tool(tool_use_id)
        # A queued close can be the last thing its turn was waiting for, and
        # this message then belongs to the next turn.
        self._guarded(self._maybe_end_root)

    def _on_assistant(self, msg: Any) -> None:
        parent = msg.parent_tool_use_id
        blocks = blocks_of(msg.content)
        # When an API call fails, the CLI writes an assistant message itself to
        # carry the error, with "<synthetic>" as its model. No model produced
        # it, so ignore that value and fall back to the requested model.
        model = None if msg.model == "<synthetic>" else msg.model
        if model:
            self._model = model

        # Tool spans open from the stream, not from a tool result: only the
        # stream carries parent_tool_use_id, which is what puts the span in the
        # right place.
        for block in blocks:
            block_id = getattr(block, "id", None)
            if block_id and getattr(block, "name", None) is not None:
                self._call(block_id).launch = block

        ctx = self._chat_context(parent)

        # A new message_id means a new API call; the same id means another
        # content block of the call already in flight.
        if ctx.invocation is None or (
            msg.message_id and msg.message_id != ctx.message_id
        ):
            self._close_chat(ctx)
            self._open_chat(ctx, msg.message_id, model)

        ctx.blocks.extend(blocks)
        if model:
            ctx.response_model = model
        if msg.usage:
            ctx.usage = msg.usage
        if msg.stop_reason:
            ctx.stop_reason = msg.stop_reason

        for block in blocks:
            if _is_tool_use(block):
                self._open_tool(block, parent)

        message = output_message(blocks, "assistant", msg.stop_reason)
        if message is not None:
            self._remember(
                ctx, [InputMessage(role="assistant", parts=message.parts)]
            )
            if parent:
                parent_call = self._calls.get(parent)
                if (
                    parent_call is not None
                    and parent_call.subagent is not None
                ):
                    parent_call.subagent.output.append(message)

    def _on_user(self, msg: UserMessage) -> None:
        parent = msg.parent_tool_use_id
        ctx = self._chat_context(parent)
        blocks = blocks_of(msg.content)

        # A user message closes the API call in flight — the next assistant
        # message belongs to a fresh request. When that call's final usage is
        # still on the wire, hold the span open for it: the CLI injects a task
        # launch receipt ahead of the message_delta, and closing here would
        # settle for the mid-stream snapshot, which undercounts by ~40x.
        if self._defers_close(ctx):
            ctx.pending_close = True
        else:
            self._close_chat(ctx)

        # A subagent's calls get no stream events, but its delegation's
        # tool_result carries the final usage of its last call, which is still
        # open. A failed tool's detail is a plain string.
        detail = msg.tool_use_result
        usage = detail.get("usage") if isinstance(detail, dict) else None

        for block in blocks:
            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id:
                continue
            # A task waits for its notification; this may only be a receipt.
            # A failed call launched nothing, so its result is final.
            call = self._calls.get(tool_use_id)
            is_error = bool(getattr(block, "is_error", None))
            if call is not None and call.task_running and not is_error:
                call.receipt_spent = True
                continue
            subagent_chat = self._chats.get(tool_use_id)
            if (
                usage
                and subagent_chat is not None
                and subagent_chat.invocation is not None
            ):
                subagent_chat.set_final_usage(usage)
            self._close_tool(
                tool_use_id,
                getattr(block, "content", None),
                "tool error" if is_error else None,
            )

        self._remember(ctx, input_messages(blocks))

    def _remember(
        self, ctx: _ChatContext, messages: list[InputMessage]
    ) -> None:
        """Add to the conversation replayed as each chat span's input.

        Trimmed from the front once it is too long. Each call's input really is
        the accumulated conversation, but the whole of it is re-serialized onto
        every chat span, so a long tool loop puts the same text on the trace over
        and over — a 40-step loop with 5 KB tool results comes to megabytes of
        gen_ai.input.messages across one run, and an exporter drops what does not
        fit without saying so. The most recent turns are the ones that describe
        what the call being recorded was working from.
        """
        ctx.history.extend(messages)
        if len(ctx.history) > _MAX_HISTORY_MESSAGES:
            del ctx.history[:-_MAX_HISTORY_MESSAGES]
        # Streaming input cannot be read without consuming what the SDK needs, so
        # the root has no input of its own in that mode. The CLI replays the
        # submitted turn as a UserMessage, which is the same text.
        root = self._root
        if (
            ctx.key == _MAIN
            and root is not None
            and not root.input_messages
            and self._handler.should_capture_content()
        ):
            first_user = next(
                (m for m in messages if m.role == "user" and m.parts), None
            )
            if first_user is not None:
                root.input_messages = [first_user]

    def _defers_close(self, ctx: _ChatContext) -> bool:
        """Whether this call's span should wait for its message_delta.

        Only worth waiting when a delta is actually coming: partial messages
        are on for this run, and the call is one they cover. Measured on CLI
        2.1.220, stream events arrive only for the main thread — a subagent's
        call never gets one, so deferring it would hold the span open until
        teardown.
        """
        return (
            ctx.invocation is not None
            and ctx.final_usage is None
            and ctx.key == _MAIN
            and self._partial_messages
        )

    def _on_stream_event(self, msg: StreamEvent) -> None:
        event = cast("StreamEventPayload", msg.event or {})
        event_type = event.get("type")

        # message_start is the first thing sent for a call. The assistant message
        # carrying the same message_id follows up to ~1.8s later (CLI 2.1.220),
        # with dozens of content_block_delta events in between, so the span has
        # to open here to cover them.
        if event_type == "message_start":
            self._on_message_start(msg, event)
            return

        key = msg.parent_tool_use_id or _MAIN
        ctx = self._chats.get(key)
        if ctx is None or ctx.invocation is None:
            return

        # message_delta closes a call on the wire: it is the one event carrying
        # the call's final usage and stop reason. Without it we are left with
        # the mid-stream snapshot, which understates output tokens badly.
        if event_type == "message_delta":
            if ctx.final_usage is not None:
                return
            usage = event.get("usage")
            if usage:
                ctx.set_final_usage(usage)
            stop = (event.get("delta") or {}).get("stop_reason")
            if stop:
                ctx.stop_reason = stop
            if ctx.pending_close:
                self._close_chat(ctx)
            return

        # Let util-genai time the stream from real chunk arrivals, which is what
        # gen_ai.client.time_to_first_chunk and time_per_output_chunk measure.
        if event_type == "content_block_delta":
            ctx.invocation.record_stream_chunk()

    def _on_message_start(
        self, msg: StreamEvent, event: StreamEventPayload
    ) -> None:
        """Open the chat span for the call this event announces.

        Both the id the call is keyed by and the model that answered sit under
        ``event.message``, so nothing has to be waited for. An assistant message
        carrying the same id then extends this span rather than rotating it.
        """
        message = event.get("message") or {}
        message_id = message.get("id")
        if not message_id:
            return
        ctx = self._chat_context(msg.parent_tool_use_id)
        if ctx.invocation is not None and ctx.message_id == message_id:
            return
        self._close_chat(ctx)
        self._open_chat(ctx, message_id, message.get("model"))

    def _on_result(self, msg: Any) -> None:
        """Record the turn's usage and outcome, and close the main thread's call
        in flight.

        In multi-turn mode this marks the turn to end once nothing it started is
        outstanding. Subagent calls and tools are left open: a backgrounded
        delegation outlives the result that announced it, and so can its current
        call, which a result can land in the middle of.
        """
        root = self._open_root()
        self._session_id = msg.session_id or self._session_id
        self._saw_result = True

        error = _result_error(msg, self._handler.should_capture_content())
        # An API failure is the one result error that belongs on the model call:
        # the CLI relays it as a synthetic assistant message, so that call's
        # span is open and would otherwise close as a success.
        api_error = error if msg.subtype == "success" else None
        main = self._chats.pop(_MAIN, None)
        if main is not None:
            self._close_chat(main, api_error)
        self._final_error = error

        if msg.model_usage:
            totals = _fold_model_usage(msg.model_usage)
            if not totals["cost_usd"]:
                totals["cost_usd"] = float(msg.total_cost_usd or 0.0)
            # A later result supersedes an earlier one rather than adding to it:
            # the counters are cumulative, so the last one holds the answer.
            self._usage_totals = totals
        if not self._inherits_usage:
            self._record_turn_usage(root)
        if msg.stop_reason:
            root.finish_reasons = [msg.stop_reason]
        if (
            self._handler.should_capture_content()
            and isinstance(msg.result, str)
            and msg.result
            and not msg.is_error
        ):
            root.output_messages = text_output(msg.result)

        if self._multi_turn:
            self._end_pending = True

    def _record_turn_usage(self, root: LocalAgentInvocation) -> None:
        turn = _usage_delta(self._usage_totals, self._usage_baseline)
        if turn["input_tokens"]:
            root.input_tokens = turn["input_tokens"]
        if turn["output_tokens"]:
            root.output_tokens = turn["output_tokens"]
        if turn["cache_read"]:
            root.attributes[GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = turn[
                "cache_read"
            ]
        if turn["cache_write"]:
            root.attributes[ATTR_USAGE_CACHE_WRITE_INPUT_TOKENS] = turn[
                "cache_write"
            ]
        if turn["reasoning"]:
            root.attributes[GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = turn[
                "reasoning"
            ]
        if turn["cost_usd"]:
            root.attributes[ATTR_USAGE_COST_USD] = turn["cost_usd"]

    # --- teardown ---------------------------------------------------------

    def _maybe_end_root(self) -> None:
        """End the turn once its result is in and nothing is outstanding.

        A backgrounded delegation outlives the result that announced it, and
        everything it does in between still belongs to this turn.
        """
        if self._end_pending and not self._work_outstanding():
            self._end_root()

    def _work_outstanding(self) -> bool:
        """Whether the turn still has to wait on any call."""
        return any(call.outstanding for call in self._calls.values())

    def _unfinished_reason(self, tool_use_id: str) -> str | None:
        """Why a tool closed before reporting back is a failure, if it is."""
        call = self._call(tool_use_id)
        status = call.terminal_status
        if status == "completed":
            return None
        if status is not None:
            return f"task {status}"
        if tool_use_id in self._detached:
            # Still running when its turn or its subagent ended. Work that never
            # reported is not work that failed.
            return None
        return "tool never returned"

    def _detach_outstanding_work(self) -> None:
        """Mark what outlives this turn: work it is still waiting on, and the
        calls its subagents are making. Closing them then is not a failure."""
        outstanding = {
            tool_use_id
            for tool_use_id, call in self._calls.items()
            if call.outstanding
        }
        self._detached |= outstanding | {
            tool_use_id
            for tool_use_id, call in self._calls.items()
            if call.owner in outstanding
        }

    def _end_root(self) -> None:
        """Close everything still open and end the agent span.

        Every close is guarded. One invocation raising as it finishes must not
        leave the rest open, and must not escape into the caller's ``async for``,
        where it would look like a transport failure and mask a real one.
        """
        self._end_pending = False
        try:
            self._detach_outstanding_work()
            open_tools = [
                tool_use_id
                for tool_use_id, call in self._calls.items()
                if call.invocation is not None
            ]
            # Closing a delegation closes its subagent's tools first, so the
            # order here does not matter.
            for tool_use_id in open_tools:
                self._guarded(
                    self._close_tool,
                    tool_use_id,
                    None,
                    self._unfinished_reason(tool_use_id),
                )
            for tool_use_id, call in list(self._calls.items()):
                if call.subagent is not None:
                    self._guarded(self._finish_subagent, tool_use_id)
            for ctx in list(self._chats.values()):
                self._guarded(self._close_chat, ctx)
            self._chats.clear()
            # Only after the closes, since _unfinished_reason treats a detached
            # call as not failed. From here on, whatever the CLI still says
            # about these calls is dropped.
            self._detached |= self._calls.keys()
            self._calls.clear()
            self._task_ids.clear()
            # Where the next turn's cumulative counters are measured from.
            self._usage_baseline = cast(
                "UsageTotals", dict(self._usage_totals)
            )
            if any(self._usage_totals.values()):
                self._inherits_usage = False
        finally:
            root, self._root = self._root, None
            self._root_context = None
            # _compacted deliberately survives: compaction is a property of the
            # conversation, not of the turn that triggered it, so a later turn is
            # still working from a compacted context. Only a reset clears it.
            error, self._final_error = self._final_error, None
            self._saw_result = False
            if root is not None:
                if error is not None:
                    self._guarded(root.fail, error)
                else:
                    self._guarded(root.stop)

    def _teardown(self, close: Any, *args: Any) -> None:
        """Run a close outside _observe: absorb what it raises and leave the
        caller's context as it was."""
        with _keep_current_context():
            self._guarded(close, *args)

    def _guarded(self, close: Any, *args: Any) -> None:
        """Run one close, absorbing whatever it raises.

        Anything left recording after this is a span that never exports, so each
        item is isolated from the last.
        """
        try:
            close(*args)
        except Exception:  # pylint: disable=broad-except
            _logger.debug(
                "claude-agent-sdk instrumentation failed closing a span",
                exc_info=True,
            )

    def _fail_root(self, exc: BaseException) -> None:
        error = Error.from_exception(exc)
        # Recorded before anything that could raise, so _end_root still fails
        # the root even if closing a chat goes wrong on the way there.
        self._final_error = error
        try:
            for ctx in list(self._chats.values()):
                self._guarded(self._close_chat, ctx, error)
            self._chats.clear()
        finally:
            self._end_root()

    def _fail_start(self, exc: Exception) -> None:
        """The SDK raised before streaming anything: report a failed run."""
        self._open_root()
        self._fail_root(exc)

    def _stop(self) -> None:
        """The caller ended the run. Whatever was still going was cut short,
        not failed."""
        self._detached |= self._calls.keys()
        self._end_root()

    def _dispose(self) -> None:
        """Close anything the stream left open."""
        if self._root is None:
            return
        if not self._saw_result and self._final_error is None:
            self._final_error = Error(
                type="incomplete_stream",
                message="agent stream ended before any result message",
            )
        self._end_root()


@contextmanager
def _keep_current_context() -> Iterator[None]:
    """Leave the caller's active context as it was.

    util-genai attaches each span it starts and detaches it when the span ends,
    and these spans do not end in the order they opened.
    """
    saved = otel_context.get_current()
    try:
        yield
    finally:
        if otel_context.get_current() is not saved:
            otel_context.attach(saved)


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


def _zero_usage() -> UsageTotals:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "cost_usd": 0.0,
    }


def _usage_delta(totals: UsageTotals, baseline: UsageTotals) -> UsageTotals:
    """One invocation's share of counts the CLI reports cumulatively.

    ``model_usage`` and ``total_cost_usd`` accumulate over the *session*, not the
    invocation: the CLI documents them that way, a resumed session carries on
    from the earlier session's totals, and a mid-session ``/clear`` zeroes them.
    In multi-turn mode a fresh root opens per turn while the counters keep
    growing, so each turn reports the difference from where the previous one
    ended.

    Clamped at zero, because a counter reset the stream did not announce would
    otherwise be reported as a negative token count.
    """
    return {
        "input_tokens": max(
            totals["input_tokens"] - baseline["input_tokens"], 0
        ),
        "output_tokens": max(
            totals["output_tokens"] - baseline["output_tokens"], 0
        ),
        "cache_read": max(totals["cache_read"] - baseline["cache_read"], 0),
        "cache_write": max(totals["cache_write"] - baseline["cache_write"], 0),
        "reasoning": max(totals["reasoning"] - baseline["reasoning"], 0),
        "cost_usd": max(totals["cost_usd"] - baseline["cost_usd"], 0.0),
    }


def _fold_model_usage(
    model_usage: dict[str, ModelUsage] | None,
) -> UsageTotals:
    """Sum per-model usage across models.

    Covers subagents and compaction, which ``result.usage`` does not.
    """
    totals = _zero_usage()
    for usage in (model_usage or {}).values():
        totals["input_tokens"] += int(usage.get("inputTokens") or 0)
        totals["output_tokens"] += int(usage.get("outputTokens") or 0)
        totals["cache_read"] += int(usage.get("cacheReadInputTokens") or 0)
        totals["cache_write"] += int(
            usage.get("cacheCreationInputTokens") or 0
        )
        # Already counted in outputTokens.
        totals["reasoning"] += int(usage.get("thinkingTokens") or 0)
        totals["cost_usd"] += float(usage.get("costUSD") or 0.0)
    # The convention wants input_tokens to include the cached counts.
    totals["input_tokens"] += totals["cache_read"] + totals["cache_write"]
    return totals


def _result_error(msg: ResultMessage, capture_content: bool) -> Error | None:
    """The error a result reports, if any.

    Which field holds the text depends on the subtype, following the rule the
    SDK applies to its own thrown error: on an API failure the subtype is still
    "success" and the text sits in ``result``; the early-stop subtypes put it
    in ``errors``.

    That text is whatever the CLI relayed and it goes into the span status, which
    is exported regardless of the capture setting, so it is gated: an
    ``invalid_request_error`` body can quote the request fragment that produced
    it. What survives either way is the subtype and the HTTP status — the SDK's
    own docstring for ``api_error_status`` calls the status code safe to log —
    which is what tells a rate limit apart from a bad request.
    """
    if not msg.is_error or msg.terminal_reason in _INTERRUPTED:
        return None
    if not capture_content:
        text = ""
    elif msg.subtype == "success":
        text = msg.result if isinstance(msg.result, str) else ""
    else:
        text = "; ".join(e for e in (msg.errors or []) if e.strip())
    status = getattr(msg, "api_error_status", None)
    if isinstance(status, int):
        text = f"{text} (HTTP {status})" if text else f"HTTP {status}"
    return Error(
        type=msg.subtype if msg.subtype != "success" else "api_error",
        message=text
        or f"claude agent run failed: {getattr(msg, 'terminal_reason', None) or msg.subtype}",
    )
