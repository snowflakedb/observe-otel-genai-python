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
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.instrumentation.genai.claude_agent_sdk._content import (
    blocks_of,
    input_messages,
    output_message,
    prompt_to_messages,
    text_output,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk._types import (
    InitData,
    StreamEventPayload,
    TaskPatch,
    TaskToolInput,
    Usage,
    UsageTotals,
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
    GenericPart,
    InputMessage,
    OutputMessage,
    SystemInstructionPart,
    TextPart,
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

# Routing env vars the CLI reads to pick a hosting platform, listed in the
# order its own resolver checks them (read off CLI 2.1.220), mapped to the
# semconv provider value. A hosting platform outranks the vendor it hosts, so a
# Bedrock-served Claude model reports aws.bedrock. Mantle is a second Bedrock
# endpoint, and the two "Claude Platform on <cloud>" flags reach Anthropic's own
# API through a marketplace — the CLI groups both with its firstParty case — so
# those stay anthropic.
#
# The CLI resolves a "gateway" provider ahead of all of these, but it keys off
# session auth state rather than the environment, so it cannot be predicted from
# here. Nothing in the convention describes a gateway either.
_PROVIDER_ENV: tuple[tuple[str, str], ...] = (
    ("CLAUDE_CODE_USE_BEDROCK", _PROVIDER.AWS_BEDROCK.value),
    ("CLAUDE_CODE_USE_FOUNDRY", _PROVIDER.AZURE_AI_INFERENCE.value),
    ("CLAUDE_CODE_USE_ANTHROPIC_AWS", _PROVIDER.ANTHROPIC.value),
    ("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", _PROVIDER.ANTHROPIC.value),
    ("CLAUDE_CODE_USE_MANTLE", _PROVIDER.AWS_BEDROCK.value),
    ("CLAUDE_CODE_USE_VERTEX", _PROVIDER.GCP_VERTEX_AI.value),
)

# The CLI's own rule for reading these flags, from the bool parser its env
# schema runs them through: an allowlist over the trimmed, lowercased value. A
# denylist would read "no" and "off" as on.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Task statuses meaning the task has finished. Mirrors the SDK's own set; the
# two lifecycle vocabularies differ (task_notification reports "stopped", while
# task_updated reports the raw "killed"), so both are listed.
_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "stopped", "killed"}
)

# The tool names that launch a subagent, and so the only ones whose input is a
# TaskToolInput. Both, because the CLI accepts either (read off CLI 2.1.220,
# which resolves a subagent_type for "Agent" and "Task" alike); every recorded
# session uses "Agent".
_DELEGATION_TOOLS = frozenset({"Agent", "Task"})

# How many messages of conversation a chat span replays as its input. Deliberately
# a constant rather than an environment variable: the bound exists to keep one
# span's attributes exportable, which is not a thing a deployment should have to
# tune. Forty covers a long tool loop — one call and its result per step —
# without letting a run's whole transcript land on every span in it. Raising it
# trades trace bytes for conversation depth.
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
    options_env = getattr(sdk_options, "env", None)
    if not isinstance(options_env, dict):
        options_env = {}
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
    if preset.get("preset"):
        parts.append(
            _SystemPromptSource(source="preset", name=preset["preset"])
        )
    # append is the caller's own text, bolted onto the preset.
    if preset.get("append"):
        parts.append(TextPart(content=preset["append"]))
    return parts


def _merge_usage(snapshot: Usage | None, final: Usage | None) -> Usage | None:
    """The call's usage, preferring ``message_delta``'s counts field by field.

    Only ``output_tokens`` is guaranteed on a ``message_delta``; the input and
    cache counters are conditional. Taking the delta wholesale therefore throws
    away whatever the assistant message's snapshot knew — a delta carrying just
    ``{"output_tokens": 812}`` reported zero input tokens and no cache counters
    against a snapshot holding 42,500. The CLI's own vendored Anthropic
    accumulator merges these the same way, per field.
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


@dataclass
class _Subagent:
    invocation: LocalAgentInvocation
    output: list[OutputMessage] = field(default_factory=list[OutputMessage])


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
        # Whether the CLI forwards stream events. Off, a call's final usage
        # never arrives and a span must not be held open waiting for it.
        self._partial_messages = bool(
            getattr(sdk_options, "include_partial_messages", False)
        )

        self._root: LocalAgentInvocation | None = None
        self._root_context: Context | None = None
        self._caller_context: Context = otel_context.get_current()

        self._chats: dict[str, _ChatContext] = {}
        self._tools: dict[str, ToolInvocation] = {}
        self._subagents: dict[str, _Subagent] = {}
        # tool_use_id -> the Task tool_use block that launched a subagent.
        self._launches: dict[str, object] = {}
        # tool_use_id -> the parent_tool_use_id of the subagent running it, so a
        # subtree can be closed from the leaves up at teardown.
        self._tool_owner: dict[str, str] = {}
        # tool_use_ids the CLI runs as a task. Their tool_result cannot be
        # trusted to mean the work finished.
        self._task_tools: set[str] = set()
        # tool_use_ids whose tool_result has already been skipped as a receipt,
        # so no later message of that kind is coming for them.
        self._task_receipts: set[str] = set()
        # tool_use_ids whose span is waiting one message for a notification that
        # may never come. See _flush_task_closes.
        self._task_closing: set[str] = set()
        # task_id -> tool_use_id, because task_updated carries only the task_id.
        self._task_ids: dict[str, str] = {}
        # tool_use_id -> the terminal status a task_updated reported, so the
        # status survives until whichever message actually closes the span.
        self._task_terminal: dict[str, str] = {}
        # parent_tool_use_ids whose subagent span has already ended. A message
        # that arrives after it must not re-open the subtree.
        self._finished_subagents: set[str] = set()
        # The caller has their answer, but background work is still running, so
        # the invocation is not over yet.
        self._end_pending = False

        # The CLI's cumulative counters as of the last result, and their value
        # when the previous invocation ended. See _usage_delta.
        self._usage_totals: UsageTotals = _zero_usage()
        self._usage_baseline: UsageTotals = _zero_usage()

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
            self._guarded(self._fail_root, exc)
            raise

        try:
            async for message in stream:
                self._observe(message)
                yield message
        except Exception as exc:
            self._guarded(self._fail_root, exc)
            raise
        finally:
            # Reached on a break, an abandoned generator, and on the way out of
            # the transport exception above, which must stay the one that
            # surfaces.
            self._guarded(self._dispose)

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

        if parent_tool_use_id:
            subagent = self._live_subagent(parent_tool_use_id)
            parent_context = (
                subagent.invocation.context
                if subagent is not None
                else self._root_ctx()
            )
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

    # --- subagents --------------------------------------------------------

    def _live_subagent(self, parent_tool_use_id: str) -> _Subagent | None:
        """The open subagent for this Task call, or None once it has ended.

        The CLI can emit a message naming a subagent after that subagent's span
        is closed. Re-opening it would start a second ``invoke_agent`` that
        nothing is left to close, so callers hang late messages off the root
        instead.
        """
        if parent_tool_use_id in self._finished_subagents:
            return None
        return self._subagent(parent_tool_use_id)

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
        launch_name = getattr(launch, "name", None)
        # Only the delegation tool's input follows TaskToolInput. _launches holds
        # every tool_use block, so reading an arbitrary tool's arguments as one
        # picks up any unrelated argument named "model" and writes it to
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
        self._subagents[parent_tool_use_id] = entry
        return entry

    def _finish_subagent(self, tool_use_id: str) -> None:
        """End a subagent span once the Task tool that spawned it has returned."""
        entry = self._subagents.pop(tool_use_id, None)
        if entry is None:
            return
        self._finished_subagents.add(tool_use_id)
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

        subagent = (
            self._subagents.get(parent_tool_use_id)
            if parent_tool_use_id
            else None
        )
        if subagent is not None and parent_tool_use_id:
            parent_context = subagent.invocation.context
            self._tool_owner[tool_call_id] = parent_tool_use_id
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
        self._task_receipts.discard(tool_call_id)
        self._task_closing.discard(tool_call_id)
        if invocation is None:
            self._maybe_end_root()
            return
        if error_message is None and status not in (None, "completed"):
            error_message = f"task {status}"
        if result is not None and invocation.should_capture_content:
            invocation.tool_result = result
        if error_message is not None:
            invocation.fail(Error(type="tool_error", message=error_message))
        else:
            invocation.stop()
        self._maybe_end_root()

    # --- message handlers -------------------------------------------------

    def _on_system(self, msg: SystemMessage) -> None:
        if msg.subtype == "init":
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
        self._compacted = False
        for ctx in self._chats.values():
            ctx.history.clear()

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

    def _on_task_updated(self, msg: TaskUpdatedMessage) -> None:
        """The task has reached a terminal state.

        What this always does is stop treating the task's tool_result as an
        untrustworthy receipt, so whichever signal arrives next closes the span.

        Whether it closes the span itself depends on what is still to come. When
        the receipt has not arrived yet it is, so this waits for it. When the
        receipt has already been and gone, nothing of that kind is left —
        measured in a recorded session, the receipt lands at 9.7s and the
        terminal update at 36.4s — and the SDK documents a terminal task whose
        notification is suppressed entirely, so the span would leak to teardown
        and close after the next two chat spans with no status and no result.
        The close is queued rather than done here, because across nine recorded
        sessions the notification carrying the real output followed the update on
        the same millisecond every time.
        """
        patch = cast("TaskPatch", msg.patch or {})
        status = msg.status or patch.get("status")
        if status not in _TERMINAL_TASK_STATUSES:
            return
        tool_use_id = self._task_ids.get(msg.task_id)
        if not tool_use_id:
            return
        self._task_tools.discard(tool_use_id)
        self._task_terminal[tool_use_id] = status
        if tool_use_id in self._task_receipts:
            self._task_closing.add(tool_use_id)
            return
        self._maybe_end_root()

    def _flush_task_closes(
        self, msg: Any, notification_type: type[Any]
    ) -> None:
        """Close a task whose terminal update arrived with its receipt spent.

        Held for exactly one message so the task_notification that usually
        follows on the same millisecond still gets to supply the real output. If
        the next message is anything else, no notification is coming.
        """
        if not self._task_closing or isinstance(msg, notification_type):
            return
        for tool_use_id in list(self._task_closing):
            self._close_tool(tool_use_id)

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
        if msg.model:
            ctx.response_model = msg.model
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
            if parent and parent in self._subagents:
                self._subagents[parent].output.append(message)

    def _on_user(self, msg: Any) -> None:
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

        for block in blocks:
            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id:
                continue
            # A task waits for its notification; this may only be a receipt.
            if tool_use_id in self._task_tools:
                self._task_receipts.add(tool_use_id)
                continue
            self._close_tool(
                tool_use_id,
                getattr(block, "content", None),
                "tool error" if getattr(block, "is_error", None) else None,
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

        # message_start opens a call on the wire, and it is the first thing sent
        # for it. Measured on CLI 2.1.220, the assistant message carrying the
        # same message_id follows 502ms to 1751ms later, once a whole content
        # block has been produced, with 21 to 35 content_block_delta events in
        # between. Opening on the assistant message dropped all of those, so
        # every chat span understated its duration and time_to_first_chunk timed
        # the gap to the *second* chunk group instead.
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
                ctx.final_usage = usage
                details = usage.get("output_tokens_details") or {}
                thinking = details.get("thinking_tokens")
                if isinstance(thinking, int) and thinking > 0:
                    ctx.thinking_tokens = thinking
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
        """A result is a checkpoint, not necessarily the end.

        It closes the main thread's model call in flight and supersedes the
        previous result. Tools and subagents are left alone — a backgrounded
        delegation legitimately outlives the result that announced it, and so
        does the call it is in the middle of: measured in a recorded session, a
        result lands between two assistant messages carrying one message_id, and
        closing the subagent's chat there split one API call into two spans.
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

        totals = _fold_model_usage(msg.model_usage)
        if not totals["cost_usd"]:
            totals["cost_usd"] = float(msg.total_cost_usd or 0.0)
        # A later result supersedes an earlier one rather than adding to it: the
        # counters are cumulative, so the last one to arrive holds the answer.
        self._usage_totals = totals
        turn = _usage_delta(totals, self._usage_baseline)
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
        if turn["cost_usd"]:
            root.attributes[ATTR_USAGE_COST_USD] = turn["cost_usd"]
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
            self._end_pending = True
            self._maybe_end_root()

    # --- teardown ---------------------------------------------------------

    def _maybe_end_root(self) -> None:
        """End the invocation once the caller has their answer and nothing is
        still running for it.

        A backgrounded delegation outlives the result that announced it —
        measured at 18.2s for the result against 36.4s for the task's real
        completion — and everything it does in between still belongs to this
        invocation. Ending on the result would close the subagent's parent out
        from under it.

        The drain is the trigger rather than a later result, because the result
        a finished task produces carries ``origin.kind == "task-notification"``
        and so is not one that answers the caller.
        """
        if self._end_pending and not (self._task_tools or self._subagents):
            self._end_root()

    def _teardown_reason(self, tool_use_id: str) -> str | None:
        """Why a tool still open when the stream ended is a failure, if it is."""
        status = self._task_terminal.get(tool_use_id)
        if status == "completed":
            return None
        if status is not None:
            return f"task {status}"
        if tool_use_id in self._task_tools:
            # Still running in the background when the stream ended. Work that
            # never reported is not work that failed.
            return None
        return "tool never returned"

    def _depth(self, tool_use_id: str) -> int:
        """How many subagents deep this tool sits.

        Walks ids the CLI supplied, so the chain is only as trustworthy as the
        stream: the visited set is there so a malformed one that points back at
        itself terminates instead of spinning.
        """
        seen = {tool_use_id}
        depth = 0
        owner = self._tool_owner.get(tool_use_id)
        while owner is not None and owner not in seen:
            seen.add(owner)
            depth += 1
            owner = self._tool_owner.get(owner)
        return depth

    def _end_root(self) -> None:
        """Close everything still open and end the agent span.

        Deepest first, because ``_close_tool`` ends a Task tool's subagent just
        before the tool itself and a subagent's own tools have to be gone by
        then: a child outliving its parent renders as a bar sticking out past it.

        Every close is guarded. One invocation raising as it finishes must not
        leave the rest open, and must not escape into the caller's ``async for``,
        where it would look like a transport failure and mask a real one.
        """
        # Re-entrant: closing a tool asks whether the invocation can now end.
        self._end_pending = False
        try:
            for tool_use_id in sorted(
                self._tools, key=self._depth, reverse=True
            ):
                self._guarded(
                    self._close_tool,
                    tool_use_id,
                    None,
                    self._teardown_reason(tool_use_id),
                )
            for tool_use_id in list(self._subagents):
                self._guarded(self._finish_subagent, tool_use_id)
            for ctx in list(self._chats.values()):
                self._guarded(self._close_chat, ctx)
            self._chats.clear()
            self._launches.clear()
            self._tool_owner.clear()
            self._task_tools.clear()
            self._task_receipts.clear()
            self._task_closing.clear()
            self._task_ids.clear()
            self._task_terminal.clear()
            self._finished_subagents.clear()
            # Where the next turn's cumulative counters are measured from.
            self._usage_baseline = cast(
                "UsageTotals", dict(self._usage_totals)
            )
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


def _zero_usage() -> UsageTotals:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cost_usd": 0.0,
    }


def _usage_delta(totals: UsageTotals, baseline: UsageTotals) -> UsageTotals:
    """One invocation's share of counts the CLI reports cumulatively.

    ``model_usage`` and ``total_cost_usd`` accumulate over the *session*, not the
    invocation: the CLI documents them that way, a resumed session starts them
    fresh, and a mid-session ``/clear`` zeroes them. With ``query()`` the session
    is the invocation, so the totals are already the invocation's. In multi-turn
    mode a fresh root opens per turn while the counters keep growing, so turn two
    reported turn one's tokens and cost as well as its own, and turn three all
    three turns'.

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
    if not msg.is_error:
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
