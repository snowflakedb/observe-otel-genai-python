# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drives the stream state machine with synthetic message sequences.

Each test builds the exact message ordering the CLI produces for one scenario,
so the rules that depend on ordering — a task's receipt versus its real
completion, a call's final usage versus its mid-stream snapshot — are exercised
directly rather than inferred from a live run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ConversationResetMessage,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from observeinc.instrumentation.genai.claude_agent_sdk._run import AgentRun
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.invocation import (
    LocalAgentInvocation,
    ToolInvocation,
)
from opentelemetry import context as otel_context
from opentelemetry import trace

SESSION = "session-abc"
NEW_CONVERSATION = "conversation-def"


# ─── builders ───────────────────────────────────────────────────────────────


def init(model: str = "claude-sonnet-4-5") -> SystemMessage:
    return SystemMessage(
        subtype="init", data={"session_id": SESSION, "model": model}
    )


def assistant(
    *blocks: Any,
    model: str = "claude-sonnet-4-5",
    message_id: str = "msg_1",
    parent: str | None = None,
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks),
        model=model,
        message_id=message_id,
        parent_tool_use_id=parent,
        usage=usage,
        stop_reason=stop_reason,
    )


def user(
    *blocks: Any,
    parent: str | None = None,
    tool_use_result: dict[str, Any] | str | None = None,
) -> UserMessage:
    return UserMessage(
        content=list(blocks),
        parent_tool_use_id=parent,
        tool_use_result=tool_use_result,
    )


def delta(
    output_tokens: int,
    *,
    input_tokens: int = 100,
    stop_reason: str = "end_turn",
    thinking: int | None = None,
    parent: str | None = None,
) -> StreamEvent:
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if thinking is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking}
    return StreamEvent(
        uuid="u",
        session_id=SESSION,
        parent_tool_use_id=parent,
        event={
            "type": "message_delta",
            "usage": usage,
            "delta": {"stop_reason": stop_reason},
        },
    )


def message_start(
    *,
    message_id: str = "msg_1",
    model: str = "claude-sonnet-4-5",
    parent: str | None = None,
) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id=SESSION,
        parent_tool_use_id=parent,
        event={
            "type": "message_start",
            "message": {
                "type": "message",
                "id": message_id,
                "model": model,
                "role": "assistant",
                "content": [],
            },
        },
    )


def chunk(parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id=SESSION,
        parent_tool_use_id=parent,
        event={"type": "content_block_delta", "delta": {"text": "x"}},
    )


def task_started(
    tool_use_id: str, task_id: str = "task-1", task_type: str = "local_agent"
) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id=task_id,
        description="delegated work",
        uuid="u",
        session_id=SESSION,
        tool_use_id=tool_use_id,
        task_type=task_type,
    )


def task_notification(
    tool_use_id: str,
    *,
    task_id: str = "task-1",
    status: str = "completed",
    summary: str = "the subagent report",
) -> TaskNotificationMessage:
    return TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id=task_id,
        status=status,
        output_file="",
        summary=summary,
        uuid="u",
        session_id=SESSION,
        tool_use_id=tool_use_id,
    )


def task_updated(
    *, task_id: str = "task-1", status: str = "killed"
) -> TaskUpdatedMessage:
    return TaskUpdatedMessage(
        subtype="task_updated",
        data={},
        task_id=task_id,
        patch={"status": status},
        status=status,
        session_id=SESSION,
        uuid="u",
    )


def conversation_reset(
    new_conversation_id: str = NEW_CONVERSATION,
) -> ConversationResetMessage:
    return ConversationResetMessage(
        new_conversation_id=new_conversation_id, uuid="u", session_id=SESSION
    )


def result(
    *,
    subtype: str = "success",
    is_error: bool = False,
    text: str | None = "done",
    model_usage: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    origin: dict[str, Any] | None = None,
    api_error_status: int | None = None,
    terminal_reason: str | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=10,
        duration_api_ms=5,
        is_error=is_error,
        num_turns=1,
        session_id=SESSION,
        result=text,
        model_usage=model_usage,
        errors=errors,
        origin=origin,
        api_error_status=api_error_status,
        terminal_reason=terminal_reason,
        stop_reason="end_turn",
    )


MODEL_USAGE = {
    "claude-sonnet-4-5": {
        "inputTokens": 1000,
        "outputTokens": 200,
        "cacheReadInputTokens": 50,
        "cacheCreationInputTokens": 20,
        "costUSD": 0.012,
    }
}


# ─── harness ────────────────────────────────────────────────────────────────


async def drive(
    handler: TelemetryHandler,
    messages: list[Any],
    *,
    prompt: str = "hi",
    multi_turn: bool = False,
    stop_after: int | None = None,
    sdk_options: Any = None,
) -> list[Any]:
    """Feed messages through an AgentRun. ``stop_after`` breaks out after that
    many, as a caller would, which skips the end-of-stream cleanup: a
    multi-turn run then exports only turns that ended by themselves."""

    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler, sdk_options=sdk_options, multi_turn=multi_turn)
    run.submit(prompt)
    messages_out = run.iterate(stream)
    seen = []
    try:
        async for message in messages_out:
            seen.append(message)
            if stop_after is not None and len(seen) >= stop_after:
                break
    finally:
        # A consumer that breaks out leaves the generator suspended; the event
        # loop finalizes it later. Do it now so the test is deterministic.
        await messages_out.aclose()
    return seen


async def roots_ended_at_each_result(
    handler: TelemetryHandler, messages: list[Any], span_exporter: Any
) -> list[int]:
    """How many turn spans had ended as each result went past the caller."""

    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler, multi_turn=True)
    ended = []
    async for message in run.iterate(stream):
        if isinstance(message, ResultMessage):
            ended.append(
                len(
                    Spans(span_exporter).named("invoke_agent claude-agent-sdk")
                )
            )
    return ended


def turn_waiting_on_a_background_command() -> list[Any]:
    return [
        init(),
        assistant(
            ToolUseBlock(id="bash", name="Bash", input={}), message_id="m1"
        ),
        task_started("bash", task_id="t-bash", task_type="local_bash"),
        user(ToolResultBlock(tool_use_id="bash", content="started")),
        result(),
    ]


def metric_points(metric_reader: Any, name: str) -> list[Any]:
    data = metric_reader.get_metrics_data()
    return [
        point
        for resource in (data.resource_metrics if data else [])
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


class Spans:
    """Query helper over the exported spans."""

    def __init__(self, exporter: Any) -> None:
        self.all = list(exporter.get_finished_spans())
        self._by_id = {s.context.span_id: s for s in self.all}

    def named(self, name: str) -> list[Any]:
        return [s for s in self.all if s.name == name]

    def one(self, name: str) -> Any:
        found = self.named(name)
        assert len(found) == 1, f"{name}: expected 1, got {len(found)}"
        return found[0]

    def parent_of(self, span: Any) -> Any | None:
        if span.parent is None:
            return None
        return self._by_id.get(span.parent.span_id)

    def parent_name(self, span: Any) -> str | None:
        parent = self.parent_of(span)
        return parent.name if parent else None


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


@pytest.fixture
def tracer(tracer_provider):
    """A tracer on the test provider. The shared fixtures deliberately leave the
    global provider unset, so a span from ``trace.get_tracer`` would not export."""
    return tracer_provider.get_tracer("test")


@pytest.fixture
def capturing_handler(tracer_provider, meter_provider, monkeypatch):
    """A handler with span content capture on.

    The capture mode is read when the handler is constructed, so the
    environment has to be set first.
    """
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    return TelemetryHandler(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )


# ─── the shape of a plain run ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_call_tree(handler, span_exporter):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hello")),
            result(model_usage=MODEL_USAGE),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    chat = spans.one("chat claude-sonnet-4-5")

    assert spans.parent_name(chat) == agent.name
    assert agent.parent is None
    # A stray span under a name no assertion queries is otherwise invisible.
    assert len(spans.all) == 2
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    # Conversation id threads the Conversations view and is required on every span.
    for span in spans.all:
        assert span.attributes["gen_ai.conversation.id"] == SESSION
    assert chat.attributes["gen_ai.provider.name"] == "anthropic"
    assert agent.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat.attributes["gen_ai.request.model"] == "claude-sonnet-4-5"
    assert chat.attributes["gen_ai.response.model"] == "claude-sonnet-4-5"
    assert chat.attributes["gen_ai.response.id"] == "msg_1"
    # Run totals fold the cache counters into input_tokens.
    assert agent.attributes["gen_ai.usage.input_tokens"] == 1070
    assert agent.attributes["gen_ai.usage.output_tokens"] == 200
    assert agent.attributes[
        "claude_agent_sdk.usage.cost_usd"
    ] == pytest.approx(0.012)


@pytest.mark.asyncio
async def test_all_messages_pass_through_unchanged(handler, span_exporter):
    messages = [init(), assistant(TextBlock("hello")), result()]
    seen = await drive(handler, messages)
    assert seen == messages
    # Messages flowing is half of it; the run has to have been observed too.
    assert Spans(span_exporter).named("invoke_agent claude-agent-sdk")


@pytest.mark.asyncio
async def test_tool_loop_children_are_siblings(handler, span_exporter):
    call = ToolUseBlock(id="call-1", name="Bash", input={"command": "ls"})
    await drive(
        handler,
        [
            init(),
            assistant(call, message_id="msg_1"),
            user(ToolResultBlock(tool_use_id="call-1", content="a.txt")),
            assistant(TextBlock("there is one file"), message_id="msg_2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    tool = spans.one("execute_tool Bash")
    chats = spans.named("chat claude-sonnet-4-5")

    assert len(chats) == 2, "one chat span per API call"
    assert len(spans.all) == 4
    assert {c.attributes["gen_ai.response.id"] for c in chats} == {
        "msg_1",
        "msg_2",
    }
    for span in [tool, *chats]:
        assert spans.parent_name(span) == agent.name
    assert tool.attributes["gen_ai.tool.name"] == "Bash"
    assert tool.attributes["gen_ai.tool.call.id"] == "call-1"
    assert tool.attributes["gen_ai.tool.type"] == "function"
    # A span without a conversation id is absent from the Conversations view,
    # and ToolInvocation carries no conversation_id field of its own.
    for span in spans.all:
        assert span.attributes["gen_ai.conversation.id"] == SESSION


@pytest.mark.asyncio
async def test_mcp_tool_is_typed_extension(handler, span_exporter):
    call = ToolUseBlock(id="c1", name="mcp__notion__search", input={})
    await drive(
        handler,
        [
            init(),
            assistant(call),
            user(ToolResultBlock(tool_use_id="c1", content="ok")),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool mcp__notion__search")
    assert tool.attributes["gen_ai.tool.type"] == "extension"


# ─── the task rule ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backgrounded_task_closes_on_notification_not_receipt(
    capturing_handler, span_exporter
):
    """A backgrounded Task's tool_result is a launch receipt.

    Closing on it would report a near-instant tool call and leave the subagent
    with no open parent to nest under.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="msg_1"),
            task_started("call-task"),
            user(
                ToolResultBlock(
                    tool_use_id="call-task",
                    content="Async agent launched successfully. agentId: a71ff",
                )
            ),
            # The subagent only starts speaking now, well after the receipt.
            assistant(
                TextBlock("subagent thinking"),
                parent="call-task",
                message_id="msg_sub",
            ),
            task_notification("call-task", summary="the subagent report"),
            assistant(TextBlock("final answer"), message_id="msg_2"),
            result(model_usage=MODEL_USAGE),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    tool = spans.one("execute_tool Task")
    subagent = spans.one("invoke_agent general-purpose")

    # The shape both contracts specify for a delegated sub-agent.
    assert spans.parent_name(tool) == agent.name
    assert spans.parent_name(subagent) == tool.name
    # Two main-thread calls plus the subagent's own.
    chats = spans.named("chat claude-sonnet-4-5")
    assert len(chats) == 3
    under_subagent = [
        c for c in chats if spans.parent_name(c) == subagent.name
    ]
    assert len(under_subagent) == 1
    assert under_subagent[0].attributes["gen_ai.response.id"] == "msg_sub"
    assert all(
        spans.parent_name(c) == agent.name
        for c in chats
        if c not in under_subagent
    )
    # The real output, not the receipt.
    assert tool.attributes["gen_ai.tool.call.result"] == "the subagent report"
    # The tool span spans the real work, so it must outlive the subagent's.
    assert tool.end_time >= subagent.end_time
    assert tool.start_time <= subagent.start_time


@pytest.mark.asyncio
async def test_backgrounded_bash_uses_the_same_rule(
    capturing_handler, span_exporter
):
    """The rule reads only whether a call is a task, never its task_type."""
    call = ToolUseBlock(id="b1", name="Bash", input={"command": "sleep 30"})
    await drive(
        capturing_handler,
        [
            init(),
            assistant(call),
            task_started("b1", task_type="local_bash"),
            user(
                ToolResultBlock(
                    tool_use_id="b1",
                    content="Command running in background with ID: bxnmsd",
                )
            ),
            task_notification("b1", summary="real command output"),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Bash")
    assert tool.attributes["gen_ai.tool.call.result"] == "real command output"


@pytest.mark.asyncio
async def test_terminal_task_updated_does_not_pre_empt_the_notification(
    capturing_handler, span_exporter
):
    """The ordering measured live on CLI 2.1.220.

    A completed task reports task_updated first and its task_notification —
    which carries the real output — lands on the same millisecond after.
    Closing on the update drops that output.
    """
    task = ToolUseBlock(
        id="t1", name="Task", input={"subagent_type": "helper"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_updated(status="completed"),
            task_notification("t1", summary="the real report"),
            user(ToolResultBlock(tool_use_id="t1", content="the real report")),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Task")
    assert tool.attributes["gen_ai.tool.call.result"] == "the real report"
    assert tool.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_terminal_task_updated_closes_a_task_whose_receipt_has_passed(
    handler, span_exporter
):
    """The ordering the receipt rule assumes is the reverse of the live one.

    Measured in a recorded session the receipt lands at 9.7s and the terminal
    update at 36.4s, so "whichever signal arrives next closes the span" leaves
    nothing to arrive. With the notification suppressed — which the SDK
    documents — the span leaked to teardown and closed after the next two chat
    spans with no status.
    """
    task = ToolUseBlock(
        id="t1", name="Task", input={"subagent_type": "helper"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("t1"),
            user(
                ToolResultBlock(
                    tool_use_id="t1",
                    content="Async agent launched successfully.",
                )
            ),
            task_updated(status="completed"),
            assistant(TextBlock("moving on"), message_id="m2"),
            assistant(TextBlock("and done"), message_id="m3"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    tool = spans.one("execute_tool Task")
    later = [
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] in {"m2", "m3"}
    ]
    assert len(later) == 2
    assert tool.status.status_code is trace.StatusCode.UNSET
    assert all(tool.end_time < c.start_time for c in later)


@pytest.mark.asyncio
async def test_a_spent_receipt_still_waits_one_message_for_the_notification(
    capturing_handler, span_exporter
):
    """The live ordering for a task that completes normally.

    Across nine recorded sessions the notification carrying the real output
    followed the terminal update on the same millisecond, long after the
    receipt. Closing on the update would drop that output — measured against a
    recorded session, the span lost a 1.8 KB report.
    """
    task = ToolUseBlock(
        id="t1", name="Task", input={"subagent_type": "helper"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("t1"),
            user(
                ToolResultBlock(
                    tool_use_id="t1",
                    content="Async agent launched successfully.",
                )
            ),
            task_updated(status="completed"),
            task_notification("t1", summary="the real report"),
            assistant(TextBlock("moving on"), message_id="m2"),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Task")
    assert tool.attributes["gen_ai.tool.call.result"] == "the real report"
    assert tool.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_killed_task_closes_without_a_notification(
    capturing_handler, span_exporter
):
    """A task stopped via TaskStop reports "killed" on task_updated and its
    notification is sometimes suppressed entirely.

    The terminal update stops the tool_result being treated as a receipt, so the
    result closes the span instead of it leaking to teardown.
    """
    task = ToolUseBlock(id="t1", name="Task", input={})
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_updated(status="killed"),
            user(ToolResultBlock(tool_use_id="t1", content="stopped")),
            assistant(TextBlock("moving on"), message_id="msg_2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    tool = spans.one("execute_tool Task")
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert "killed" in (tool.status.description or "")
    # Closed on the result, not swept up by teardown, so it ends before the run.
    assert tool.end_time < spans.one("invoke_agent claude-agent-sdk").end_time


@pytest.mark.asyncio
async def test_killed_task_with_no_result_at_all_is_closed_at_teardown(
    capturing_handler, span_exporter
):
    task = ToolUseBlock(id="t1", name="Task", input={})
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_updated(status="killed"),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Task")
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert "killed" in (tool.status.description or "")


@pytest.mark.asyncio
async def test_failed_task_marks_the_tool_span(handler, span_exporter):
    task = ToolUseBlock(id="t1", name="Task", input={})
    await drive(
        handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_notification("t1", status="failed", summary="it broke"),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Task")
    assert tool.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_tool_that_never_returns_is_closed_at_teardown(
    capturing_handler, span_exporter
):
    call = ToolUseBlock(id="c1", name="Read", input={})
    await drive(capturing_handler, [init(), assistant(call), result()])
    tool = Spans(span_exporter).one("execute_tool Read")
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert "never returned" in (tool.status.description or "")


# ─── token accuracy ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_delta_supplies_the_calls_final_usage(
    handler, span_exporter
):
    """The assistant message's usage is a mid-stream snapshot; message_delta
    carries the real figure."""
    await drive(
        handler,
        [
            init(),
            assistant(
                TextBlock("hi"),
                usage={"input_tokens": 100, "output_tokens": 3},
            ),
            delta(output_tokens=181, input_tokens=100, thinking=40),
            result(),
        ],
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.output_tokens"] == 181
    assert chat.attributes["gen_ai.usage.input_tokens"] == 100
    assert "claude_agent_sdk.usage.is_partial" not in chat.attributes
    assert chat.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)


@pytest.mark.asyncio
async def test_snapshot_only_usage_is_flagged_partial(handler, span_exporter):
    await drive(
        handler,
        [
            init(),
            assistant(
                TextBlock("hi"),
                usage={"input_tokens": 100, "output_tokens": 3},
            ),
            result(),
        ],
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["claude_agent_sdk.usage.is_partial"] is True
    assert chat.attributes["gen_ai.usage.output_tokens"] == 3


@pytest.mark.asyncio
async def test_a_sparse_message_delta_keeps_the_snapshots_input_counts(
    handler, span_exporter, metric_reader
):
    """Only output_tokens is guaranteed on a message_delta.

    Every delta in the recorded sessions happens to carry all five counters, so
    taking the delta wholesale looked safe; on one that carries only the field
    the API promises, it discarded 42,500 input tokens and both cache counters,
    and recorded a zero input observation on the token histogram.
    """
    await drive(
        handler,
        [
            init(),
            assistant(
                TextBlock("hi"),
                usage={
                    "input_tokens": 12000,
                    "cache_read_input_tokens": 30000,
                    "cache_creation_input_tokens": 500,
                    "output_tokens": 1,
                },
            ),
            StreamEvent(
                uuid="u",
                session_id=SESSION,
                event={
                    "type": "message_delta",
                    "usage": {"output_tokens": 812},
                    "delta": {"stop_reason": "end_turn"},
                },
            ),
            result(),
        ],
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.output_tokens"] == 812
    assert chat.attributes["gen_ai.usage.input_tokens"] == 42500
    assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 30000
    assert chat.attributes["gen_ai.usage.cache_write.input_tokens"] == 500
    inputs = [
        point
        for point in metric_points(metric_reader, "gen_ai.client.token.usage")
        if point.attributes.get("gen_ai.token.type") == "input"
    ]
    assert [point.sum for point in inputs] == [42500]


@pytest.mark.asyncio
async def test_task_receipt_does_not_pre_empt_the_calls_final_usage(
    handler, span_exporter
):
    """The CLI injects a Task launch receipt ahead of the message_delta.

    Measured on CLI 2.1.220: the receipt and `message_stop` land on the same
    tick, the receipt first. Closing the chat on it would settle for the
    mid-stream snapshot — 6 output tokens against a real 255.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(
                task,
                message_id="msg_1",
                usage={"input_tokens": 100, "output_tokens": 6},
            ),
            task_started("call-task"),
            user(
                ToolResultBlock(
                    tool_use_id="call-task",
                    content="Async agent launched successfully.",
                )
            ),
            delta(
                output_tokens=255,
                input_tokens=100,
                stop_reason="tool_use",
                thinking=83,
            ),
            task_notification("call-task", summary="the subagent report"),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.output_tokens"] == 255
    assert chat.attributes["gen_ai.usage.reasoning.output_tokens"] == 83
    assert chat.attributes["gen_ai.response.finish_reasons"] == ("tool_use",)
    assert "claude_agent_sdk.usage.is_partial" not in chat.attributes


@pytest.mark.asyncio
async def test_tool_result_still_closes_the_call_without_stream_events(
    handler, span_exporter
):
    """Without include_partial_messages no delta is ever coming, so the
    tool_result has to close the call itself."""
    await drive(
        handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="call-1", name="Read", input={}),
                message_id="msg_1",
                usage={"input_tokens": 100, "output_tokens": 6},
            ),
            user(ToolResultBlock(tool_use_id="call-1", content="file body")),
            assistant(
                TextBlock("done"),
                message_id="msg_2",
                usage={"input_tokens": 200, "output_tokens": 9},
            ),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    chats = spans.named("chat claude-sonnet-4-5")
    assert len(chats) == 2
    first = next(
        c for c in chats if c.attributes["gen_ai.response.id"] == "msg_1"
    )
    tool = spans.one("execute_tool Read")
    # The snapshot is all there is, and the span ends on the tool_result rather
    # than hanging until the next call.
    assert first.attributes["gen_ai.usage.output_tokens"] == 6
    assert first.attributes["claude_agent_sdk.usage.is_partial"] is True
    assert first.end_time <= tool.end_time


@pytest.mark.asyncio
async def test_subagent_tool_result_closes_without_waiting_for_a_delta(
    handler, span_exporter
):
    """A subagent's call never gets a stream event, so its chat span must not
    be held open for one."""
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="msg_1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            delta(output_tokens=255, stop_reason="tool_use"),
            assistant(
                ToolUseBlock(id="call-sub", name="Read", input={}),
                parent="call-task",
                message_id="msg_sub",
                usage={"input_tokens": 50, "output_tokens": 4},
            ),
            user(
                ToolResultBlock(tool_use_id="call-sub", content="body"),
                parent="call-task",
            ),
            task_notification("call-task", summary="report"),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    spans = Spans(span_exporter)
    subagent = spans.one("invoke_agent general-purpose")
    sub_chat = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "msg_sub"
    )
    sub_tool = spans.one("execute_tool Read")
    assert spans.parent_name(sub_chat) == subagent.name
    assert sub_chat.attributes["claude_agent_sdk.usage.is_partial"] is True
    assert sub_chat.end_time <= sub_tool.end_time


@pytest.mark.asyncio
async def test_cache_counters_fold_into_input_tokens(handler, span_exporter):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hi")),
            StreamEvent(
                uuid="u",
                session_id=SESSION,
                event={
                    "type": "message_delta",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 1000,
                        "cache_creation_input_tokens": 100,
                    },
                    "delta": {"stop_reason": "end_turn"},
                },
            ),
            result(),
        ],
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.input_tokens"] == 1110
    assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 1000


@pytest.mark.asyncio
async def test_agent_and_chat_spans_name_the_cache_counters_alike(
    handler, span_exporter
):
    """The convention renamed cache_creation to cache_write and only the new
    name is in the registry. util-genai writes the new one on chat spans, so the
    agent span writing the old one had a trace reporting one quantity under two
    keys."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hi")),
            StreamEvent(
                uuid="u",
                session_id=SESSION,
                event={
                    "type": "message_delta",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 20,
                    },
                    "delta": {"stop_reason": "end_turn"},
                },
            ),
            result(model_usage=MODEL_USAGE),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    chat = spans.one("chat claude-sonnet-4-5")
    for key in (
        "gen_ai.usage.cache_write.input_tokens",
        "gen_ai.usage.cache_read.input_tokens",
    ):
        assert agent.attributes[key] == chat.attributes[key], key
    assert agent.attributes["gen_ai.usage.cache_write.input_tokens"] == 20
    assert "gen_ai.usage.cache_creation.input_tokens" not in agent.attributes


@pytest.mark.asyncio
async def test_per_call_tokens_sum_to_the_run_total(handler, span_exporter):
    """The property the contract actually checks."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            delta(output_tokens=120, input_tokens=500),
            assistant(TextBlock("two"), message_id="m2"),
            delta(output_tokens=80, input_tokens=570),
            result(
                model_usage={
                    "claude-sonnet-4-5": {
                        "inputTokens": 1070,
                        "outputTokens": 200,
                        "cacheReadInputTokens": 0,
                        "cacheCreationInputTokens": 0,
                        "costUSD": 0.01,
                    }
                }
            ),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    chats = spans.named("chat claude-sonnet-4-5")
    assert (
        sum(c.attributes["gen_ai.usage.output_tokens"] for c in chats) == 200
    )
    assert agent.attributes["gen_ai.usage.output_tokens"] == 200
    assert (
        sum(c.attributes["gen_ai.usage.input_tokens"] for c in chats) == 1070
    )


@pytest.mark.asyncio
async def test_thinking_tokens_recorded(handler, span_exporter):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hi")),
            delta(output_tokens=50, thinking=30),
            result(),
        ],
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.reasoning.output_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_span_opens_on_message_start(
    handler, span_exporter, metric_reader
):
    """The call begins at message_start, not at the assistant message.

    Measured on CLI 2.1.220 the assistant message lands 502ms to 1751ms later,
    once a whole content block is finished, with 21 to 35 chunks already gone
    past. Opening there dropped every one of them, so the span understated its
    duration and time_to_first_chunk timed the gap to the next content block
    instead of the wait for the first token.
    """
    await drive(
        handler,
        [
            init(),
            message_start(message_id="msg_1"),
            chunk(),
            chunk(),
            chunk(),
            assistant(TextBlock("hi"), message_id="msg_1"),
            delta(output_tokens=10),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.response.id"] == "msg_1"
    # The first chunk is the time_to_first_chunk observation; the rest are gaps.
    assert (
        len(
            metric_points(
                metric_reader, "gen_ai.client.operation.time_to_first_chunk"
            )
        )
        == 1
    )
    gaps = metric_points(
        metric_reader, "gen_ai.client.operation.time_per_output_chunk"
    )
    assert sum(p.count for p in gaps) == 2


@pytest.mark.asyncio
async def test_message_start_identifies_a_call_with_no_assistant_message(
    handler, span_exporter
):
    """message_start carries both the id and the model, so a call cut short
    before any content block still reports which call it was."""
    await drive(
        handler,
        [
            init(model="claude-haiku-4-5"),
            message_start(message_id="msg_1", model="claude-sonnet-4-5"),
            chunk(),
            delta(output_tokens=4),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.response.id"] == "msg_1"
    assert chat.attributes["gen_ai.response.model"] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_assistant_message_extends_the_span_message_start_opened(
    handler, span_exporter
):
    """One API call arrives as several assistant messages sharing a message_id,
    each carrying one finished content block."""
    await drive(
        handler,
        [
            init(),
            message_start(message_id="msg_1"),
            assistant(TextBlock("first block"), message_id="msg_1"),
            assistant(TextBlock("second block"), message_id="msg_1"),
            delta(output_tokens=10),
            message_start(message_id="msg_2"),
            assistant(TextBlock("next call"), message_id="msg_2"),
            delta(output_tokens=3),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    chats = Spans(span_exporter).named("chat claude-sonnet-4-5")
    assert len(chats) == 2
    assert {c.attributes["gen_ai.response.id"] for c in chats} == {
        "msg_1",
        "msg_2",
    }


@pytest.mark.asyncio
async def test_streaming_chunks_produce_timing_metrics(
    handler, span_exporter, metric_reader
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hi")),
            chunk(),
            chunk(),
            delta(output_tokens=10),
            result(),
        ],
    )
    names = {
        metric.name
        for rm in [metric_reader.get_metrics_data()]
        if rm
        for resource in rm.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert "gen_ai.client.operation.time_to_first_chunk" in names
    assert "gen_ai.client.token.usage" in names


# ─── errors ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_early_stop_result_fails_only_the_agent_span(
    handler, span_exporter
):
    """The loop hit a limit; the model call before it went fine."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("partial")),
            delta(output_tokens=10),
            result(
                subtype="error_max_turns",
                is_error=True,
                text=None,
                errors=["max turns reached"],
            ),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    chat = spans.one("chat claude-sonnet-4-5")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert agent.attributes["error.type"] == "error_max_turns"
    assert chat.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_api_error_result_fails_the_model_call_too(
    capturing_handler, span_exporter
):
    """An API failure keeps subtype "success" and puts the text in result; the
    CLI relays it as a synthetic assistant message, so that call's span is open.
    """
    await drive(
        capturing_handler,
        [
            init(),
            assistant(TextBlock("API Error: 400")),
            result(
                is_error=True,
                text="API Error: 400 invalid model",
                api_error_status=400,
            ),
        ],
    )
    spans = Spans(span_exporter)
    assert (
        spans.one("chat claude-sonnet-4-5").status.status_code
        is trace.StatusCode.ERROR
    )
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert "HTTP 400" in (agent.status.description or "")


@pytest.mark.asyncio
async def test_the_clis_error_text_is_content(handler, span_exporter):
    """An ``invalid_request_error`` body can quote the request that produced
    it, so on a span that carries no content only the subtype stays."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("API Error: 400")),
            result(
                is_error=True,
                text='API Error: 400 {"error":{"message":"bad value for '
                "system: You are Acme's internal assistant\"}}",
                api_error_status=400,
            ),
        ],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.status.description is None
    assert agent.attributes["error.type"] == "api_error"


@pytest.mark.asyncio
async def test_the_clis_error_text_is_kept_when_content_is_captured(
    capturing_handler, span_exporter
):
    await drive(
        capturing_handler,
        [
            init(),
            assistant(TextBlock("API Error: 400")),
            result(
                is_error=True,
                text="API Error: 400 invalid model",
                api_error_status=400,
            ),
        ],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert (
        agent.status.description == "API Error: 400 invalid model (HTTP 400)"
    )


@pytest.mark.asyncio
async def test_an_early_stops_reason_survives_without_its_text(
    handler, span_exporter
):
    """With nothing content-free to report, the subtype still names the
    failure."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("partial")),
            result(
                subtype="error_max_turns",
                is_error=True,
                text=None,
                errors=["max turns reached while editing /srv/secret.py"],
            ),
        ],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.attributes["error.type"] == "error_max_turns"
    assert "secret" not in (agent.status.description or "")


@pytest.mark.asyncio
async def test_stream_raising_fails_the_run(handler, span_exporter):
    async def stream():
        yield init()
        yield assistant(TextBlock("hi"))
        raise RuntimeError("transport died")

    run = AgentRun(handler)
    with pytest.raises(RuntimeError):
        run.submit("hi")
        async for _ in run.iterate(stream):
            pass
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert agent.attributes["error.type"].endswith("RuntimeError")


@pytest.mark.asyncio
async def test_stream_ending_without_result_is_an_error(
    handler, span_exporter
):
    await drive(handler, [init(), assistant(TextBlock("hi"))])
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert agent.attributes["error.type"] == "incomplete_stream"


@pytest.mark.asyncio
async def test_abandoned_stream_still_closes_its_spans(handler, span_exporter):
    """The caller breaks out of the loop partway."""
    call = ToolUseBlock(id="c1", name="Bash", input={})
    await drive(
        handler,
        [
            init(),
            assistant(call),
            user(ToolResultBlock(tool_use_id="c1")),
            result(),
        ],
        stop_after=2,
    )
    spans = Spans(span_exporter)
    assert spans.named("execute_tool Bash")
    assert all(s.end_time is not None for s in spans.all)
    # The caller chose to stop, so nothing it left running failed.
    for span in spans.all:
        assert span.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_a_subagents_tools_close_before_the_subagent_at_teardown(
    handler, span_exporter
):
    """The run is interrupted while a subagent has a tool open.

    Draining subagents before tools left the subagent's own tool ending after the
    span it belongs to, which renders as a bar sticking out past its parent.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            assistant(
                ToolUseBlock(id="call-sub", name="Read", input={}),
                parent="call-task",
                message_id="m_sub",
            ),
        ],
    )
    spans = Spans(span_exporter)
    sub_tool = spans.one("execute_tool Read")
    subagent = spans.one("invoke_agent general-purpose")
    task_tool = spans.one("execute_tool Task")
    assert spans.parent_name(sub_tool) == subagent.name
    assert sub_tool.end_time <= subagent.end_time <= task_tool.end_time


@pytest.mark.asyncio
async def test_one_span_failing_to_close_does_not_strand_the_others(
    handler, span_exporter
):
    """A close raising must not abandon everything after it: the root would
    stay recording and never export, and the exception would reach the caller's
    ``async for``, where it looks like a transport failure."""

    def explode(*_: Any, **__: Any) -> None:
        raise RuntimeError("_apply_finish blew up")

    messages = [
        init(),
        assistant(ToolUseBlock(id="c1", name="Read", input={})),
        result(),
    ]
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(ToolInvocation, "_apply_finish", explode)
        seen = await drive(handler, messages)

    assert seen == messages
    spans = Spans(span_exporter)
    # The tool's finish was abandoned, so it carries none of its attributes...
    tool = spans.one("execute_tool Read")
    assert "gen_ai.tool.call.id" not in tool.attributes
    # ...but everything around it still closed, and still as one tree.
    agent = spans.one("invoke_agent claude-agent-sdk")
    chat = spans.one("chat claude-sonnet-4-5")
    assert spans.parent_name(chat) == agent.name
    assert len({s.context.trace_id for s in spans.all}) == 1
    assert trace.get_current_span() is trace.INVALID_SPAN


@pytest.mark.asyncio
async def test_instrumentation_failure_does_not_break_the_stream(
    handler, span_exporter, tracer
):
    """A message the mapping cannot handle must not interrupt the caller's
    iteration, break the span tree, or leave a span attached."""

    class Hostile(dict):  # type: ignore[type-arg]
        """``SystemMessage.data`` is a raw CLI payload; reading it can fail."""

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("cannot read this message")

    messages = [
        init(),
        SystemMessage(subtype="init", data=Hostile(session_id="x")),
        result(),
    ]
    with tracer.start_as_current_span("caller") as caller:
        seen = await drive(handler, messages)
        assert trace.get_current_span() is caller

    assert seen == messages
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert spans.parent_name(agent) == "caller"
    assert len({s.context.trace_id for s in spans.all}) == 1


# ─── boundaries and context ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_opens_one_agent_span_per_human_result(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("answer one"), message_id="m1"),
            result(origin={"kind": "human"}),
            assistant(TextBlock("answer two"), message_id="m2"),
            result(origin={"kind": "human"}),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    agents = spans.named("invoke_agent claude-agent-sdk")
    chats = spans.named("chat claude-sonnet-4-5")
    assert len(agents) == 2
    assert len(chats) == 2
    # Each turn is its own trace, threaded by the conversation id.
    assert len({a.context.trace_id for a in agents}) == 2
    for span in spans.all:
        assert span.attributes["gen_ai.conversation.id"] == SESSION
    # Matched by span id, since both agent spans share the same name: a chat
    # mis-parented onto the previous turn's agent would still pass a name check.
    pairs = {
        c.attributes["gen_ai.response.id"]: spans.parent_of(c) for c in chats
    }
    assert pairs["m1"] is not pairs["m2"]
    assert {id(a) for a in pairs.values()} == {id(a) for a in agents}
    for chat in chats:
        assert chat.context.trace_id == spans.parent_of(chat).context.trace_id


def cumulative(
    input_tokens: int, output_tokens: int, cost: float
) -> dict[str, Any]:
    """``model_usage`` as the CLI reports it: totals for the whole session."""
    return {
        "claude-sonnet-4-5": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "costUSD": cost,
        }
    }


@pytest.mark.asyncio
async def test_each_turn_reports_only_its_own_tokens(handler, span_exporter):
    """``model_usage`` and ``total_cost_usd`` accumulate over the session.

    A fresh root opens per turn while the counters keep growing, so reading them
    as they stand had turn two report turn one's tokens and cost as well as its
    own, and turn three all three turns'.
    """
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            result(
                origin={"kind": "human"}, model_usage=cumulative(100, 10, 0.01)
            ),
            assistant(TextBlock("two"), message_id="m2"),
            result(
                origin={"kind": "human"}, model_usage=cumulative(350, 25, 0.04)
            ),
            assistant(TextBlock("three"), message_id="m3"),
            result(
                origin={"kind": "human"}, model_usage=cumulative(900, 60, 0.10)
            ),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    agents = sorted(
        spans.named("invoke_agent claude-agent-sdk"), key=lambda s: s.end_time
    )
    assert [a.attributes["gen_ai.usage.input_tokens"] for a in agents] == [
        100,
        250,
        550,
    ]
    assert [a.attributes["gen_ai.usage.output_tokens"] for a in agents] == [
        10,
        15,
        35,
    ]
    assert [
        a.attributes["claude_agent_sdk.usage.cost_usd"] for a in agents
    ] == [
        pytest.approx(0.01),
        pytest.approx(0.03),
        pytest.approx(0.06),
    ]


@pytest.mark.asyncio
async def test_a_conversation_reset_rebases_the_cumulative_counters(
    handler, span_exporter
):
    """``/clear`` zeroes the CLI's running totals, so the baseline the next
    turn is measured against has to go back to zero with them."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            result(
                origin={"kind": "human"}, model_usage=cumulative(500, 40, 0.05)
            ),
            conversation_reset(),
            assistant(TextBlock("two"), message_id="m2"),
            result(
                origin={"kind": "human"}, model_usage=cumulative(120, 8, 0.01)
            ),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    agents = sorted(
        spans.named("invoke_agent claude-agent-sdk"), key=lambda s: s.end_time
    )
    assert [a.attributes["gen_ai.usage.input_tokens"] for a in agents] == [
        500,
        120,
    ]


@pytest.mark.asyncio
async def test_multi_turn_waits_for_background_work_before_ending_a_turn(
    handler, span_exporter
):
    """The result that answers the caller arrives while a delegation is still
    running — measured at 18.2s against the task's real completion at 36.4s.

    Ending the turn there closes the subagent's parent out from under it, so
    the Task tool is failed as never having returned and the subagent, its
    launch block already discarded, reappears under the next turn's agent span
    under a placeholder name.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            result(origin={"kind": "human"}),
            # The subagent only starts speaking after the caller was answered.
            assistant(
                TextBlock("subagent working"),
                parent="call-task",
                message_id="m_sub",
            ),
            task_notification("call-task", summary="the subagent report"),
            assistant(TextBlock("next turn"), message_id="m2"),
            result(origin={"kind": "human"}),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    tool = spans.one("execute_tool Task")
    subagent = spans.one("invoke_agent general-purpose")
    agents = spans.named("invoke_agent claude-agent-sdk")

    assert tool.status.status_code is not trace.StatusCode.ERROR
    # The delegation stays where it belongs instead of being re-rooted.
    assert spans.parent_name(subagent) == tool.name
    assert spans.parent_name(tool) == "invoke_agent claude-agent-sdk"
    assert len(agents) == 2
    # Turn one owns the delegation; turn two is a separate trace.
    turn_one = spans.parent_of(tool)
    assert turn_one.context.trace_id == subagent.context.trace_id
    assert len({a.context.trace_id for a in agents}) == 2


@pytest.mark.asyncio
async def test_background_task_open_at_teardown_is_not_an_error(
    handler, span_exporter
):
    """A tool whose receipt was deliberately ignored is not a tool that hung."""
    await drive(
        handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="bg-1", name="Bash", input={}),
                message_id="m1",
            ),
            task_started("bg-1", task_type="local_bash"),
            user(ToolResultBlock(tool_use_id="bg-1", content="started")),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Bash")
    assert tool.status.status_code is not trace.StatusCode.ERROR
    assert "tool never returned" not in str(tool.status.description)


@pytest.mark.asyncio
async def test_late_message_does_not_reopen_a_finished_subagent(
    handler, span_exporter
):
    """Re-opening the subtree would start a second invoke_agent that nothing is
    left to close, so it leaks to teardown parented to the root."""
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            assistant(
                TextBlock("subagent working"),
                parent="call-task",
                message_id="m_sub",
            ),
            task_notification("call-task", summary="report"),
            user(ToolResultBlock(tool_use_id="call-task", content="report")),
            # Straggler naming a subagent whose span has already ended.
            assistant(
                TextBlock("late"), parent="call-task", message_id="m_late"
            ),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    subagent = spans.one("invoke_agent general-purpose")
    late = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m_late"
    )
    assert spans.parent_name(late) == "invoke_agent claude-agent-sdk"
    assert subagent.end_time <= late.start_time


@pytest.mark.asyncio
async def test_a_result_does_not_split_a_subagents_call_in_two(
    handler, span_exporter
):
    """A result closes the main thread's call, not a subagent's.

    Measured in a recorded session: a result lands between two assistant
    messages carrying one message_id, so closing every chat there rebuilt the
    subagent's context from scratch and emitted a second span — half a call
    each, the first flagged partial.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            assistant(
                TextBlock("thinking"), parent="call-task", message_id="m_sub"
            ),
            result(origin={"kind": "task-notification"}),
            assistant(
                TextBlock("and the answer"),
                parent="call-task",
                message_id="m_sub",
                usage={"input_tokens": 50, "output_tokens": 9},
            ),
            task_notification("call-task", summary="report"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    sub_chats = [
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m_sub"
    ]
    assert len(sub_chats) == 1, "one API call is one chat span"
    assert sub_chats[0].attributes["gen_ai.usage.output_tokens"] == 9


@pytest.mark.asyncio
async def test_a_turn_the_cli_injects_is_its_own_invocation(
    handler, span_exporter
):
    """A finished background task makes the CLI run a turn of its own, with its
    own model calls and its own share of the session's usage."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            result(model_usage=cumulative(100, 10, 0.1)),
            assistant(TextBlock("task finished"), message_id="m2"),
            result(
                origin={"kind": "task-notification"},
                model_usage=cumulative(200, 20, 0.2),
            ),
            assistant(TextBlock("two"), message_id="m3"),
            result(model_usage=cumulative(300, 30, 0.3)),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    agents = spans.named("invoke_agent claude-agent-sdk")
    assert [a.attributes["gen_ai.usage.output_tokens"] for a in agents] == [
        10,
        10,
        10,
    ]
    parents = {
        chat.attributes["gen_ai.response.id"]: spans.parent_of(chat)
        for chat in spans.named("chat claude-sonnet-4-5")
    }
    assert [parents[m] for m in ("m1", "m2", "m3")] == agents


@pytest.mark.asyncio
async def test_a_turn_waiting_on_background_work_ends_when_the_next_begins(
    handler, span_exporter
):
    """A background task can outlive every later turn — a dev server never
    reports finishing — so waiting for it cannot hold its turn open for good."""
    await drive(
        handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="bg", name="Bash", input={}), message_id="m1"
            ),
            task_started("bg", task_type="local_bash"),
            user(ToolResultBlock(tool_use_id="bg", content="started")),
            result(model_usage=cumulative(100, 10, 0.1)),
            init(),
            assistant(TextBlock("two"), message_id="m2"),
            result(model_usage=cumulative(200, 20, 0.2)),
            init(),
            assistant(TextBlock("three"), message_id="m3"),
            result(model_usage=cumulative(300, 30, 0.3)),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    agents = spans.named("invoke_agent claude-agent-sdk")
    assert [a.attributes["gen_ai.usage.output_tokens"] for a in agents] == [
        10,
        10,
        10,
    ]
    tool = spans.one("execute_tool Bash")
    assert spans.parent_of(tool) is agents[0]
    assert tool.end_time <= agents[0].end_time
    # Still running is not failed.
    assert tool.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_background_work_reporting_after_its_turn_ended_is_dropped(
    handler, span_exporter
):
    """The turn that owned it has ended, so a late subagent message has no
    parent left: attaching it to the current turn would credit that turn with
    work it never asked for."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            result(),
            assistant(TextBlock("early"), parent="call-task", message_id="s1"),
            init(),
            assistant(TextBlock("next turn"), message_id="m2"),
            assistant(TextBlock("late"), parent="call-task", message_id="s2"),
            task_notification("call-task", summary="the report"),
            result(),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    assert len(spans.named("invoke_agent claude-agent-sdk")) == 2
    assert len(spans.named("invoke_agent researcher")) == 1
    responses = {
        c.attributes["gen_ai.response.id"]
        for c in spans.all
        if c.name.startswith("chat ")
    }
    assert responses == {"m1", "s1", "m2"}


@pytest.mark.asyncio
async def test_parallel_tools_closing_out_of_order(
    handler, span_exporter, tracer
):
    """Two tools open, and the second one's result arrives first.

    Closing out of order must still leave both parented to the agent and the
    caller's active context untouched.
    """
    messages = [
        init(),
        assistant(
            ToolUseBlock(id="a", name="Read", input={}),
            ToolUseBlock(id="b", name="Grep", input={}),
        ),
        user(
            ToolResultBlock(tool_use_id="b", content="grep out"),
            ToolResultBlock(tool_use_id="a", content="read out"),
        ),
        result(),
    ]

    async def stream():
        for message in messages:
            yield message

    with tracer.start_as_current_span("caller") as caller:
        before = otel_context.get_current()
        run = AgentRun(handler)
        # The invariant is per-yield, not just per-run: these spans do not finish
        # in the order they opened, and util-genai attaches each one it starts.
        run.submit("hi")
        async for _ in run.iterate(stream):
            assert trace.get_current_span() is caller
            assert otel_context.get_current() == before
        assert trace.get_current_span() is caller

    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert spans.parent_name(agent) == "caller"
    for name in ("execute_tool Read", "execute_tool Grep"):
        assert spans.parent_name(spans.one(name)) == agent.name


@pytest.mark.asyncio
async def test_agent_span_nests_under_the_callers_span(
    handler, span_exporter, tracer
):
    with tracer.start_as_current_span("server-request"):
        await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert spans.parent_name(agent) == "server-request"
    # One trace for the whole interaction.
    assert len({s.context.trace_id for s in spans.all}) == 1


@pytest.mark.asyncio
async def test_no_context_leak_after_a_run(handler, span_exporter):
    before = otel_context.get_current()
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert trace.get_current_span() is trace.INVALID_SPAN
    assert otel_context.get_current() == before


@pytest.mark.asyncio
async def test_no_warning_logs_during_a_run(handler, span_exporter, caplog):
    task = ToolUseBlock(
        id="t1", name="Task", input={"subagent_type": "helper"}
    )
    with caplog.at_level(0):
        await drive(
            handler,
            [
                init(),
                assistant(task),
                task_started("t1"),
                user(ToolResultBlock(tool_use_id="t1", content="launched")),
                assistant(TextBlock("sub"), parent="t1", message_id="ms"),
                task_notification("t1"),
                result(),
            ],
        )
    noisy = [r for r in caplog.records if r.levelno >= 30]
    assert not noisy, [r.getMessage() for r in noisy]


# ─── compaction and resets ──────────────────────────────────────────────────


def compact_boundary() -> SystemMessage:
    return SystemMessage(
        subtype="compact_boundary",
        data={"compact_metadata": {"trigger": "auto"}},
    )


@pytest.mark.asyncio
async def test_compaction_marks_the_run_and_every_call_after_it(
    handler, span_exporter
):
    """A compacted context is not the one the earlier calls saw, so the marker
    belongs on the run and on every call from the boundary on."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("before"), message_id="m1"),
            compact_boundary(),
            assistant(TextBlock("after"), message_id="m2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    chats = {
        c.attributes["gen_ai.response.id"]: c
        for c in spans.named("chat claude-sonnet-4-5")
    }
    assert agent.attributes["gen_ai.conversation.compacted"] is True
    assert chats["m2"].attributes["gen_ai.conversation.compacted"] is True
    assert "gen_ai.conversation.compacted" not in chats["m1"].attributes


@pytest.mark.asyncio
async def test_compaction_outlives_the_turn_that_triggered_it(
    handler, span_exporter
):
    """Compaction is a property of the conversation, not of one turn: the next
    turn is still working from the compacted context."""
    await drive(
        handler,
        [
            init(),
            compact_boundary(),
            assistant(TextBlock("one"), message_id="m1"),
            result(origin={"kind": "human"}),
            assistant(TextBlock("two"), message_id="m2"),
            result(origin={"kind": "human"}),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    for span in spans.all:
        assert span.attributes["gen_ai.conversation.compacted"] is True, (
            span.name
        )


@pytest.mark.asyncio
async def test_a_reset_starts_a_new_conversation_id(handler, span_exporter):
    """``/clear`` replaces the conversation mid-session. A call made before it
    belongs to the old one."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("before"), message_id="m1"),
            conversation_reset(),
            assistant(TextBlock("after"), message_id="m2"),
            result(),
        ],
    )
    chats = {
        c.attributes["gen_ai.response.id"]: c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
    }
    assert chats["m1"].attributes["gen_ai.conversation.id"] == SESSION
    assert chats["m2"].attributes["gen_ai.conversation.id"] == NEW_CONVERSATION


@pytest.mark.asyncio
async def test_a_reset_clears_the_compacted_marker(handler, span_exporter):
    """An empty conversation is not a compacted one."""
    await drive(
        handler,
        [
            init(),
            compact_boundary(),
            assistant(TextBlock("one"), message_id="m1"),
            result(origin={"kind": "human"}),
            conversation_reset(),
            assistant(TextBlock("two"), message_id="m2"),
            result(origin={"kind": "human"}),
        ],
        multi_turn=True,
    )
    chats = {
        c.attributes["gen_ai.response.id"]: c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
    }
    assert chats["m1"].attributes["gen_ai.conversation.compacted"] is True
    assert "gen_ai.conversation.compacted" not in chats["m2"].attributes


# ─── content ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_content_shape_matches_the_contract(
    capturing_handler, span_exporter
):
    call = ToolUseBlock(id="call-1", name="Bash", input={"command": "ls"})
    await drive(
        capturing_handler,
        [
            init(),
            assistant(call, message_id="m1"),
            user(ToolResultBlock(tool_use_id="call-1", content="a.txt")),
            assistant(TextBlock("one file"), message_id="m2"),
            result(text="one file"),
        ],
    )
    spans = Spans(span_exporter)
    first, second = sorted(
        spans.named("chat claude-sonnet-4-5"),
        key=lambda s: s.attributes["gen_ai.response.id"],
    )
    out = json.loads(first.attributes["gen_ai.output.messages"])
    assert out[0]["role"] == "assistant"
    part = out[0]["parts"][0]
    assert part["type"] == "tool_call"
    assert part["name"] == "Bash"
    # The id ties the assistant tool_call, the execute_tool span, and the response.
    assert part["id"] == "call-1"
    assert (
        spans.one("execute_tool Bash").attributes["gen_ai.tool.call.id"]
        == "call-1"
    )

    replayed = json.loads(second.attributes["gen_ai.input.messages"])
    response_parts = [
        p
        for message in replayed
        for p in message["parts"]
        if p["type"] == "tool_call_response"
    ]
    assert response_parts and response_parts[0]["id"] == "call-1"
    # Only the canonical part types.
    for message in replayed + out:
        for p in message["parts"]:
            assert p["type"] in {
                "text",
                "reasoning",
                "tool_call",
                "tool_call_response",
            }


@pytest.mark.asyncio
async def test_a_turn_with_text_and_a_tool_result_is_two_messages(
    capturing_handler, span_exporter
):
    """One user turn can be both the tool answering and the user speaking.
    Folded into one message, the text got role ``tool``."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="c1", name="Read", input={}), message_id="m1"
            ),
            user(
                ToolResultBlock(tool_use_id="c1", content="file body"),
                TextBlock("also, hurry up"),
            ),
            assistant(TextBlock("done"), message_id="m2"),
            result(),
        ],
    )
    second = next(
        c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m2"
    )
    replayed = json.loads(second.attributes["gen_ai.input.messages"])
    by_role = {m["role"]: m for m in replayed}
    assert by_role["tool"]["parts"][0]["type"] == "tool_call_response"
    assert by_role["user"]["parts"][-1] == {
        "type": "text",
        "content": "also, hurry up",
    }


@pytest.mark.asyncio
async def test_the_replayed_conversation_is_bounded(
    capturing_handler, span_exporter
):
    """Each call's input is the whole conversation, but it is re-serialized onto
    every chat span, so an unbounded one puts megabytes on a single trace and
    the exporter silently drops what does not fit."""
    messages: list[Any] = [init()]
    for step in range(30):
        messages.append(
            assistant(
                ToolUseBlock(id=f"c{step}", name="Read", input={}),
                message_id=f"m{step}",
            )
        )
        messages.append(
            user(ToolResultBlock(tool_use_id=f"c{step}", content="x" * 64))
        )
    messages.append(assistant(TextBlock("done"), message_id="last"))
    messages.append(result())
    await drive(capturing_handler, messages)

    last = next(
        c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "last"
    )
    replayed = json.loads(last.attributes["gen_ai.input.messages"])
    assert len(replayed) == 40
    # The bound trims the oldest, so the most recent step is still there.
    assert any(
        part.get("id") == "c29"
        for message in replayed
        for part in message["parts"]
    )


@pytest.mark.asyncio
async def test_compaction_drops_the_turns_it_replaced(
    capturing_handler, span_exporter
):
    """After compaction the recorded input was strictly larger than what the
    model saw."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(TextBlock("before the boundary"), message_id="m1"),
            compact_boundary(),
            assistant(TextBlock("after"), message_id="m2"),
            assistant(TextBlock("after again"), message_id="m3"),
            result(),
        ],
    )
    third = next(
        c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m3"
    )
    replayed = third.attributes["gen_ai.input.messages"]
    assert "before the boundary" not in replayed
    assert "after" in replayed


@pytest.mark.asyncio
async def test_streaming_input_seeds_the_root_from_the_replayed_turn(
    capturing_handler, span_exporter
):
    """An async-iterable prompt cannot be read without starving the SDK, so the
    root has no input of its own. The CLI replays the submitted turn as a
    UserMessage, which is the same text."""

    async def prompt():  # pragma: no cover - never consumed
        yield {"type": "user", "message": {"role": "user", "content": "hi"}}

    await drive(
        capturing_handler,
        [
            init(),
            user(TextBlock("summarize the incident report")),
            assistant(TextBlock("here you go"), message_id="m1"),
            result(text="here you go"),
        ],
        prompt=prompt(),
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    seeded = json.loads(agent.attributes["gen_ai.input.messages"])
    assert seeded == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "summarize the incident report"}
            ],
            "name": None,
        }
    ]


@pytest.mark.asyncio
async def test_a_server_tool_is_content_not_an_execute_tool_span(
    capturing_handler, span_exporter
):
    """The API runs a server tool on the model's behalf and never comes back for
    a result, so a span opened for one would never be closed on time: it would be
    swept up at teardown with an ERROR status. ``_is_tool_use`` guards on a class
    name prefix, so this is what notices if that stops matching."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ThinkingBlock(thinking="let me look that up", signature="sig"),
                ServerToolUseBlock(
                    id="srv-1", name="web_search", input={"query": "otel"}
                ),
                message_id="m1",
            ),
            user(
                ServerToolResultBlock(
                    tool_use_id="srv-1", content={"type": "web_search_result"}
                )
            ),
            assistant(TextBlock("found it"), message_id="m2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    assert not [s for s in spans.all if s.name.startswith("execute_tool")]
    first = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m1"
    )
    types = [
        part["type"]
        for message in json.loads(first.attributes["gen_ai.output.messages"])
        for part in message["parts"]
    ]
    assert types == ["reasoning", "server_tool_call"]
    second = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m2"
    )
    replayed = [
        part["type"]
        for message in json.loads(second.attributes["gen_ai.input.messages"])
        for part in message["parts"]
    ]
    assert "server_tool_call_response" in replayed


@pytest.mark.asyncio
async def test_an_empty_assistant_turn_is_still_a_turn(
    capturing_handler, span_exporter
):
    """A turn whose only block is an empty TextBlock, or a ThinkingBlock the
    model signed but left blank, used to produce no parts — which took the whole
    message out of the output and out of the replayed history."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ThinkingBlock(thinking="", signature="sig"), message_id="m1"
            ),
            assistant(TextBlock(""), message_id="m2"),
            result(),
        ],
    )
    chats = {
        c.attributes["gen_ai.response.id"]: c
        for c in Spans(span_exporter).named("chat claude-sonnet-4-5")
    }
    first = json.loads(chats["m1"].attributes["gen_ai.output.messages"])
    assert first[0]["parts"] == [{"type": "reasoning", "content": ""}]
    second = json.loads(chats["m2"].attributes["gen_ai.output.messages"])
    assert second[0]["parts"] == [{"type": "text", "content": ""}]
    # ...and the earlier turn is in the later call's replayed input.
    assert (
        len(json.loads(chats["m2"].attributes["gen_ai.input.messages"])) == 2
    )


@pytest.mark.asyncio
async def test_a_subagents_own_work_nests_under_the_subagent(
    capturing_handler, span_exporter
):
    """Everything a subagent does belongs inside its own invoke_agent span.

    The delta is included for the parented-stream-event path, which the CLI does
    not currently take: measured across nine recorded sessions every stream event
    carried ``parent_tool_use_id: null``.
    """
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "general-purpose"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            assistant(
                ToolUseBlock(
                    id="call-sub", name="Grep", input={"pattern": "TODO"}
                ),
                parent="call-task",
                message_id="m_sub",
                usage={"input_tokens": 50, "output_tokens": 4},
            ),
            chunk(parent="call-task"),
            delta(
                output_tokens=77, stop_reason="tool_use", parent="call-task"
            ),
            user(
                ToolResultBlock(tool_use_id="call-sub", content="two hits"),
                parent="call-task",
            ),
            task_notification("call-task", summary="reviewed"),
            assistant(TextBlock("all done"), message_id="m2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    subagent = spans.one("invoke_agent general-purpose")
    sub_tool = spans.one("execute_tool Grep")
    sub_chat = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m_sub"
    )
    assert spans.parent_name(sub_tool) == subagent.name
    assert spans.parent_name(sub_chat) == subagent.name
    assert sub_chat.attributes["gen_ai.usage.output_tokens"] == 77
    assert "claude_agent_sdk.usage.is_partial" not in sub_chat.attributes
    assert sub_chat.attributes["gen_ai.response.finish_reasons"] == (
        "tool_use",
    )
    # The subagent's own tool_result belongs to its history, not the main
    # thread's.
    main_second = next(
        c
        for c in spans.named("chat claude-sonnet-4-5")
        if c.attributes["gen_ai.response.id"] == "m2"
    )
    assert "two hits" not in main_second.attributes["gen_ai.input.messages"]


@pytest.mark.asyncio
async def test_a_subagents_tool_is_attributed_to_the_subagent(
    handler, span_exporter
):
    """On execute_tool ``gen_ai.agent.name`` is the agent doing the executing."""
    task = ToolUseBlock(
        id="call-task", name="Task", input={"subagent_type": "code-reviewer"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            assistant(
                ToolUseBlock(id="call-sub", name="Read", input={}),
                parent="call-task",
                message_id="m_sub",
            ),
            user(
                ToolResultBlock(tool_use_id="call-sub", content="body"),
                parent="call-task",
            ),
            task_notification("call-task", summary="reviewed"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    assert (
        spans.one("execute_tool Read").attributes["gen_ai.agent.name"]
        == "code-reviewer"
    )
    assert (
        spans.one("execute_tool Task").attributes["gen_ai.agent.name"]
        == "claude-agent-sdk"
    )


# ─── the options a caller passes ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_custom_system_prompt_is_captured(
    capturing_handler, span_exporter
):
    """Identical content was recorded or dropped depending on which API shape
    the caller reached for: the bare string was captured and the custom form,
    which holds the real full prompt, returned nothing."""
    await drive(
        capturing_handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            system_prompt={"type": "custom", "prompt": "You are a linter."},
            model="claude-sonnet-4-5",
        ),
    )
    spans = Spans(span_exporter)
    for span in (
        spans.one("chat claude-sonnet-4-5"),
        spans.one("invoke_agent claude-agent-sdk"),
    ):
        assert json.loads(span.attributes["gen_ai.system_instructions"]) == [
            {"type": "text", "content": "You are a linter."}
        ]


@pytest.mark.asyncio
async def test_a_preset_system_prompt_is_named_not_quoted(
    capturing_handler, span_exporter
):
    """ "preset: claude_code" is a label, not something the model was told. The
    ``append`` text beside it is the caller's own instruction."""
    await drive(
        capturing_handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": "Always answer in French.",
            },
            model="claude-sonnet-4-5",
        ),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert json.loads(chat.attributes["gen_ai.system_instructions"]) == [
        {
            "type": "claude_agent_sdk.system_prompt_source",
            "source": "preset",
            "name": "claude_code",
        },
        {"type": "text", "content": "Always answer in French."},
    ]


@pytest.mark.asyncio
async def test_a_system_prompt_file_records_its_path(
    capturing_handler, span_exporter
):
    await drive(
        capturing_handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            system_prompt={"type": "file", "path": "/etc/agent/prompt.md"},
            model="claude-sonnet-4-5",
        ),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert json.loads(chat.attributes["gen_ai.system_instructions"]) == [
        {
            "type": "claude_agent_sdk.system_prompt_source",
            "source": "file",
            "name": "/etc/agent/prompt.md",
        }
    ]


@pytest.mark.asyncio
async def test_a_bare_string_system_prompt_is_captured(
    capturing_handler, span_exporter
):
    await drive(
        capturing_handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            system_prompt="You are terse.", model="claude-sonnet-4-5"
        ),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert json.loads(chat.attributes["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "You are terse."}
    ]


@pytest.mark.asyncio
async def test_the_system_prompt_is_absent_without_content_capture(
    handler, span_exporter
):
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            system_prompt="You are terse.", model="claude-sonnet-4-5"
        ),
    )
    for span in Spans(span_exporter).all:
        assert "gen_ai.system_instructions" not in span.attributes


@pytest.mark.asyncio
async def test_effort_becomes_the_reasoning_level(handler, span_exporter):
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            effort="high", model="claude-sonnet-4-5"
        ),
    )
    chat = Spans(span_exporter).one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.request.reasoning.level"] == "high"


@pytest.mark.asyncio
async def test_a_caller_supplied_conversation_id_overrides_the_session(
    handler, span_exporter
):
    async def stream():
        for message in [init(), assistant(TextBlock("hi")), result()]:
            yield message

    run = AgentRun(handler)
    run.submit("hi", "ticket-42")
    async for _ in run.iterate(stream):
        pass
    spans = Spans(span_exporter).all
    assert spans
    for span in spans:
        assert span.attributes["gen_ai.conversation.id"] == "ticket-42"


@pytest.mark.asyncio
async def test_options_model_is_the_fallback_when_init_carries_none(
    handler, span_exporter
):
    await drive(
        handler,
        [
            SystemMessage(subtype="init", data={"session_id": SESSION}),
            assistant(TextBlock("hi"), model="", message_id="m1"),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(model="claude-opus-4-5"),
    )
    chat = Spans(span_exporter).one("chat claude-opus-4-5")
    assert chat.attributes["gen_ai.request.model"] == "claude-opus-4-5"


@pytest.mark.asyncio
async def test_content_is_absent_by_default(handler, span_exporter):
    await drive(handler, [init(), assistant(TextBlock("secret")), result()])
    for span in Spans(span_exporter).all:
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" not in span.attributes
        assert "gen_ai.tool.call.result" not in span.attributes


@pytest.mark.asyncio
async def test_span_names_carry_no_content(handler, span_exporter):
    await drive(
        handler,
        [init(), assistant(TextBlock("a very secret answer")), result()],
        prompt="my secret prompt",
    )
    for span in Spans(span_exporter).all:
        assert "secret" not in span.name
        assert SESSION not in span.name


@pytest.mark.asyncio
async def test_the_task_description_is_content(handler, span_exporter):
    """``description`` is the model's own summary of the user's request, so it
    is content — and per-invocation data in a field semconv describes as a
    static application-provided description."""
    task = ToolUseBlock(
        id="t1",
        name="Task",
        input={
            "subagent_type": "general-purpose",
            "description": "Refactor the billing reconciliation in ledger.py",
        },
    )
    messages = [
        init(),
        assistant(task, message_id="m1"),
        task_started("t1"),
        assistant(TextBlock("working"), parent="t1", message_id="ms"),
        task_notification("t1"),
        result(),
    ]
    await drive(handler, messages)
    subagent = Spans(span_exporter).one("invoke_agent general-purpose")
    assert "gen_ai.agent.description" not in subagent.attributes


@pytest.mark.asyncio
async def test_the_task_description_is_recorded_when_content_is_captured(
    capturing_handler, span_exporter
):
    task = ToolUseBlock(
        id="t1",
        name="Task",
        input={
            "subagent_type": "general-purpose",
            "description": "do a thing",
        },
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("t1"),
            assistant(TextBlock("working"), parent="t1", message_id="ms"),
            task_notification("t1"),
            result(),
        ],
    )
    subagent = Spans(span_exporter).one("invoke_agent general-purpose")
    assert subagent.attributes["gen_ai.agent.description"] == "do a thing"


@pytest.mark.asyncio
async def test_a_non_delegation_tools_arguments_are_not_read_as_a_task(
    handler, span_exporter
):
    """``_launches`` holds every tool_use block, so reading an arbitrary one as
    a TaskToolInput picked up whatever argument happened to be named "model" and
    wrote it to ``gen_ai.request.model`` — always emitted, and a metric
    dimension besides."""
    fetch = ToolUseBlock(
        id="c1",
        name="WebFetch",
        input={"model": "summarizer-v2", "url": "https://example.com"},
    )
    await drive(
        handler,
        [
            init(),
            assistant(fetch, message_id="m1"),
            assistant(TextBlock("nested"), parent="c1", message_id="ms"),
            user(ToolResultBlock(tool_use_id="c1", content="fetched")),
            result(),
        ],
    )
    nested = Spans(span_exporter).one("invoke_agent WebFetch")
    assert nested.attributes["gen_ai.request.model"] == "claude-sonnet-4-5", (
        "the session's model, not a tool argument"
    )


@pytest.mark.asyncio
async def test_an_unknown_model_is_left_off_rather_than_invented(
    handler, span_exporter
):
    """A fabricated identifier reaches the span name, both model attributes and
    the metric attribute set; semconv makes both attributes optional."""
    await drive(
        handler,
        [
            SystemMessage(subtype="init", data={"session_id": SESSION}),
            assistant(TextBlock("hi"), model="", message_id="m1"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    chat = spans.one("chat")
    assert "gen_ai.request.model" not in chat.attributes
    assert "gen_ai.response.model" not in chat.attributes
    assert not spans.named("chat unknown")


@pytest.mark.asyncio
async def test_a_requested_model_is_not_reported_as_the_response_model(
    handler, span_exporter
):
    """``options.model`` is what was asked for. Only the wire says what
    answered."""
    await drive(
        handler,
        [
            SystemMessage(subtype="init", data={"session_id": SESSION}),
            assistant(TextBlock("hi"), model="", message_id="m1"),
            result(),
        ],
        sdk_options=ClaudeAgentOptions(model="sonnet"),
    )
    chat = Spans(span_exporter).one("chat sonnet")
    assert chat.attributes["gen_ai.request.model"] == "sonnet"
    assert "gen_ai.response.model" not in chat.attributes


# ─── provider resolution ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hosting_platform_outranks_the_vendor(
    handler, span_exporter, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    for span in Spans(span_exporter).all:
        assert span.attributes["gen_ai.provider.name"] == "aws.bedrock"


@pytest.mark.asyncio
async def test_options_env_sets_a_routing_flag(handler, span_exporter):
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            env={"CLAUDE_CODE_USE_VERTEX": "true"},
            model="claude-sonnet-4-5",
        ),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "gcp.vertex_ai"
    )


@pytest.mark.asyncio
async def test_ambient_routing_survives_an_options_object(
    handler, span_exporter, monkeypatch
):
    """``options.env`` defaults to an empty dict, not None.

    Treating it as the whole environment whenever it is a dict hides the
    ambient routing flags behind every options object a caller passes.
    """
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(model="claude-sonnet-4-5"),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "aws.bedrock"
    )


@pytest.mark.asyncio
async def test_options_env_merges_with_ours_key_by_key(
    handler, span_exporter, monkeypatch
):
    """The CLI merges ``options.env`` over the inherited environment, so an
    unrelated key does not clear the ambient routing flag, and a key the
    options do set overrides it."""
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            env={"ANTHROPIC_API_KEY": "sk-test"},
            model="claude-sonnet-4-5",
        ),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "aws.bedrock"
    )


@pytest.mark.asyncio
async def test_options_env_can_disable_an_ambient_routing_flag(
    handler, span_exporter, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            env={"CLAUDE_CODE_USE_BEDROCK": "0"},
            model="claude-sonnet-4-5",
        ),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "anthropic"
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
@pytest.mark.asyncio
async def test_routing_flag_is_on_for_the_values_the_cli_accepts(
    handler, span_exporter, monkeypatch, value
):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", value)
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "aws.bedrock"
    )


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", ""])
@pytest.mark.asyncio
async def test_disabled_routing_flag_is_not_a_platform(
    handler, span_exporter, monkeypatch, value
):
    """The CLI reads these flags through an allowlist, so anything outside it
    is off. A denylist of just "0" and "false" would route "no" to Bedrock."""
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", value)
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "anthropic"
    )


@pytest.mark.parametrize(
    ("flag", "provider"),
    [
        ("CLAUDE_CODE_USE_BEDROCK", "aws.bedrock"),
        ("CLAUDE_CODE_USE_MANTLE", "aws.bedrock"),
        ("CLAUDE_CODE_USE_VERTEX", "gcp.vertex_ai"),
        ("CLAUDE_CODE_USE_FOUNDRY", "azure.ai.inference"),
        ("CLAUDE_CODE_USE_ANTHROPIC_AWS", "anthropic"),
        ("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "anthropic"),
    ],
)
@pytest.mark.asyncio
async def test_every_routing_flag_the_cli_reads_is_mapped(
    handler, span_exporter, monkeypatch, flag, provider
):
    monkeypatch.setenv(flag, "1")
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == provider
    )


@pytest.mark.parametrize(
    ("also_set", "provider"),
    [
        # The CLI checks foundry before vertex, and vertex last of all.
        ("CLAUDE_CODE_USE_VERTEX", "azure.ai.inference"),
        # ...and bedrock before foundry.
        ("CLAUDE_CODE_USE_BEDROCK", "aws.bedrock"),
    ],
)
@pytest.mark.asyncio
async def test_two_flags_resolve_the_way_the_cli_resolves_them(
    handler, span_exporter, monkeypatch, also_set, provider
):
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv(also_set, "1")
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == provider
    )


@pytest.mark.asyncio
async def test_options_env_clears_an_inherited_flag_with_an_empty_value(
    handler, span_exporter, monkeypatch
):
    """The CLI documents the empty string as the way to clear a routing flag
    exported by a shell profile you don't control."""
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=ClaudeAgentOptions(
            env={"CLAUDE_CODE_USE_VERTEX": ""},
            model="claude-sonnet-4-5",
        ),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "anthropic"
    )


@pytest.mark.asyncio
async def test_a_delegation_tool_named_agent_names_its_subagent(
    capturing_handler, span_exporter
):
    """The CLI calls the delegation tool ``Agent``, not ``Task``.

    Both names resolve a subagent_type, and every recorded session uses
    ``Agent``, so the subagent takes its name from the input either way.
    """
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(
                    id="call-agent",
                    name="Agent",
                    input={
                        "subagent_type": "researcher",
                        "description": "dig through the ledger",
                    },
                ),
                message_id="msg_1",
            ),
            task_started("call-agent"),
            assistant(
                TextBlock("subagent working"),
                parent="call-agent",
                message_id="msg_sub",
            ),
            task_notification("call-agent"),
            user(
                ToolResultBlock(tool_use_id="call-agent", content="the report")
            ),
            result(model_usage=MODEL_USAGE),
        ],
    )
    spans = Spans(span_exporter)
    subagent = spans.one("invoke_agent researcher")
    assert subagent.attributes["gen_ai.agent.name"] == "researcher"
    # Nested under the Agent tool's span, not hung off the root.
    assert (
        subagent.parent.span_id
        == spans.one("execute_tool Agent").context.span_id
    )


@pytest.mark.asyncio
async def test_a_caller_that_stops_reading_after_the_result_is_not_an_error(
    handler, span_exporter
):
    """Abandoning the stream throws GeneratorExit in at the yield.

    The caller walking away once they have their answer is not a failed run, so
    it must not reach the handler that records a cancellation.
    """
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result(model_usage=MODEL_USAGE)],
        stop_after=3,
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.UNSET
    assert "error.type" not in agent.attributes


@pytest.mark.asyncio
async def test_a_subagent_named_after_its_tool_span_closed_keeps_its_identity(
    capturing_handler, span_exporter
):
    """A subagent reads its name and model from the Task call that launched it.

    The CLI can name a subagent for the first time after that Task's own span
    has closed, so the launch block has to outlive it.
    """
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(
                    id="cA",
                    name="Agent",
                    input={
                        "subagent_type": "archivist",
                        "model": "claude-opus-4",
                    },
                ),
                message_id="msg_1",
            ),
            task_started("cA"),
            task_notification("cA"),
            user(ToolResultBlock(tool_use_id="cA", content="done")),
            # Only now does anything name the subagent.
            assistant(TextBlock("late"), parent="cA", message_id="msg_sub"),
            result(model_usage=MODEL_USAGE),
        ],
    )
    subagent = Spans(span_exporter).one("invoke_agent archivist")
    assert subagent.attributes["gen_ai.request.model"] == "claude-opus-4"


@pytest.mark.asyncio
async def test_a_foreground_delegation_closes_on_its_tool_result(
    capturing_handler, span_exporter
):
    """A foreground task reports finishing before its tool_result arrives, and
    only the tool_result carries the full output."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="cA", name="Agent", input={}), message_id="m1"
            ),
            task_started("cA"),
            task_updated(status="completed"),
            task_notification("cA", summary="short summary"),
            user(ToolResultBlock(tool_use_id="cA", content="the full report")),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Agent")
    assert tool.attributes["gen_ai.tool.call.result"] == "the full report"


@pytest.mark.asyncio
async def test_a_delegations_result_completes_its_subagents_last_call(
    handler, span_exporter
):
    """A subagent's calls get no stream events, so none of them sees its final
    usage — except the last, whose usage the delegation's tool_result carries.

    It is that one call's usage, not the subagent's total: recorded sessions
    where the subagent made two calls match the second call exactly.
    """
    await drive(
        handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="cA", name="Agent", input={}), message_id="m1"
            ),
            task_started("cA"),
            assistant(
                TextBlock("first"),
                parent="cA",
                message_id="s1",
                usage={"input_tokens": 10, "output_tokens": 6},
            ),
            assistant(
                TextBlock("last"),
                parent="cA",
                message_id="s2",
                usage={"input_tokens": 12, "output_tokens": 1},
            ),
            task_updated(status="completed"),
            task_notification("cA"),
            user(
                ToolResultBlock(tool_use_id="cA", content="report"),
                tool_use_result={
                    "status": "completed",
                    "usage": {"input_tokens": 12, "output_tokens": 479},
                },
            ),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    calls = {
        c.attributes["gen_ai.response.id"]: c
        for c in spans.all
        if c.name.startswith("chat ")
    }
    assert calls["s2"].attributes["gen_ai.usage.output_tokens"] == 479
    assert "claude_agent_sdk.usage.is_partial" not in calls["s2"].attributes
    assert calls["s1"].attributes["claude_agent_sdk.usage.is_partial"] is True
    # A total for the subagent would need every call's usage, which only the
    # last one has.
    agent = spans.one("invoke_agent Agent")
    assert "gen_ai.usage.output_tokens" not in agent.attributes


@pytest.mark.asyncio
async def test_a_subagent_ended_mid_stream_closes_its_own_tools_first(
    capturing_handler, span_exporter
):
    """A subagent killed while one of its tools is open ends then and there, not
    at teardown, so its tools have to be closed with it."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="launched")),
            assistant(
                ToolUseBlock(id="call-sub", name="Read", input={}),
                parent="call-task",
                message_id="m_sub",
            ),
            task_notification("call-task", status="killed"),
            assistant(TextBlock("moving on"), message_id="m2"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    sub_tool = spans.one("execute_tool Read")
    subagent = spans.one("invoke_agent researcher")
    assert sub_tool.end_time <= subagent.end_time
    assert sub_tool.status.description == "tool never returned"


@pytest.mark.asyncio
async def test_a_failed_tools_string_result_still_closes_it(
    capturing_handler, span_exporter
):
    """The CLI sends a failed tool's detail as a plain string, not a dict."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(ToolUseBlock(id="b", name="Bash", input={})),
            user(
                ToolResultBlock(
                    tool_use_id="b", content="boom", is_error=True
                ),
                tool_use_result="Error: Exit code 1",
            ),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Bash")
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert tool.status.description == "tool error"


def _backgrounded_delegation() -> list[Any]:
    """A turn that ends at its result with a delegation still running."""
    return [
        init(),
        assistant(
            ToolUseBlock(id="outer", name="Agent", input={}), message_id="m1"
        ),
        task_started("outer", task_id="t-outer"),
        user(ToolResultBlock(tool_use_id="outer", content="launched")),
        result(model_usage=cumulative(100, 10, 0.1)),
    ]


def _backgrounded_delegation_then_next_turn() -> list[Any]:
    """Turn two starts while turn one's delegation is still running."""
    return _backgrounded_delegation() + [
        init(),
        assistant(TextBlock("two"), message_id="m2"),
    ]


@pytest.mark.asyncio
async def test_a_task_started_by_detached_work_does_not_hold_the_next_turn(
    handler, span_exporter
):
    """The detached subagent starts a background command of its own. Task
    messages name no parent, so without tracking that command as detached too,
    turn two would wait on it instead of ending at its result."""
    messages = _backgrounded_delegation_then_next_turn() + [
        assistant(
            ToolUseBlock(id="bash", name="Bash", input={}),
            parent="outer",
            message_id="s1",
        ),
        task_started("bash", task_id="t-bash", task_type="local_bash"),
        result(model_usage=cumulative(200, 20, 0.2)),
    ]

    ended_at_result = await roots_ended_at_each_result(
        handler, messages, span_exporter
    )
    # Turn one waits on its delegation; turn two has nothing of its own left.
    assert ended_at_result == [0, 2]


@pytest.mark.asyncio
async def test_a_delegation_launched_by_detached_work_is_dropped(
    handler, span_exporter
):
    await drive(
        handler,
        _backgrounded_delegation_then_next_turn()
        + [
            assistant(
                ToolUseBlock(id="inner", name="Agent", input={}),
                parent="outer",
                message_id="s1",
            ),
            task_started("inner", task_id="t-inner"),
            assistant(TextBlock("nested"), parent="inner", message_id="n1"),
            result(model_usage=cumulative(200, 20, 0.2)),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    assert not spans.named("invoke_agent Agent")
    responses = {
        s.attributes["gen_ai.response.id"]
        for s in spans.all
        if s.name.startswith("chat ")
    }
    assert responses == {"m1", "m2"}


@pytest.mark.asyncio
async def test_the_next_turns_replayed_prompt_opens_the_next_turn(
    capturing_handler, span_exporter
):
    """With ``extra_args={"replay-user-messages": None}`` the CLI echoes each
    prompt back. A turn left open by background work must end before that echo
    arrives, or the prompt is recorded on the wrong turn."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="bg", name="Bash", input={}), message_id="m1"
            ),
            task_started("bg", task_type="local_bash"),
            user(ToolResultBlock(tool_use_id="bg", content="started")),
            result(),
            init(),
            user(TextBlock("second question")),
            assistant(TextBlock("two"), message_id="m2"),
            result(),
        ],
        multi_turn=True,
        prompt=None,
    )
    first, second = Spans(span_exporter).named("invoke_agent claude-agent-sdk")
    assert "gen_ai.input.messages" not in first.attributes
    assert json.loads(second.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "second question"}],
            "name": None,
        }
    ]


@pytest.mark.asyncio
async def test_a_result_without_usage_does_not_reset_the_count(
    handler, span_exporter
):
    """The counters are cumulative, so a result that reports none has not
    zeroed them: the next turn is still measured from the last real count."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            result(model_usage=cumulative(100, 10, 0.1)),
            assistant(TextBlock("two"), message_id="m2"),
            result(model_usage=None),
            assistant(TextBlock("three"), message_id="m3"),
            result(model_usage=cumulative(300, 30, 0.3)),
        ],
        multi_turn=True,
    )
    agents = Spans(span_exporter).named("invoke_agent claude-agent-sdk")
    assert [
        a.attributes.get("gen_ai.usage.output_tokens") for a in agents
    ] == [
        10,
        None,
        20,
    ]


@pytest.mark.asyncio
async def test_a_tool_running_inside_detached_work_is_not_a_failure(
    handler, span_exporter
):
    await drive(
        handler,
        _backgrounded_delegation()
        + [
            assistant(
                ToolUseBlock(id="slow", name="Bash", input={}),
                parent="outer",
                message_id="s1",
            ),
            init(),
            result(model_usage=cumulative(200, 20, 0.2)),
        ],
        multi_turn=True,
    )
    tool = Spans(span_exporter).one("execute_tool Bash")
    assert tool.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_a_delegation_launched_before_its_turn_ended_is_detached_with_its_owner(
    handler, span_exporter
):
    """The detached subagent launched it while its turn was still open, so it
    was never dropped as a message; it is detached along with its owner."""
    await drive(
        handler,
        _backgrounded_delegation()
        + [
            assistant(
                ToolUseBlock(id="inner", name="Agent", input={}),
                parent="outer",
                message_id="s1",
            ),
            init(),
            task_started("inner", task_id="t-inner"),
            assistant(TextBlock("nested"), parent="inner", message_id="n1"),
            assistant(TextBlock("two"), message_id="m2"),
            result(model_usage=cumulative(200, 20, 0.2)),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    responses = {
        s.attributes["gen_ai.response.id"]
        for s in spans.all
        if s.name.startswith("chat ")
    }
    assert responses == {"m1", "s1", "m2"}


@pytest.mark.asyncio
async def test_a_killed_background_subagent_ends_before_its_turn(
    capturing_handler, span_exporter
):
    """In the CLI's order: task_updated, then the notification with the summary.
    The turn ends only after the notification, once, and a late message from
    the killed subagent does not reopen it in the next turn."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    messages = [
        init(),
        assistant(task, message_id="m1"),
        task_started("call-task"),
        user(ToolResultBlock(tool_use_id="call-task", content="launched")),
        result(),
        assistant(
            ToolUseBlock(id="call-sub", name="Read", input={}),
            parent="call-task",
            message_id="m_sub",
        ),
        task_updated(status="killed"),
        task_notification("call-task", status="killed", summary="partial"),
        assistant(TextBlock("late"), parent="call-task", message_id="m_late"),
        init(),
        assistant(TextBlock("two"), message_id="m2"),
        result(),
    ]
    assert await roots_ended_at_each_result(
        capturing_handler, messages, span_exporter
    ) == [0, 2]
    spans = Spans(span_exporter)
    delegation = spans.one("execute_tool Agent")
    assert delegation.status.description == "task killed"
    assert delegation.attributes["gen_ai.tool.call.result"] == "partial"
    assert (
        spans.one("invoke_agent researcher").end_time
        <= delegation.end_time
        <= spans.named("invoke_agent claude-agent-sdk")[0].end_time
    )
    assert not spans.named("invoke_agent subagent")


@pytest.mark.asyncio
async def test_a_subagents_background_task_outlives_it_without_failing(
    handler, span_exporter
):
    """A foreground subagent can return while a background command it started
    is still running. That command has not failed, and it no longer holds the
    turn open: its parent is gone, so it is detached."""
    messages = [
        init(),
        assistant(
            ToolUseBlock(id="agent", name="Agent", input={}), message_id="m1"
        ),
        task_started("agent", task_id="t-agent"),
        assistant(
            ToolUseBlock(id="bash", name="Bash", input={}),
            parent="agent",
            message_id="s1",
        ),
        task_started("bash", task_id="t-bash", task_type="local_bash"),
        user(
            ToolResultBlock(tool_use_id="bash", content="started"),
            parent="agent",
        ),
        task_notification("agent", task_id="t-agent"),
        user(ToolResultBlock(tool_use_id="agent", content="report")),
        result(),
    ]

    ended_at_result = await roots_ended_at_each_result(
        handler, messages, span_exporter
    )
    assert ended_at_result == [1]
    bash = Spans(span_exporter).one("execute_tool Bash")
    assert bash.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_a_late_message_from_a_finished_subagent_opens_no_turn(
    handler, span_exporter
):
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    await drive(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            task_started("call-task"),
            assistant(
                TextBlock("working"), parent="call-task", message_id="s1"
            ),
            task_notification("call-task"),
            user(ToolResultBlock(tool_use_id="call-task", content="report")),
            result(),
            assistant(TextBlock("late"), parent="call-task", message_id="s2"),
            init(),
            assistant(TextBlock("two"), message_id="m2"),
            result(),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    assert len(spans.named("invoke_agent claude-agent-sdk")) == 2
    assert len(spans.named("invoke_agent researcher")) == 1
    responses = {
        s.attributes["gen_ai.response.id"]
        for s in spans.all
        if s.name.startswith("chat ")
    }
    assert responses == {"m1", "s1", "m2"}


@pytest.mark.asyncio
async def test_a_subagents_tool_failing_to_close_does_not_strand_the_delegation(
    handler, span_exporter
):
    """The subagent's own tools close on the way to closing the delegation, so
    one of them raising must not leave the delegation behind."""
    finish = ToolInvocation._apply_finish  # pylint: disable=protected-access

    def explode_for_read(
        self: ToolInvocation, *args: Any, **kwargs: Any
    ) -> Any:
        if self.tool_call_id == "read":
            raise RuntimeError("_apply_finish blew up")
        return finish(self, *args, **kwargs)

    messages = [
        init(),
        assistant(
            ToolUseBlock(id="agent", name="Agent", input={}), message_id="m1"
        ),
        task_started("agent"),
        assistant(
            ToolUseBlock(id="read", name="Read", input={}),
            parent="agent",
            message_id="s1",
        ),
        result(),
    ]
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(ToolInvocation, "_apply_finish", explode_for_read)
        await drive(handler, messages)
    assert Spans(span_exporter).one("execute_tool Agent")


@pytest.mark.asyncio
async def test_a_foreground_delegation_without_its_tool_result_does_not_hold_the_turn(
    handler, span_exporter
):
    """Its notification says it finished; the turn can end at its result."""
    messages = [
        init(),
        assistant(
            ToolUseBlock(id="agent", name="Agent", input={}), message_id="m1"
        ),
        task_started("agent"),
        assistant(TextBlock("sub"), parent="agent", message_id="s1"),
        task_notification("agent", status="stopped"),
        result(),
    ]

    ended_at_result = await roots_ended_at_each_result(
        handler, messages, span_exporter
    )
    assert ended_at_result == [1]


@pytest.mark.asyncio
async def test_a_stopped_foreground_delegation_reports_that_it_was_stopped(
    capturing_handler, span_exporter
):
    """Its tool_result is flagged as an error too, but the notification says
    why."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="agent", name="Agent", input={}),
                message_id="m1",
            ),
            task_started("agent"),
            task_notification("agent", status="stopped"),
            user(
                ToolResultBlock(
                    tool_use_id="agent", content="stopped", is_error=True
                )
            ),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Agent")
    assert tool.status.description == "task stopped"


@pytest.mark.asyncio
async def test_a_foreground_delegation_inside_a_killed_subagent_never_returned(
    capturing_handler, span_exporter
):
    """Only background work outlives the subagent that started it. A
    foreground delegation still waiting on its own subagent never returned."""
    await drive(
        capturing_handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="outer", name="Agent", input={}),
                message_id="m1",
            ),
            task_started("outer", task_id="t-outer"),
            assistant(
                ToolUseBlock(id="inner", name="Task", input={}),
                parent="outer",
                message_id="s1",
            ),
            task_started("inner", task_id="t-inner"),
            task_notification("outer", task_id="t-outer", status="killed"),
            user(ToolResultBlock(tool_use_id="outer", content="killed")),
            result(),
        ],
    )
    inner = Spans(span_exporter).one("execute_tool Task")
    assert inner.status.description == "tool never returned"


@pytest.mark.asyncio
async def test_cancelling_the_reader_of_a_finished_turn_is_not_a_failure(
    handler, span_exporter
):
    """The turn has its result and is only waiting on background work."""
    messages = turn_waiting_on_a_background_command()
    drained = asyncio.Event()

    async def stream():
        for message in messages:
            yield message
        drained.set()
        await asyncio.Event().wait()

    run = AgentRun(handler, multi_turn=True)
    run.submit("hi")

    async def read():
        async for _ in run.iterate(stream):
            pass

    reader = asyncio.create_task(read())
    await drained.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    root = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert root.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_close_leaves_the_callers_span_current(
    handler, span_exporter, tracer
):
    messages = turn_waiting_on_a_background_command()

    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler, multi_turn=True)
    with tracer.start_as_current_span("turn"):
        run.submit("hi")
        # Stops at the result, as receive_response() does, leaving the turn
        # open on its background command.
        reader = run.iterate(stream)
        async for message in reader:
            if isinstance(message, ResultMessage):
                break
        await reader.aclose()
    assert not Spans(span_exporter).named("invoke_agent claude-agent-sdk")
    with tracer.start_as_current_span("cleanup") as cleanup:
        run.close()
        assert trace.get_current_span() is cleanup
    assert Spans(span_exporter).one("invoke_agent claude-agent-sdk")


@pytest.mark.asyncio
async def test_a_foreground_task_stopped_after_its_turns_result_ends_the_turn(
    capturing_handler, span_exporter
):
    """Only the notification arrives, with no task_updated before it."""
    messages = [
        init(),
        assistant(
            ToolUseBlock(id="agent", name="Agent", input={}), message_id="m1"
        ),
        task_started("agent"),
        assistant(TextBlock("sub"), parent="agent", message_id="s1"),
        result(),
        task_notification("agent", status="stopped"),
    ]
    await drive(
        capturing_handler, messages, multi_turn=True, stop_after=len(messages)
    )
    assert Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    delegation = Spans(span_exporter).one("execute_tool Agent")
    assert delegation.status.description == "task stopped"


@pytest.mark.asyncio
async def test_a_subagent_its_turn_closed_stays_closed_in_the_next_turn(
    handler, span_exporter
):
    """The turn closes a stopped foreground subagent itself, since no
    tool_result came. A late message from it must not reopen it."""
    messages = [
        init(),
        assistant(
            ToolUseBlock(
                id="agent", name="Agent", input={"subagent_type": "researcher"}
            ),
            message_id="m1",
        ),
        task_started("agent"),
        assistant(TextBlock("sub"), parent="agent", message_id="s1"),
        result(),
        task_notification("agent", status="stopped"),
        assistant(TextBlock("late"), parent="agent", message_id="s2"),
        init(),
        result(),
    ]
    assert await roots_ended_at_each_result(
        handler, messages, span_exporter
    ) == [0, 2]
    spans = Spans(span_exporter)
    assert len(spans.named("invoke_agent researcher")) == 1
    assert not spans.named("invoke_agent subagent")


@pytest.mark.asyncio
async def test_a_stream_error_while_a_turn_waits_on_background_work_is_a_failure(
    handler, span_exporter
):
    messages = turn_waiting_on_a_background_command()

    async def stream():
        for message in messages:
            yield message
        raise RuntimeError("CLI exited")

    run = AgentRun(handler, multi_turn=True)
    run.submit("hi")
    with pytest.raises(RuntimeError):
        async for _ in run.iterate(stream):
            pass
    root = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert root.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_a_later_read_that_ends_the_turn_leaves_its_span_current(
    handler, span_exporter, tracer
):
    messages = turn_waiting_on_a_background_command()

    async def first_turn():
        for message in messages:
            yield message

    async def nothing_more():
        return
        yield

    run = AgentRun(handler, multi_turn=True)
    with tracer.start_as_current_span("turn"):
        run.submit("hi")
        reader = run.iterate(first_turn)
        async for message in reader:
            if isinstance(message, ResultMessage):
                break
        await reader.aclose()
    with tracer.start_as_current_span("later") as later:
        async for _ in run.iterate(nothing_more):
            pass
        assert trace.get_current_span() is later
    assert Spans(span_exporter).one("invoke_agent claude-agent-sdk")


@pytest.mark.asyncio
async def test_a_message_after_the_close_its_turn_waited_on_opens_nothing_in_it(
    handler, span_exporter
):
    """task_updated with no notification after it: the next message closes the
    delegation, ends the turn, and is itself dropped as late."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    messages = [
        init(),
        assistant(task, message_id="m1"),
        task_started("call-task"),
        user(ToolResultBlock(tool_use_id="call-task", content="launched")),
        result(),
        task_updated(status="killed"),
        assistant(TextBlock("late"), parent="call-task", message_id="m_late"),
        init(),
        assistant(TextBlock("two"), message_id="m2"),
        result(),
    ]
    assert await roots_ended_at_each_result(
        handler, messages, span_exporter
    ) == [0, 2]
    spans = Spans(span_exporter)
    assert not spans.named("invoke_agent researcher")
    assert {
        s.attributes["gen_ai.response.id"]
        for s in spans.all
        if s.name.startswith("chat ")
    } == {"m1", "m2"}


@pytest.mark.asyncio
async def test_a_delegation_killed_before_its_subagent_spoke_stays_closed(
    handler, span_exporter
):
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "researcher"}
    )
    messages = [
        init(),
        assistant(task, message_id="m1"),
        task_started("call-task"),
        user(ToolResultBlock(tool_use_id="call-task", content="launched")),
        result(),
        task_notification("call-task", status="killed"),
        init(),
        assistant(TextBlock("late"), parent="call-task", message_id="m_late"),
        result(),
    ]
    assert await roots_ended_at_each_result(
        handler, messages, span_exporter
    ) == [0, 2]
    spans = Spans(span_exporter)
    assert not spans.named("invoke_agent researcher")
    assert not spans.named("invoke_agent subagent")


@pytest.mark.asyncio
async def test_a_foreground_task_inside_a_killed_subagent_does_not_hold_the_turn(
    handler, span_exporter
):
    messages = [
        init(),
        assistant(
            ToolUseBlock(id="outer", name="Agent", input={}), message_id="m1"
        ),
        task_started("outer", task_id="t-outer"),
        user(ToolResultBlock(tool_use_id="outer", content="launched")),
        result(),
        assistant(
            ToolUseBlock(id="inner", name="Task", input={}),
            parent="outer",
            message_id="s1",
        ),
        task_started("inner", task_id="t-inner"),
        task_notification("outer", task_id="t-outer", status="killed"),
    ]
    await drive(handler, messages, multi_turn=True, stop_after=len(messages))
    assert Spans(span_exporter).one("invoke_agent claude-agent-sdk")


@pytest.mark.asyncio
async def test_a_stream_failing_to_start_leaves_the_callers_span_current(
    handler, span_exporter, tracer
):
    def start():
        raise RuntimeError("no CLI")

    run = AgentRun(handler)
    with tracer.start_as_current_span("caller") as caller:
        with pytest.raises(RuntimeError):
            async for _ in run.iterate(start):
                pass
        assert trace.get_current_span() is caller
    root = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert root.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("multi_turn", [False, True])
async def test_a_subagent_failing_to_close_does_not_strand_its_delegation(
    handler, span_exporter, monkeypatch, multi_turn
):
    def explode(self: LocalAgentInvocation, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("_apply_finish blew up")

    monkeypatch.setattr(LocalAgentInvocation, "_apply_finish", explode)
    await drive(
        handler,
        [
            init(),
            assistant(
                ToolUseBlock(id="agent", name="Agent", input={}),
                message_id="m1",
            ),
            task_started("agent"),
            assistant(TextBlock("sub"), parent="agent", message_id="s1"),
            result(),
        ],
        multi_turn=multi_turn,
    )
    assert Spans(span_exporter).one("execute_tool Agent")


@pytest.mark.asyncio
async def test_a_subagent_that_owns_its_own_launcher_still_closes(
    handler, span_exporter
):
    """Malformed: the delegation's tool_use arrives inside the subagent it
    launches."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("x"), parent="A", message_id="s1"),
            assistant(
                ToolUseBlock(id="A", name="Agent", input={}),
                parent="A",
                message_id="s2",
            ),
            user(ToolResultBlock(tool_use_id="A", content="report")),
            result(),
        ],
    )
    assert Spans(span_exporter).one("execute_tool Agent")


# ─── resumed sessions ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        ClaudeAgentOptions(resume="earlier-session"),
        ClaudeAgentOptions(continue_conversation=True),
    ],
    ids=["resume", "continue"],
)
async def test_a_resumed_session_reports_no_run_totals(
    handler, span_exporter, options
):
    """The first result of a resumed session already carries the earlier
    session's usage, by an amount the stream never states."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("hi"), usage={"input_tokens": 12}),
            result(model_usage=MODEL_USAGE),
        ],
        sdk_options=options,
    )
    spans = Spans(span_exporter)
    agent = spans.one("invoke_agent claude-agent-sdk")
    for key in (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_write.input_tokens",
        "claude_agent_sdk.usage.cost_usd",
    ):
        assert key not in agent.attributes
    # Per-call counts are not cumulative, so they stay.
    chat = spans.one("chat claude-sonnet-4-5")
    assert chat.attributes["gen_ai.usage.input_tokens"] == 12


@pytest.mark.asyncio
async def test_a_resumed_client_measures_later_turns_from_the_first(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("one"), message_id="m1"),
            result(model_usage=cumulative(5000, 400, 0.5)),
            init(),
            assistant(TextBlock("two"), message_id="m2"),
            result(model_usage=cumulative(5300, 425, 0.53)),
        ],
        multi_turn=True,
        sdk_options=ClaudeAgentOptions(resume="earlier-session"),
    )
    agents = sorted(
        Spans(span_exporter).named("invoke_agent claude-agent-sdk"),
        key=lambda s: s.end_time,
    )
    assert "gen_ai.usage.input_tokens" not in agents[0].attributes
    assert agents[1].attributes["gen_ai.usage.input_tokens"] == 300
    assert agents[1].attributes["gen_ai.usage.output_tokens"] == 25
    assert agents[1].attributes[
        "claude_agent_sdk.usage.cost_usd"
    ] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_a_resumed_turn_that_ends_without_usage_sets_no_baseline(
    handler, span_exporter
):
    """A turn that never saw the session's totals leaves the baseline at zero,
    so the next turn's totals would still include the earlier session's."""
    run = AgentRun(
        handler,
        sdk_options=ClaudeAgentOptions(resume="earlier-session"),
        multi_turn=True,
    )

    async def failing_read():
        yield init()
        yield assistant(TextBlock("one"), message_id="m1")
        raise ConnectionError("dropped")

    async def next_read():
        yield init()
        yield assistant(TextBlock("two"), message_id="m2")
        yield result(model_usage=cumulative(5300, 425, 0.53))

    with pytest.raises(ConnectionError):
        async for _ in run.iterate(failing_read):
            pass
    async for _ in run.iterate(next_read):
        pass

    agents = Spans(span_exporter).named("invoke_agent claude-agent-sdk")
    assert len(agents) == 2
    for agent in agents:
        assert "gen_ai.usage.input_tokens" not in agent.attributes


@pytest.mark.asyncio
async def test_a_reset_after_resuming_starts_the_totals_fresh(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(),
            conversation_reset(),
            assistant(TextBlock("hi")),
            result(model_usage=cumulative(120, 8, 0.01)),
        ],
        sdk_options=ClaudeAgentOptions(resume="earlier-session"),
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.attributes["gen_ai.usage.input_tokens"] == 120


# ─── interrupts ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["aborted_streaming", "aborted_tools"])
async def test_an_interrupted_turn_is_not_a_failure(
    handler, span_exporter, reason
):
    """The result CLI 2.1.229 and 2.1.277 send after ``interrupt()``."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("1\n2\n3")),
            result(
                subtype="error_during_execution",
                is_error=True,
                text=None,
                terminal_reason=reason,
            ),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    for span in spans.all:
        assert span.status.status_code is trace.StatusCode.UNSET
        assert "error.type" not in span.attributes


@pytest.mark.asyncio
async def test_an_error_result_that_was_not_interrupted_still_fails(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("...")),
            result(
                subtype="error_max_turns",
                is_error=True,
                text=None,
                terminal_reason="max_turns",
            ),
        ],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert agent.attributes["error.type"] == "error_max_turns"


# ─── thinking tokens ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_totals_sum_thinking_tokens_across_models(
    handler, span_exporter
):
    """``thinkingTokens`` is a share of ``outputTokens``, not added to it."""
    model_usage = {
        "claude-sonnet-4-5": {
            "inputTokens": 100,
            "outputTokens": 80,
            "thinkingTokens": 30,
            "costUSD": 0.01,
        },
        "claude-haiku-4-5": {
            "inputTokens": 50,
            "outputTokens": 20,
            "thinkingTokens": 5,
            "costUSD": 0.001,
        },
    }
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result(model_usage=model_usage)],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.attributes["gen_ai.usage.reasoning.output_tokens"] == 35
    assert agent.attributes["gen_ai.usage.output_tokens"] == 100


@pytest.mark.asyncio
async def test_no_thinking_tokens_reported_when_the_cli_omits_them(
    handler, span_exporter
):
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result(model_usage=MODEL_USAGE)],
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert "gen_ai.usage.reasoning.output_tokens" not in agent.attributes


# ─── tasks known from the call ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_delegation_receipt_ahead_of_task_started_is_still_a_receipt(
    capturing_handler, span_exporter
):
    """Closing the delegation on its receipt would drop every later message
    from its subagent."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "helper"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task, message_id="m1"),
            user(
                ToolResultBlock(
                    tool_use_id="call-task", content="Async agent launched"
                )
            ),
            task_started("call-task"),
            assistant(
                TextBlock("working"), parent="call-task", message_id="s1"
            ),
            task_notification("call-task", summary="the subagent report"),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    tool = spans.one("execute_tool Agent")
    subagent = spans.one("invoke_agent helper")
    assert spans.parent_name(subagent) == tool.name
    assert tool.attributes["gen_ai.tool.call.result"] == "the subagent report"
    assert tool.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_a_background_command_receipt_ahead_of_task_started_is_held(
    capturing_handler, span_exporter
):
    call = ToolUseBlock(
        id="b1",
        name="Bash",
        input={"command": "sleep 3", "run_in_background": True},
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(call),
            user(ToolResultBlock(tool_use_id="b1", content="running")),
            task_started("b1", task_type="local_bash"),
            task_notification("b1", summary="real command output"),
            result(),
        ],
    )
    tool = Spans(span_exporter).one("execute_tool Bash")
    assert tool.attributes["gen_ai.tool.call.result"] == "real command output"


@pytest.mark.asyncio
async def test_a_delegation_that_fails_to_launch_closes_on_its_result(
    handler, span_exporter
):
    """An error result launched nothing, so no notification will follow and
    the turn must not wait for one."""
    task = ToolUseBlock(
        id="call-task", name="Agent", input={"subagent_type": "missing"}
    )
    ended = await roots_ended_at_each_result(
        handler,
        [
            init(),
            assistant(task, message_id="m1"),
            user(
                ToolResultBlock(
                    tool_use_id="call-task",
                    content="Agent type 'missing' not found",
                    is_error=True,
                )
            ),
            assistant(TextBlock("sorry"), message_id="m2"),
            result(),
        ],
        span_exporter,
    )
    assert ended == [1]
    tool = Spans(span_exporter).one("execute_tool Agent")
    assert tool.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_other_tools_passing_run_in_background_close_on_their_result(
    handler, span_exporter
):
    """Only Bash backgrounds on that argument; any other tool that takes one
    returns its result directly, with no task behind it."""
    call = ToolUseBlock(
        id="c1", name="mcp__jobs__start", input={"run_in_background": True}
    )
    ended = await roots_ended_at_each_result(
        handler,
        [
            init(),
            assistant(call),
            user(ToolResultBlock(tool_use_id="c1", content="job 7 queued")),
            result(),
        ],
        span_exporter,
    )
    assert ended == [1]


# ─── the caller stopping ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disconnecting_mid_turn_is_not_a_failure(handler, span_exporter):
    """After an interrupt the caller can disconnect before the turn's result
    arrives, or disconnect mid-turn outright. Either way they chose to stop."""
    run = AgentRun(handler, multi_turn=True)
    messages = [
        init(),
        assistant(ToolUseBlock(id="c1", name="Bash", input={})),
    ]

    async def stream():
        for message in messages:
            yield message

    reader = run.iterate(stream)
    async for message in reader:
        if message is messages[-1]:
            break
    await reader.aclose()
    run.close()

    spans = Spans(span_exporter)
    assert spans.one("execute_tool Bash")
    for span in spans.all:
        assert span.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_stopping_after_an_error_result_still_fails_the_run(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("..."), message_id="m1"),
            result(subtype="error_max_turns", is_error=True, text=None),
            assistant(TextBlock("more"), message_id="m2"),
        ],
        stop_after=3,
    )
    agent = Spans(span_exporter).one("invoke_agent claude-agent-sdk")
    assert agent.status.status_code is trace.StatusCode.ERROR
    assert agent.attributes["error.type"] == "error_max_turns"


@pytest.mark.asyncio
async def test_stopping_does_not_hide_a_task_that_already_failed(
    capturing_handler, span_exporter
):
    task = ToolUseBlock(id="t1", name="Agent", input={"subagent_type": "x"})
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_updated(status="failed"),
            assistant(TextBlock("next"), message_id="m2"),
        ],
        stop_after=4,
    )
    tool = Spans(span_exporter).one("execute_tool Agent")
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert tool.status.description == "task failed"
