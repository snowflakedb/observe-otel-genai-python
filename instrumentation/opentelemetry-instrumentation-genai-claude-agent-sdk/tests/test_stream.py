# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Drives the stream state machine with synthetic message sequences.

Each test builds the exact message ordering the CLI produces for one scenario,
so the rules that depend on ordering — a task's receipt versus its real
completion, a call's final usage versus its mid-stream snapshot — are exercised
directly rather than inferred from a live run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.genai.claude_agent_sdk._run import AgentRun
from opentelemetry.util.genai.handler import TelemetryHandler

SESSION = "session-abc"


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


def user(*blocks: Any, parent: str | None = None) -> UserMessage:
    return UserMessage(content=list(blocks), parent_tool_use_id=parent)


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


def result(
    *,
    subtype: str = "success",
    is_error: bool = False,
    text: str | None = "done",
    model_usage: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    origin: dict[str, Any] | None = None,
    api_error_status: int | None = None,
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
    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler, sdk_options=sdk_options, multi_turn=multi_turn)
    messages_out = run.iterate(stream, prompt)
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
    assert agent.attributes["gen_ai.usage.cost_usd"] == pytest.approx(0.012)


@pytest.mark.asyncio
async def test_all_messages_pass_through_unchanged(handler, span_exporter):
    messages = [init(), assistant(TextBlock("hello")), result()]
    seen = await drive(handler, messages)
    assert seen == messages


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
async def test_inline_task_notification_first_then_receipt_is_noop(
    capturing_handler, span_exporter
):
    """Run inline the notification simply arrives first."""
    task = ToolUseBlock(
        id="t1", name="Task", input={"subagent_type": "helper"}
    )
    await drive(
        capturing_handler,
        [
            init(),
            assistant(task),
            task_started("t1"),
            task_notification("t1", summary="inline report"),
            user(ToolResultBlock(tool_use_id="t1", content="inline report")),
            result(),
        ],
    )
    spans = Spans(span_exporter)
    tool = spans.one("execute_tool Task")
    assert tool.attributes["gen_ai.tool.call.result"] == "inline report"
    assert tool.status.status_code is trace.StatusCode.UNSET


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
async def test_killed_task_closes_without_a_notification(
    handler, span_exporter
):
    """A task stopped via TaskStop reports "killed" on task_updated and its
    notification is sometimes suppressed entirely.

    The terminal update stops the tool_result being treated as a receipt, so the
    result closes the span instead of it leaking to teardown.
    """
    task = ToolUseBlock(id="t1", name="Task", input={})
    await drive(
        handler,
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
    handler, span_exporter
):
    task = ToolUseBlock(id="t1", name="Task", input={})
    await drive(
        handler,
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
    handler, span_exporter
):
    call = ToolUseBlock(id="c1", name="Read", input={})
    await drive(handler, [init(), assistant(call), result()])
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
    assert "genai.usage.is_partial" not in chat.attributes
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
    assert chat.attributes["genai.usage.is_partial"] is True
    assert chat.attributes["gen_ai.usage.output_tokens"] == 3


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
    handler, span_exporter
):
    """An API failure keeps subtype "success" and puts the text in result; the
    CLI relays it as a synthetic assistant message, so that call's span is open.
    """
    await drive(
        handler,
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
async def test_stream_raising_fails_the_run(handler, span_exporter):
    async def stream():
        yield init()
        yield assistant(TextBlock("hi"))
        raise RuntimeError("transport died")

    run = AgentRun(handler)
    with pytest.raises(RuntimeError):
        async for _ in run.iterate(stream, "hi"):
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
    assert spans.named("invoke_agent claude-agent-sdk")
    assert spans.named("execute_tool Bash")
    assert all(s.end_time is not None for s in spans.all)


@pytest.mark.asyncio
async def test_instrumentation_failure_does_not_break_the_stream(
    handler, span_exporter
):
    """A malformed message must not interrupt the caller's iteration."""

    class Broken(SystemMessage):
        pass

    messages = [init(), Broken(subtype="init", data=None), result()]
    seen = await drive(handler, messages)
    assert seen == messages


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
    assert len(agents) == 2
    assert len(spans.named("chat claude-sonnet-4-5")) == 2
    # Each turn is its own trace, threaded by the conversation id.
    for span in spans.all:
        assert span.attributes["gen_ai.conversation.id"] == SESSION


@pytest.mark.asyncio
async def test_injected_turn_result_is_only_a_checkpoint(
    handler, span_exporter
):
    """A background-task notification produces a result of its own. It does not
    answer the caller's prompt, so it must not end the invocation."""
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock("working"), message_id="m1"),
            result(origin={"kind": "task-notification"}),
            assistant(TextBlock("real answer"), message_id="m2"),
            result(origin={"kind": "human"}),
        ],
        multi_turn=True,
    )
    spans = Spans(span_exporter)
    assert len(spans.named("invoke_agent claude-agent-sdk")) == 1
    assert len(spans.named("chat claude-sonnet-4-5")) == 2


@pytest.mark.asyncio
async def test_parallel_tools_closing_out_of_order(
    handler, span_exporter, tracer
):
    """Two tools open, and the second one's result arrives first.

    Closing out of order must still leave both parented to the agent and the
    caller's active context untouched.
    """
    with tracer.start_as_current_span("caller") as caller:
        await drive(
            handler,
            [
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
            ],
        )
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
async def test_options_env_wins_over_our_own(handler, span_exporter):
    class Options:
        env = {"CLAUDE_CODE_USE_VERTEX": "true"}
        model = "claude-sonnet-4-5"

    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=Options(),
    )
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "gcp.vertex_ai"
    )


@pytest.mark.asyncio
async def test_disabled_routing_flag_is_not_a_platform(
    handler, span_exporter, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "0")
    await drive(handler, [init(), assistant(TextBlock("hi")), result()])
    assert (
        Spans(span_exporter)
        .one("chat claude-sonnet-4-5")
        .attributes["gen_ai.provider.name"]
        == "anthropic"
    )
