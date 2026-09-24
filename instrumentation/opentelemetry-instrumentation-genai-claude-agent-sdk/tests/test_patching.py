# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Drives the real SDK entry points through the installed wrappers.

test_stream.py builds an ``AgentRun`` by hand, so nothing there touches the
patching layer: which symbols are replaced, that they come back off again, that
a prompt handed to one half of ``ClaudeSDKClient`` reaches the other, and that
``receive_response()`` produces one invocation rather than two.

The seam is the subprocess boundary, the deepest one that needs no CLI: the
client is given a fake transport and message source, and ``query()`` a fake
``InternalClient``. Everything above that — the real functions, the real message
parser, the real SDK message types — runs.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
from collections import Counter
from typing import Any

import claude_agent_sdk
import pytest
import wrapt
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from claude_agent_sdk._internal.message_parser import parse_message

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

SESSION = "session-abc"
MODEL = "claude-sonnet-4-5"
ROOT = "invoke_agent claude-agent-sdk"
CHAT = f"chat {MODEL}"


# ─── raw CLI messages ───────────────────────────────────────────────────────
# Written as the dicts the CLI puts on stdout rather than as SDK dataclasses, so
# the real parser decides what the stream reader sees.


def raw_init() -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "session_id": SESSION,
        "model": MODEL,
    }


def raw_assistant(text: str = "4") -> dict[str, Any]:
    return {
        "type": "assistant",
        "session_id": SESSION,
        "message": {
            "id": "msg_1",
            "model": MODEL,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "stop_reason": "end_turn",
        },
    }


def raw_result(
    text: str = "4", output_tokens_so_far: int | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 5,
        "is_error": False,
        "num_turns": 1,
        "session_id": SESSION,
        "result": text,
        "stop_reason": "end_turn",
    }
    if output_tokens_so_far is not None:
        # The CLI reports usage for the whole session so far, not the turn.
        message["modelUsage"] = {
            MODEL: {
                "inputTokens": 0,
                "outputTokens": output_tokens_so_far,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "costUSD": 0.0,
            }
        }
    return message


ONE_TURN = [raw_init(), raw_assistant(), raw_result()]


def raw_call(
    call_id: str,
    name: str = "Bash",
    *,
    message_id: str = "msg_call",
    parent: str | None = None,
) -> dict[str, Any]:
    message = raw_assistant()
    message["message"]["id"] = message_id
    message["message"]["content"] = [
        {"type": "tool_use", "id": call_id, "name": name, "input": {}}
    ]
    message["parent_tool_use_id"] = parent
    return message


def raw_text(
    text: str, *, message_id: str, parent: str | None = None
) -> dict[str, Any]:
    message = raw_assistant(text)
    message["message"]["id"] = message_id
    message["parent_tool_use_id"] = parent
    return message


def raw_tool_result(
    call_id: str, content: str = "ok", *, parent: str | None = None
) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                }
            ],
        },
        "parent_tool_use_id": parent,
    }


def raw_task(subtype: str, call_id: str) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": subtype,
        "task_id": f"task-{call_id}",
        "tool_use_id": call_id,
        "description": "d",
        "status": "completed",
        "output_file": "",
        "summary": "done",
        "uuid": "u",
        "session_id": SESSION,
    }


# ─── fakes ──────────────────────────────────────────────────────────────────


class FakeTransport:
    """Records what the SDK writes to the CLI's stdin."""

    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []

    async def write(self, data: str) -> None:
        self.written.extend(json.loads(line) for line in data.splitlines())


class FakeQuery:
    """Stands in for the client's internal reader, yielding raw CLI messages.

    One queue for the whole session, like the real reader, so each receive
    picks up where the last one stopped.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = iter(messages)

    async def receive_messages(self):
        for message in self._messages:
            yield message

    async def close(self) -> None:
        pass

    def close_receive_stream(self) -> None:
        pass


def first_input_text(span: Any) -> str:
    """The text of the first input message on a span."""
    parsed = json.loads(span.attributes["gen_ai.input.messages"])
    return parsed[0]["parts"][0]["content"]


def connected_client(
    messages: list[dict[str, Any]] | None = None,
    options: ClaudeAgentOptions | None = None,
) -> ClaudeSDKClient:
    """A client that behaves as connected without spawning the CLI."""
    client = ClaudeSDKClient(options=options)
    client._query = FakeQuery(messages if messages is not None else ONE_TURN)  # pylint: disable=protected-access
    client._transport = FakeTransport()  # pylint: disable=protected-access
    return client


class FakeInternalClient:
    """Stands in for the ``InternalClient`` that ``query()`` builds per call."""

    calls: list[dict[str, Any]] = []

    async def process_query(
        self, *, prompt: Any, options: Any, transport: Any
    ):
        FakeInternalClient.calls.append(
            {"prompt": prompt, "options": options, "transport": transport}
        )
        for message in ONE_TURN:
            yield parse_message(message)


@pytest.fixture
def fake_internal_client(monkeypatch: pytest.MonkeyPatch):
    """Cut ``query()`` off above the subprocess, leaving its own body running."""
    FakeInternalClient.calls = []
    # By module object, not by dotted string: ``claude_agent_sdk.query`` as an
    # attribute is the re-exported function, not the submodule.
    monkeypatch.setattr(
        sys.modules["claude_agent_sdk.query"],
        "InternalClient",
        FakeInternalClient,
    )
    return FakeInternalClient


# ─── span helpers ───────────────────────────────────────────────────────────


def span_names(span_exporter: Any) -> Counter[str]:
    return Counter(span.name for span in span_exporter.get_finished_spans())


def one_span(span_exporter: Any, name: str) -> Any:
    found = [s for s in span_exporter.get_finished_spans() if s.name == name]
    assert len(found) == 1, f"{name}: expected 1, got {len(found)}"
    return found[0]


def message_text(span: Any, attribute: str) -> str:
    return json.dumps(json.loads(span.attributes[attribute]))


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def instrumented(tracer_provider, logger_provider, meter_provider):
    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrumented_with_content(
    tracer_provider, logger_provider, meter_provider
):
    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


# ─── which symbols are patched ──────────────────────────────────────────────


def patched_symbols() -> dict[str, Any]:
    """Every binding the instrumentor replaces, keyed by how to say where.

    Class members are read out of ``__dict__``: attribute access on the class
    runs the wrapper's own descriptor and hands back a ``BoundFunctionWrapper``,
    which hides whether the class attribute itself was replaced. Absent
    attributes are skipped so a test that removes one can still compare.
    """
    symbols = {
        "claude_agent_sdk.query (package re-export)": claude_agent_sdk.query,
        "claude_agent_sdk.query.query (submodule)": sys.modules[
            "claude_agent_sdk.query"
        ].query,
    }
    for attribute in ("query", "receive_messages", "disconnect"):
        if attribute in vars(ClaudeSDKClient):
            symbols[f"ClaudeSDKClient.{attribute}"] = vars(ClaudeSDKClient)[
                attribute
            ]
    return symbols


def test_every_entry_point_is_wrapped_and_handed_back(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    before = patched_symbols()
    assert len(before) == 5

    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        during = patched_symbols()
        assert sorted(during) == sorted(before)
        for name, symbol in during.items():
            assert isinstance(symbol, wrapt.FunctionWrapper), name
            assert symbol.__wrapped__ is before[name], name

    assert patched_symbols() == before


def test_a_failure_part_way_through_instrument_unwinds(
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
    uninstrument_claude_agent_sdk,
):
    """A future SDK that relocates receive_messages must not strand wrappers.

    instrument() marks the instrumentor as instrumented only once _instrument
    returns, so a wrapper left installed here could never be removed —
    uninstrument() declines as already uninstrumented — while a retry is let
    through and wraps the wrapper, which double-counts every call.
    """
    before = patched_symbols()
    monkeypatch.delattr(ClaudeSDKClient, "receive_messages")
    instrumentor = ClaudeAgentSDKInstrumentor()

    # wrapt raises its own PathResolutionError here, not AttributeError.
    with pytest.raises(Exception):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )

    assert not instrumentor.is_instrumented_by_opentelemetry
    assert instrumentor._handler is None  # pylint: disable=protected-access
    remaining = patched_symbols()
    assert "ClaudeSDKClient.receive_messages" not in remaining
    for name, symbol in remaining.items():
        assert symbol is before[name], name


# ─── query() ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry_point",
    ["package", "submodule"],
)
async def test_query_builds_one_trace_through_either_binding(
    instrumented, fake_internal_client, span_exporter, entry_point
):
    """The package re-export and the submodule attribute are separate bindings
    of the same function, and a call goes through exactly one of them."""
    query = (
        claude_agent_sdk.query
        if entry_point == "package"
        else sys.modules["claude_agent_sdk.query"].query
    )

    seen = [message async for message in query(prompt="what is 2 + 2?")]

    assert len(seen) == 3
    assert span_names(span_exporter) == Counter({ROOT: 1, CHAT: 1})
    assert fake_internal_client.calls[0]["prompt"] == "what is 2 + 2?"


@pytest.mark.asyncio
async def test_query_prompt_reaches_the_spans(
    instrumented_with_content, fake_internal_client, span_exporter
):
    async for _ in claude_agent_sdk.query(prompt="what is 2 + 2?"):
        pass

    for name in (ROOT, CHAT):
        assert "what is 2 + 2?" in message_text(
            one_span(span_exporter, name), "gen_ai.input.messages"
        )


def test_a_caller_argument_error_still_raises_at_the_call_site(
    instrumented, fake_internal_client, span_exporter
):
    """The wrapper defers the call into the stream, so without binding the
    arguments up front a misspelled keyword would surface from the first
    ``__anext__`` — after a root span had opened and been failed, reporting a
    pure caller bug as an errored agent invocation."""
    with pytest.raises(TypeError):
        claude_agent_sdk.query(prmopt="what is 2 + 2?")

    assert not span_exporter.get_finished_spans()
    assert not fake_internal_client.calls


# ─── ClaudeSDKClient ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_receive_response_is_one_invocation(instrumented, span_exporter):
    """receive_response() iterates receive_messages(), so wrapping both would
    rebuild the same tree twice — two agent spans and two chat spans, in two
    traces."""
    client = connected_client()

    seen = [message async for message in client.receive_response()]

    assert len(seen) == 3
    assert span_names(span_exporter) == Counter({ROOT: 1, CHAT: 1})
    trace_ids = {
        span.context.trace_id for span in span_exporter.get_finished_spans()
    }
    assert len(trace_ids) == 1


@pytest.mark.asyncio
async def test_client_query_prompt_reaches_the_spans(
    instrumented_with_content, span_exporter
):
    """The client splits sending from receiving, so the prompt is only in hand
    in query(); and in client mode the CLI emits no UserMessage to recover it
    from. Without the handoff both spans lose gen_ai.input.messages."""
    client = connected_client()

    await client.query("what is 2 + 2?", session_id="s-42")
    async for _ in client.receive_response():
        pass

    for name in (ROOT, CHAT):
        span = one_span(span_exporter, name)
        assert "what is 2 + 2?" in message_text(span, "gen_ai.input.messages")
        # The caller's own session id names the conversation; the CLI's init
        # session id is the fallback.
        assert span.attributes["gen_ai.conversation.id"] == "s-42"


@pytest.mark.asyncio
async def test_default_session_id_leaves_the_conversation_to_the_cli(
    instrumented, span_exporter
):
    """query()'s own default is the literal "default", which is a placeholder
    rather than a session id — passing it through would collapse every session
    in the Conversations view into one."""
    client = connected_client()

    await client.query("what is 2 + 2?")
    async for _ in client.receive_response():
        pass

    assert (
        one_span(span_exporter, ROOT).attributes["gen_ai.conversation.id"]
        == SESSION
    )


@pytest.mark.asyncio
async def test_a_streaming_prompt_is_not_consumed(
    instrumented_with_content, span_exporter
):
    """In streaming-input mode the prompt is an async iterable that the SDK has
    to read, so the wrapper may only hold onto it."""
    turns = [
        {"type": "user", "message": {"role": "user", "content": "first"}},
        {"type": "user", "message": {"role": "user", "content": "second"}},
    ]

    async def prompts():
        for turn in turns:
            yield turn

    client = connected_client()
    await client.query(prompts())

    assert client._transport.written == [  # pylint: disable=protected-access
        dict(turn, session_id="default") for turn in turns
    ]


@pytest.mark.asyncio
async def test_a_receive_without_a_new_query_replays_no_prompt(
    instrumented_with_content, span_exporter
):
    """A prompt is taken by the turn it opens, not read: a second turn that
    submitted nothing must not be reported as having asked the first turn's
    question."""
    client = connected_client(ONE_TURN + ONE_TURN)

    await client.query("what is 2 + 2?")
    async for _ in client.receive_response():
        pass
    async for _ in client.receive_response():
        pass

    roots = [s for s in span_exporter.get_finished_spans() if s.name == ROOT]
    assert len(roots) == 2
    assert "what is 2 + 2?" in message_text(roots[0], "gen_ai.input.messages")
    assert "gen_ai.input.messages" not in roots[1].attributes


@pytest.mark.asyncio
async def test_one_long_lived_receive_reports_each_turns_own_prompt(
    instrumented_with_content, span_exporter
):
    """receive_messages() is the continuous multi-turn API: one iterator spans
    every turn, and the caller sends each turn's prompt while it is open."""
    client = connected_client(
        [raw_init(), raw_assistant("4"), raw_result("4")]
        + [raw_assistant("6"), raw_result("6")]
    )
    # Opened before the first query, as a reader task would be.
    messages = client.receive_messages()

    async def read_turn() -> None:
        while not isinstance(await messages.__anext__(), ResultMessage):
            pass

    await client.query("what is 2 + 2?")
    await read_turn()
    await client.query("what is 3 + 3?")
    await read_turn()
    await messages.aclose()

    roots = [s for s in span_exporter.get_finished_spans() if s.name == ROOT]
    chats = [s for s in span_exporter.get_finished_spans() if s.name == CHAT]
    assert [first_input_text(r) for r in roots] == [
        "what is 2 + 2?",
        "what is 3 + 3?",
    ]
    assert [first_input_text(c) for c in chats] == [
        "what is 2 + 2?",
        "what is 3 + 3?",
    ]


@pytest.mark.asyncio
async def test_each_receive_response_reports_only_its_own_turns_usage(
    instrumented, span_exporter
):
    """One receive_response() per turn is the SDK's documented multi-turn loop,
    while the CLI's usage counters keep growing over the whole connection."""
    client = connected_client(
        [raw_init(), raw_assistant("4"), raw_result("4", 10)]
        + [raw_assistant("6"), raw_result("6", 20)]
        + [raw_assistant("8"), raw_result("8", 30)]
    )
    for question in ("2 + 2?", "3 + 3?", "4 + 4?"):
        await client.query(question)
        async for _ in client.receive_response():
            pass

    roots = [s for s in span_exporter.get_finished_spans() if s.name == ROOT]
    assert [r.attributes["gen_ai.usage.output_tokens"] for r in roots] == [
        10,
        10,
        10,
    ]


def root_output_tokens(span_exporter: Any) -> list[Any]:
    return [
        span.attributes.get("gen_ai.usage.output_tokens")
        for span in span_exporter.get_finished_spans()
        if span.name == ROOT
    ]


async def read_turns(
    client: ClaudeSDKClient, prompts: list[str], *, collect: bool = True
) -> None:
    for prompt in prompts:
        await client.query(prompt)
        async for _ in client.receive_response():
            pass
        # receive_response() returns without closing the generator it read.
        # collect=True finalizes it now, as an immediate garbage collection
        # would; collect=False leaves it open into the next turn.
        if collect:
            gc.collect()
            for _ in range(3):
                await asyncio.sleep(0)


BACKGROUND_TURN = [
    raw_init(),
    raw_call("bg"),
    raw_task("task_started", "bg"),
    raw_tool_result("bg", "started"),
    raw_text("started it", message_id="m1"),
    raw_result("started it", 10),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collect", [True, False], ids=["closed-between-turns", "closed-later"]
)
async def test_a_turn_left_waiting_is_still_counted_once(
    instrumented, span_exporter, collect
):
    """receive_response() stops at each result without closing its stream, so
    the turn waiting on background work is still open when the next call
    starts reading. Both calls read one connection, and count it once."""
    client = connected_client(
        BACKGROUND_TURN
        + [raw_init(), raw_text("two", message_id="m2"), raw_result("", 30)]
        + [raw_init(), raw_text("three", message_id="m3"), raw_result("", 60)]
    )
    await read_turns(client, ["one", "two", "three"], collect=collect)
    assert root_output_tokens(span_exporter) == [10, 20, 30]


@pytest.mark.asyncio
async def test_a_reconnected_client_counts_from_zero_again(
    instrumented, span_exporter
):
    """connect() starts a new CLI, whose counters start over."""
    client = connected_client(
        [raw_init(), raw_assistant(), raw_result("", 100)]
    )
    await read_turns(client, ["one"])
    # What connect() does: a new reader for a new CLI process.
    client._query = FakeQuery(  # pylint: disable=protected-access
        [raw_init(), raw_assistant(), raw_result("", 30)]
    )
    await read_turns(client, ["two"])
    assert root_output_tokens(span_exporter) == [100, 30]


@pytest.mark.asyncio
async def test_a_background_delegation_stays_in_the_turn_that_started_it(
    instrumented, span_exporter
):
    """The order the CLI uses (recordings/base.jsonl): the caller's result, the
    subagent working, its notification, then the next turn's init."""
    client = connected_client(
        [
            raw_init(),
            raw_call("agent", "Agent"),
            raw_task("task_started", "agent"),
            raw_tool_result("agent", "launched"),
            raw_result("", 10),
            raw_text("working", message_id="s1", parent="agent"),
            raw_task("task_notification", "agent"),
            raw_init(),
            raw_text("it finished", message_id="m2"),
            raw_result("", 30),
        ]
    )
    await read_turns(client, ["one", "two"])
    subagent = one_span(span_exporter, "invoke_agent Agent")
    tool = one_span(span_exporter, "execute_tool Agent")
    first_turn = [
        s for s in span_exporter.get_finished_spans() if s.name == ROOT
    ][0]
    assert subagent.parent.span_id == tool.context.span_id
    assert tool.parent.span_id == first_turn.context.span_id
    assert root_output_tokens(span_exporter) == [10, 20]


@pytest.mark.asyncio
async def test_disconnect_ends_a_turn_still_waiting(
    instrumented, span_exporter
):
    client = connected_client(BACKGROUND_TURN)
    await read_turns(client, ["one"])
    assert root_output_tokens(span_exporter) == []
    await client.disconnect()
    assert root_output_tokens(span_exporter) == [10]


@pytest.mark.asyncio
async def test_each_turn_opens_in_the_callers_context_for_it(
    instrumented, span_exporter, tracer_provider
):
    tracer = tracer_provider.get_tracer("app")
    client = connected_client(
        [raw_init(), raw_assistant(), raw_result()]
        + [raw_init(), raw_text("again", message_id="m2"), raw_result()]
    )
    for turn in ("first", "second"):
        with tracer.start_as_current_span(turn):
            await read_turns(client, [turn])

    spans = span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span.name for span in spans}
    roots = [span for span in spans if span.name == ROOT]
    assert [by_id[root.parent.span_id] for root in roots] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_prompts_sent_before_their_turns_open_keep_their_order(
    instrumented_with_content, span_exporter, tracer_provider
):
    """Each queued prompt keeps the context it was sent in."""
    tracer = tracer_provider.get_tracer("app")
    client = connected_client(
        [raw_init(), raw_assistant(), raw_result()]
        + [raw_init(), raw_text("again", message_id="m2"), raw_result()]
    )
    for prompt in ("first", "second"):
        with tracer.start_as_current_span(prompt):
            await client.query(prompt)
    for _ in range(2):
        async for _ in client.receive_response():
            pass

    spans = span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span.name for span in spans}
    roots = [span for span in spans if span.name == ROOT]
    assert [first_input_text(r) for r in roots] == ["first", "second"]
    assert [by_id[r.parent.span_id] for r in roots] == ["first", "second"]


@pytest.mark.asyncio
async def test_a_turn_the_cli_starts_opens_where_the_caller_last_was(
    instrumented, span_exporter, tracer_provider
):
    """With the caller's last span, and the caller's conversation id."""
    tracer = tracer_provider.get_tracer("app")
    client = connected_client(
        [raw_init(), raw_assistant(), raw_result()]
        + [raw_init(), raw_text("two", message_id="m2"), raw_result()]
        + [raw_init(), raw_text("task done", message_id="m3"), raw_result()]
    )
    for turn in ("first", "second"):
        with tracer.start_as_current_span(turn):
            await client.query(turn, session_id="ticket-42")
    for _ in range(3):
        async for _ in client.receive_response():
            pass

    spans = span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span.name for span in spans}
    roots = [span for span in spans if span.name == ROOT]
    assert [by_id[r.parent.span_id] for r in roots] == [
        "first",
        "second",
        "second",
    ]
    assert [r.attributes["gen_ai.conversation.id"] for r in roots] == [
        "ticket-42"
    ] * 3


@pytest.mark.asyncio
async def test_a_streamed_prompt_does_not_hold_a_place_in_the_queue(
    instrumented_with_content, span_exporter
):
    """An async-iterable prompt can hold any number of turns, including none,
    so it cannot be matched to one. The next plain prompt still lands on the
    next turn."""

    async def nothing():
        return
        yield  # pragma: no cover - makes this an async generator

    client = connected_client([raw_init(), raw_assistant(), raw_result()])
    await client.query(nothing())
    await client.query("second")
    async for _ in client.receive_response():
        pass

    root = one_span(span_exporter, ROOT)
    assert first_input_text(root) == "second"


# ─── naming the root agent span ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [0, 1, 2])
async def test_declared_subagents_never_name_the_root(
    instrumented, span_exporter, declared
):
    """``options.agents`` defines "custom subagents invokable via the Agent
    tool" — delegation targets, not the agent being invoked. Naming the root
    after the only declared one made ``invoke_agent assistant`` appear as both
    root and descendant, and declaring a second silently renamed the root."""
    agents = {
        name: AgentDefinition(description=name, prompt=name, tools=["Read"])
        for name in ("assistant", "reviewer")[:declared]
    }
    client = connected_client(options=ClaudeAgentOptions(agents=agents))

    async for _ in client.receive_response():
        pass

    assert span_names(span_exporter) == Counter({ROOT: 1, CHAT: 1})


@pytest.mark.asyncio
async def test_agent_name_is_an_explicit_opt_in(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    uninstrument_claude_agent_sdk,
):
    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        agent_name="checkout-bot",
    )
    try:
        async for _ in connected_client().receive_response():
            pass
    finally:
        instrumentor.uninstrument()

    assert span_names(span_exporter) == Counter(
        {"invoke_agent checkout-bot": 1, CHAT: 1}
    )


# ─── handler and completion hook lifecycle ──────────────────────────────────


def test_constructing_a_second_instrumentor_keeps_the_live_handler(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    """BaseInstrumentor.__new__ returns the one live instance but __init__ still
    runs, so an __init__ that reset the handler would wipe it out from under an
    instrumentation already in flight."""
    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        handler = instrumentor._handler  # pylint: disable=protected-access
        assert handler is not None
        assert (
            ClaudeAgentSDKInstrumentor()._handler  # pylint: disable=protected-access
            is handler
        )
