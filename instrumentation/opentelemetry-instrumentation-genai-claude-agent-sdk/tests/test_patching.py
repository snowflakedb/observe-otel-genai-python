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

import json
import sys
from collections import Counter
from typing import Any
from unittest.mock import MagicMock, patch

import claude_agent_sdk
import pytest
import wrapt
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
)
from claude_agent_sdk._internal.message_parser import parse_message

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.completion_hook import _SafeCompletionHook

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


def raw_result(text: str = "4") -> dict[str, Any]:
    return {
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


ONE_TURN = [raw_init(), raw_assistant(), raw_result()]


# ─── fakes ──────────────────────────────────────────────────────────────────


class FakeTransport:
    """Records what the SDK writes to the CLI's stdin."""

    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []

    async def write(self, data: str) -> None:
        self.written.extend(json.loads(line) for line in data.splitlines())


class FakeQuery:
    """Stands in for the client's internal reader, yielding raw CLI messages."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    async def receive_messages(self):
        for message in self._messages:
            yield message


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
    for attribute in ("query", "receive_messages"):
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
    assert len(before) == 4

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
    """The stashed prompt is consumed, not read: a second turn that submitted
    nothing must not be reported as having asked the first turn's question."""
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


def test_uninstrument_shuts_down_the_hook_it_loaded(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    """The upload hook owns a thread pool and load_completion_hook() builds a
    fresh one per _instrument(), so every cycle that only dropped the handler
    leaked a pool and abandoned whatever it had not uploaded."""
    hook = MagicMock()
    with patch(
        "opentelemetry.instrumentation.genai.claude_agent_sdk.load_completion_hook",
        # What the real loader returns: the hook behind util-genai's private
        # safety wrapper, which forwards on_completion and nothing else.
        return_value=_SafeCompletionHook(hook),
    ):
        with instrument(
            ClaudeAgentSDKInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            hook.shutdown.assert_not_called()

    hook.shutdown.assert_called_once()


def test_a_caller_supplied_hook_is_left_alone(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    """A hook passed to instrument() belongs to the caller, who may well be
    using it for more than this one instrumentation."""
    hook = MagicMock()
    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        completion_hook=hook,
    ):
        pass

    hook.shutdown.assert_not_called()
