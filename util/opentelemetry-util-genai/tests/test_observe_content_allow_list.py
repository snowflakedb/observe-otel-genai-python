# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ``ALLOW_LIST`` capture mode keeps only allowlisted attributes on
everything the util records. Specific to this fork."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from observeinc.test_util_genai._observe_allow_list import (
    assert_allow_listed,
    attribute_keys,
)
from observeinc.util.genai._instruments import _Instruments
from observeinc.util.genai._invocation import get_content_attributes
from observeinc.util.genai._observe_content_allow_list import (
    ALLOWED_KEYS,
    CONTENT_KEYS,
    allow_list_telemetry,
    warn_completion_hook_ignored,
)
from observeinc.util.genai._observe_metric_conversation_id import (
    OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID,
)
from observeinc.util.genai._observe_metric_user_id import (
    OBSERVE_GENAI_METRICS_INCLUDE_USER_ID,
)
from observeinc.util.genai.completion_hook import (
    _NoOpCompletionHook,
    load_completion_hook,
)
from observeinc.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.invocation import GenAIInvocation
from observeinc.util.genai.types import (
    ContentCapturingMode,
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    RetrievalDocument,
    TextPart,
)
from observeinc.util.genai.user_id import using_user_id
from observeinc.util.genai.utils import (
    _should_emit_event,
    get_content_capturing_mode,
)
from opentelemetry import trace
from opentelemetry.metrics import Instrument
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Link, SpanKind, Status, StatusCode

SECRET = "secret-content"
UNKNOWN_KEY = "gen_ai.future.thing"

# Keys the util emits that are in neither list of _content_allow_list.json.
# They are dropped under ALLOW_LIST until a privacy review adds them to `keep`
# (or to `content`).
NOT_ALLOW_LISTED = {
    # Content the shared `content` list does not name yet.
    "gen_ai.retrieval.query.text",
    "gen_ai.retrieval.documents",
    "gen_ai.prompt.variable.city",
    # Free text, so left out on purpose, as in the TypeScript exporter.
    "gen_ai.request.stop_sequences",
    # Metadata pending review.
    "gen_ai.conversation.compacted",
    "gen_ai.data_source.id",
    "gen_ai.embeddings.dimension.count",
    "gen_ai.prompt.name",
    "gen_ai.prompt.version",
    "gen_ai.request.previous_response.id",
    "gen_ai.request.reasoning.level",
    "gen_ai.request.stream_cursor",
    "gen_ai.request.top_k",
    "gen_ai.response.status",
    "gen_ai.response.time_to_first_chunk",
    "gen_ai.retrieval.top_k",
    "gen_ai.token.modality",
    "gen_ai.usage.audio.cache_read.input_tokens",
    "gen_ai.usage.audio.input_tokens",
    "gen_ai.usage.audio.output_tokens",
    "gen_ai.usage.cache_write.input_tokens",
    "gen_ai.usage.image.cache_read.input_tokens",
    "gen_ai.usage.image.input_tokens",
    "gen_ai.usage.image.output_tokens",
    "gen_ai.usage.text.cache_read.input_tokens",
    "gen_ai.usage.text.input_tokens",
    "gen_ai.usage.text.output_tokens",
    "gen_ai.workflow.name",
    # The key every scenario adds to check that unknown keys are dropped.
    UNKNOWN_KEY,
}


class RecordingHook:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class Telemetry:
    def __init__(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        self.logger_provider = LoggerProvider()
        self.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        self.hook = RecordingHook()

    def handler(self) -> TelemetryHandler:
        return TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
            logger_provider=self.logger_provider,
            completion_hook=self.hook,
        )

    def spans(self) -> tuple[ReadableSpan, ...]:
        return self.span_exporter.get_finished_spans()

    def assert_allow_listed(self) -> None:
        assert_allow_listed(
            self.spans(),
            self.log_exporter.get_finished_logs(),
            self.metric_reader.get_metrics_data(),
        )

    def metric_points(self) -> dict[str, list[dict[str, Any]]]:
        data = self.metric_reader.get_metrics_data()
        points: dict[str, list[dict[str, Any]]] = {}
        for resource_metrics in data.resource_metrics if data else ():
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    points.setdefault(metric.name, []).extend(
                        dict(point.attributes or {})
                        for point in metric.data.data_points
                    )
        return points

    def keys(self) -> set[str]:
        return {
            key
            for _, key in attribute_keys(
                self.spans(),
                self.log_exporter.get_finished_logs(),
                self.metric_reader.get_metrics_data(),
            )
        }


def _mode_env(
    monkeypatch: pytest.MonkeyPatch, mode: ContentCapturingMode
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, mode.name
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "true")
    monkeypatch.setenv(OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID, "true")
    monkeypatch.setenv(OBSERVE_GENAI_METRICS_INCLUDE_USER_ID, "true")


@pytest.fixture
def telemetry() -> Iterator[Telemetry]:
    telemetry = Telemetry()
    yield telemetry
    telemetry.tracer_provider.shutdown()
    telemetry.logger_provider.shutdown()
    telemetry.meter_provider.shutdown()


@pytest.fixture
def allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _mode_env(monkeypatch, ContentCapturingMode.ALLOW_LIST)


# ─── worst-case scenarios, one per invocation type ──────────────────────────


def _inputs() -> list[InputMessage]:
    return [InputMessage(role="user", parts=[TextPart(content=SECRET)])]


def _outputs() -> list[OutputMessage]:
    return [
        OutputMessage(
            role="assistant",
            parts=[TextPart(content=SECRET)],
            finish_reason="stop",
        )
    ]


def _tools() -> list[FunctionToolDefinition]:
    return [
        FunctionToolDefinition(
            name="lookup", description=SECRET, parameters={"q": SECRET}
        )
    ]


def _finish(invocation: GenAIInvocation) -> None:
    invocation.attributes[UNKNOWN_KEY] = SECRET
    invocation.metric_attributes[UNKNOWN_KEY] = SECRET
    invocation.fail(RuntimeError(SECRET))


def _inference(handler: TelemetryHandler, *, stream: bool) -> None:
    chat = handler.inference(
        "openai",
        request_model="gpt-5",
        server_address="api.openai.com",
        server_port=443,
        agent_name="support-bot",
        conversation_id="conv-1",
    )
    chat.input_messages = _inputs()
    chat.output_messages = _outputs()
    chat.system_instruction = [TextPart(content=SECRET)]
    chat.tool_definitions = _tools()
    chat.prompt_variables = {"city": SECRET}
    chat.prompt_name = "greeting"
    chat.prompt_version = "3"
    chat.response_model_name = "gpt-5-0801"
    chat.response_id = "resp-1"
    chat.finish_reasons = ["stop"]
    chat.temperature = 0.5
    chat.top_p = 0.9
    chat.top_k = 40
    chat.frequency_penalty = 0.1
    chat.presence_penalty = 0.2
    chat.max_tokens = 100
    chat.stop_sequences = [SECRET]
    chat.seed = 7
    chat.request_choice_count = 2
    chat.output_type = "text"
    chat.reasoning_level = "high"
    chat.previous_response_id = "resp-0"
    chat.conversation_compacted = True
    chat.input_tokens = 30
    chat.output_tokens = 20
    chat.thinking_tokens = 5
    chat.cache_read_input_tokens = 4
    chat.cache_write_input_tokens = 3
    chat.set_input_tokens([("text", 20), ("image", 5), ("audio", 5)])
    chat.set_output_tokens([("text", 10), ("image", 5), ("audio", 5)])
    chat.set_cache_read_input_tokens([("text", 2), ("image", 1), ("audio", 1)])
    if stream:
        chat.record_stream_chunk()
        chat.record_stream_chunk()
    _finish(chat)


def _embedding(handler: TelemetryHandler) -> None:
    embedding = handler.embedding(
        "openai",
        request_model="embed-3",
        server_address="api.openai.com",
        server_port=443,
        agent_name="support-bot",
    )
    embedding.encoding_formats = ["float"]
    embedding.input_tokens = 12
    embedding.dimension_count = 1536
    embedding.response_model_name = "embed-3"
    _finish(embedding)


def _retrieval(handler: TelemetryHandler) -> None:
    retrieval = handler.retrieval(
        data_source_id="kb",
        provider="pinecone",
        request_model="embed-3",
        server_address="kb.example.com",
        server_port=443,
        agent_name="support-bot",
        conversation_id="conv-1",
    )
    retrieval.top_k = 5
    retrieval.query_text = SECRET
    retrieval.documents = [RetrievalDocument(id="doc-1", score=0.9)]
    _finish(retrieval)


def _tool(handler: TelemetryHandler) -> None:
    tool = handler.tool(
        "lookup",
        tool_type="function",
        agent_name="support-bot",
        conversation_id="conv-1",
    )
    tool.tool_call_id = "call-1"
    tool.tool_description = "Looks things up"
    tool.arguments = {"q": SECRET}
    tool.tool_result = SECRET
    _finish(tool)


def _fill_agent(agent: Any) -> None:
    agent.agent_description = "Answers questions"
    agent.data_source_id = "kb"
    agent.output_type = "text"
    agent.temperature = 0.5
    agent.top_p = 0.9
    agent.frequency_penalty = 0.1
    agent.presence_penalty = 0.2
    agent.max_tokens = 100
    agent.stop_sequences = [SECRET]
    agent.seed = 7
    agent.choice_count = 2
    agent.finish_reasons = ["stop"]
    agent.input_tokens = 30
    agent.output_tokens = 20
    agent.input_messages = _inputs()
    agent.output_messages = _outputs()
    agent.system_instruction = [TextPart(content=SECRET)]
    agent.tool_definitions = _tools()


def _local_agent(handler: TelemetryHandler) -> None:
    agent = handler.invoke_local_agent(
        request_model="gpt-5",
        agent_name="support-bot",
        conversation_id="conv-1",
    )
    _fill_agent(agent)
    _finish(agent)


def _remote_agent(handler: TelemetryHandler) -> None:
    agent = handler.invoke_remote_agent(
        "openai",
        request_model="gpt-5",
        server_address="api.openai.com",
        server_port=443,
        agent_name="support-bot",
        conversation_id="conv-1",
    )
    _fill_agent(agent)
    agent.agent_id = "agent-1"
    agent.agent_version = "2"
    agent.previous_response_id = "resp-0"
    agent.cache_read_input_tokens = 4
    agent.cache_write_input_tokens = 3
    _finish(agent)


def _workflow(handler: TelemetryHandler) -> None:
    workflow = handler.workflow("triage", conversation_id="conv-1")
    workflow.input_messages = _inputs()
    workflow.output_messages = _outputs()
    _finish(workflow)


def _fetch_response(handler: TelemetryHandler) -> None:
    fetch = handler.fetch_response(
        "openai",
        response_id="resp-1",
        request_stream=True,
        server_address="api.openai.com",
        server_port=443,
        agent_name="support-bot",
    )
    fetch.response_model_name = "gpt-5-0801"
    fetch.response_status = "completed"
    fetch.finish_reasons = ["stop"]
    fetch.stream_cursor = "cursor-1"
    fetch.output_messages = _outputs()
    fetch.system_instruction = [TextPart(content=SECRET)]
    fetch.tool_definitions = _tools()
    _finish(fetch)


_CLIENT_DURATION = (
    "gen_ai.client.operation.duration",
    {"gen_ai.operation.name", "gen_ai.provider.name"},
)
# Each scenario's duration metric, and allowlisted keys it must keep.
DURATION_METRICS: dict[str, tuple[str, set[str]]] = {
    "inference": _CLIENT_DURATION,
    "streaming_inference": _CLIENT_DURATION,
    "embedding": _CLIENT_DURATION,
    "retrieval": _CLIENT_DURATION,
    "tool": ("gen_ai.execute_tool.duration", {"gen_ai.tool.name"}),
    "local_agent": ("gen_ai.invoke_agent.duration", {"gen_ai.agent.name"}),
    "remote_agent": _CLIENT_DURATION,
    "workflow": ("gen_ai.invoke_workflow.duration", set()),
    "fetch_response": _CLIENT_DURATION,
}

SCENARIOS: dict[str, Callable[[TelemetryHandler], None]] = {
    "inference": lambda handler: _inference(handler, stream=False),
    "streaming_inference": lambda handler: _inference(handler, stream=True),
    "embedding": _embedding,
    "retrieval": _retrieval,
    "tool": _tool,
    "local_agent": _local_agent,
    "remote_agent": _remote_agent,
    "workflow": _workflow,
    "fetch_response": _fetch_response,
}


@pytest.mark.usefixtures("allow_list")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_only_allowlisted_attributes_are_recorded(
    telemetry: Telemetry, scenario: str
) -> None:
    with using_user_id("user-1"):
        SCENARIOS[scenario](telemetry.handler())

    telemetry.assert_allow_listed()
    (span,) = telemetry.spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes["gen_ai.operation.name"]
    assert span.attributes["user.id"] == "user-1"
    assert telemetry.hook.calls == []

    points = telemetry.metric_points()
    for name, attributes in points.items():
        for point in attributes:
            assert point["user.id"] == "user-1", name
    metric, keys = DURATION_METRICS[scenario]
    (duration,) = points[metric]
    assert duration["error.type"] == "RuntimeError"
    assert keys <= duration.keys()


@pytest.mark.usefixtures("allow_list")
def test_a_successful_operation(telemetry: Telemetry) -> None:
    with telemetry.handler().tool("lookup", tool_type="function") as tool:
        tool.arguments = {"q": SECRET}
        tool.attributes[UNKNOWN_KEY] = SECRET

    telemetry.assert_allow_listed()
    (span,) = telemetry.spans()
    assert span.status.status_code is StatusCode.UNSET
    assert dict(span.attributes) == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.type": "function",
    }
    (duration,) = telemetry.metric_points()["gen_ai.execute_tool.duration"]
    assert duration == {
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.type": "function",
    }


@pytest.mark.usefixtures("allow_list")
def test_the_inference_event_is_still_emitted_filtered(
    telemetry: Telemetry,
) -> None:
    _inference(telemetry.handler(), stream=False)

    (log,) = telemetry.log_exporter.get_finished_logs()
    assert (
        log.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    )
    assert log.log_record.attributes["gen_ai.request.model"] == "gpt-5"
    telemetry.assert_allow_listed()


def test_every_emitted_key_is_classified(
    telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails when the util starts emitting a key nobody has decided on yet:
    add it to `keep` or `content` in the shared list, or to NOT_ALLOW_LISTED."""
    _mode_env(monkeypatch, ContentCapturingMode.SPAN_AND_EVENT)
    handler = telemetry.handler()
    with using_user_id("user-1"):
        for scenario in SCENARIOS.values():
            scenario(handler)

    keys = telemetry.keys()
    assert keys - ALLOWED_KEYS - CONTENT_KEYS - NOT_ALLOW_LISTED == set()
    assert NOT_ALLOW_LISTED - keys == set(), "no longer emitted"
    assert telemetry.hook.calls


def test_allow_list_is_a_mode_of_its_own() -> None:
    """Two members with the same value make the second an alias of the
    first, which a merge with upstream could do without anyone noticing."""
    assert len(ContentCapturingMode.__members__) == len(ContentCapturingMode)
    assert ContentCapturingMode["ALLOW_LIST"].name == "ALLOW_LIST"


def test_the_lists_do_not_overlap() -> None:
    assert not ALLOWED_KEYS & CONTENT_KEYS
    assert not ALLOWED_KEYS & NOT_ALLOW_LISTED
    assert "gen_ai.input.messages" in CONTENT_KEYS


# ─── the mode itself ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["ALLOW_LIST", "allow_list", " Allow_List "])
def test_the_mode_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, value
    )
    assert get_content_capturing_mode() is ContentCapturingMode.ALLOW_LIST


@pytest.mark.usefixtures("allow_list")
def test_content_is_not_captured(telemetry: Telemetry) -> None:
    handler = telemetry.handler()
    assert handler.should_capture_content() is False
    with handler.tool("lookup") as tool:
        assert tool.should_capture_content is False
    assert (
        get_content_attributes(
            input_messages=_inputs(),
            output_messages=_outputs(),
            system_instruction=[],
            tool_definitions=None,
            for_span=True,
        )
        == {}
    )


@pytest.mark.usefixtures("allow_list")
def test_a_configured_hook_is_reported_once(
    telemetry: Telemetry, caplog: pytest.LogCaptureFixture
) -> None:
    warn_completion_hook_ignored.cache_clear()
    telemetry.handler()
    telemetry.handler()
    TelemetryHandler(tracer_provider=telemetry.tracer_provider)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "completion hook is not called" in warnings[0].getMessage()


@pytest.mark.usefixtures("allow_list")
def test_the_hook_named_in_the_environment_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    warn_completion_hook_ignored.cache_clear()
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, "upload")
    with patch(
        "observeinc.util.genai.completion_hook.entry_points"
    ) as entry_points:
        assert isinstance(load_completion_hook(), _NoOpCompletionHook)
    entry_points.assert_not_called()
    assert "completion hook is not called" in caplog.text


def test_loading_a_hook_does_not_repeat_the_invalid_mode_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "bogus"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, "missing")
    load_completion_hook()
    assert "is not a valid option" not in caplog.text


def test_events_follow_the_emit_event_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _should_emit_event(ContentCapturingMode.ALLOW_LIST) is False
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "true")
    assert _should_emit_event(ContentCapturingMode.ALLOW_LIST) is True


@pytest.mark.usefixtures("allow_list")
def test_the_current_span_is_filtered_for_other_code(
    telemetry: Telemetry,
) -> None:
    tracer = telemetry.tracer_provider.get_tracer("other")
    with telemetry.handler().invoke_local_agent(agent_name="support-bot"):
        current = trace.get_current_span()
        current.set_attribute(UNKNOWN_KEY, SECRET)
        current.set_attribute("gen_ai.agent.id", "agent-1")
        current.record_exception(ValueError(SECRET))
        tracer.start_span("child").end()

    child, agent = telemetry.spans()
    assert child.parent is not None
    assert child.parent.span_id == agent.context.span_id
    assert UNKNOWN_KEY not in agent.attributes
    assert agent.attributes["gen_ai.agent.id"] == "agent-1"
    telemetry.assert_allow_listed()


def test_other_modes_record_the_real_span(
    telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mode_env(monkeypatch, ContentCapturingMode.NO_CONTENT)
    with telemetry.handler().tool("lookup") as tool:
        assert isinstance(tool.span, ReadableSpan)
        tool.attributes[UNKNOWN_KEY] = "kept"
    (span,) = telemetry.spans()
    assert span.attributes[UNKNOWN_KEY] == "kept"


def test_every_instrument_is_filtered(telemetry: Telemetry) -> None:
    instruments = _Instruments(telemetry.meter_provider.get_meter("test"))
    _, filtered, _ = allow_list_telemetry(
        telemetry.tracer_provider.get_tracer("test"),
        instruments,
        telemetry.logger_provider.get_logger("test"),
    )
    for name, instrument in vars(instruments).items():
        if isinstance(instrument, Instrument):
            assert getattr(filtered, name) is not instrument, name


# ─── the span wrapper ────────────────────────────────────────────────────────


def _allow_list_tracer(telemetry: Telemetry) -> trace.Tracer:
    tracer, _, _ = allow_list_telemetry(
        telemetry.tracer_provider.get_tracer("test"),
        _Instruments(telemetry.meter_provider.get_meter("test")),
        telemetry.logger_provider.get_logger("test"),
    )
    return tracer


def _allow_list_span(telemetry: Telemetry, **kwargs: Any) -> trace.Span:
    return _allow_list_tracer(telemetry).start_span(
        "chat", kind=SpanKind.CLIENT, **kwargs
    )


def test_span_attributes_are_filtered(telemetry: Telemetry) -> None:
    span = _allow_list_span(
        telemetry,
        attributes={"gen_ai.request.model": "gpt-5", UNKNOWN_KEY: SECRET},
    )
    span.set_attribute("gen_ai.response.id", "resp-1")
    span.set_attribute("gen_ai.input.messages", SECRET)
    span.set_attributes({"server.port": 443, UNKNOWN_KEY: SECRET})
    span.end()

    (exported,) = telemetry.spans()
    assert dict(exported.attributes) == {
        "gen_ai.request.model": "gpt-5",
        "gen_ai.response.id": "resp-1",
        "server.port": 443,
    }
    assert exported.kind is SpanKind.CLIENT


def test_event_and_link_attributes_are_filtered(telemetry: Telemetry) -> None:
    linked = telemetry.tracer_provider.get_tracer("test").start_span("linked")
    linked.end()
    span = _allow_list_span(
        telemetry,
        links=[
            Link(
                linked.get_span_context(),
                {"gen_ai.agent.name": "a", UNKNOWN_KEY: SECRET},
            )
        ],
    )
    span.add_event("progress", {"error.type": "Timeout", UNKNOWN_KEY: SECRET})
    span.add_link(
        linked.get_span_context(),
        {"gen_ai.tool.name": "lookup", UNKNOWN_KEY: SECRET},
    )
    span.end()

    exported = telemetry.spans()[-1]
    (event,) = exported.events
    assert event.name == "progress"
    assert dict(event.attributes) == {"error.type": "Timeout"}
    assert [dict(link.attributes) for link in exported.links] == [
        {"gen_ai.agent.name": "a"},
        {"gen_ai.tool.name": "lookup"},
    ]


@pytest.mark.parametrize(
    "set_status",
    [
        lambda span: span.set_status(Status(StatusCode.ERROR, SECRET)),
        lambda span: span.set_status(StatusCode.ERROR, SECRET),
    ],
    ids=["status", "code_and_description"],
)
def test_the_status_keeps_its_code_only(
    telemetry: Telemetry, set_status: Callable[[trace.Span], None]
) -> None:
    span = _allow_list_span(telemetry)
    set_status(span)
    span.end()

    (exported,) = telemetry.spans()
    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None


def test_an_exception_records_its_type_only(telemetry: Telemetry) -> None:
    span = _allow_list_span(telemetry)
    span.record_exception(
        ValueError(SECRET), attributes={UNKNOWN_KEY: SECRET}, escaped=True
    )
    span.end()

    (event,) = telemetry.spans()[0].events
    assert event.name == "exception"
    assert dict(event.attributes) == {"exception.type": "ValueError"}


def test_the_span_as_a_context_manager(telemetry: Telemetry) -> None:
    span = _allow_list_span(telemetry)
    with pytest.raises(ValueError), span as entered:
        assert entered is span
        assert entered.is_recording()
        entered.update_name("chat gpt-5")
        raise ValueError(SECRET)

    (exported,) = telemetry.spans()
    assert exported.name == "chat gpt-5"
    assert exported.context == span.get_span_context()
    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None
    assert [dict(e.attributes) for e in exported.events] == [
        {"exception.type": "ValueError"}
    ]


def test_the_span_started_as_current(telemetry: Telemetry) -> None:
    tracer = _allow_list_tracer(telemetry)
    with (
        pytest.raises(ValueError),
        tracer.start_as_current_span(
            "chat", attributes={"server.port": 443, UNKNOWN_KEY: SECRET}
        ) as span,
    ):
        assert trace.get_current_span() is span
        trace.get_current_span().set_attribute(UNKNOWN_KEY, SECRET)
        raise ValueError(SECRET)

    (exported,) = telemetry.spans()
    assert dict(exported.attributes) == {"server.port": 443}
    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None
    assert [dict(e.attributes) for e in exported.events] == [
        {"exception.type": "ValueError"}
    ]
