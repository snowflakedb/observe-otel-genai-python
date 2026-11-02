# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""An error message can quote the prompt, so spans that carry no content keep
the error's type and the ERROR status but not its message. Specific to this
fork."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from observeinc.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.invocation import GenAIInvocation
from observeinc.util.genai.types import ContentCapturingMode
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

SECRET = "the prompt said secret-content"

OPERATIONS: dict[str, Callable[[TelemetryHandler], GenAIInvocation]] = {
    "inference": lambda h: h.inference("openai", request_model="gpt-4o"),
    "embedding": lambda h: h.embedding("openai"),
    "fetch_response": lambda h: h.fetch_response("openai", response_id="r"),
    "retrieval": lambda h: h.retrieval(),
    "tool": lambda h: h.tool("lookup"),
    "local_agent": lambda h: h.invoke_local_agent(),
    "remote_agent": lambda h: h.invoke_remote_agent("openai"),
    "workflow": lambda h: h.workflow("wf"),
}


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def log_exporter() -> InMemoryLogRecordExporter:
    return InMemoryLogRecordExporter()


@pytest.fixture
def make_handler(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
) -> Iterator[Callable[[ContentCapturingMode], TelemetryHandler]]:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    def make(mode: ContentCapturingMode) -> TelemetryHandler:
        monkeypatch.setenv(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, mode.name
        )
        return TelemetryHandler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        )

    yield make
    tracer_provider.shutdown()
    logger_provider.shutdown()


@pytest.mark.parametrize(
    "mode", [ContentCapturingMode.NO_CONTENT, ContentCapturingMode.EVENT_ONLY]
)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_span_without_content_drops_the_message(
    make_handler: Callable[[ContentCapturingMode], TelemetryHandler],
    span_exporter: InMemorySpanExporter,
    mode: ContentCapturingMode,
    operation: str,
) -> None:
    OPERATIONS[operation](make_handler(mode)).fail(RuntimeError(SECRET))

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert span.attributes is not None
    assert span.attributes["error.type"] == "RuntimeError"
    assert not span.events
    assert SECRET not in str(dict(span.attributes))


@pytest.mark.parametrize(
    "mode",
    [ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT],
)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_span_with_content_keeps_the_message(
    make_handler: Callable[[ContentCapturingMode], TelemetryHandler],
    span_exporter: InMemorySpanExporter,
    mode: ContentCapturingMode,
    operation: str,
) -> None:
    OPERATIONS[operation](make_handler(mode)).fail(RuntimeError(SECRET))

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == SECRET


def test_the_event_only_log_event_carries_no_error_text(
    make_handler: Callable[[ContentCapturingMode], TelemetryHandler],
    log_exporter: InMemoryLogRecordExporter,
) -> None:
    invocation = make_handler(ContentCapturingMode.EVENT_ONLY).inference(
        "openai"
    )
    invocation.fail(RuntimeError(SECRET))

    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.attributes is not None
    assert record.attributes["error.type"] == "RuntimeError"
    assert SECRET not in str(dict(record.attributes))
    assert record.body is None
