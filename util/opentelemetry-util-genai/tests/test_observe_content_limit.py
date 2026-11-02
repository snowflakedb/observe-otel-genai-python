# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""OBSERVE_GENAI_MAX_CONTENT_LENGTH caps each captured content attribute and
keeps its JSON valid. Specific to this fork."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from observeinc.util.genai._observe_content_limit import (
    DEFAULT_MAX_CONTENT_LENGTH,
    OBSERVE_GENAI_MAX_CONTENT_LENGTH,
)
from observeinc.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    TextPart,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

LIMIT = 200
INPUT = GenAI.GEN_AI_INPUT_MESSAGES


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
        self.hook = RecordingHook()

    def handler(self) -> TelemetryHandler:
        return TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
            completion_hook=self.hook,
        )

    def span(self) -> ReadableSpan:
        (span,) = self.span_exporter.get_finished_spans()
        return span

    def chat(
        self,
        inputs: list[InputMessage],
        outputs: list[OutputMessage] | None = None,
    ) -> ReadableSpan:
        with self.handler().inference("openai", request_model="gpt-5") as chat:
            chat.input_messages = inputs
            chat.output_messages = outputs or []
        return self.span()


@pytest.fixture
def telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[Telemetry]:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    telemetry = Telemetry()
    yield telemetry
    telemetry.tracer_provider.shutdown()
    telemetry.logger_provider.shutdown()


@pytest.fixture
def small_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OBSERVE_GENAI_MAX_CONTENT_LENGTH, str(LIMIT))


def _user(text: str) -> InputMessage:
    return InputMessage(role="user", parts=[TextPart(content=text)])


def _assert_trimmed(span: ReadableSpan, key: str, original: str) -> str:
    attributes = span.attributes or {}
    value = attributes[key]
    assert isinstance(value, str)
    assert len(value) <= LIMIT
    assert attributes[f"{key}.truncated"] is True
    assert attributes[f"{key}.original_size_bytes"] == len(
        original.encode("utf-8")
    )
    return value


def _untrimmed_json(*messages: InputMessage) -> str:
    return json.dumps(
        [
            {
                "role": m.role,
                "parts": [
                    {"content": p.content, "type": p.type}
                    for p in m.parts
                    if isinstance(p, TextPart)
                ],
                "name": m.name,
            }
            for m in messages
        ],
        separators=(",", ":"),
    )


@pytest.mark.usefixtures("small_limit")
def test_messages_keep_their_structure(telemetry: Telemetry) -> None:
    messages = [_user("a" * 500), _user("short")]
    span = telemetry.chat(messages)

    value = _assert_trimmed(span, INPUT, _untrimmed_json(*messages))
    trimmed = json.loads(value)
    assert [m["role"] for m in trimmed] == ["user", "user"]
    assert [p["type"] for m in trimmed for p in m["parts"]] == ["text"] * 2
    long_text = trimmed[0]["parts"][0]["content"]
    assert long_text.startswith("aaa")
    assert long_text.endswith("\u2026")


@pytest.mark.usefixtures("small_limit")
def test_content_under_the_limit_is_unchanged(telemetry: Telemetry) -> None:
    span = telemetry.chat([_user("hello")])

    attributes = span.attributes or {}
    assert attributes[INPUT] == _untrimmed_json(_user("hello"))
    assert f"{INPUT}.truncated" not in attributes
    assert f"{INPUT}.original_size_bytes" not in attributes


@pytest.mark.usefixtures("small_limit")
def test_only_the_attribute_over_the_limit_is_trimmed(
    telemetry: Telemetry,
) -> None:
    output = OutputMessage(
        role="assistant", parts=[TextPart(content="ok")], finish_reason="stop"
    )
    span = telemetry.chat([_user("a" * 500)], [output])

    attributes = span.attributes or {}
    assert f"{INPUT}.truncated" in attributes
    assert f"{GenAI.GEN_AI_OUTPUT_MESSAGES}.truncated" not in attributes
    assert json.loads(str(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])) == [
        {
            "role": "assistant",
            "parts": [{"content": "ok", "type": "text"}],
            "finish_reason": "stop",
            "name": None,
        }
    ]


@pytest.mark.usefixtures("small_limit")
def test_escaped_text_still_fits(telemetry: Telemetry) -> None:
    span = telemetry.chat([_user("你好" * 300)])

    value = (span.attributes or {})[INPUT]
    assert isinstance(value, str)
    assert len(value) <= LIMIT
    assert json.loads(value)[0]["role"] == "user"


@pytest.mark.usefixtures("small_limit")
def test_the_oldest_messages_go_when_structure_alone_is_too_big(
    telemetry: Telemetry,
) -> None:
    messages = [
        InputMessage(role="user", parts=[TextPart(content="x")], name=f"m{i}")
        for i in range(20)
    ]
    span = telemetry.chat(messages)

    trimmed = json.loads(
        _assert_trimmed(span, INPUT, _untrimmed_json(*messages))
    )
    assert 0 < len(trimmed) < 20
    assert trimmed[-1]["name"] == "m19"


@pytest.mark.usefixtures("small_limit")
def test_tool_definitions_drop_whole_fields(telemetry: Telemetry) -> None:
    tool = FunctionToolDefinition(
        name="lookup",
        description="d" * 100,
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    with telemetry.handler().inference(
        "openai", request_model="gpt-5"
    ) as chat:
        chat.tool_definitions = [tool, tool]

    span = telemetry.span()
    key = GenAI.GEN_AI_TOOL_DEFINITIONS
    attributes = span.attributes or {}
    assert attributes[f"{key}.truncated"] is True
    assert json.loads(str(attributes[key])) == [
        {"name": "lookup", "type": "function"},
        {"name": "lookup", "type": "function"},
    ]


@pytest.mark.usefixtures("small_limit")
def test_tool_arguments_stay_valid_json(telemetry: Telemetry) -> None:
    arguments = {"query": "q" * 500, "limit": 5}
    with telemetry.handler().tool("lookup") as tool:
        tool.arguments = arguments
        tool.tool_result = "r" * 500

    span = telemetry.span()
    trimmed = json.loads(
        _assert_trimmed(
            span,
            GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
            json.dumps(arguments, separators=(",", ":")),
        )
    )
    assert trimmed["limit"] == 5
    assert trimmed["query"].startswith("qqq")

    result = _assert_trimmed(span, GenAI.GEN_AI_TOOL_CALL_RESULT, "r" * 500)
    head, marker, tail = result.split("\n")
    assert head.startswith("rrr")
    assert tail.endswith("rrr")
    assert marker == f"...[truncated {500 - len(head) - len(tail)} chars]..."


@pytest.mark.usefixtures("small_limit")
def test_the_event_keeps_structured_content(
    telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "true")
    telemetry.chat([_user("a" * 500)])

    (log,) = telemetry.log_exporter.get_finished_logs()
    attributes = log.log_record.attributes or {}
    messages = attributes[INPUT]
    assert isinstance(messages, tuple | list)
    assert messages[0]["role"] == "user"
    assert len(json.dumps(messages, separators=(",", ":"))) <= LIMIT
    assert attributes[f"{INPUT}.truncated"] is True


@pytest.mark.usefixtures("small_limit")
def test_the_completion_hook_gets_the_full_content(
    telemetry: Telemetry,
) -> None:
    telemetry.chat([_user("a" * 500)])

    (call,) = telemetry.hook.calls
    assert call["inputs"] == [_user("a" * 500)]


def test_the_default_limit_applies(telemetry: Telemetry) -> None:
    text = "a" * DEFAULT_MAX_CONTENT_LENGTH
    span = telemetry.chat([_user(text)])

    attributes = span.attributes or {}
    assert len(str(attributes[INPUT])) <= DEFAULT_MAX_CONTENT_LENGTH
    assert attributes[f"{INPUT}.truncated"] is True


def test_zero_turns_the_limit_off(
    telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OBSERVE_GENAI_MAX_CONTENT_LENGTH, "0")
    text = "a" * DEFAULT_MAX_CONTENT_LENGTH
    span = telemetry.chat([_user(text)])

    attributes = span.attributes or {}
    assert attributes[INPUT] == _untrimmed_json(_user(text))
    assert f"{INPUT}.truncated" not in attributes


@pytest.mark.parametrize("value", ["lots", "-1"])
def test_an_invalid_value_falls_back_to_the_default(
    telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    monkeypatch.setenv(OBSERVE_GENAI_MAX_CONTENT_LENGTH, value)
    with caplog.at_level(logging.WARNING):
        span = telemetry.chat([_user("a" * DEFAULT_MAX_CONTENT_LENGTH)])

    assert f"{INPUT}.truncated" in (span.attributes or {})
    assert OBSERVE_GENAI_MAX_CONTENT_LENGTH in caplog.text
