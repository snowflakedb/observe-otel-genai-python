# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A failed tool or agent keeps the error message off its span when the span
carries no content, since the message can quote the prompt. Specific to this
fork."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from observeinc.instrumentation.genai.langchain import LangChainInstrumentor
from observeinc.test_util_genai.instrumentor import instrument
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

SECRET = "no weather for the secret-city"


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    raise RuntimeError(SECRET)


def _failed_spans(
    content_capture: str,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
) -> dict[str, ReadableSpan]:
    def run_agent(_: Any, config: RunnableConfig) -> str:
        return get_weather.invoke({"city": "Paris"}, config=config)

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_capture,
    ):
        with pytest.raises(RuntimeError):
            RunnableLambda(run_agent).invoke(
                {}, config={"metadata": {"agent_name": "weather_agent"}}
            )

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert set(spans) == {
        "invoke_agent weather_agent",
        "execute_tool get_weather",
    }
    return spans


@pytest.mark.parametrize("content_capture", ["NO_CONTENT", "EVENT_ONLY"])
def test_a_span_without_content_drops_the_message(
    content_capture: str,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    spans = _failed_spans(
        content_capture,
        tracer_provider,
        logger_provider,
        meter_provider,
        span_exporter,
    )

    for span in spans.values():
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description is None
        assert span.attributes is not None
        assert span.attributes["error.type"] == "RuntimeError"
        assert not span.events
        assert SECRET not in str(dict(span.attributes))


def test_a_span_with_content_keeps_the_message(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    spans = _failed_spans(
        "SPAN_ONLY",
        tracer_provider,
        logger_provider,
        meter_provider,
        span_exporter,
    )

    for span in spans.values():
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description == SECRET
