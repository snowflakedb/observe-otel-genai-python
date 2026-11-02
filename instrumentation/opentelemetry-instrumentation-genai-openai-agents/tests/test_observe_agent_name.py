# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""A tool run by an agent records that agent's name. The processor never passes
one, so it comes from util-genai's agent context. Specific to this fork."""

from __future__ import annotations

from typing import Any

from agents.tracing.span_data import AgentSpanData, FunctionSpanData

from observeinc.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)
from observeinc.util.genai.handler import TelemetryHandler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class _Span:
    def __init__(self, span_data: Any) -> None:
        self.span_data = span_data
        self.span_id = f"span-{id(self)}"


def test_a_tool_inside_an_agent_records_its_name(
    tracer_provider: TracerProvider, span_exporter: InMemorySpanExporter
) -> None:
    processor = GenAITracingProcessor(
        TelemetryHandler(tracer_provider=tracer_provider), provider="openai"
    )
    agent = _Span(AgentSpanData(name="triage"))
    tool = _Span(FunctionSpanData(name="get_weather", input=None, output=None))

    processor.on_span_start(agent)
    processor.on_span_start(tool)
    processor.on_span_end(tool)
    processor.on_span_end(agent)

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    tool_span = spans["execute_tool get_weather"]
    assert tool_span.attributes is not None
    assert tool_span.attributes["gen_ai.agent.name"] == "triage"
