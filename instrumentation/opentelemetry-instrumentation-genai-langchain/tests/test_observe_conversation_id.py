# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""A tool run inside an agent or workflow carries the run's conversation id.
Specific to this fork, whose util-genai sets it on every span."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler


@pytest.mark.parametrize(
    ("chain_name", "metadata", "parent_span"),
    [
        (
            "weather_agent",
            {"agent_name": "weather_agent", "thread_id": "thread-1"},
            "invoke_agent weather_agent",
        ),
        ("LangGraph", {"thread_id": "thread-1"}, "invoke_workflow LangGraph"),
    ],
    ids=["agent", "workflow"],
)
def test_a_tool_carries_its_runs_conversation_id(
    chain_name: str, metadata: dict[str, Any], parent_span: str
) -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(tracer_provider=tracer_provider)
    )

    run_id, tool_run_id = uuid4(), uuid4()
    handler.on_chain_start(
        serialized={"name": chain_name},
        inputs={},
        run_id=run_id,
        parent_run_id=None,
        metadata=metadata,
    )
    handler.on_tool_start(
        serialized={"name": "get_weather"},
        input_str="",
        run_id=tool_run_id,
        parent_run_id=run_id,
        inputs={"city": "Paris"},
    )
    output = MagicMock()
    output.content = "sunny"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=tool_run_id)
    handler.on_chain_end(outputs={}, run_id=run_id)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert parent_span in spans
    tool = spans["execute_tool get_weather"]
    assert tool.attributes is not None
    assert tool.attributes["gen_ai.conversation.id"] == "thread-1"
