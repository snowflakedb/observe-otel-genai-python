# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool and retrieval spans carry the run's conversation id, whether they run
inside an agent or workflow or are called on their own. Specific to this fork,
whose util-genai sets it on every span."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument
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


def test_a_retrieval_carries_its_agents_conversation_id() -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(tracer_provider=tracer_provider)
    )

    run_id, retriever_run_id = uuid4(), uuid4()
    handler.on_chain_start(
        serialized={"name": "weather_agent"},
        inputs={},
        run_id=run_id,
        parent_run_id=None,
        metadata={"agent_name": "weather_agent", "thread_id": "thread-1"},
    )
    handler.on_retriever_start(
        serialized={},
        query="weather",
        run_id=retriever_run_id,
        parent_run_id=run_id,
    )
    handler.on_retriever_end(documents=[], run_id=retriever_run_id)
    handler.on_chain_end(outputs={}, run_id=run_id)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    retrieval = spans["retrieval"]
    assert retrieval.attributes is not None
    assert retrieval.attributes["gen_ai.conversation.id"] == "thread-1"


class _Retriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document(page_content="Paris is sunny.")]


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


def _target(kind: str) -> tuple[Any, Any]:
    if kind == "retriever":
        return _Retriever(), "weather"
    return get_weather, {"city": "Paris"}


def _call(kind: str, config: dict[str, Any], use_async: bool) -> None:
    runnable, call_input = _target(kind)
    if use_async:
        asyncio.run(runnable.ainvoke(call_input, config=config))
    else:
        runnable.invoke(call_input, config=config)


@pytest.fixture(name="instrumented")
def fixture_instrumented(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
) -> Iterator[None]:
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        yield


_SPAN_NAMES = {
    "retriever": "retrieval",
    "tool": "execute_tool get_weather",
}


@pytest.mark.usefixtures("instrumented")
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["retriever", "tool"])
@pytest.mark.parametrize("key", ["thread_id", "session_id", "conversation_id"])
def test_a_standalone_call_carries_its_metadata_conversation_id(
    span_exporter: InMemorySpanExporter, key: str, kind: str, use_async: bool
) -> None:
    _call(kind, {"metadata": {key: "t-1"}}, use_async)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == _SPAN_NAMES[kind]
    assert span.attributes is not None
    assert span.attributes["gen_ai.conversation.id"] == "t-1"


@pytest.mark.usefixtures("instrumented")
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["retriever", "tool"])
def test_a_standalone_call_without_metadata_has_no_conversation_id(
    span_exporter: InMemorySpanExporter, kind: str, use_async: bool
) -> None:
    _call(kind, {}, use_async)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert "gen_ai.conversation.id" not in span.attributes


@pytest.mark.usefixtures("instrumented")
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["retriever", "tool"])
def test_a_call_inside_an_agent_carries_the_agents_conversation_id(
    span_exporter: InMemorySpanExporter, kind: str, use_async: bool
) -> None:
    def run_agent(_: Any, config: RunnableConfig) -> str:
        _call(kind, config, use_async=False)
        return "done"

    async def arun_agent(_: Any, config: RunnableConfig) -> str:
        runnable, call_input = _target(kind)
        await runnable.ainvoke(call_input, config=config)
        return "done"

    agent = RunnableLambda(run_agent, afunc=arun_agent)
    config: dict[str, Any] = {
        "metadata": {"agent_name": "weather_agent", "thread_id": "t-1"}
    }
    if use_async:
        asyncio.run(agent.ainvoke({}, config=config))
    else:
        agent.invoke({}, config=config)

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    for name in ("invoke_agent weather_agent", _SPAN_NAMES[kind]):
        attributes = spans[name].attributes
        assert attributes is not None
        assert attributes["gen_ai.conversation.id"] == "t-1"


@pytest.mark.usefixtures("instrumented")
@pytest.mark.parametrize("kind", ["retriever", "tool"])
def test_a_calls_own_conversation_id_wins_over_the_agents(
    span_exporter: InMemorySpanExporter, kind: str
) -> None:
    def run_agent(_: Any, config: RunnableConfig) -> str:
        _call(kind, {**config, "metadata": {"thread_id": "t-2"}}, False)
        return "done"

    RunnableLambda(run_agent).invoke(
        {},
        config={
            "metadata": {"agent_name": "weather_agent", "thread_id": "t-1"}
        },
    )

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    agent = spans["invoke_agent weather_agent"].attributes
    child = spans[_SPAN_NAMES[kind]].attributes
    assert agent is not None and child is not None
    assert agent["gen_ai.conversation.id"] == "t-1"
    assert child["gen_ai.conversation.id"] == "t-2"
