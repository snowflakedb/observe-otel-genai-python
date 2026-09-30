# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""A run's ``user_id`` metadata is recorded as ``user.id`` when
OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY is on. Specific to this fork."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.user_id import using_user_id

CAPTURE_FLAG = "OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY"


class _Retriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document(page_content="Paris is sunny.")]


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


def _chat_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content="Hi.")]))


def _agent(config: RunnableConfig) -> None:
    """An agent that calls a chat model, a tool and a retriever."""
    _chat_model().invoke(
        [HumanMessage(content="Hi")],
        config={
            **config,
            "metadata": {**config["metadata"], "ls_model_name": "fake"},
        },
    )
    get_weather.invoke({"city": "Paris"}, config=config)
    _Retriever().invoke("weather", config=config)


async def _aagent(config: RunnableConfig) -> None:
    await _chat_model().ainvoke(
        [HumanMessage(content="Hi")],
        config={
            **config,
            "metadata": {**config["metadata"], "ls_model_name": "fake"},
        },
    )
    await get_weather.ainvoke({"city": "Paris"}, config=config)
    await _Retriever().ainvoke("weather", config=config)


SPAN_NAMES = {
    "invoke_agent weather_agent",
    "chat fake",
    "execute_tool get_weather",
    "retrieval",
}


def _run_agent(metadata: dict[str, Any], use_async: bool) -> None:
    def run(_: Any, config: RunnableConfig) -> str:
        _agent(config)
        return "done"

    async def arun(_: Any, config: RunnableConfig) -> str:
        await _aagent(config)
        return "done"

    agent = RunnableLambda(run, afunc=arun)
    config: RunnableConfig = {
        "metadata": {"agent_name": "weather_agent", **metadata}
    }
    if use_async:
        asyncio.run(agent.ainvoke({}, config=config))
    else:
        agent.invoke({}, config=config)


def _user_ids(span_exporter: InMemorySpanExporter) -> dict[str, object]:
    return {
        span.name: (span.attributes or {}).get("user.id")
        for span in span_exporter.get_finished_spans()
    }


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


@pytest.fixture(name="capture_from_library")
def fixture_capture_from_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CAPTURE_FLAG, "true")


@pytest.mark.usefixtures("instrumented", "capture_from_library")
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_every_span_records_the_metadata_user_id(
    span_exporter: InMemorySpanExporter, use_async: bool
) -> None:
    _run_agent({"user_id": "user-123"}, use_async)
    assert _user_ids(span_exporter) == dict.fromkeys(SPAN_NAMES, "user-123")


@pytest.mark.usefixtures("instrumented")
def test_the_metadata_user_id_is_left_off_by_default(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CAPTURE_FLAG, raising=False)
    _run_agent({"user_id": "user-123"}, use_async=False)
    assert _user_ids(span_exporter) == dict.fromkeys(SPAN_NAMES)


@pytest.mark.usefixtures("instrumented", "capture_from_library")
def test_the_metadata_user_id_wins_over_using_user_id(
    span_exporter: InMemorySpanExporter,
) -> None:
    with using_user_id("from-helper"):
        _run_agent({"user_id": "from-metadata"}, use_async=False)
    assert _user_ids(span_exporter) == dict.fromkeys(
        SPAN_NAMES, "from-metadata"
    )


@pytest.mark.usefixtures("instrumented", "capture_from_library")
def test_using_user_id_applies_without_metadata(
    span_exporter: InMemorySpanExporter,
) -> None:
    with using_user_id("from-helper"):
        _run_agent({}, use_async=False)
    assert _user_ids(span_exporter) == dict.fromkeys(SPAN_NAMES, "from-helper")


def _call_standalone(kind: str, metadata: dict[str, Any]) -> None:
    config: RunnableConfig = {"metadata": metadata}
    if kind == "chat":
        _chat_model().invoke(
            [HumanMessage(content="Hi")],
            config={"metadata": {**metadata, "ls_model_name": "fake"}},
        )
    elif kind == "workflow":
        RunnableLambda(lambda _: "done", name="LangGraph").invoke(
            {}, config=config
        )
    elif kind == "retriever":
        _Retriever().invoke("weather", config=config)
    else:
        get_weather.invoke({"city": "Paris"}, config=config)


@pytest.mark.usefixtures("instrumented", "capture_from_library")
@pytest.mark.parametrize("kind", ["chat", "workflow", "retriever", "tool"])
def test_a_standalone_call_records_its_metadata_user_id(
    span_exporter: InMemorySpanExporter, kind: str
) -> None:
    _call_standalone(kind, {"user_id": "user-123"})
    (span,) = span_exporter.get_finished_spans()
    assert (span.attributes or {}).get("user.id") == "user-123"


@pytest.mark.usefixtures("instrumented", "capture_from_library")
def test_a_non_string_metadata_user_id_is_ignored(
    span_exporter: InMemorySpanExporter,
) -> None:
    with using_user_id("from-helper"):
        _call_standalone("tool", {"user_id": {"id": 1}})
    (span,) = span_exporter.get_finished_spans()
    assert (span.attributes or {}).get("user.id") == "from-helper"
