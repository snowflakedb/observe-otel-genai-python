# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""gen_ai.provider.name follows the host the OpenAI client points at.
Specific to this fork."""

from __future__ import annotations

import asyncio
import json

import openai
import openai._base_client as base_client
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)

# Newer openai releases vendor httpx as httpx2.
httpx = getattr(base_client, "httpx2", None) or base_client.httpx

CORTEX_BASE_URL = (
    "https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1"
)
CORTEX_HOST = "myorg-myaccount.snowflakecomputing.com"
MESSAGES = [{"role": "user", "content": "hi"}]

_CHAT = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "claude-haiku-4-5",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
}
_CHAT_CHUNK = {
    "id": "c",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "claude-haiku-4-5",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "delta": {"role": "assistant", "content": "hi"},
        }
    ],
}
_RESPONSE = {
    "id": "r",
    "object": "response",
    "created_at": 0,
    "model": "claude-haiku-4-5",
    "status": "completed",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_EMBEDDING = {
    "object": "list",
    "model": "e5-base-v2",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


def _respond(request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/embeddings"):
        return httpx.Response(200, json=_EMBEDDING)
    if path.endswith("/responses"):
        return httpx.Response(200, json=_RESPONSE)
    if json.loads(request.content).get("stream"):
        body = f"data: {json.dumps(_CHAT_CHUNK)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
        )
    return httpx.Response(200, json=_CHAT)


def _client(base_url: str = CORTEX_BASE_URL, respond=_respond):
    return openai.OpenAI(
        api_key="x",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def _async_client(base_url: str = CORTEX_BASE_URL):
    return openai.AsyncOpenAI(
        api_key="x",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_respond)),
    )


def _only_span(span_exporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _duration_providers(metric_reader) -> set[str]:
    providers = set()
    for resource_metrics in metric_reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if (
                    metric.name
                    != GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION
                ):
                    continue
                for point in metric.data.data_points:
                    providers.add(
                        point.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
                    )
    return providers


def _assert_provider(span_exporter, metric_reader, expected: str) -> None:
    span = _only_span(span_exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == expected
    assert _duration_providers(metric_reader) == {expected}


def test_chat_completion(span_exporter, metric_reader, instrument_no_content):
    _client().chat.completions.create(
        model="claude-haiku-4-5", messages=MESSAGES
    )

    _assert_provider(span_exporter, metric_reader, "snowflake")
    span = _only_span(span_exporter)
    assert span.attributes[ServerAttributes.SERVER_ADDRESS] == CORTEX_HOST


def test_async_chat_completion(
    span_exporter, metric_reader, instrument_no_content
):
    asyncio.run(
        _async_client().chat.completions.create(
            model="claude-haiku-4-5", messages=MESSAGES
        )
    )

    _assert_provider(span_exporter, metric_reader, "snowflake")


def test_chat_completion_stream(
    span_exporter, metric_reader, instrument_no_content
):
    stream = _client().chat.completions.create(
        model="claude-haiku-4-5", messages=MESSAGES, stream=True
    )
    for _ in stream:
        pass

    _assert_provider(span_exporter, metric_reader, "snowflake")


def test_chat_completion_error(
    span_exporter, metric_reader, instrument_no_content
):
    def fail(request):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(openai.InternalServerError):
        _client(respond=fail).chat.completions.create(
            model="claude-haiku-4-5", messages=MESSAGES
        )

    _assert_provider(span_exporter, metric_reader, "snowflake")


@pytest.mark.skipif(
    not hasattr(openai.OpenAI, "responses"),
    reason="Responses API requires a newer openai SDK",
)
def test_responses(span_exporter, metric_reader, instrument_no_content):
    _client().responses.create(model="claude-haiku-4-5", input="hi")

    _assert_provider(span_exporter, metric_reader, "snowflake")


def test_embeddings(span_exporter, metric_reader, instrument_no_content):
    _client().embeddings.create(model="e5-base-v2", input="hi")

    _assert_provider(span_exporter, metric_reader, "snowflake")


def test_default_base_url_stays_openai(
    span_exporter, metric_reader, instrument_no_content
):
    _client(base_url="https://api.openai.com/v1").chat.completions.create(
        model="gpt-4o-mini", messages=MESSAGES
    )

    _assert_provider(span_exporter, metric_reader, "openai")
