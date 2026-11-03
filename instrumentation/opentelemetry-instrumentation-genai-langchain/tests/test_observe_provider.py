# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""A ChatOpenAI pointed at another provider reports that provider. Specific to
this fork."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)

CORTEX_BASE_URL = (
    "https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1"
)

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


def _chat_model(base_url: str, status: int = 200) -> ChatOpenAI:
    body = _CHAT if status == 200 else {"error": {"message": "boom"}}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json=body)
    )
    return ChatOpenAI(
        model="claude-haiku-4-5",
        base_url=base_url,
        api_key="x",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )


def _only_span(span_exporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _duration_providers(metric_reader) -> set[str]:
    return {
        point.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION
        for point in metric.data.data_points
    }


def test_cortex_base_url(span_exporter, metric_reader, start_instrumentation):
    _chat_model(CORTEX_BASE_URL).invoke([HumanMessage(content="hi")])

    span = _only_span(span_exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "snowflake"
    assert _duration_providers(metric_reader) == {"snowflake"}


def test_cortex_base_url_on_error(
    span_exporter, metric_reader, start_instrumentation
):
    with pytest.raises(openai.InternalServerError):
        _chat_model(CORTEX_BASE_URL, status=500).invoke(
            [HumanMessage(content="hi")]
        )

    span = _only_span(span_exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "snowflake"
    assert _duration_providers(metric_reader) == {"snowflake"}


def test_unknown_host_stays_openai(
    span_exporter, metric_reader, start_instrumentation
):
    _chat_model("http://localhost:4242/v1").invoke(
        [HumanMessage(content="hi")]
    )

    span = _only_span(span_exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
