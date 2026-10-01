# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""When the API rejects a request, the CLI relays the error as an assistant
message it writes itself, with the model set to ``<synthetic>``. Specific to
this fork."""

from __future__ import annotations

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    SystemMessage,
    TextBlock,
)

from observeinc.util.genai.handler import TelemetryHandler
from opentelemetry import trace

from .test_stream import (
    SESSION,
    Spans,
    assistant,
    drive,
    init,
    metric_points,
    result,
)


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


def api_error(message_id: str = "msg_err") -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock("API Error: 404 model not found")],
        model="<synthetic>",
        message_id=message_id,
        error="unknown",
    )


def failed_result():
    return result(
        is_error=True,
        text="API Error: 404 model not found",
        api_error_status=404,
    )


@pytest.mark.asyncio
async def test_a_rejected_call_reports_the_requested_model(
    handler, span_exporter, metric_reader
):
    await drive(
        handler,
        [init(model="no-such-model"), api_error(), failed_result()],
        sdk_options=ClaudeAgentOptions(model="no-such-model"),
    )

    chat = Spans(span_exporter).one("chat no-such-model")
    assert chat.attributes["gen_ai.request.model"] == "no-such-model"
    assert "gen_ai.response.model" not in chat.attributes
    assert chat.status.status_code is trace.StatusCode.ERROR
    assert chat.attributes["error.type"] == "api_error"

    durations = metric_points(
        metric_reader, "gen_ai.client.operation.duration"
    )
    chat_durations = [
        p for p in durations if p.attributes["gen_ai.operation.name"] == "chat"
    ]
    assert len(chat_durations) == 1
    assert (
        chat_durations[0].attributes["gen_ai.request.model"] == "no-such-model"
    )
    assert "gen_ai.response.model" not in chat_durations[0].attributes


@pytest.mark.asyncio
async def test_a_real_call_after_a_rejected_one_reports_its_own_model(
    handler, span_exporter
):
    await drive(
        handler,
        [
            init(model="claude-sonnet-4-5"),
            api_error(),
            failed_result(),
            assistant(
                TextBlock("hello"),
                model="claude-sonnet-4-5-20250929",
                message_id="msg_2",
            ),
            result(),
        ],
        multi_turn=True,
    )

    spans = Spans(span_exporter)
    rejected = spans.one("chat claude-sonnet-4-5")
    assert rejected.attributes["gen_ai.request.model"] == "claude-sonnet-4-5"
    assert "gen_ai.response.model" not in rejected.attributes
    assert rejected.status.status_code is trace.StatusCode.ERROR

    answered = spans.one("chat claude-sonnet-4-5-20250929")
    assert (
        answered.attributes["gen_ai.request.model"]
        == "claude-sonnet-4-5-20250929"
    )
    assert (
        answered.attributes["gen_ai.response.model"]
        == "claude-sonnet-4-5-20250929"
    )
    assert answered.status.status_code is trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_nothing_else_knows_the_model(handler, span_exporter):
    await drive(
        handler,
        [
            SystemMessage(subtype="init", data={"session_id": SESSION}),
            api_error(),
            failed_result(),
        ],
    )

    spans = Spans(span_exporter)
    assert "gen_ai.request.model" not in spans.one("chat").attributes
    agent = spans.one("invoke_agent claude-agent-sdk")
    assert "gen_ai.request.model" not in agent.attributes
