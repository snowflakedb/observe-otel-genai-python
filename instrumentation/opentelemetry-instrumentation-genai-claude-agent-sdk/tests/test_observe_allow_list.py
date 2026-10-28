# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A failed run in the ``ALLOW_LIST`` capture mode keeps only allowlisted
attributes and no error text. Specific to this fork."""

from __future__ import annotations

import pytest
from claude_agent_sdk.types import TextBlock, ToolUseBlock

from observeinc.instrumentation.genai.claude_agent_sdk._run import AgentRun
from observeinc.test_util_genai._observe_allow_list import (
    assert_allow_listed,
)
from observeinc.util.genai.handler import TelemetryHandler
from opentelemetry import trace

from .test_stream import USAGE, assistant, drive, init, result

SECRET = "secret-content"


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider, monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "ALLOW_LIST"
    )
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


def _assert_allow_listed(span_exporter, log_exporter, metric_reader):
    spans = span_exporter.get_finished_spans()
    assert spans
    assert_allow_listed(
        spans,
        log_exporter.get_finished_logs(),
        metric_reader.get_metrics_data(),
    )
    return spans


@pytest.mark.asyncio
async def test_an_api_error_result(
    handler, span_exporter, log_exporter, metric_reader
):
    await drive(
        handler,
        [
            init(),
            assistant(TextBlock(SECRET)),
            result(
                is_error=True,
                text=f"API Error: 400 {SECRET}",
                api_error_status=400,
                usage=USAGE,
            ),
        ],
    )

    spans = _assert_allow_listed(span_exporter, log_exporter, metric_reader)
    failed = [
        s for s in spans if s.status.status_code is trace.StatusCode.ERROR
    ]
    assert {s.name for s in failed} == {
        "chat claude-sonnet-4-5",
        "invoke_agent claude-agent-sdk",
    }
    for span in failed:
        assert span.attributes["error.type"] == "api_error"


@pytest.mark.asyncio
async def test_a_stream_failure_with_work_in_flight(
    handler, span_exporter, log_exporter, metric_reader
):
    async def stream():
        yield init()
        yield assistant(
            ToolUseBlock(id="bash", name="Bash", input={"command": SECRET})
        )
        raise RuntimeError(SECRET)

    run = AgentRun(handler)
    run.submit(SECRET)
    with pytest.raises(RuntimeError):
        async for _ in run.iterate(stream):
            pass

    spans = _assert_allow_listed(span_exporter, log_exporter, metric_reader)
    assert {s.name: s.status.status_code for s in spans} == {
        "chat claude-sonnet-4-5": trace.StatusCode.ERROR,
        "execute_tool Bash": trace.StatusCode.ERROR,
        "invoke_agent claude-agent-sdk": trace.StatusCode.ERROR,
    }
