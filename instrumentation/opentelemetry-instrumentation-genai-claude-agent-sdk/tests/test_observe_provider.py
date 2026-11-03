# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A CLI pointed at another provider's Anthropic API through
``ANTHROPIC_BASE_URL`` reports that provider. Specific to this fork."""

from __future__ import annotations

import pytest
from claude_agent_sdk.types import ClaudeAgentOptions, TextBlock

from observeinc.util.genai.handler import TelemetryHandler

from .test_stream import Spans, assistant, drive, init, metric_points, result

CORTEX_BASE_URL = (
    "https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/anthropic"
)


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


async def _run(handler, sdk_options=None):
    await drive(
        handler,
        [init(), assistant(TextBlock("hi")), result()],
        sdk_options=sdk_options,
    )


def _providers(span_exporter) -> set[str]:
    return {
        span.attributes["gen_ai.provider.name"]
        for span in Spans(span_exporter).all
    }


@pytest.mark.asyncio
async def test_ambient_base_url(
    handler, span_exporter, metric_reader, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CORTEX_BASE_URL)
    await _run(handler)

    assert _providers(span_exporter) == {"snowflake"}
    durations = metric_points(
        metric_reader, "gen_ai.client.operation.duration"
    )
    assert durations
    assert {p.attributes["gen_ai.provider.name"] for p in durations} == {
        "snowflake"
    }


@pytest.mark.asyncio
async def test_options_env_base_url(handler, span_exporter):
    await _run(
        handler,
        ClaudeAgentOptions(env={"ANTHROPIC_BASE_URL": CORTEX_BASE_URL}),
    )

    assert _providers(span_exporter) == {"snowflake"}


@pytest.mark.asyncio
async def test_routing_flag_outranks_the_base_url(
    handler, span_exporter, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CORTEX_BASE_URL)
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    await _run(handler)

    assert _providers(span_exporter) == {"aws.bedrock"}


@pytest.mark.asyncio
async def test_unknown_base_url_stays_anthropic(
    handler, span_exporter, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:4000")
    await _run(handler)

    assert _providers(span_exporter) == {"anthropic"}
