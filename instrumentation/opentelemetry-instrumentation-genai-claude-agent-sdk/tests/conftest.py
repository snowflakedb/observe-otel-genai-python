# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Claude Agent SDK instrumentation tests."""
# pylint: disable=redefined-outer-name

import pytest

pytest_plugins = ["observeinc.test_util_genai.fixtures"]


@pytest.fixture
def instrument_claude_agent_sdk(
    tracer_provider, logger_provider, meter_provider
):
    """Fixture to instrument Claude Agent SDK with test providers."""
    # pylint: disable=import-outside-toplevel
    from observeinc.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture
def uninstrument_claude_agent_sdk():
    """Fixture to ensure Claude Agent SDK is uninstrumented after test."""
    yield
    # pylint: disable=import-outside-toplevel
    from observeinc.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    ClaudeAgentSDKInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def no_ambient_base_url(monkeypatch):
    """A developer running the CLI through a gateway has ANTHROPIC_BASE_URL
    set, which would change the provider every test expects."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
