# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ClaudeAgentSDKInstrumentor class.

Every test that instruments takes ``uninstrument_claude_agent_sdk``. The
instrumentor is a singleton and these tests call instrument()/uninstrument() as
bare statements, so without the guard an assertion failure in between would
leave the SDK wrapped around a handler bound to a shut-down TracerProvider for
the rest of the session — which test_patching.py, going through the patched
entry points, would then be running against.
"""

from observeinc.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)


def test_instrumentor_instantiation():
    """Test that the instrumentor can be instantiated."""
    instrumentor = ClaudeAgentSDKInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = ClaudeAgentSDKInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "claude-agent-sdk >= 0.2.157, < 0.3" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # First instrumentation
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # First uninstrumentation
    instrumentor.uninstrument()

    # Second instrumentation (should work)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second uninstrumentation
    instrumentor.uninstrument()


def test_multiple_instrumentation_calls(
    tracer_provider,
    logger_provider,
    meter_provider,
    uninstrument_claude_agent_sdk,
):
    """Test that multiple instrument() calls don't cause issues."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # First call
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second call (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_uninstrument_without_instrument():
    """Test that uninstrument() can be called without prior instrument()."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # This should not raise an error
    instrumentor.uninstrument()


def test_instrument_with_no_providers(uninstrument_claude_agent_sdk):
    """Test that instrument() works without explicit providers."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # Should use global providers
    instrumentor.instrument()

    # Clean up
    instrumentor.uninstrument()


def test_instrumentor_has_required_attributes():
    """Test that the instrumentor has the required methods."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "instrumentation_dependencies")
    assert callable(instrumentor.instrument)
    assert callable(instrumentor.uninstrument)
    assert callable(instrumentor.instrumentation_dependencies)
