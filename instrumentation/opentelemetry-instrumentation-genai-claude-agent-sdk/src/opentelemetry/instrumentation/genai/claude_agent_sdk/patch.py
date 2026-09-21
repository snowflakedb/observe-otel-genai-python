# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wraps the SDK's stream-producing entry points.

Two functions are wrapped, and they define different agent-invocation
boundaries:

* ``query()`` — one call is one invocation; the span ends with the stream.
* ``ClaudeSDKClient.receive_messages()`` — continuous over many turns, so the
  boundary is a result that answers a prompt the caller submitted.

``ClaudeSDKClient.receive_response()`` is deliberately **not** wrapped. It is a
convenience method whose whole body iterates ``receive_messages()`` and stops at
the first result, so wrapping it as well would build the tree twice from the
same messages — measured as two agent spans and two chat spans per turn, in
separate traces. Wrapping the one primitive covers both callers.

Each wrapper returns an async iterator yielding the same messages in the same
order. Nothing is injected into the caller's options, and no message is altered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.instrumentation.genai.claude_agent_sdk._run import AgentRun
from opentelemetry.util.genai.handler import TelemetryHandler

if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition, ClaudeAgentOptions


def _agent_name(options: ClaudeAgentOptions | None) -> str | None:
    """Use the single agent definition's name when the caller declared exactly
    one, since that is unambiguously the agent being invoked."""
    agents: dict[str, AgentDefinition] | None = getattr(
        options, "agents", None
    )
    if isinstance(agents, dict) and len(agents) == 1:
        return next(iter(agents))
    return None


def wrap_query(handler: TelemetryHandler) -> Any:
    """Build a replacement for ``claude_agent_sdk.query``."""

    def traced_query(
        wrapped: Any, _instance: Any, args: Any, kwargs: Any
    ) -> Any:
        options = kwargs.get("options")
        prompt = kwargs.get("prompt")
        run = AgentRun(
            handler,
            sdk_options=options,
            agent_name=_agent_name(options),
        )
        return run.iterate(lambda: wrapped(*args, **kwargs), prompt)

    return traced_query


def wrap_receive_messages(handler: TelemetryHandler) -> Any:
    """Build a replacement for ``ClaudeSDKClient.receive_messages``."""

    def traced(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        run = AgentRun(
            handler,
            sdk_options=getattr(instance, "options", None),
            agent_name=_agent_name(getattr(instance, "options", None)),
            multi_turn=True,
        )
        return run.iterate(lambda: wrapped(*args, **kwargs))

    return traced
