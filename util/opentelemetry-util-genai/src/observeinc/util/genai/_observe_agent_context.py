# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry ``gen_ai.agent.name`` on the OTel Context. Specific to this fork.

Semconv puts the agent name on ``invoke_agent`` and ``execute_tool`` only, but
Observe groups the inference, embeddings and retrieval calls an agent makes, and
their metrics, by agent. An agent invocation publishes its name here, the way
``_conversation_context`` carries the conversation id, so those calls record it
even when another instrumentation package creates them. Callers reach this
through ``handler.*(agent_name=...)``.
"""

from __future__ import annotations

from opentelemetry.context import (
    Context,
    create_key,
    get_value,
    set_value,
)

AGENT_NAME_KEY = create_key("observe.util.genai.agent_name")


def get_ambient_agent_name(context: Context | None = None) -> str | None:
    value = get_value(AGENT_NAME_KEY, context=context)
    return value if isinstance(value, str) else None


def with_agent_name(
    agent_name: str | None, context: Context | None = None
) -> Context:
    return set_value(AGENT_NAME_KEY, agent_name, context=context)
