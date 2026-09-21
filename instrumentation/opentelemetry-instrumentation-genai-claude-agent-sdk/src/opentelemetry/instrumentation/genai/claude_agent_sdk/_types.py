# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shapes of the CLI payloads the SDK passes through as plain dicts.

``StreamEvent.event``, ``SystemMessage.data``, ``TaskUpdatedMessage.patch`` and
``ToolUseBlock.input`` are typed ``dict[str, Any]`` by the SDK: they carry the
CLI's raw JSON, which the SDK does not model. These declarations record the
fields this instrumentation reads, so the assumptions are written down and
checked rather than implied.

Every key is optional. The CLI sends what it sends, and an absent field means
the corresponding attribute is left off the span.
"""

from __future__ import annotations

from typing import TypedDict


class OutputTokenDetails(TypedDict, total=False):
    thinking_tokens: int


class Usage(TypedDict, total=False):
    """Anthropic token counts.

    Arrives twice per call: as a mid-stream snapshot on the assistant message,
    then as the final counts on the ``message_delta`` stream event.
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens_details: OutputTokenDetails


class MessageDelta(TypedDict, total=False):
    stop_reason: str


class StreamEventPayload(TypedDict, total=False):
    """``StreamEvent.event`` — one raw Anthropic streaming event."""

    type: str
    usage: Usage
    delta: MessageDelta


class InitData(TypedDict, total=False):
    """``SystemMessage.data`` for the ``init`` subtype."""

    session_id: str
    model: str


class TaskPatch(TypedDict, total=False):
    """``TaskUpdatedMessage.patch`` — the fields that changed on a task."""

    status: str


class TaskToolInput(TypedDict, total=False):
    """``ToolUseBlock.input`` for the Task tool that launches a subagent."""

    subagent_type: str
    agent_type: str
    description: str
    prompt: str
    model: str


class UsageTotals(TypedDict):
    """Per-model usage summed across an invocation by ``_fold_model_usage``."""

    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    cost_usd: float
