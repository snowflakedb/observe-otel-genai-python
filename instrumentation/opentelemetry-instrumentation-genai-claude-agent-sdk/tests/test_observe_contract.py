# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checks the fields LLM Explorer needs, separately from the upstream tests.

The upstream test files check that the instrumentation follows the GenAI
semantic conventions. This file checks the narrower thing our product depends
on: that the specific fields LLM Explorer reads are actually present on the
spans we emit. Keeping it in its own file means a rebase never has to
reconcile our product requirements with upstream's test expectations.

Add a test here when LLM Explorer starts depending on a field, and a
``test_observe_*`` file next to the upstream tests when we carry a fix or an
intentional difference from upstream.

Most of the contract is already exercised incidentally by test_stream.py: the
conversation id is asserted on every span of several runs, cost comes off
``modelUsage``, and the span hierarchy is what most of that file is about. What
nothing there covers is ``gen_ai.agent.name``, because every test finds an agent
span by matching its *name string* — so dropping the attribute while keeping the
span-name suffix leaves the whole suite green and LLM Explorer blank.

Time to first token is not asserted here: it looks unimplemented rather than
untested, since ``_run.py`` never reads ``ttft_ms`` off the result message.
``gen_ai.client.operation.time_to_first_chunk`` is recorded from stream-event
arrivals, but that is a metric rather than the span field LLM Explorer reads.
"""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from observeinc.instrumentation.genai.claude_agent_sdk._run import AgentRun
from observeinc.util.genai.handler import TelemetryHandler

SESSION = "session-contract"


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


def _assistant(
    *blocks: Any, message_id: str, parent: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks),
        model="claude-sonnet-4-5",
        message_id=message_id,
        parent_tool_use_id=parent,
    )


@pytest.mark.asyncio
async def test_every_span_that_names_an_agent_carries_the_attribute(
    handler, span_exporter
):
    """``gen_ai.agent.name`` is what LLM Explorer groups a conversation by.

    The span name is not a substitute: it is a display string, and a reader that
    parses the agent back out of it is reading a label rather than a field.
    """
    messages = [
        SystemMessage(
            subtype="init",
            data={"session_id": SESSION, "model": "claude-sonnet-4-5"},
        ),
        _assistant(
            ToolUseBlock(
                id="call-task",
                name="Task",
                input={"subagent_type": "code-reviewer"},
            ),
            message_id="m1",
        ),
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="task-1",
            description="review",
            uuid="u",
            session_id=SESSION,
            tool_use_id="call-task",
        ),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="call-task", content="launched")
            ]
        ),
        _assistant(
            ToolUseBlock(id="call-sub", name="Read", input={}),
            message_id="m_sub",
            parent="call-task",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="call-sub", content="body")],
            parent_tool_use_id="call-task",
        ),
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="task-1",
            status="completed",
            output_file="",
            summary="looks fine",
            uuid="u",
            session_id=SESSION,
            tool_use_id="call-task",
        ),
        _assistant(TextBlock("all reviewed"), message_id="m2"),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=SESSION,
            result="all reviewed",
        ),
    ]

    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler)
    run.submit("review the ledger")
    async for _ in run.iterate(stream):
        pass

    spans = list(span_exporter.get_finished_spans())
    named = {
        span.name: span
        for span in spans
        if span.name.startswith(("invoke_agent", "execute_tool"))
    }
    assert set(named) == {
        "invoke_agent claude-agent-sdk",
        "invoke_agent code-reviewer",
        "execute_tool Task",
        "execute_tool Read",
    }
    assert (
        named["invoke_agent claude-agent-sdk"].attributes["gen_ai.agent.name"]
        == "claude-agent-sdk"
    )
    assert (
        named["invoke_agent code-reviewer"].attributes["gen_ai.agent.name"]
        == "code-reviewer"
    )
    # On execute_tool the attribute is the agent that ran the tool, so a
    # delegated call is attributed to the subagent rather than the run.
    assert (
        named["execute_tool Task"].attributes["gen_ai.agent.name"]
        == "claude-agent-sdk"
    )
    assert (
        named["execute_tool Read"].attributes["gen_ai.agent.name"]
        == "code-reviewer"
    )
    # Every span in the trace is reachable by conversation, which is how the
    # Explorer assembles one.
    for span in spans:
        assert span.attributes["gen_ai.conversation.id"] == SESSION
