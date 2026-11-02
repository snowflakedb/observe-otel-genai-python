# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Each ``invoke_agent`` span reports its own model calls, for the recorded CLI
sessions: the root's come from the result, a subagent's from its chat spans,
and neither includes the other's. Specific to this fork."""

from __future__ import annotations

from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from observeinc.util.genai.handler import TelemetryHandler

from .test_recordings import ATTR_IS_PARTIAL, RECORDINGS, replay
from .test_stream import USAGE_BREAKDOWNS

ROOT = "invoke_agent claude-agent-sdk"
TOTALS = ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")


@pytest.mark.parametrize(
    "multi_turn", [False, True], ids=["one-shot", "multi"]
)
@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_each_agent_reports_the_calls_directly_under_it(
    tracer_provider, span_exporter, path: Path, multi_turn: bool
):
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    # Every session was recorded with stream events on, which is what gives the
    # main thread's calls their final usage.
    await replay(
        handler,
        path,
        multi_turn=multi_turn,
        sdk_options=ClaudeAgentOptions(include_partial_messages=True),
    )
    spans = list(span_exporter.get_finished_spans())
    agents = [s for s in spans if s.name.startswith("invoke_agent ")]

    subagents_checked = 0
    for agent in agents:
        calls = [
            s
            for s in spans
            if s.name.startswith("chat ")
            and s.parent.span_id == agent.context.span_id
        ]
        if not calls:
            assert not any(
                key.startswith("gen_ai.usage.") for key in agent.attributes
            ), f"{path.stem}: {agent.name} reports usage without calls"
            continue
        for key in TOTALS:
            assert agent.attributes[key] == sum(
                c.attributes[key] for c in calls
            ), f"{path.stem}: {agent.name} {key}"
        assert set(USAGE_BREAKDOWNS).isdisjoint(agent.attributes), agent.name
        if agent.name == ROOT:
            assert ATTR_IS_PARTIAL not in agent.attributes
            continue
        subagents_checked += 1
        # A subagent's calls get no stream events, so none has final usage.
        assert agent.attributes.get(ATTR_IS_PARTIAL) is True, agent.name
        assert all(c.attributes.get(ATTR_IS_PARTIAL) is True for c in calls), (
            f"{path.stem}: {agent.name} has a call with final usage"
        )
    # bashbg backgrounds a shell command, not a subagent.
    assert subagents_checked or path.stem == "bashbg", path.stem

    for span in spans:
        assert not any("cost" in key for key in span.attributes), span.name
