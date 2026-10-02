# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A subagent's first chat span reports the prompt it was launched with.

The CLI forwards that prompt for a foreground subagent but not for a background
one. Specific to this fork.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observeinc.util.genai.handler import TelemetryHandler

from .test_recordings import RECORDINGS, replay

BY_NAME = {path.stem: path for path in RECORDINGS}

# bgfix passes run_in_background; the other background sessions are
# backgrounded by the CLI without it, and in bg4 the subagent speaks in the
# middle of a main-thread call.
BACKGROUND = ["bgfix", "base", "bg2", "bg3", "bg4"]
FOREGROUND = ["fix1", "fix2", "after"]


def _handler(tracer_provider, monkeypatch, capture: str) -> TelemetryHandler:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture
    )
    return TelemetryHandler(tracer_provider=tracer_provider)


def _launch_prompt(path: Path) -> str:
    for line in path.read_text().splitlines():
        message = json.loads(line)
        if message["type"] != "assistant":
            continue
        for block in message["message"]["content"]:
            if block.get("type") == "tool_use" and block["name"] == "Agent":
                return block["input"]["prompt"]
    raise AssertionError(f"{path.stem}: no Agent call")


def _subagent_chats(spans):
    subagent_ids = {
        span.context.span_id
        for span in spans
        if span.name.startswith("invoke_agent")
        and span.name != "invoke_agent claude-agent-sdk"
    }
    chats = [
        span
        for span in spans
        if span.name.startswith("chat ")
        and span.parent.span_id in subagent_ids
    ]
    assert chats, "no subagent chat spans"
    return sorted(chats, key=lambda span: span.start_time)


def _user_texts(span) -> list[str]:
    return [
        part["content"]
        for message in json.loads(span.attributes["gen_ai.input.messages"])
        if message["role"] == "user"
        for part in message["parts"]
        if part["type"] == "text"
    ]


@pytest.mark.parametrize(
    "multi_turn", [False, True], ids=["one-shot", "multi"]
)
@pytest.mark.parametrize("name", BACKGROUND + FOREGROUND)
@pytest.mark.asyncio
async def test_the_first_subagent_call_reports_its_prompt_once(
    tracer_provider, span_exporter, monkeypatch, name: str, multi_turn: bool
):
    path = BY_NAME[name]
    handler = _handler(tracer_provider, monkeypatch, "SPAN_ONLY")
    await replay(handler, path, multi_turn=multi_turn)

    chats = _subagent_chats(span_exporter.get_finished_spans())
    prompt = _launch_prompt(path)
    assert _user_texts(chats[0]) == [prompt]
    for later in chats[1:]:
        assert _user_texts(later).count(prompt) == 1


@pytest.mark.parametrize("name", BACKGROUND)
@pytest.mark.asyncio
async def test_the_prompt_is_absent_without_content_capture(
    tracer_provider, span_exporter, monkeypatch, name: str
):
    handler = _handler(tracer_provider, monkeypatch, "NO_CONTENT")
    await replay(handler, BY_NAME[name], multi_turn=False)

    for span in _subagent_chats(span_exporter.get_finished_spans()):
        assert "gen_ai.input.messages" not in span.attributes
