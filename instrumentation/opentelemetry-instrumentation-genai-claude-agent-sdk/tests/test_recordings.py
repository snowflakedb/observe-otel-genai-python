# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Replays the recorded CLI sessions in tests/recordings.

The rest of the suite drives synthetic message sequences, which prove the code
does what it was written to do but not that the CLI sends what the code expects.
These are real streams, so they are where a wrong assumption about ordering
shows up.

The assertions are invariants rather than a stored span tree: a snapshot would
have to be regenerated on every intentional change, which is how a snapshot
stops being read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk._internal.message_parser import parse_message

from opentelemetry.instrumentation.genai.claude_agent_sdk._run import AgentRun
from opentelemetry.util.genai.handler import TelemetryHandler

RECORDINGS = sorted(
    Path(__file__).parent.joinpath("recordings").glob("*.jsonl")
)

ATTR_IS_PARTIAL = "claude_agent_sdk.usage.is_partial"
ATTR_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    # Local rather than in conftest.py, which comes from upstream.
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


def load(path: Path) -> list[Any]:
    """The messages the SDK would have handed the caller, in order."""
    messages = []
    for line in path.read_text().splitlines():
        if line.strip():
            messages.append(parse_message(json.loads(line)))
    return messages


async def replay(
    handler: TelemetryHandler, path: Path, *, multi_turn: bool
) -> None:
    messages = load(path)

    async def stream():
        for message in messages:
            yield message

    run = AgentRun(handler, multi_turn=multi_turn)
    run.submit("replay")
    async for _ in run.iterate(stream):
        pass


def assert_recordings_exist() -> None:
    assert RECORDINGS, "no recordings found; tests/recordings is empty"


@pytest.mark.parametrize(
    "multi_turn", [False, True], ids=["one-shot", "multi"]
)
@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_a_recorded_session_builds_a_well_formed_tree(
    handler, span_exporter, path: Path, multi_turn: bool
):
    assert_recordings_exist()
    await replay(handler, path, multi_turn=multi_turn)
    spans = list(span_exporter.get_finished_spans())
    assert spans, f"{path.stem}: no spans"

    by_id = {span.context.span_id: span for span in spans}
    for span in spans:
        assert span.end_time is not None, f"{span.name} never ended"
        if span.parent is None:
            continue
        parent = by_id.get(span.parent.span_id)
        if parent is None:
            # Parented outside this run only when nothing here opened it.
            continue
        # A child rendering past its parent means teardown closed them in the
        # wrong order.
        assert span.end_time <= parent.end_time, (
            f"{path.stem}: {span.name} ends after its parent {parent.name}"
        )

    roots = [
        span
        for span in spans
        if span.name.startswith("invoke_agent")
        and (span.parent is None or span.parent.span_id not in by_id)
    ]
    assert roots, f"{path.stem}: no root invoke_agent span"
    assert all(root.name == "invoke_agent claude-agent-sdk" for root in roots)


@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_a_recorded_chat_span_with_final_usage_also_has_chunk_timings(
    handler, span_exporter, path: Path
):
    """A call's final usage and its chunk timings both come from stream events.

    So a span that is not marked partial had events flowing, and must carry the
    timings too. The reverse does not hold: a call whose events were flowing
    when the stream ended is timed and still partial, having never received the
    ``message_delta`` that closes its usage.
    """
    assert_recordings_exist()
    await replay(handler, path, multi_turn=False)
    chats = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("chat ")
    ]
    assert chats, f"{path.stem}: no chat spans"
    for chat in chats:
        if chat.attributes.get(ATTR_IS_PARTIAL):
            continue
        assert ATTR_TIME_TO_FIRST_CHUNK in chat.attributes, (
            f"{path.stem}: {chat.name} reports final usage but no chunk timing"
        )


@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_no_recorded_chunk_is_dropped_before_its_span_opens(
    handler, span_exporter, metric_reader, path: Path
):
    """Every content block delta the CLI sent is timed.

    A chat span has to be open before its call's first delta arrives, which is
    why it opens on ``message_start`` rather than on the assistant message that
    follows a whole content block later. Anything that opens it later silently
    drops the deltas in between, and the chunk timings are the only place that
    shows.

    The first delta of each call is timed as the time to first chunk, the rest
    as time per output chunk, so the two counts together account for all of
    them.
    """
    assert_recordings_exist()
    sent = sum(
        1
        for line in path.read_text().splitlines()
        if line.strip()
        and (message := json.loads(line)).get("type") == "stream_event"
        and (message.get("event") or {}).get("type") == "content_block_delta"
        # Subagent calls get no stream events of their own to time.
        and not message.get("parent_tool_use_id")
    )
    await replay(handler, path, multi_turn=False)

    first = sum(
        1
        for span in span_exporter.get_finished_spans()
        if ATTR_TIME_TO_FIRST_CHUNK in span.attributes
    )
    rest = sum(
        point.count
        for resource in (
            metric_reader.get_metrics_data().resource_metrics or []
        )
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name.endswith("time_per_output_chunk")
        for point in metric.data.data_points
    )
    assert first + rest == sent, (
        f"{path.stem}: CLI sent {sent} deltas, {first + rest} were timed"
    )


@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
def test_a_recorded_task_update_carries_its_status_only_in_the_patch(
    path: Path,
):
    """The CLI leaves ``status`` unset on task_updated and puts it in ``patch``.

    _on_task_updated reads both. This records the real shape so a re-recording
    against a newer CLI shows up as a failure here; it does not exercise the
    read itself, because in every recorded session the notification arrives on
    the next line and closes the tool before the status is needed.
    """
    assert_recordings_exist()
    updates = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and '"task_updated"' in line
    ]
    for update in updates:
        assert update.get("status") is None, (
            f"{path.stem}: task_updated has a top-level status; the test "
            f"builder in test_stream.py may now be the more realistic shape"
        )
        assert (update.get("patch") or {}).get("status"), (
            f"{path.stem}: task_updated patch carries no status"
        )
