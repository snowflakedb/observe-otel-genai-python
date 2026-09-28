# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics add up to the token counts on the spans, for
the recorded CLI sessions. Specific to this fork."""

from __future__ import annotations

from pathlib import Path

import pytest

from opentelemetry.test_util_genai._observe_token_metrics import (
    assert_token_metrics_match_spans,
)
from opentelemetry.util.genai.handler import TelemetryHandler

from .test_recordings import RECORDINGS, replay


@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_a_recorded_session(
    tracer_provider, meter_provider, span_exporter, metric_reader, path: Path
):
    handler = TelemetryHandler(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    await replay(handler, path, multi_turn=False)
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    for token_type in usage:
        assert usage[token_type], f"{path.stem}: no {token_type}"
