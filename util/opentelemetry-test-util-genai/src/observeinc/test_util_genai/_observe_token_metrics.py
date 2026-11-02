# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics this fork records ahead of the pinned semconv.

The pinned registry predates semantic-conventions-genai#374, so weaver reports
these metrics and their ``gen_ai.token.modality`` attribute as unknown.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_INFERENCE_TOKEN_METRICS = frozenset(
    {
        "gen_ai.client.inference.usage.input_tokens",
        "gen_ai.client.inference.usage.output_tokens",
        "gen_ai.client.inference.usage.cache_read.input_tokens",
        "gen_ai.client.inference.usage.cache_write.input_tokens",
        "gen_ai.client.inference.usage.reasoning.output_tokens",
        "gen_ai.client.inference.operation.input_tokens",
        "gen_ai.client.inference.operation.output_tokens",
    }
)


def without_inference_token_metrics(metrics: Iterable[Metric]) -> list[Metric]:
    """The metrics upstream records, for tests that count them."""
    return [m for m in metrics if m.name not in _INFERENCE_TOKEN_METRICS]


def is_ahead_of_pinned_semconv(violation: dict[str, Any]) -> bool:
    if violation.get("signal_name") not in _INFERENCE_TOKEN_METRICS:
        return False
    if violation.get("id") == "missing_metric":
        return True
    context = violation.get("context") or {}
    return (
        violation.get("id") == "missing_attribute"
        and context.get("attribute_key") == "gen_ai.token.modality"
    )


_TOKEN_OPERATIONS = frozenset(
    {"chat", "text_completion", "generate_content", "embeddings"}
)
_MODALITIES = ("text", "image", "audio")
# Each counter, the span attribute with its total, and the per-modality ones.
_USAGE_COUNTERS: dict[str, tuple[str, str | None]] = {
    "gen_ai.client.inference.usage.input_tokens": (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.{}.input_tokens",
    ),
    "gen_ai.client.inference.usage.output_tokens": (
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.{}.output_tokens",
    ),
    "gen_ai.client.inference.usage.cache_read.input_tokens": (
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.{}.cache_read.input_tokens",
    ),
    "gen_ai.client.inference.usage.cache_write.input_tokens": (
        "gen_ai.usage.cache_write.input_tokens",
        None,
    ),
    "gen_ai.client.inference.usage.reasoning.output_tokens": (
        "gen_ai.usage.reasoning.output_tokens",
        None,
    ),
}
_OPERATION_HISTOGRAMS = {
    "gen_ai.client.inference.operation.input_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.client.inference.operation.output_tokens": "gen_ai.usage.output_tokens",
}


def assert_token_metrics_match_spans(
    span_exporter: InMemorySpanExporter, metric_reader: InMemoryMetricReader
) -> dict[str, dict[str, int]]:
    """Assert the inference token metrics add up to the token counts on the
    inference and embeddings spans, split by modality the same way.

    Returns each counter's value per modality, for a test to check the token
    types it expects are there.
    """
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data
    metrics = {
        m.name: m
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    token_spans = [
        s.attributes or {}
        for s in span_exporter.get_finished_spans()
        if (s.attributes or {}).get("gen_ai.operation.name")
        in _TOKEN_OPERATIONS
    ]
    assert token_spans, "no inference or embeddings spans"

    usage: dict[str, dict[str, int]] = {}
    for name, (total_key, modality_key) in _USAGE_COUNTERS.items():
        expected: Counter[str] = Counter()
        for attributes in token_spans:
            expected.update(_split(attributes, total_key, modality_key))
        actual: Counter[str] = Counter()
        for point in _points(metrics, name):
            actual[str(point.attributes["gen_ai.token.modality"])] += (
                point.value
            )
        assert actual == expected, (
            f"{name}: {dict(actual)} != {dict(expected)}"
        )
        usage[name.removeprefix("gen_ai.client.inference.usage.")] = dict(
            actual
        )

    for name, key in _OPERATION_HISTOGRAMS.items():
        counts = [int(a[key]) for a in token_spans if a.get(key) is not None]
        points = _points(metrics, name)
        assert (
            sum(p.count for p in points),
            sum(p.sum for p in points),
        ) == (len(counts), sum(counts)), name
    return usage


def _points(metrics: dict[str, Metric], name: str) -> list[Any]:
    return list(metrics[name].data.data_points) if name in metrics else []


def _split(
    attributes: Mapping[str, Any], total_key: str, modality_key: str | None
) -> dict[str, int]:
    total = attributes.get(total_key)
    if not total:
        return {}
    parts = {
        m: int(attributes[modality_key.format(m)])
        for m in _MODALITIES
        if modality_key and attributes.get(modality_key.format(m))
    }
    remainder = int(total) - sum(parts.values())
    if remainder < 0:
        return {"unknown": int(total)}
    if remainder:
        parts["unknown"] = remainder
    return parts
