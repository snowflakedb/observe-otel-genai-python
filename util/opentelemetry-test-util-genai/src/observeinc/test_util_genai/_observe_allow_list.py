# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks for telemetry recorded in the ``ALLOW_LIST`` capture mode. Specific
to this fork."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from observeinc.util.genai._observe_content_allow_list import ALLOWED_KEYS
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk.metrics.export import MetricsData
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_TYPE,
)


def attribute_keys(
    spans: Sequence[ReadableSpan],
    logs: Sequence[ReadableLogRecord] = (),
    metrics: MetricsData | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(where, key)`` for every attribute on the given telemetry."""
    for span in spans:
        for key in span.attributes or {}:
            yield f"span {span.name!r}", key
        for event in span.events:
            for key in event.attributes or {}:
                yield f"event {event.name!r} on {span.name!r}", key
        for link in span.links:
            for key in link.attributes or {}:
                yield f"link on {span.name!r}", key
    for log in logs:
        for key in log.log_record.attributes or {}:
            yield f"log {log.log_record.event_name!r}", key
    for resource_metrics in metrics.resource_metrics if metrics else ():
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for point in metric.data.data_points:
                    for key in point.attributes or {}:
                        yield f"metric {metric.name!r}", key


def assert_allow_listed(
    spans: Sequence[ReadableSpan],
    logs: Sequence[ReadableLogRecord] = (),
    metrics: MetricsData | None = None,
) -> None:
    """Fail unless the telemetry carries only allowlisted attributes, no
    status description, no exception detail and no log body."""
    problems = [
        f"{where} carries {key}"
        for where, key in attribute_keys(spans, logs, metrics)
        if key not in ALLOWED_KEYS
    ]
    for span in spans:
        if span.status.description:
            problems.append(f"span {span.name!r} has a status description")
        for event in span.events:
            if event.name == "exception" and set(event.attributes or {}) != {
                EXCEPTION_TYPE
            }:
                problems.append(
                    f"exception event on {span.name!r} carries more than its type"
                )
    for log in logs:
        if log.log_record.body is not None:
            problems.append(f"log {log.log_record.event_name!r} has a body")
    assert not problems, "\n".join(problems)
