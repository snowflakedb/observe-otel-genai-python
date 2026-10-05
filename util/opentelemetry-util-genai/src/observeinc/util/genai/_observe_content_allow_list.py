# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ``ALLOW_LIST`` content capturing mode. Specific to this fork.

Under ``ALLOW_LIST`` nothing is captured, as with ``NO_CONTENT``, and on top of
that every span, event and metric point an invocation records keeps only the
attributes named in ``_content_allow_list.json``. A key missing from the list,
including one added in a later release, is dropped.

Only the ``keep`` list in ``_content_allow_list.json`` is enforced; ``content``
names the keys known to carry content, for tests and docs. Observe's TypeScript
SDK applies the same ``keep`` list (``src/content/allow-list.ts`` in genai-otel),
so change both together.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping, Sequence
from functools import cache
from importlib.resources import files
from types import TracebackType
from typing import TypeVar
from weakref import WeakKeyDictionary

from observeinc.util.genai._instruments import _Instruments
from observeinc.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from observeinc.util.genai.types import ContentCapturingMode
from observeinc.util.genai.utils import fq_exception_type
from opentelemetry._logs import Logger, LogRecord, SeverityNumber
from opentelemetry.context import Context
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_TYPE,
)
from opentelemetry.trace import Link, Span, SpanContext, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import AnyValue, Attributes, AttributeValue

_ALLOW_LIST = json.loads(
    files("observeinc.util.genai")
    .joinpath("_content_allow_list.json")
    .read_text(encoding="utf-8")
)
ALLOWED_KEYS: frozenset[str] = frozenset(_ALLOW_LIST["keep"])
CONTENT_KEYS: frozenset[str] = frozenset(_ALLOW_LIST["content"])

_V = TypeVar("_V")

_logger = logging.getLogger(__name__)


def start_span(
    tracer: Tracer,
    mode: ContentCapturingMode,
    name: str,
    *,
    context: Context | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Attributes = None,
    links: Sequence[Link] | None = None,
) -> Span:
    """Start a span, filtered by the allowlist when ``mode`` is ``ALLOW_LIST``."""
    if mode is not ContentCapturingMode.ALLOW_LIST:
        return tracer.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
        )
    return _AllowListSpan(
        tracer.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=_keep_allowed(attributes),
            links=[
                Link(link.context, _keep_allowed(link.attributes))
                for link in links or ()
            ],
        )
    )


def allow_list_telemetry(
    instruments: _Instruments, logger: Logger
) -> tuple[_Instruments, Logger]:
    """The instruments and logger an ``ALLOW_LIST`` invocation records through."""
    return _allow_listed_instruments(instruments), _AllowListLogger(logger)


def allow_list_configured() -> bool:
    """Whether the environment selects ``ALLOW_LIST``. Unlike
    ``get_content_capturing_mode``, it doesn't warn about an invalid value."""
    value = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ""
    )
    return value.strip().upper() == ContentCapturingMode.ALLOW_LIST.name


@cache
def warn_completion_hook_ignored() -> None:
    _logger.warning(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is ALLOW_LIST, "
        "so the configured completion hook is not called."
    )


_INSTRUMENTS_CACHE: WeakKeyDictionary[_Instruments, _Instruments] = (
    WeakKeyDictionary()
)


def _allow_listed_instruments(instruments: _Instruments) -> _Instruments:
    filtered = _INSTRUMENTS_CACHE.get(instruments)
    if filtered is None:
        filtered = copy.copy(instruments)
        for name, instrument in vars(instruments).items():
            if isinstance(instrument, Histogram):
                setattr(filtered, name, _AllowListHistogram(instrument))
            elif isinstance(instrument, Counter):
                setattr(filtered, name, _AllowListCounter(instrument))
        _INSTRUMENTS_CACHE[instruments] = filtered
    return filtered


def _keep_allowed(attributes: Mapping[str, _V] | None) -> dict[str, _V]:
    if not attributes:
        return {}
    return {k: v for k, v in attributes.items() if k in ALLOWED_KEYS}


class _AllowListSpan(Span):
    """Wraps a span so only allowlisted attributes, and no status description
    or exception message, reach it."""

    def __init__(self, span: Span) -> None:
        self._span = span

    def end(self, end_time: int | None = None) -> None:
        self._span.end(end_time)

    def get_span_context(self) -> SpanContext:
        return self._span.get_span_context()

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        self._span.set_attributes(_keep_allowed(attributes))

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        if key in ALLOWED_KEYS:
            self._span.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        attributes: Attributes = None,
        timestamp: int | None = None,
    ) -> None:
        self._span.add_event(name, _keep_allowed(attributes), timestamp)

    def add_link(
        self, context: SpanContext, attributes: Attributes = None
    ) -> None:
        self._span.add_link(context, _keep_allowed(attributes))

    def update_name(self, name: str) -> None:
        self._span.update_name(name)

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def set_status(
        self,
        status: Status | StatusCode,
        description: str | None = None,
    ) -> None:
        code = status.status_code if isinstance(status, Status) else status
        self._span.set_status(Status(code))

    def record_exception(
        self,
        exception: BaseException,
        attributes: Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        self._span.add_event(
            "exception",
            {EXCEPTION_TYPE: fq_exception_type(exception)},
            timestamp,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # The wrapped span's own __exit__ would record the exception message.
        if exc_val is not None and self.is_recording():
            self.record_exception(exc_val)
            self.set_status(StatusCode.ERROR)
        self.end()


class _AllowListHistogram(Histogram):
    def __init__(self, histogram: Histogram) -> None:
        self._histogram = histogram

    def record(
        self,
        amount: float,
        attributes: Attributes | None = None,
        context: Context | None = None,
    ) -> None:
        self._histogram.record(amount, _keep_allowed(attributes), context)


class _AllowListCounter(Counter):
    def __init__(self, counter: Counter) -> None:
        self._counter = counter

    def add(
        self,
        amount: float,
        attributes: Attributes | None = None,
        context: Context | None = None,
    ) -> None:
        self._counter.add(amount, _keep_allowed(attributes), context)


class _AllowListLogger(Logger):
    """Emits records with allowlisted attributes only, and no body."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    # Abstract from opentelemetry-api 1.45, so it can't forward on older
    # versions; the wrapped logger still drops what it doesn't want.
    def enabled(
        self,
        *,
        context: Context | None = None,
        severity_number: SeverityNumber | None = None,
        event_name: str | None = None,
    ) -> bool:
        return True

    def emit(
        self,
        record: LogRecord | None = None,
        *,
        timestamp: int | None = None,
        observed_timestamp: int | None = None,
        context: Context | None = None,
        severity_number: SeverityNumber | None = None,
        severity_text: str | None = None,
        body: AnyValue | None = None,
        attributes: Mapping[str, AnyValue] | None = None,
        event_name: str | None = None,
        exception: BaseException | None = None,
    ) -> None:
        if record is not None:
            timestamp = record.timestamp
            observed_timestamp = record.observed_timestamp
            context = record.context
            severity_number = record.severity_number
            severity_text = record.severity_text
            attributes = record.attributes
            event_name = record.event_name
        self._logger.emit(
            LogRecord(
                timestamp=timestamp,
                observed_timestamp=observed_timestamp,
                context=context,
                severity_number=severity_number,
                severity_text=severity_text,
                attributes=_keep_allowed(attributes),
                event_name=event_name,
            )
        )
