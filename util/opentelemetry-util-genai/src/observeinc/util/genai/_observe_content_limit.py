# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A size cap on each captured content attribute. Specific to this fork.

``OBSERVE_GENAI_MAX_CONTENT_LENGTH`` sets the cap in characters, and ``0``
turns it off. A value over the cap keeps its JSON valid: only the text inside
messages, tool arguments and results, and the like is shortened, while roles,
types, ids and names are kept. A value that isn't JSON is cut in the middle.
A trimmed attribute ``<key>`` is joined by ``<key>.truncated`` and
``<key>.original_size_bytes``. Other Observe SDKs trim the same way, so keep
them in sync.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Final, cast

from observeinc.util.genai._invocation import _GEN_AI_PROMPT_VARIABLE_PREFIX
from observeinc.util.genai.utils import gen_ai_json_dumps
from opentelemetry._logs import Logger, LogRecord, SeverityNumber
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    SpanKind,
    Tracer,
    use_span,
)
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AnyValue, Attributes, AttributeValue

OBSERVE_GENAI_MAX_CONTENT_LENGTH: Final = "OBSERVE_GENAI_MAX_CONTENT_LENGTH"
DEFAULT_MAX_CONTENT_LENGTH: Final = 1024 * 1024

# In these and in retrieval documents only the strings under _CONTENT_FIELDS
# are content. In the other content values every string is.
_MESSAGE_KEYS = frozenset(
    {
        GenAI.GEN_AI_INPUT_MESSAGES,
        GenAI.GEN_AI_OUTPUT_MESSAGES,
        GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
    }
)
_CONTENT_KEYS = _MESSAGE_KEYS | {
    GenAI.GEN_AI_TOOL_DEFINITIONS,
    GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
    GenAI.GEN_AI_TOOL_CALL_RESULT,
    GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT,
    GenAI.GEN_AI_RETRIEVAL_DOCUMENTS,
}
_CONTENT_FIELDS = frozenset(
    {
        "content",
        "arguments",
        "response",
        "text",
        "server_tool_call",
        "server_tool_call_response",
    }
)
_ELLIPSIS = "\u2026"
_ELLIPSIS_SIZE = len(gen_ai_json_dumps(_ELLIPSIS)) - 2

_logger = logging.getLogger(__name__)


def _max_content_length() -> int:
    value = os.environ.get(OBSERVE_GENAI_MAX_CONTENT_LENGTH, "").strip()
    if not value:
        return DEFAULT_MAX_CONTENT_LENGTH
    try:
        limit = int(value)
    except ValueError:
        limit = -1
    if limit < 0:
        _logger.warning(
            "%s is not a valid value for %s. Defaulting to %s.",
            value,
            OBSERVE_GENAI_MAX_CONTENT_LENGTH,
            DEFAULT_MAX_CONTENT_LENGTH,
        )
        return DEFAULT_MAX_CONTENT_LENGTH
    return limit


def limit_content_length(
    tracer: Tracer, logger: Logger
) -> tuple[Tracer, Logger]:
    """The tracer and logger a handler records content through."""
    limit = _max_content_length()
    if limit == 0:
        return tracer, logger
    return _ContentLimitTracer(tracer, limit), _ContentLimitLogger(
        logger, limit
    )


def _limit_attributes(
    attributes: Mapping[str, Any], limit: int
) -> dict[str, Any]:
    limited = dict(attributes)
    for key, value in attributes.items():
        if key not in _CONTENT_KEYS and not key.startswith(
            _GEN_AI_PROMPT_VARIABLE_PREFIX
        ):
            continue
        encoded = value if isinstance(value, str) else gen_ai_json_dumps(value)
        if len(encoded) <= limit:
            continue
        trimmed = _trim(key, encoded, limit)
        # Even an empty list can be over a tiny limit.
        if len(trimmed) > limit:
            del limited[key]
            continue
        limited[key] = (
            trimmed if isinstance(value, str) else json.loads(trimmed)
        )
        limited[f"{key}.truncated"] = True
        limited[f"{key}.original_size_bytes"] = len(encoded.encode("utf-8"))
    return limited


def _trim(key: str, encoded: str, limit: int) -> str:
    try:
        data = json.loads(encoded)
    except ValueError:
        data = None
    if not isinstance(data, (list, dict)):
        return _trim_middle(encoded, limit)
    data = cast("list[Any] | dict[str, Any]", data)
    if key == GenAI.GEN_AI_TOOL_DEFINITIONS and isinstance(data, list):
        return _trim_tool_definitions(data, limit)
    return _trim_structured(
        data,
        encoded,
        limit,
        whole_value_is_content=key not in _MESSAGE_KEYS
        and key != GenAI.GEN_AI_RETRIEVAL_DOCUMENTS,
        drop_from_end=key not in _MESSAGE_KEYS,
    )


def _trim_middle(text: str, limit: int) -> str:
    """Keeps the first 70% and the last 30% of what fits."""
    overhead = len(f"\n...[truncated {len(text)} chars]...\n")
    if limit <= overhead:
        return text[:limit]
    head = (limit - overhead) * 7 // 10
    tail = limit - overhead - head
    removed = len(text) - head - tail
    return f"{text[:head]}\n...[truncated {removed} chars]...\n{text[len(text) - tail :]}"


def _trim_tool_definitions(definitions: list[Any], limit: int) -> str:
    """Drops whole fields rather than cut into a schema: ``parameters``, then
    ``description``, then definitions from the end."""
    trimmed = gen_ai_json_dumps(definitions)
    for field in ("parameters", "description"):
        for definition in definitions:
            if isinstance(definition, dict):
                cast("dict[str, Any]", definition).pop(field, None)
        trimmed = gen_ai_json_dumps(definitions)
        if len(trimmed) <= limit:
            return trimmed
    while definitions and len(trimmed) > limit:
        definitions.pop()
        trimmed = gen_ai_json_dumps(definitions)
    return trimmed


def _trim_structured(
    data: list[Any] | dict[str, Any],
    encoded: str,
    limit: int,
    *,
    whole_value_is_content: bool,
    drop_from_end: bool,
) -> str:
    """Shortens content strings in proportion to their size, then drops whole
    list items (the oldest messages, or the last documents) if the rest of the
    value alone is over the limit."""
    leaves: list[tuple[Any, Any]] = []
    _collect_content_leaves(data, whole_value_is_content, leaves)
    # Measured as encoded, since escaping can make a string longer.
    total = sum(
        len(gen_ai_json_dumps(parent[key])) - 2 for parent, key in leaves
    )
    budget = limit - (len(encoded) - total) - _ELLIPSIS_SIZE * len(leaves)
    if total and budget > 0:
        for parent, key in leaves:
            text: str = parent[key]
            keep = len(text) * budget // total
            if keep < len(text):
                parent[key] = text[:keep] + _ELLIPSIS
    trimmed = gen_ai_json_dumps(data)

    # Escaping and rounding can still push it over.
    for _ in range(500):
        if len(trimmed) <= limit or not leaves:
            break
        parent, key = max(leaves, key=lambda leaf: len(leaf[0][leaf[1]]))
        text = parent[key]
        if len(text) <= 1:
            break
        parent[key] = text[: len(text) // 2]
        trimmed = gen_ai_json_dumps(data)

    if isinstance(data, list):
        while data and len(trimmed) > limit:
            data.pop(-1 if drop_from_end else 0)
            trimmed = gen_ai_json_dumps(data)
    return trimmed


def _collect_content_leaves(
    node: Any, in_content: bool, leaves: list[tuple[Any, Any]]
) -> None:
    parent: Any = node
    items: Iterable[tuple[Any, Any]]
    if isinstance(node, list):
        items = enumerate(cast("list[Any]", node))
    elif isinstance(node, dict):
        items = cast("dict[Any, Any]", node).items()
    else:
        return
    for key, value in items:
        content = in_content or key in _CONTENT_FIELDS
        if isinstance(value, str):
            if content:
                leaves.append((parent, key))
        else:
            _collect_content_leaves(value, content, leaves)


class _ContentLimitTracer(Tracer):
    def __init__(self, tracer: Tracer, limit: int) -> None:
        self._tracer = tracer
        self._limit = limit

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        return _ContentLimitSpan(
            self._tracer.start_span(
                name,
                context,
                kind,
                attributes,
                links,
                start_time,
                record_exception,
                set_status_on_exception,
            ),
            self._limit,
        )

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        span = self.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ):
            yield span


class _ContentLimitSpan(Span):
    def __init__(self, span: Span, limit: int) -> None:
        self._span = span
        self._limit = limit

    def __getattr__(self, name: str) -> object:
        return getattr(self._span, name)

    def end(self, end_time: int | None = None) -> None:
        self._span.end(end_time)

    def get_span_context(self) -> SpanContext:
        return self._span.get_span_context()

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        self._span.set_attributes(_limit_attributes(attributes, self._limit))

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.set_attributes({key: value})

    def add_event(
        self,
        name: str,
        attributes: Attributes = None,
        timestamp: int | None = None,
    ) -> None:
        self._span.add_event(name, attributes, timestamp)

    def add_link(
        self, context: SpanContext, attributes: Attributes = None
    ) -> None:
        self._span.add_link(context, attributes)

    def update_name(self, name: str) -> None:
        self._span.update_name(name)

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def set_status(
        self,
        status: Status | StatusCode,
        description: str | None = None,
    ) -> None:
        self._span.set_status(status, description)

    def record_exception(
        self,
        exception: BaseException,
        attributes: Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        self._span.record_exception(exception, attributes, timestamp, escaped)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._span.__exit__(exc_type, exc_val, exc_tb)


class _ContentLimitLogger(Logger):
    def __init__(self, logger: Logger, limit: int) -> None:
        self._logger = logger
        self._limit = limit

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
        if record is None:
            record = LogRecord(
                timestamp=timestamp,
                observed_timestamp=observed_timestamp,
                context=context,
                severity_number=severity_number,
                severity_text=severity_text,
                body=body,
                attributes=attributes,
                event_name=event_name,
            )
        else:
            record = copy.copy(record)
        if record.attributes:
            record.attributes = _limit_attributes(
                record.attributes, self._limit
            )
        self._logger.emit(record)
