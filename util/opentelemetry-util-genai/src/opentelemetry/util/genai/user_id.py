# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Record ``user.id`` on GenAI telemetry. Specific to this fork.

The id travels on the OTel Context, so every GenAI span started under it
records it, whichever instrumentation package creates the span. It is kept
off baggage on purpose: baggage is sent in the headers of outgoing requests,
which would leak the id to the model provider.

Wrap the code that serves a user in ``using_user_id``::

    with using_user_id("user-123"):
        agent.run(...)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry.context import (
    Context,
    attach,
    create_key,
    detach,
    get_value,
    set_value,
)
from opentelemetry.util.genai._observe_library_user_id import (
    capture_user_id_from_library,
)

_USER_ID_KEY = create_key("observe.util.genai.user_id")


def get_user_id(context: Context | None = None) -> str | None:
    """Return the user id on ``context``, or on the current context."""
    value = get_value(_USER_ID_KEY, context=context)
    return value if isinstance(value, str) else None


def with_user_id(user_id: str, context: Context | None = None) -> Context:
    """Return a copy of ``context``, or of the current context, carrying ``user_id``."""
    return set_value(_USER_ID_KEY, user_id, context=context)


@contextmanager
def using_user_id(user_id: str) -> Iterator[None]:
    """Record ``user_id`` on the GenAI operations started inside the block."""
    token = attach(with_user_id(user_id))
    try:
        yield
    finally:
        detach(token)


def with_library_user_id(
    user_id: str | None, context: Context | None = None
) -> Context | None:
    """For instrumentations: add a user id the instrumented library was given.

    It overrides one set with ``using_user_id``. Returns ``context`` unchanged
    when ``user_id`` is empty or ``OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY``
    is not ``true``, so the result can be passed straight to a
    ``TelemetryHandler`` factory's ``context``.
    """
    if not user_id or not capture_user_id_from_library():
        return context
    return with_user_id(user_id, context)
