# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wraps the SDK's stream-producing entry points.

Two of them produce a stream, and they define different agent-invocation
boundaries:

* ``query()`` — one call is one invocation; the span ends with the stream.
* ``ClaudeSDKClient.receive_messages()`` — continuous over the connection's
  turns, so each turn is one invocation.

``ClaudeSDKClient.query()`` and ``disconnect()`` are wrapped as well, but open
no span. ``query()`` only writes a prompt to the CLI's stdin. It is wrapped to
hand the connection's run that prompt and the span active when it was sent,
which is where the turn's span opens. By default the CLI does not echo a
client-mode prompt back as a ``UserMessage``, so this is the only place it is
seen. A prompt passed to ``connect()`` is written by the SDK itself and is not
captured. ``disconnect()`` ends whatever the connection left open.

``ClaudeSDKClient.receive_response()`` is deliberately **not** wrapped. It is a
convenience method whose whole body iterates ``receive_messages()``, so wrapping
it too would build the tree twice from the same messages. Wrapping the one
primitive covers both callers.

The root ``invoke_agent`` span is always named for the SDK itself, never for
anything in ``options.agents``. The SDK documents that field as defining
"custom **subagents** invokable via the Agent tool", so its entries are
delegation targets rather than the agent being invoked — _run.py opens their
spans from the Task calls that reach them. A caller whose own application is
the agent can name the root explicitly with ``instrument(agent_name=...)``.

Each stream wrapper returns an async iterator yielding the same messages in the
same order. Nothing is injected into the caller's options, and no message is
altered.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from opentelemetry.instrumentation.genai.claude_agent_sdk._run import AgentRun
from opentelemetry.util.genai.handler import TelemetryHandler

_logger = logging.getLogger(__name__)

# Where a client connection keeps its AgentRun: on the SDK's reader for that
# connection, which ``connect()`` builds and ``disconnect()`` drops. Every read
# of one connection, however many ``receive_response()`` calls it takes, feeds
# the same run, the way the CLI sends it: as one stream.
_RUN = "_otel_run"

# ``query``'s own default session id. The CLI does not use it as its own session
# id — it reports a UUID on the init message — so passing the placeholder
# through as a conversation id would collapse every session in the
# Conversations view into one.
_DEFAULT_SESSION_ID = "default"


def _bind(wrapped: Any, args: Any, kwargs: Any) -> inspect.BoundArguments:
    """The call's arguments, with the signature's defaults filled in."""
    bound = inspect.signature(wrapped).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _conversation_id(session_id: Any) -> str | None:
    if (
        isinstance(session_id, str)
        and session_id
        and session_id != _DEFAULT_SESSION_ID
    ):
        return session_id
    return None


def _existing_run(client: Any) -> AgentRun | None:
    return getattr(getattr(client, "_query", None), _RUN, None)


def _connection_run(
    handler: TelemetryHandler, client: Any, agent_name: str | None
) -> AgentRun | None:
    """The run for the client's current connection, or None when it is not
    connected."""
    reader = getattr(client, "_query", None)
    if reader is None:
        return None
    run: AgentRun | None = getattr(reader, _RUN, None)
    if run is None:
        run = AgentRun(
            handler,
            sdk_options=getattr(client, "options", None),
            agent_name=agent_name,
            multi_turn=True,
        )
        setattr(reader, _RUN, run)
    return run


def wrap_query(
    handler: TelemetryHandler, agent_name: str | None = None
) -> Any:
    """Build a replacement for ``claude_agent_sdk.query``."""

    def traced_query(
        wrapped: Any, _instance: Any, args: Any, kwargs: Any
    ) -> Any:
        # Bound here rather than read out of kwargs, because the call itself is
        # deferred into AgentRun.iterate: without this a caller's misspelled
        # keyword would surface from the first ``__anext__``, by which point a
        # root span has opened and the caller's own bug is reported as a failed
        # agent invocation.
        bound = _bind(wrapped, args, kwargs)
        run = AgentRun(
            handler,
            sdk_options=bound.arguments.get("options"),
            agent_name=agent_name,
        )
        run.submit(bound.arguments.get("prompt"))
        return run.iterate(lambda: wrapped(*args, **kwargs))

    return traced_query


def wrap_client_query(
    handler: TelemetryHandler, agent_name: str | None = None
) -> Any:
    """Build a replacement for ``ClaudeSDKClient.query`` that hands the prompt
    to the connection's run."""

    def traced(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        try:
            # Bound through the signature: prompt and session_id can come
            # positionally or by keyword. The prompt is only stored: in
            # streaming-input mode it is an async iterable, and reading it would
            # consume what the SDK has to send.
            bound = _bind(wrapped, args, kwargs)
            run = _connection_run(handler, instance, agent_name)
            if run is not None:
                run.submit(
                    bound.arguments.get("prompt"),
                    _conversation_id(bound.arguments.get("session_id")),
                )
        except Exception:  # pylint: disable=broad-except
            _logger.debug(
                "claude-agent-sdk instrumentation could not record the prompt",
                exc_info=True,
            )
        return wrapped(*args, **kwargs)

    return traced


def wrap_receive_messages(
    handler: TelemetryHandler, agent_name: str | None = None
) -> Any:
    """Build a replacement for ``ClaudeSDKClient.receive_messages``."""

    def traced(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        run = _connection_run(handler, instance, agent_name)
        if run is None:
            # Not connected: the SDK raises for itself.
            return wrapped(*args, **kwargs)
        return run.iterate(lambda: wrapped(*args, **kwargs))

    return traced


def wrap_disconnect() -> Any:
    """Build a replacement for ``ClaudeSDKClient.disconnect`` that ends what
    the connection left open, such as a turn waiting on background work."""

    def traced(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        run = _existing_run(instance)
        if run is not None:
            run.close()
        return wrapped(*args, **kwargs)

    return traced
