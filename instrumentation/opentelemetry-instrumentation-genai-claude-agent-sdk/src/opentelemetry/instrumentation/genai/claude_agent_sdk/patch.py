# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wraps the SDK's stream-producing entry points.

Two of them produce a stream, and they define different agent-invocation
boundaries:

* ``query()`` — one call is one invocation; the span ends with the stream.
* ``ClaudeSDKClient.receive_messages()`` — continuous over many turns, so the
  boundary is a result that answers a prompt the caller submitted.

``ClaudeSDKClient.query()`` is wrapped as well, but opens no span: it only
writes a prompt to the CLI's stdin, and the invocation it starts is spanned by
the receiving half. It is wrapped because it is the only place the client's
prompt is in hand.

``ClaudeSDKClient.receive_response()`` is deliberately **not** wrapped. It is a
convenience method whose whole body iterates ``receive_messages()`` and stops at
the first result, so wrapping it as well would build the tree twice from the
same messages — measured as two agent spans and two chat spans per turn, in
separate traces. Wrapping the one primitive covers both callers.

The root ``invoke_agent`` span is always named for the SDK itself, never for
anything in ``options.agents``. The SDK documents that field as defining
"custom **subagents** invokable via the Agent tool", so its entries are
delegation targets rather than the agent being invoked — _run.py opens their
spans from the Task calls that reach them. A caller whose own application is
the agent can name the root explicitly with ``instrument(agent_name=...)``.

Each wrapper returns an async iterator yielding the same messages in the same
order. Nothing is injected into the caller's options, and no message is altered.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from opentelemetry.instrumentation.genai.claude_agent_sdk._run import AgentRun
from opentelemetry.util.genai.handler import TelemetryHandler

_logger = logging.getLogger(__name__)

# Where ``ClaudeSDKClient.query`` leaves what it was asked to send. The client
# splits sending from receiving — the caller awaits ``query()`` and then
# iterates ``receive_response()`` — so the receiving half has no prompt of its
# own, and in client mode the CLI emits no UserMessage to recover one from: a
# recorded session's raw stream held only SystemMessage, StreamEvent,
# AssistantMessage, ResultMessage and HookEventMessage.
_PENDING = "_otel_pending_query"

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


def _take_pending_query(instance: Any) -> tuple[Any, str | None]:
    """Consume what the last ``query`` on this client left behind.

    Popped rather than read, so a receive that follows no new query does not
    replay the previous turn's prompt.
    """
    pending = getattr(instance, _PENDING, None)
    if pending is None:
        return None, None
    try:
        delattr(instance, _PENDING)
    except AttributeError:
        pass
    return pending


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
        return run.iterate(
            lambda: wrapped(*args, **kwargs), bound.arguments.get("prompt")
        )

    return traced_query


def wrap_client_query() -> Any:
    """Build a replacement for ``ClaudeSDKClient.query`` that records the
    prompt for the receiving half to report."""

    def traced(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        try:
            # ``query(self, prompt, session_id="default")`` takes both
            # positionally, unchanged across every 0.1.14-0.2.157 release, so
            # they are read through the signature. The prompt is only ever
            # stored: in streaming-input mode it is an async iterable and
            # reading it would consume what the SDK has to send.
            bound = _bind(wrapped, args, kwargs)
            setattr(
                instance,
                _PENDING,
                (
                    bound.arguments.get("prompt"),
                    _conversation_id(bound.arguments.get("session_id")),
                ),
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
        prompt, conversation_id = _take_pending_query(instance)
        run = AgentRun(
            handler,
            sdk_options=getattr(instance, "options", None),
            agent_name=agent_name,
            conversation_id=conversation_id,
            multi_turn=True,
        )
        return run.iterate(lambda: wrapped(*args, **kwargs), prompt)

    return traced
