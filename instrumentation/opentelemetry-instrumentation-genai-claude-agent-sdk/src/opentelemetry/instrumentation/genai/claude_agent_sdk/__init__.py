# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Claude Agent SDK Instrumentation
===============================================

Instrumentation for the `Claude Agent SDK
<https://github.com/anthropics/claude-agent-sdk-python>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        AgentDefinition,
        AssistantMessage,
        TextBlock,
        query,
    )

    # Enable instrumentation
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK normally
    import anyio


    async def main():
        options = ClaudeAgentOptions(
            agents={
                "assistant": AgentDefinition(
                    description="A helpful assistant",
                    prompt="You are a helpful assistant.",
                    tools=["Read"],
                    model="sonnet",
                ),
            },
        )

        async for message in query(prompt="Hello!", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


    anyio.run(main)

Configuration
-------------

Message content capture is off by default. Enable it by setting
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY`` or ``SPAN_AND_EVENT``. Any other
value, ``true`` included, is not a mode and reads as ``NO_CONTENT``.

The root ``invoke_agent`` span is named ``claude-agent-sdk``. Pass
``ClaudeAgentSDKInstrumentor().instrument(agent_name="...")`` to name it after
the calling application instead. Subagents declared in ``options.agents`` do
not name it: they are delegation targets, and each gets its own nested
``invoke_agent`` span when the Agent tool reaches it.

For accurate per-call token counts, stop reasons, and streaming timings, pass
``include_partial_messages=True`` in ``ClaudeAgentOptions``. It is off by
default and is not set for you, because it changes what the caller's own loop
receives. Without it each ``chat`` span's token counts are a mid-stream
snapshot and the span is marked ``claude_agent_sdk.usage.is_partial``; the run
totals on the ``invoke_agent`` span are accurate either way.

API
---
"""

import sys
from collections.abc import Collection
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.claude_agent_sdk.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk.patch import (
    wrap_client_query,
    wrap_disconnect,
    wrap_query,
    wrap_receive_messages,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk.version import (
    __version__,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """An instrumentor for the Claude Agent SDK.

    This instrumentor will automatically trace Anthropic API calls and
    optionally capture message content as events.
    """

    # Class-level defaults rather than an ``__init__`` that assigns them:
    # ``BaseInstrumentor.__new__`` hands back the one live instance and
    # ``__init__`` still runs on it, so constructing a second instrumentor
    # anywhere in the process would otherwise wipe the state of the
    # instrumentation already in flight.
    _handler: TelemetryHandler | None = None

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Claude Agent SDK instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance
                - agent_name: name for the root ``invoke_agent`` span, for a
                  caller whose own application is the agent being invoked
        """

        # Get providers from kwargs
        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        meter_provider = kwargs.get("meter_provider")

        self._handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )
        agent_name = kwargs.get("agent_name")

        import claude_agent_sdk
        from claude_agent_sdk import client as client_module

        # instrument() flips _is_instrumented_by_opentelemetry only once this
        # returns, so a wrapper left behind by a failure part-way through can
        # never be removed: uninstrument() declines as "already uninstrumented"
        # while a retry is let through and wraps the wrapper. Hence the unwind.
        installed: list[tuple[Any, str]] = []
        try:
            # ``claude_agent_sdk.query`` is both a submodule and the function
            # the package re-exports under the same name, so the two bindings
            # are wrapped separately. A call goes through exactly one of them.
            # The submodule is always in sys.modules by now: importing the
            # package runs ``from .query import query``.
            for module in (
                claude_agent_sdk,
                sys.modules["claude_agent_sdk.query"],
            ):
                wrap_function_wrapper(
                    module, "query", wrap_query(self._handler, agent_name)
                )
                installed.append((module, "query"))

            client = client_module.ClaudeSDKClient
            for attribute, wrapper in (
                # query() opens no span; it hands the prompt to the
                # connection's run.
                ("query", wrap_client_query(self._handler, agent_name)),
                # Only receive_messages: receive_response() is a convenience
                # wrapper that iterates it, so wrapping both would build the
                # tree twice.
                (
                    "receive_messages",
                    wrap_receive_messages(self._handler, agent_name),
                ),
                ("disconnect", wrap_disconnect()),
            ):
                wrap_function_wrapper(client, attribute, wrapper)
                installed.append((client, attribute))
        except BaseException:
            for target, attribute in reversed(installed):
                unwrap(target, attribute)
            self._handler = None
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation.

        This removes all patches applied during instrumentation.
        """
        import claude_agent_sdk
        from claude_agent_sdk import client as client_module

        unwrap(claude_agent_sdk, "query")
        unwrap(sys.modules["claude_agent_sdk.query"], "query")
        unwrap(client_module.ClaudeSDKClient, "query")
        unwrap(client_module.ClaudeSDKClient, "receive_messages")
        unwrap(client_module.ClaudeSDKClient, "disconnect")
        self._handler = None
