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

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

For accurate per-call token counts, stop reasons, and streaming timings, pass
``include_partial_messages=True`` in ``ClaudeAgentOptions``. It is off by
default and is not set for you, because it changes what the caller's own loop
receives. Without it each ``chat`` span's token counts are a mid-stream
snapshot and the span is marked ``genai.usage.is_partial``; the run totals on
the ``invoke_agent`` span are accurate either way.

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

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None

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

        import claude_agent_sdk
        from claude_agent_sdk import client as client_module

        # ``claude_agent_sdk.query`` is both a submodule and the function the
        # package re-exports under the same name, so the two bindings are
        # wrapped separately. A call goes through exactly one of them.
        wrap_function_wrapper(
            claude_agent_sdk, "query", wrap_query(self._handler)
        )
        query_module = sys.modules.get("claude_agent_sdk.query")
        if query_module is not None:
            wrap_function_wrapper(
                query_module, "query", wrap_query(self._handler)
            )

        # Only receive_messages: receive_response() is a convenience wrapper
        # that iterates it, so wrapping both would build the tree twice.
        wrap_function_wrapper(
            client_module,
            "ClaudeSDKClient.receive_messages",
            wrap_receive_messages(self._handler),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation.

        This removes all patches applied during instrumentation.
        """
        import claude_agent_sdk
        from claude_agent_sdk import client as client_module

        unwrap(claude_agent_sdk, "query")
        query_module = sys.modules.get("claude_agent_sdk.query")
        if query_module is not None:
            unwrap(query_module, "query")
        unwrap(client_module.ClaudeSDKClient, "receive_messages")
        self._handler = None
