Observe Claude Agent SDK Instrumentation
========================================

This library traces LLM requests made through the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_ and
emits `OpenTelemetry GenAI semantic conventions
<https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.

It is Observe's replacement for upstream's
``opentelemetry-instrumentation-genai-claude-agent-sdk``, not an addition to it.
The two distributions install to the same paths, so an environment holding both
gets a per-file mix of them and uninstalling either one breaks the other.
Uninstall the upstream package first. The `repository README <../../README.md>`_
describes what a mixed install actually does.

Installation
------------

::

    pip uninstall opentelemetry-instrumentation-genai-claude-agent-sdk
    pip install observe-instrumentation-genai-claude-agent-sdk

If you don't have a Claude Agent SDK application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Claude Agent SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock, query

    # Instrument Claude Agent SDK
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK as normal
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
            include_partial_messages=True,
        )

        async for message in query(
            prompt="Hello, Claude!",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    anyio.run(main)


With ``ClaudeSDKClient``, each turn of a connection gets its own
``invoke_agent`` span, however many ``receive_response()`` calls read it. The
span opens under whatever span was active when you called ``query()`` for that
turn. A turn that starts background work stays open until that work finishes or
the next turn begins, so its span can include the time you spent before sending
the next prompt.

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Any other value is rejected with a warning and falls back to ``NO_CONTENT``.

Naming the Agent
****************

The root ``invoke_agent`` span is named ``claude-agent-sdk``. To name it after
the calling application instead:

.. code-block:: python

    ClaudeAgentSDKInstrumentor().instrument(agent_name="support-triage")

Subagents declared in ``options.agents`` do not name it. They are delegation
targets, and each gets its own nested ``invoke_agent`` span when the Agent tool
reaches it.

Claude Agent SDK Options
************************

Two ``ClaudeAgentOptions`` fields change how much of a run the instrumentation
can see. Neither is set on your behalf, because both change what your own
message loop receives.

``include_partial_messages=True`` makes the CLI forward stream events, which is
what the ``chat`` spans need to report a call's final token counts and stop
reason. Without it, those spans carry the mid-stream usage snapshot instead —
an undercount, marked on the span so a consumer can tell the difference — and
no ``gen_ai.client.operation.time_to_first_chunk`` or
``gen_ai.client.operation.time_per_output_chunk`` timings are recorded. The run
totals on the ``invoke_agent`` span are accurate either way. Turning it on adds
``StreamEvent`` messages to the stream your loop iterates.

``forward_subagent_text=True`` makes the CLI forward the text and thinking
blocks of a subagent spawned through the Agent tool. By default it forwards
only that subagent's ``tool_use`` and ``tool_result`` blocks, which is enough
for a progress heartbeat: the delegation still gets its own nested
``invoke_agent`` span, but the content recorded on it is thin, and empty if the
subagent never called a tool. Either way the forwarded messages arrive as
``AssistantMessage`` / ``UserMessage`` objects whose ``parent_tool_use_id`` is
the spawning Agent ``tool_use`` id, so turning the option on adds the
subagent's own text to your loop. It requires ``claude-agent-sdk >= 0.2.148``,
above this package's declared minimum of 0.2.137, so guard the option if you
support that whole range.


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
