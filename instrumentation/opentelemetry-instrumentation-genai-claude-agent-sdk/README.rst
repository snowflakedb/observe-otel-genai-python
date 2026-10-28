Observe Claude Agent SDK Instrumentation
========================================

This library traces LLM requests made through the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_ and
emits `OpenTelemetry GenAI semantic conventions
<https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.

Don't install it next to the OpenTelemetry project's
``opentelemetry-instrumentation-genai-claude-agent-sdk``: both instrument the
same SDK, so every call would be recorded twice.

Installation
------------

This package and its ``observe-util-genai`` dependency are not published to
PyPI yet. Install both from this repository in one command, so pip does not
look for either name on PyPI:

::

    pip install \
        "observe-util-genai @ git+https://github.com/Snowflake-Labs/observe-otel-genai-python#subdirectory=util/opentelemetry-util-genai" \
        "observe-instrumentation-genai-claude-agent-sdk @ git+https://github.com/Snowflake-Labs/observe-otel-genai-python#subdirectory=instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk"

If you don't have a Claude Agent SDK application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Claude Agent SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from observeinc.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor

    # Instrument Claude Agent SDK
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK as normal
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock, query

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

Call ``instrument()`` before ``from claude_agent_sdk import query``. The
instrumentation replaces ``query`` on the ``claude_agent_sdk`` module, so a
``query`` imported earlier still points at the original function and its calls
are not traced. Calls to ``claude_agent_sdk.query(...)`` and ``ClaudeSDKClient`` are
traced regardless of import order. This only matters for manual setup:
``opentelemetry-instrument`` instruments before your code runs.

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
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, ``SPAN_AND_EVENT``, or ``ALLOW_LIST``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

``ALLOW_LIST`` captures no content, like ``NO_CONTENT``, and keeps only a fixed list of
attributes on everything recorded, as described in the ``observe-util-genai`` README. It drops
``claude_agent_sdk.usage.is_partial``, so under it you cannot tell which token counts are undercounts.

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
``gen_ai.client.operation.time_per_output_chunk`` timings are recorded. The
token counts on the root ``invoke_agent`` span are accurate either way. Turning
it on adds ``StreamEvent`` messages to the stream your loop iterates.

A subagent's calls get no stream events even with the option on. When the
delegation runs in the foreground, its last call still reports final counts,
taken from the delegation's result; every other subagent call stays marked as
a snapshot.

``forward_subagent_text=True`` makes the CLI forward the text and thinking
blocks of a subagent spawned through the Agent tool. By default it forwards
only that subagent's ``tool_use`` and ``tool_result`` blocks, which is enough
for a progress heartbeat: the delegation still gets its own nested
``invoke_agent`` span, but the content recorded on it is thin, and empty if the
subagent never called a tool. Either way the forwarded messages arrive as
``AssistantMessage`` / ``UserMessage`` objects whose ``parent_tool_use_id`` is
the spawning Agent ``tool_use`` id, so turning the option on adds the
subagent's own text to your loop.

Token Usage
-----------

Each ``invoke_agent`` span reports only the tokens of its own agent's calls. A
subagent's calls count on the subagent's span, not on the span of the agent that
delegated to it. To total a run, add up every ``invoke_agent`` span in the trace.
Don't also add the ``chat`` spans: they cover the same calls, so each call would
be counted twice.

- **The root agent** takes its counts from the ``usage`` field of each
  ``ResultMessage``, which counts the root agent's own calls since the previous
  result, not its subagents'. They are accurate whether or not stream events are
  on, and for resumed sessions too.
- **A subagent** gets no total of its own from the CLI, so its span adds up
  its ``chat`` spans. Input and cache counts are accurate. Output is accurate
  only for a foreground subagent's last call: the CLI forwards no stream
  events for a subagent, so each earlier call's output count is the near-zero
  figure from when that call started. A background subagent has no accurate
  output count at all. When any of its calls was undercounted, the subagent
  span carries ``claude_agent_sdk.usage.is_partial`` set to ``true``, as the
  ``chat`` span of that call does.

Calls the CLI makes for itself, such as the one that shortens a long
conversation (compaction), are counted on no span. Only the session-wide
``ResultMessage.model_usage`` includes them.

No span reports cost. ``ResultMessage.total_cost_usd`` covers the whole session
and every agent in it, so it fits no single span. Read it in your own code if
you need it.


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
