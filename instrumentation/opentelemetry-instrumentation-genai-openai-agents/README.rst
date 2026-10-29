Observe OpenAI Agents Instrumentation
=====================================

This library traces applications built with the
`OpenAI Agents SDK <https://pypi.org/project/openai-agents/>`_. It registers a
tracing processor with the Agents runtime and turns its trace/span callbacks
into spans that mirror the structure of your agent run:

* **Workflow spans** for an agent run — one per ``Runner.run`` / ``run_sync``
  invocation, named after the run's ``RunConfig.workflow_name``.
* **Agent spans** for each agent invoked within the workflow, including the
  agent name.
* **Tool spans** for function tool calls, capturing the tool arguments and
  result.

The spans nest to reflect the run, so a single ``Runner.run`` produces a
workflow span with the agent and tool calls it triggered as children.

LLM-level spans (``chat`` / ``embeddings``) are **not** emitted by this package.
They are produced by
``observe-instrumentation-genai-openai``
when it is installed and instrumented alongside this package.

.. note::
   This package continues the project previously published as
   ``opentelemetry-instrumentation-openai-agents-v2``.

This package is Observe's replacement for upstream's
``opentelemetry-instrumentation-genai-openai-agents``, not an addition to it. The two
distributions install to the same paths, so an environment holding both gets a
per-file mix of them and uninstalling either one breaks the other. This package
depends on ``observe-util-genai``, which replaces upstream's
``opentelemetry-util-genai`` in the same way. Uninstall both upstream packages
first; an upstream instrumentation that still needs ``opentelemetry-util-genai``
cannot share an environment with this one.

Installation
------------

This package and its ``observe-util-genai`` dependency are not published to
PyPI yet. Install both from this repository in one command, so pip does not
look for ``observe-util-genai`` on PyPI:

::

    pip uninstall opentelemetry-instrumentation-genai-openai-agents opentelemetry-util-genai
    pip install \
        "observe-util-genai @ git+https://github.com/snowflake-eng/observe-otel-genai-python#subdirectory=util/opentelemetry-util-genai" \
        "observe-instrumentation-genai-openai-agents @ git+https://github.com/snowflake-eng/observe-otel-genai-python#subdirectory=instrumentation/opentelemetry-instrumentation-genai-openai-agents"

.. note::
   Model calls (``chat`` spans and their token counts) come from
   ``observe-instrumentation-genai-openai``, not from this package. If your
   agents call models through the ``openai`` package, which is the default and
   includes OpenAI-compatible endpoints set with ``base_url``, install it the
   same way and call ``OpenAIInstrumentor().instrument()`` next to
   ``OpenAIAgentsInstrumentor().instrument()``:

   ::

       pip install "observe-instrumentation-genai-openai @ git+https://github.com/snowflake-eng/observe-otel-genai-python#subdirectory=instrumentation/opentelemetry-instrumentation-genai-openai"

   With other model clients, such as ``LitellmModel``, no model calls are
   recorded. To ask for your client to be supported,
   `open an issue <https://github.com/snowflake-eng/observe-otel-genai-python/issues>`_.

See the `examples <examples>`_ directory for runnable ``manual`` and
``zero-code`` scenarios.

Usage
-----

Call ``OpenAIAgentsInstrumentor().instrument()`` once during startup, then run
your agent as usual. The ``Runner.run_sync(...)`` call below is recorded as a
workflow span with the agent and tool spans nested underneath.

.. code-block:: python

    from agents import Agent, Runner, function_tool

    from observeinc.instrumentation.genai.openai_agents import (
        OpenAIAgentsInstrumentor,
    )

    OpenAIAgentsInstrumentor().instrument()


    @function_tool
    def get_weather(city: str) -> str:
        return f"The forecast for {city} is sunny with pleasant temperatures."


    assistant = Agent(
        name="Travel Concierge",
        instructions="You are a concise travel concierge.",
        model="<your-model>",
        tools=[get_weather],
    )

    result = Runner.run_sync(
        assistant, "I'm visiting Barcelona this weekend. How should I pack?"
    )
    print(result.final_output)

Configuration
-------------

By default the instrumentor keeps the Agents SDK's built-in trace exporter
(which uploads traces to OpenAI's hosted tracing backend when ``OPENAI_API_KEY``
is set) active alongside OpenTelemetry emission. Pass
``disable_openai_trace_export=True`` to route traces only through OpenTelemetry:

.. code-block:: python

    OpenAIAgentsInstrumentor().instrument(disable_openai_trace_export=True)

The workflow span name comes from the Agents SDK's own
``Runner.run(..., run_config=RunConfig(workflow_name=...))`` knob. Content
capture is controlled through
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment variable.

Prompts and completions can instead be uploaded to external storage via a
completion hook: set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` (install the ``upload`` extra by
changing ``observe-util-genai`` to ``observe-util-genai[upload]`` in the install
command above), or pass a custom ``CompletionHook`` programmatically, which
takes precedence over the environment variable::

    OpenAIAgentsInstrumentor().instrument(completion_hook=my_hook)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `OpenAI Agents SDK (Python) <https://github.com/openai/openai-agents-python>`_
