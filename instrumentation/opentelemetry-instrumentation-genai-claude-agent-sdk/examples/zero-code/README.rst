OpenTelemetry Claude Agent SDK Zero-Code Instrumentation Example
================================================================

This is an example of how to instrument Claude Agent SDK calls with zero
OpenTelemetry code, using ``opentelemetry-instrument``.

Based on the `claude-agent-sdk-python agents example
<https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/agents.py>`_,
this example defines two custom agents — a **code reviewer** and a
**documentation writer** — using ``AgentDefinition``, then runs queries
against them via the ``query()`` API. When `main.py <main.py>`_ is run with
the CLI, it exports traces, logs, and metrics to an OTLP compatible endpoint.
Traces capture the run as a whole, each model call, each tool call, and — when
the prompt delegates to one of the declared agents — a nested span for the
subagent's own turns.

Note: `.env.example <.env.example>`_ configures additional environment
variables:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT``
  configures Claude Agent SDK instrumentation to capture prompt and completion
  content both on spans and as log events. The other accepted values are
  ``NO_CONTENT``, ``SPAN_ONLY`` and ``EVENT_ONLY``; anything else falls back to
  ``NO_CONTENT`` with a warning.
- ``OTEL_LOGS_EXPORTER`` is left unset, because ``opentelemetry-instrument``
  already defaults it to ``otlp``. The commented-out ``console`` line is there
  to uncomment if your OTLP endpoint doesn't accept logs.

`main.py <main.py>`_ passes ``include_partial_messages=True`` in
``ClaudeAgentOptions``. That is a Claude Agent SDK option rather than
OpenTelemetry setup, so the example still contains no OpenTelemetry code.
Without it the CLI forwards no stream events, so each model call's span reports
a mid-stream token snapshot rather than the call's final usage and stop reason,
and no streaming timings are recorded. The instrumentation does not set it for
you, because it also adds ``StreamEvent`` messages to the stream the caller's
own ``async for`` loop iterates — the loops here ignore them.

Setup
-----

Copy `.env.example <.env.example>`_ to ``.env`` and put your real
``ANTHROPIC_API_KEY`` in it. ``.env`` is git-ignored, so the key stays out of
version control. An OTLP compatible endpoint should be listening for traces and
logs on http://localhost:4317. If not, update ``OTEL_EXPORTER_OTLP_ENDPOINT`` as
well.

Next, set up a virtual environment like this:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install "python-dotenv[cli]"
    pip install -r requirements.txt
    cp .env.example .env

Run
---

Run the example with zero-code instrumentation like this:

::

    dotenv run -- opentelemetry-instrument python main.py

You should see the code reviewer and documentation writer agents respond in
the console while traces and logs export to your configured observability tool.
No OpenTelemetry setup code in ``main.py`` was required!

Learn More
----------

See the `OpenTelemetry Python automatic instrumentation docs
<https://opentelemetry.io/docs/languages/python/automatic/>`_ for more
information about zero-code instrumentation.
