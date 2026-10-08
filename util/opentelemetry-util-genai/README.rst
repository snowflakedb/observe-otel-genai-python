Observe Util for GenAI
======================

Downstream fork of ``opentelemetry-util-genai``. It is Observe's replacement for
that package, not an addition to it. The two install to the same paths, so an
environment holding both gets a per-file mix of them and uninstalling either one
breaks the other. Uninstall ``opentelemetry-util-genai`` first. Every upstream GenAI
instrumentation depends on it, and ``opentelemetry-bootstrap -a install`` can pull
one in, so install instrumentations explicitly rather than through bootstrap.

The GenAI Utils package provides boilerplate and helpers to standardize instrumentation for Generative AI.
It offers APIs to minimize the work needed to instrument GenAI libraries,
while providing standardization for generating spans, metrics, and events.


Key Components
--------------

- ``TelemetryHandler`` -- manages LLM invocation lifecycles (spans, metrics, events)
- ``InferenceInvocation`` and message types (``TextPart``, ``ReasoningPart``, ``BlobPart``, etc.) -- structured data model for GenAI interactions
- ``CompletionHook`` -- protocol for uploading content to external storage (built-in ``fsspec`` support)
- Metrics -- ``gen_ai.client.operation.duration`` and ``gen_ai.client.token.usage`` histograms, plus
  the streaming timing histograms ``gen_ai.client.operation.time_to_first_chunk`` and
  ``gen_ai.client.operation.time_per_output_chunk``


Usage
-----

See the module docstring in ``observeinc.util.genai.handler`` for usage examples,
including context manager and manual lifecycle patterns.


Context Management and Propagation
----------------------------------

Invocation factory methods on ``TelemetryHandler`` (such as ``inference``, ``workflow``, ``tool``,
``embedding``, ``retrieval``, and ``invoke_local_agent``) accept an optional ``context`` keyword argument
to manage context:

- ``context``: An explicit OpenTelemetry ``Context`` to parent the span. When omitted, the current
  ambient context is used.

Conversation ID on Every Span
-----------------------------

This fork sets ``gen_ai.conversation.id`` on every span, including tool, embeddings, retrieval
and fetch-response spans, where semconv omits it. The value is the invocation's
``conversation_id``, which defaults to the one on the OTel Context. Setting ``conversation_id``
after creation also reaches the operations started under the invocation afterwards: those
parented on its ``context``, and, while nothing else has been attached on top of it, those using
the current context. Set it in the thread or task that created the invocation, as with
``stop()``. A value an instrumentation sets on ``attributes`` takes precedence.

Agent Name on Child Operations
------------------------------

This fork adds ``gen_ai.agent.name`` to the inference, embeddings, retrieval and fetch-response
spans an agent runs, and to the metrics they record, which semconv does not define. It travels on
the OTel Context the way the conversation id does: ``handler.inference()``, ``embedding()``,
``retrieval()``, ``fetch_response()`` and ``tool()`` take an ``agent_name``, and default to the
enclosing agent's, so the name reaches operations another instrumentation package creates.

As with the conversation id, an explicit ``agent_name`` also reaches the operations under that
invocation. An agent always sets its own name, so the nearest agent wins, and one with no name
hides an outer agent's. A value an instrumentation sets itself, on ``attributes`` or
``metric_attributes``, takes precedence.

User ID
-------

This fork records the standard OpenTelemetry ``user.id`` attribute on every GenAI span started
while a user id is set. Wrap the code that serves a user in ``using_user_id``:

.. code-block:: python

    from observeinc.util.genai.user_id import using_user_id

    with using_user_id("user-123"):
        agent.run(...)

The id travels on the OTel Context, so it reaches spans created by any instrumentation package,
and by ``TelemetryHandler`` directly. It is not put in baggage, because baggage is sent in the
headers of outgoing requests, including those to the model provider. To set it for a single
operation, and the operations it starts, pass ``context=with_user_id("user-123")`` to a
``TelemetryHandler`` factory such as ``handler.inference(...)``. A value an instrumentation sets on
``attributes`` takes precedence.

Some libraries accept a user id of their own, such as a LangChain run's ``user_id`` metadata. Set
``OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY`` to ``true`` to let instrumentations record it. It is
off by default, so an id passed to a library does not start showing up in telemetry on upgrade.
When it is on, the library's id takes precedence over one set with ``using_user_id``. The flag is
read each time an instrumentation starts an operation.


Inference Token Metrics
-----------------------

This fork also records the token metrics semconv introduced in
`semantic-conventions-genai#374 <https://github.com/open-telemetry/semantic-conventions-genai/pull/374>`_,
on inference and embeddings operations. ``gen_ai.client.token.usage`` is still recorded alongside them.

- ``gen_ai.client.inference.usage.input_tokens``, ``.output_tokens``, ``.cache_read.input_tokens``,
  ``.cache_write.input_tokens`` and ``.reasoning.output_tokens``: counters, for totals and cost. Each
  carries ``gen_ai.token.modality``, taken from the invocation's per-modality token counts; the part
  no modality accounts for is ``unknown``.
- ``gen_ai.client.inference.operation.input_tokens`` and ``.output_tokens``: histograms, one
  measurement per operation.

Cache and reasoning tokens are already included in the input and output totals, so don't add those
counters to the input or output ones.


Modalities
----------

``observeinc.util.genai.types.Modality`` provides string enum members
``TEXT``, ``IMAGE``, ``VIDEO``, ``AUDIO``, and ``DOCUMENT``. Use these constants
when constructing ``BlobPart``, ``FilePart``, and ``UriPart`` or passing
modality/token-count pairs to the inference invocation's token setters:

.. code-block:: python

    from observeinc.util.genai.types import Modality, UriPart

    image = UriPart(
        mime_type="image/png",
        modality=Modality.IMAGE,
        uri="https://example.com/image.png",
    )

Members serialize and format as their string values, such as ``"image"``.
Message parts continue to accept plain strings, including provider-specific
modalities. Token setters continue to record only ``text``, ``image``, and
``audio``; other modalities are ignored.

``Modality`` replaces the previous ``Literal`` type alias. Annotations that
also accept raw strings should use ``Modality | str`` rather than ``Modality``
alone. Custom modalities remain strings, not additional enum members.


Retrieval Documents
-------------------

Set ``RetrievalInvocation.documents`` using
``observeinc.util.genai.types.RetrievalDocument`` objects:

.. code-block:: python

    from observeinc.util.genai.types import RetrievalDocument

    with handler.retrieval(data_source_id="my-index") as invocation:
        invocation.documents = [RetrievalDocument(id="doc-1", score=0.9)]

The model contains only the optional ``id`` and ``score`` fields; unset
fields serialize as JSON ``null``. Documents are recorded in
``gen_ai.retrieval.documents`` only in ``SPAN_ONLY`` or ``SPAN_AND_EVENT``
content-capture mode. Passing dictionaries is deprecated, but existing
dictionary payloads continue to serialize unchanged.


Environment Variables
---------------------

This package relies on environment variables to configure capturing of message content.
By default, message content will not be captured.
Set the environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of:

- ``NO_CONTENT``: Do not capture message content (default).
- ``SPAN_ONLY``: Capture message content in spans only.
- ``EVENT_ONLY``: Capture message content in events only.
- ``SPAN_AND_EVENT``: Capture message content in both spans and events.
- ``ALLOW_LIST``: Do not capture message content, and keep only allowlisted attributes. Specific
  to this fork; see `Allowlist Mode`_.

An error message can quote the prompt, so under ``NO_CONTENT`` and ``EVENT_ONLY``, which put no
message content on spans, a failed span keeps its ERROR status and ``error.type`` but no status
description. The ``gen_ai.client.inference.operation.details`` event carries ``error.type`` and
never the message, in every mode. Specific to this fork.

To control event emission, you can optionally set ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` to ``true`` or ``false`` (case-insensitive).
This variable controls whether to emit ``gen_ai.client.inference.operation.details`` events.
If not explicitly set, the default value is automatically determined by ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``:

- When ``NO_CONTENT`` or ``SPAN_ONLY`` is set: defaults to ``false``
- When ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` is set: defaults to ``true``

If explicitly set, the user's value takes precedence over the default.

When ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` mode is enabled and a LoggerProvider is configured,
the package also emits ``gen_ai.client.inference.operation.details`` events with structured
message content (as dictionaries instead of JSON strings). Note that when using ``EVENT_ONLY``
or ``SPAN_AND_EVENT``, the ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` environment variable defaults
to ``true``, so events will be emitted automatically unless explicitly set to ``false``.

Conversation ID on Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID`` to ``true`` (case-insensitive) to add
``gen_ai.conversation.id`` to every metric this package records, so a backend can break metrics
down by conversation. It is off by default: semconv keeps the conversation id off metrics because
each conversation adds new time series. Specific to this fork.

User ID on Metrics
^^^^^^^^^^^^^^^^^^

Set ``OBSERVE_GENAI_METRICS_INCLUDE_USER_ID`` to ``true`` (case-insensitive) to add ``user.id``
to every metric this package records, not only to spans. It is off by default, because each user
adds new time series. Specific to this fork.

Allowlist Mode
^^^^^^^^^^^^^^

Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to ``ALLOW_LIST`` (case-insensitive)
when the telemetry must not hold message content: prompts, completions, system instructions, and
tool arguments and results. No message content is captured, as with ``NO_CONTENT``, and every
span, event and metric this package records keeps only the attributes on a fixed list. It holds
keys such as model and provider names, token counts, request settings such as temperature, finish
reasons, ``server.address`` and ``server.port``, ``error.type``, ``user.id``, response and
conversation ids, and agent and tool ids, names and descriptions. Those values are kept as the
application or library set them. Every other attribute is dropped, including ones an
instrumentation adds through ``attributes`` or ``metric_attributes``, and ones a later release
starts recording. Specific to this fork.

In this mode:

- A span keeps its status code but not its status description, and a recorded exception keeps
  only ``exception.type``.
- Code that writes to the current span while a GenAI operation runs, such as
  ``trace.get_current_span().set_attribute(...)``, goes through the same filter.
- The ``gen_ai.client.inference.operation.details`` event is emitted only when
  ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` is ``true``, as with ``NO_CONTENT``. It then carries
  the allowed attributes and no body.
- The completion hook is never called, even when one is configured, and
  ``should_capture_content()`` returns ``False``. A configured hook gets a warning, logged once,
  and the hook named by ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` is not loaded at all, so
  the ``upload`` hook does not start.

The list is fixed in this version. Some attributes this package records are not on it yet
and are dropped, such as ``gen_ai.request.top_k``, ``gen_ai.usage.cache_write.input_tokens``
and the per-modality token counts. ``gen_ai.token.modality`` is one of them, so token metrics
are not split by modality.
Instrumentation-specific keys are dropped too, such as the OpenAI package's
``openai.api.type``, ``openai.request.service_tier``, ``openai.response.service_tier`` and
``openai.response.system_fingerprint``.

Not filtered: span names, resource attributes, attributes that the application's own span
processors add, spans, logs and metrics that other libraries record, such as HTTP client spans,
and the Python ``logging`` messages instrumentations write, such as debug logs that include an
exception.

Completion Hook / Upload
^^^^^^^^^^^^^^^^^^^^^^^^

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``: Name of the completion hook entry point to load (e.g. ``upload``).
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``: An ``fsspec``-compatible URI/path for uploading prompts and completions
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``). Required when using the ``upload`` hook.
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_FORMAT``: Format for uploaded data -- ``json`` (default) or ``jsonl``.
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_MAX_QUEUE_SIZE``: Maximum number of concurrent uploads to queue (default: ``20``).


Span Attributes
---------------

This package sets the following span attributes on LLM invocations:

**Common attributes:**

- ``gen_ai.operation.name``: Str(chat)
- ``gen_ai.provider.name``: Str(openai)
- ``gen_ai.request.model``: Str(gpt-4o)
- ``server.address``: Str(api.openai.com)
- ``server.port``: Int(443)

**Response attributes:**

- ``gen_ai.response.finish_reasons``: Slice(["stop"])
- ``gen_ai.response.model``: Str(gpt-4o-2024-05-13)
- ``gen_ai.response.id``: Str(chatcmpl-Bz8yrvPnydD9pObv625n2CGBPHS13)
- ``gen_ai.usage.input_tokens``: Int(24)
- ``gen_ai.usage.output_tokens``: Int(7)
- ``gen_ai.usage.cache_write.input_tokens``: Int(10)
- ``gen_ai.usage.cache_read.input_tokens``: Int(5)

**Request parameter attributes (when provided):**

- ``gen_ai.request.temperature``: Float(0.7)
- ``gen_ai.request.top_p``: Float(1.0)
- ``gen_ai.request.frequency_penalty``: Float(0.0)
- ``gen_ai.request.presence_penalty``: Float(0.0)
- ``gen_ai.request.max_tokens``: Int(1024)
- ``gen_ai.request.stop_sequences``: Slice(["\\n"])
- ``gen_ai.request.seed``: Int(42)

**Content attributes (sensitive, requires content capturing enabled):**

- ``gen_ai.input.messages``: Str('[{"role": "user", "parts": [{"content": "hello world", "type": "text"}]}]')
- ``gen_ai.output.messages``: Str('[{"role": "assistant", "parts": [{"content": "hello back", "type": "text"}], "finish_reason": "stop"}]')
- ``gen_ai.system_instructions``: Str('[{"content": "You are a helpful assistant.", "type": "text"}]')

**Error attributes:**

- ``error.type``: Str(TimeoutError)

Embedding Span Attributes
-------------------------

This package also supports embedding invocation spans via the ``embedding`` context manager.
For embedding invocations, the following attributes are set:

**Common attributes:**

- ``gen_ai.operation.name``: Str(embeddings)
- ``gen_ai.provider.name``: Str(openai)
- ``server.address``: Str(api.openai.com)
- ``server.port``: Int(443)

**Request attributes:**

- ``gen_ai.request.model``: Str(text-embedding-3-small)
- ``gen_ai.embeddings.dimension.count``: Int(1536)
- ``gen_ai.request.encoding_formats``: Slice(["float"])

**Response attributes:**

- ``gen_ai.response.model``: Str(text-embedding-3-small)
- ``gen_ai.usage.input_tokens``: Int(24)


Installation
------------

This package is not published to PyPI yet. Install it from this repository:

::

    pip install "observe-util-genai @ git+https://github.com/snowflake-eng/observe-otel-genai-python#subdirectory=util/opentelemetry-util-genai"

For upload support (requires ``fsspec``)::

    pip install "observe-util-genai[upload] @ git+https://github.com/snowflake-eng/observe-otel-genai-python#subdirectory=util/opentelemetry-util-genai"


Design Document
---------------

The design document for the OpenTelemetry GenAI Utils can be found at: `Design Document <https://docs.google.com/document/d/1LzNGylxot5zaIV1goOJZ2mz3LI0weFu1SgaMnyDv7gg/edit?usp=sharing>`_

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_