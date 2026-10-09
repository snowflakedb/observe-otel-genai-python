Observe LangChain Instrumentation
=================================

This library traces `LangChain <https://pypi.org/project/langchain/>`_ and
`LangGraph`_ applications. It hooks into
LangChain's callback manager to emit spans that mirror the structure of your
application:

* **Workflow spans** for a graph or chain run — for example an invocation of a
  LangGraph ``StateGraph`` — capturing the overall input and output of the run.
* **Agent spans** for agent invocations nested inside a workflow, including the
  agent name, id, description, and conversation/session id when available.
* **Tool spans** for tool calls made during a run.
* **Retrieval spans** for retriever invocations, capturing the query and retrieved
  document IDs and relevance scores when available.

The spans nest to reflect the graph, so a single graph invocation produces a
workflow span with the agent, tool, and model calls it triggered as children.

This package is Observe's replacement for upstream's
``opentelemetry-instrumentation-genai-langchain``, not an addition to it. The two
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

    pip uninstall opentelemetry-instrumentation-genai-langchain opentelemetry-util-genai
    pip install \
        "observe-util-genai @ git+https://github.com/Snowflake-Labs/observe-otel-genai-python#subdirectory=util/opentelemetry-util-genai" \
        "observe-instrumentation-genai-langchain @ git+https://github.com/Snowflake-Labs/observe-otel-genai-python#subdirectory=instrumentation/opentelemetry-instrumentation-genai-langchain"

See the `examples <examples>`_ directory for runnable ``workflow``, ``agent``,
``tools``, and ``zero-code`` scenarios.

Usage
-----

Call ``LangChainInstrumentor().instrument()`` once during startup, then build
and invoke your graph as usual. The example below traces a simple two-node
LangGraph ``StateGraph`` (``START → researcher → summariser → END``); the
``graph.invoke(...)`` call is recorded as a workflow span with the node model
calls nested underneath.

.. code-block:: python

    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    from observeinc.instrumentation.genai.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="<your-model>", temperature=0)


    class State(TypedDict):
        messages: Annotated[list, add_messages]
        research: str


    def researcher(state: State) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content="Provide 2-3 factual sentences."),
                HumanMessage(content=state["messages"][-1].content),
            ]
        )
        return {"research": response.content, "messages": [response]}


    def summariser(state: State) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content="Condense the text into one sentence."),
                HumanMessage(content=state["research"]),
            ]
        )
        return {"messages": [response]}


    builder = StateGraph(State)
    builder.add_node("researcher", researcher)
    builder.add_node("summariser", summariser)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "summariser")
    builder.add_edge("summariser", END)
    graph = builder.compile()

    # Recorded as a workflow span with the two node LLM calls nested underneath.
    graph.invoke(
        {
            "messages": [HumanMessage(content="What is the capital of France?")],
            "research": "",
        }
    )

Using with model-provider instrumentations
------------------------------------------

This package records a ``chat`` span for each LangChain chat model call. If
the provider SDK underneath is instrumented too, each call is recorded twice:
once here, and again by the SDK's instrumentation, nested under it. Token usage
and duration metrics count the call twice as well. Known cases:

- ``observe-instrumentation-genai-openai``, under ``ChatOpenAI``
- ``opentelemetry-instrumentation-botocore``, under ``ChatBedrock`` and
  ``ChatBedrockConverse``. ``opentelemetry-bootstrap -a install`` installs it
  whenever ``boto3`` is present.
- other tools that instrument the provider SDK, such as OpenInference or
  OpenLLMetry

``opentelemetry-instrument`` enables every installed instrumentation, so having
both installed is enough. Turn off the provider's instrumentation, for
example::

    export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=openai,botocore

or uninstall it. Disabling ``botocore`` removes spans for every AWS call (S3,
DynamoDB, and so on), not just Bedrock. If you set up instrumentation in code,
don't call ``instrument()`` on the provider's instrumentor.

Keep the provider's instrumentation if the same process also needs it:

- calls to the SDK outside LangChain
- LangChain features this package doesn't trace, such as embeddings (for
  example ``OpenAIEmbeddings``)
- ``openai-agents``, whose model spans come only from the OpenAI
  instrumentation

LangChain chat calls are then still counted twice.

Retrieval Spans and Document Scores
-----------------------------------

When invoking LangChain retrievers (e.g., vectorstores, knowledge bases, or contextual compression retrievers),
retrieval spans are recorded with the query and retrieved document IDs and scores.

When message content capture is enabled (``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY``
or ``SPAN_AND_EVENT``), the retrieved documents are serialized into the
``gen_ai.retrieval.documents`` span attribute using the shared ``RetrievalDocument``
model, as a JSON array of objects with only ``id`` and ``score``.
Document text (previously recorded as ``content``) and metadata are not captured.
Query text capture is unchanged.

When available, relevance and similarity scores are captured in each document object under ``score``:

* **Direct retrieval scores**: extracted from ``metadata["score"]`` (populated by retrievers such as
  ``AmazonKnowledgeBasesRetriever``, ``TavilySearchAPIRetriever``, and vectorstore score-threshold searches).
* **Reranking scores**: extracted from ``metadata["relevance_score"]`` (populated when retrievers are wrapped
  with rerankers via ``ContextualCompressionRetriever``, such as ``CohereRerank``).
* **Duck-typed / custom documents**: extracted from a top-level ``score`` attribute or mapping key.

If a document has no score, or if the score is non-numeric or non-finite (``NaN``, ``Infinity``),
``score`` is recorded as JSON ``null`` to ensure RFC 8259 JSON compliance.
Missing document IDs are also recorded as ``null``.

Configuration
-------------

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``.

Prompts and completions can instead be uploaded to external storage via a completion hook: set
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` (install the ``upload`` extra by
changing ``observe-util-genai`` to ``observe-util-genai[upload]`` in the install
command above), or pass a custom ``CompletionHook`` programmatically, which
takes precedence over the environment variable::

    LangChainInstrumentor().instrument(completion_hook=my_hook)

Recording the User
******************

To record ``user.id``, pass it as ``user_id`` in the run's metadata and set
``OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY=true``. LangChain passes the metadata on to the
run's children, so every span the instrumentation creates for the run records it::

    graph.invoke(inputs, config={"metadata": {"user_id": "user-123"}})

It is off by default. Without it, or when the metadata has no ``user_id``, spans record the id set
with ``observeinc.util.genai.user_id.using_user_id``, if any. When both are set, the metadata
value wins. With the async API, spans created by other instrumentations under the run, such as
HTTP client spans, do not get the metadata value, for the same reason as the context propagation
limitation below.

Known Limitations
-----------------

Context propagation to nested calls (such as auto-instrumented HTTP clients
or database queries within tools) is not supported when using LangChain async API.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `LangGraph <https://langchain-ai.github.io/langgraph/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_
