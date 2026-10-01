# observe-otel-genai-python

Python instrumentation for GenAI frameworks, emitting [OpenTelemetry GenAI
semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) in the
shape Observe's LLM Explorer expects.

## Packages

| Package | Import path | Instruments |
| --- | --- | --- |
| [`observe-instrumentation-genai-claude-agent-sdk`](instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk/README.rst) | `observeinc.instrumentation.genai.claude_agent_sdk` | `claude-agent-sdk >= 0.2.157, < 0.3` |
| [`observe-instrumentation-genai-langchain`](instrumentation/opentelemetry-instrumentation-genai-langchain/README.rst) | `observeinc.instrumentation.genai.langchain` | `langchain >= 0.3.21, < 2` |
| [`observe-instrumentation-genai-openai`](instrumentation/opentelemetry-instrumentation-genai-openai/README.rst) | `observeinc.instrumentation.genai.openai` | `openai >= 1.26.0, < 4` |
| [`observe-instrumentation-genai-openai-agents`](instrumentation/opentelemetry-instrumentation-genai-openai-agents/README.rst) | `observeinc.instrumentation.genai.openai_agents` | `openai-agents >= 0.3.3, < 1` |
| [`observe-util-genai`](util/opentelemetry-util-genai/README.rst) | `observeinc.util.genai` | — (shared telemetry layer every package uses) |

## Usage

Each package's README, linked above, shows how to install and enable it, with
runnable examples in the `examples/` directory next to it. Settings that apply to
every package, such as the opt-in conversation id on metrics, are described in the
[`observe-util-genai` README](util/opentelemetry-util-genai/README.rst).

## License

Apache-2.0. Derived from
[opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai);
see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
