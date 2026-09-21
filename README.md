# observe-otel-genai-python

Python instrumentation for GenAI frameworks, emitting [OpenTelemetry GenAI
semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) in the
shape LLM Explorer expects.

This repository is a downstream fork of
[open-telemetry/opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai).
It carries only the packages we ship and keeps upstream's directory layout,
import paths, and tests unchanged, so tracking upstream stays cheap. Only the
distribution names differ: we publish `observe-util-genai` where upstream
publishes `opentelemetry-util-genai`, and so on.

Upstream is the default home for anything vendor-neutral. Code lands here when a
requirement can't wait for upstream review, or when upstream won't take it.

## Packages

| Package | Import path | Instruments | Upstream counterpart |
| --- | --- | --- | --- |
| `observe-instrumentation-genai-claude-agent-sdk` | `opentelemetry.instrumentation.genai.claude_agent_sdk` | `claude-agent-sdk >= 0.1.14, < 1` | [`opentelemetry-instrumentation-genai-claude-agent-sdk`](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk) |
| `observe-util-genai` | `opentelemetry.util.genai` | — (shared span/metric/lifecycle layer) | [`opentelemetry-util-genai`](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/util/opentelemetry-util-genai) |
| `observe-test-util-genai` | `opentelemetry.test_util_genai` | — (test fixtures, not published) | [`opentelemetry-test-util-genai`](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/util/opentelemetry-test-util-genai) |

LangChain and OpenAI Agents SDK are not here yet. We have not decided whether to
own them or stay on upstream's packages.

## Using it

Zero-code activation, the same as any OpenTelemetry instrumentation:

```shell
pip install observe-instrumentation-genai-claude-agent-sdk
opentelemetry-instrument python app.py
```

Or activate it yourself:

```python
from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)

ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)
```

Automatic activation only discovers instrumentors. Exporters, resources, and
samplers stay part of your existing OpenTelemetry setup.

### Don't install ours and upstream's side by side

Our distributions install to the same paths as the upstream ones they replace, so
they are mutually exclusive. Remove upstream's first:

```shell
pip uninstall opentelemetry-instrumentation-genai-claude-agent-sdk
pip install observe-instrumentation-genai-claude-agent-sdk
```

Nothing stops you from installing both — Python packaging has no way to declare a
conflict, and `pip check` reports a mixed install as healthy. What actually
happens, verified:

- The two do not cleanly replace each other. You get a **per-file mix**: one
  package's `__init__.py` next to the other's `handler.py`.
- **Uninstalling either one breaks the other.** Both list the same paths in their
  `RECORD`, so removing one deletes files the other still needs.

The case to watch is an app that needs our Claude Agent SDK instrumentation *and*
an upstream instrumentation for another framework, since every upstream GenAI
instrumentation depends on `opentelemetry-util-genai`. Today that combination is
not supported. It stops being a problem once we ship our own packages for the
other frameworks, because then nothing pulls upstream's util in.

`opentelemetry-bootstrap -a install` is the one path that can produce this
silently: it installs instrumentation based on what it finds in the environment,
so an app that merely has `anthropic` present can end up with upstream's
instrumentation and util. Pin your instrumentation explicitly instead.

## Layout

```
observe-otel-genai-python/
├── instrumentation/
│   └── observe-instrumentation-genai-claude-agent-sdk/
│       ├── src/opentelemetry/instrumentation/genai/claude_agent_sdk/
│       ├── tests/
│       └── examples/{manual,zero-code}/
├── util/
│   ├── observe-util-genai/       # forked shared layer, our changes live here
│   └── observe-test-util-genai/  # test fixtures, workspace-internal
├── policies/                     # weaver live-check policies for conformance
├── scripts/
│   └── sync_upstream.sh
└── tox.ini
```

## Development

```shell
uv sync                                   # create the workspace venv
uvx --with tox-uv tox -l                  # list environments
uvx --with tox-uv tox -e py310-test-instrumentation-genai-claude-agent-sdk-latest
uvx --with tox-uv tox -e py310-test-util-genai
uvx --with tox-uv tox -e ruff,typecheck   # lint and pyright
```

Test layout follows upstream: per-package `tests/`, with
`requirements.oldest.txt` and `requirements.latest.txt` driving the `-oldest` and
`-latest` environments so the declared dependency floors are what actually get
exercised.

## Tracking upstream

Upstream is the source of truth. Keep our patch set small, and drop a local
change as soon as the upstream equivalent ships.

```shell
./scripts/sync_upstream.sh          # fetch, merge, keep pruned packages deleted
uvx --with tox-uv tox               # then make sure everything still passes
```

Because the layout and import paths match upstream exactly, a sync produces only
two kinds of conflict:

- **Deleted packages.** Upstream keeps changing the packages we pruned. These are
  the bulk of them and the sync script resolves them by keeping them deleted, so
  they need no review. Measured against one 27-commit upstream window: 71 files.
- **Files we also changed.** Measured against the same window: 3 files, all of
  them in `opentelemetry-util-genai`, which is where our patches live.

### What differs from upstream

Only the distribution name and its metadata:

| | upstream | here |
| --- | --- | --- |
| distribution | `opentelemetry-util-genai` | `observe-util-genai` |
| directory | `util/opentelemetry-util-genai/` | same |
| import path | `opentelemetry.util.genai` | same |
| instrumentor class | `ClaudeAgentSDKInstrumentor` | same |

Everything a user writes — imports, class names, the `OTEL_*` environment
variables, the `opentelemetry_instrumentor` and
`opentelemetry_genai_completion_hook` entry point groups — is identical to
upstream. Switching between the two is `pip uninstall` plus `pip install`, with
no application changes.

The cost of that is the section above: our distributions and upstream's write to
the same files, so they cannot share an environment.

## Still to decide

- **Versioning.** Package versions currently match upstream's (`1.2b0.dev`) so
  version files don't conflict on every rebase. That can't distinguish two of our
  releases built on the same upstream version, so it needs revisiting before we
  publish.
- **Publishing.** No release workflow is wired up. Upstream's release automation
  was removed rather than half-ported. `scripts/build.sh` still builds dists
  locally. Publishing also needs OSS publication review (go/oss-publish) and
  product security review.
- **Conformance tests.** The weaver tooling is here (`policies/`, `.weaver.toml`,
  `versions.env`, `dev-requirements-conformance.txt`) but no package has a
  `tests/test_conformance.py` yet. `tox.ini` shows how to wire one up.
- **LLM Explorer contract tests.** See
  `instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk/tests/test_observe_contract.py`.

## License

Apache-2.0, same as upstream. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
