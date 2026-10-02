# Contributing

This is a downstream fork of
[open-telemetry/opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai).
Read [README.md](README.md) first for which packages it carries.

If you are using AI agents to help, read [AGENTS.md](AGENTS.md) too.

## Before you write code here

Upstream is the default home for anything vendor-neutral: framework version
support, correctness fixes, missing standard attributes, new semantic
conventions. Those belong in an upstream PR, not here.

Change this repo when one of these is true:

- Upstream has rejected the change, or won't get to it in time for our ship date.
- The behaviour is specific to what Observe needs and isn't generally useful.
- We carry a fix while the upstream PR is still in review.

For the third case, link the upstream PR in your PR description and delete the
local change once upstream ships it. Every patch we carry is a merge conflict we
pay for on each sync.

The Claude Agent SDK package
(`instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk`) is our
own, not a fork: a sync leaves it alone, and the rules above don't apply to it.
Files we write there start with `# Copyright Snowflake Inc.`; the few that came
from upstream keep their `Copyright The OpenTelemetry Authors` header.

## Prerequisites

- [Python](https://www.python.org/downloads/) — see [`tox.ini`](tox.ini) for the
  supported versions.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — manages the
  workspace and backs the `tox` environments.

## Code structure

```
├── instrumentation/
│   └── opentelemetry-instrumentation-genai-<name>/  # one package per GenAI library
│       ├── src/observeinc/instrumentation/genai/<name>/
│       ├── tests/
│       ├── CHANGELOG.md
│       └── pyproject.toml
└── util/
    ├── opentelemetry-util-genai/                    # shared GenAI utilities
    │   ├── src/observeinc/util/genai/
    │   ├── tests/
    │   ├── CHANGELOG.md
    │   └── pyproject.toml
    └── opentelemetry-test-util-genai/               # test fixtures, not published
```

The monorepo uses `uv` workspaces; each package owns its own `pyproject.toml`,
version, and entry points. `tox.ini` defines the test matrix.

## Package naming

Instrumentation packages are named `observe-instrumentation-genai-<name>` and
import as `observeinc.instrumentation.genai.<name>`.

The import paths are our own so that our packages and upstream's can be
installed together without overwriting each other's files.
`scripts/rename_namespace.py` moves upstream's `opentelemetry.util.genai` and
`opentelemetry.instrumentation.genai.*` under `observeinc`, and
`scripts/sync_upstream.sh` applies the same rename to upstream's code before
bringing in its changes, so the rename itself never conflicts. Class names,
environment variables, entry point groups, and directory names are upstream's,
unchanged. Don't rename anything else: every extra difference is a conflict on
the next sync.

Package versions still match upstream's. We plan to move to our own version
line, shared by all packages, before the first release.

## Making a change

### 1. Set up the environment

```sh
uv sync --frozen --all-packages
```

### 2. Lint

```sh
uv run pre-commit run ruff --all-files
```

### 3. Test

Run the environment for the package you changed (`-oldest` and `-latest` pick up
`tests/requirements.{oldest,latest}.txt`):

```sh
uv run tox -e py310-test-instrumentation-genai-claude-agent-sdk-latest
uv run tox -e py310-test-util-genai
```

Type check the workspace:

```sh
uv run tox -e typecheck
```

Keep upstream's test files and test names intact where you can. A test that
still matches upstream's is a test that merges cleanly. When we carry a fix or an
intentional difference from upstream, add a separate `test_observe_*.py` file
next to the upstream tests rather than editing them, so the difference is obvious
during a rebase.

#### Managing cassettes (test recordings)

Tests replay recorded HTTP interactions (cassettes) under each package's
`tests/cassettes/`. The Claude Agent SDK has no HTTP layer in our process — it
talks to a CLI subprocess — so it drives synthetic messages instead, but any
package that does call a provider follows these rules:

- **Run**: nothing extra — cassettes replay automatically when present. Tests
  that need a cassette skip if it is missing and no real API key is set.
- **Record**: delete the target `tests/cassettes/<test_name>.yaml`, export a real
  provider API key (e.g. `ANTHROPIC_API_KEY`), and rerun the test. `pytest-vcr`
  writes the cassette on the live call.
- **Sanitize**: every package's `vcr_config()` in `tests/conftest.py` must scrub
  auth via `filter_headers` and strip identifying response headers via
  `scrub_response_headers(...)` from `observeinc.test_util_genai.vcr`. Diff each new
  cassette before committing — leaked API keys, org ids, or `Set-Cookie` values
  block the PR.
- **CI**: replay only.

### 4. Update the changelog

This repo uses [towncrier](https://towncrier.readthedocs.io/). Add a fragment
under the affected package's `.changelog/` directory instead of editing
`CHANGELOG.md`.

**Types:** `added`, `changed`, `deprecated`, `removed`, `fixed`.

There are two fragment names, and which one you use matters here:

- `<package>/.changelog/<PR_NUMBER>.<type>` — for a change that has an upstream
  PR with that number. Changelog links point at upstream, because most entries in
  this repo come from upstream.
- `<package>/.changelog/+<short-name>.<type>` — for a fork-only change. The `+`
  tells towncrier to render it without a link, which is what we want since our PR
  numbers are not upstream's.

The file holds a one-line description. For example,
`util/opentelemetry-util-genai/.changelog/+parent-context-agent-inference.added`:

```
carry parent context through agent and inference invocations
```

Don't include the PR number in the body — towncrier appends it from the filename.

Preview locally:

```sh
uv run tox -e changelog-preview
```

If your change needs no entry (pure docs or tooling), add the `Skip Changelog`
label.

## Syncing with upstream

```sh
./scripts/sync_upstream.sh
```

It fetches upstream, merges without committing, and keeps the packages we pruned
deleted. Then review the result, run the tests, and commit the merge yourself.
The header of `scripts/sync_upstream.sh` explains the conflicts to expect.

Don't cherry-pick individual upstream commits. Merging keeps the shared history
that makes the next sync cheap.

## Skills

`.github/skills/` carries upstream's skills for the heavy, repeatable flows:

- [**`migrate-from-openinference`**](.github/skills/migrate-from-openinference/SKILL.md)
  — port an `openinference-instrumentation-*` package into this layout, or
  augment an existing package with the coverage OpenInference adds.
- [**`review-migration`**](.github/skills/review-migration/SKILL.md) — review a
  ported package against its original implementation.
- [**`write-conformance-tests`**](.github/skills/write-conformance-tests/SKILL.md)
  — author conformance scenarios and a `test_conformance.py` runner.

## Keep PRs small

One logical change per PR. Don't bundle unrelated fixes, refactors, or features —
split them so each can be reviewed and reverted independently.
