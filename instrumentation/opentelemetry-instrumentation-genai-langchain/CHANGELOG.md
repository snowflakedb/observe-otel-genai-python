# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Do *NOT* add changelog entries here!

This changelog is managed by towncrier and is compiled at release time.

See [CONTRIBUTING.md](../../CONTRIBUTING.md#4-update-the-changelog) for details.
-->

<!-- changelog start -->

## Version 0.1.0 (2026-11-02)

First release. It is based on upstream
[opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai)
at commit `068606f9`, and the entries below describe how it differs from
upstream's last release, `1.3b0`.

### Added

- Add support for Python 3.15.
  ([#992](https://github.com/open-telemetry/opentelemetry-python-genai/pull/992))
- Record a run's `user_id` metadata as `user.id` on its spans when
  `OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY` is `true`.

### Changed

- Import as `observeinc.instrumentation.genai.langchain` instead of upstream's
  path, so this package can be installed next to the upstream OpenTelemetry
  GenAI packages without overwriting their files.
- Publish as `observe-instrumentation-genai-langchain`, depending on
  `observe-util-genai` in place of `opentelemetry-util-genai`. Use it in place of
  upstream's `opentelemetry-instrumentation-genai-langchain`.

### Removed

- Remove ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` environment variable.
  ``gen_ai.client.inference.operation.details`` events are now emitted by
  default whenever the logger is enabled and can be disabled via logger
  configuration, see
  https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/examples/suppress-inference-event/README.rst
  for examples.
  ([#836](https://github.com/open-telemetry/opentelemetry-python-genai/pull/836))
- Drop support for Python 3.10.
  ([#992](https://github.com/open-telemetry/opentelemetry-python-genai/pull/992))

### Fixed

- Record cache and reasoning token counts for langchain-openai flex and
  priority service tier requests.
  ([#1004](https://github.com/open-telemetry/opentelemetry-python-genai/pull/1004))
- Set `gen_ai.conversation.id` on retrieval and execute_tool spans from a
  standalone retriever or tool call's `thread_id`, `session_id` or
  `conversation_id` metadata.
