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

### Changed

- Import as `observeinc.instrumentation.genai.openai` instead of upstream's
  path, so this package can be installed next to the upstream OpenTelemetry
  GenAI packages without overwriting their files.
- Publish as `observe-instrumentation-genai-openai`, depending on
  `observe-util-genai` in place of `opentelemetry-util-genai`. Use it in place of
  upstream's `opentelemetry-instrumentation-genai-openai`.

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

- Fix embedding dimension counts for explicitly requested base64 responses
  without changing the SDK's returned embeddings.
  ([#981](https://github.com/open-telemetry/opentelemetry-python-genai/pull/981))
