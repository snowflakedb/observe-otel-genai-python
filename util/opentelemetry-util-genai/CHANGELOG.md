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

- Add agent call count metrics on LocalAgentInvocation.
  ([#634](https://github.com/open-telemetry/opentelemetry-python-genai/pull/634))
- Check if logger is enabled before emitting inference details event, and set
  event severity to DEBUG per GenAI semantic conventions.
  ([#836](https://github.com/open-telemetry/opentelemetry-python-genai/pull/836))
- Add support for Python 3.15.
  ([#992](https://github.com/open-telemetry/opentelemetry-python-genai/pull/992))
- Add `OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY`, which lets instrumentations
  record a user id passed to the instrumented library as `user.id`, through
  `observeinc.util.genai.user_id.with_library_user_id`.
- Add `OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID`, which puts
  `gen_ai.conversation.id` on every metric when set to `true`.
- Add `OBSERVE_GENAI_METRICS_INCLUDE_USER_ID`, which puts `user.id` on every
  metric when set to `true`.
- Add a `conversation_id` parameter to `handler.tool()` and
  `handler.retrieval()`, matching `inference()` and `workflow()`.
- Add the `ALLOW_LIST` value for
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. It captures no content,
  and every span, event and metric keeps only the attributes on a fixed
  allowlist, with no status description or exception message.
- Cap each captured content attribute at 1,048,576 characters by default,
  keeping JSON values valid. Set `OBSERVE_GENAI_MAX_CONTENT_LENGTH` to change
  the cap, or to `0` to turn it off.
- Record `user.id` on every GenAI span started inside
  `observeinc.util.genai.user_id.using_user_id`.
- Record the enclosing agent's `gen_ai.agent.name` on inference, embeddings,
  retrieval and fetch-response spans and their metrics, and on tools created
  without one. It travels on the OTel Context like the conversation id, and
  those handler factories accept an explicit `agent_name`.
- Record the inference token metrics of semantic-conventions-genai#374,
  alongside `gen_ai.client.token.usage`.
- Set `gen_ai.conversation.id` on every span, including tool, embeddings,
  retrieval and fetch-response spans, so all of a conversation's spans appear
  together in Observe.

### Changed

- Import as `observeinc.util.genai` instead of upstream's path, so this package
  can be installed next to the upstream OpenTelemetry GenAI packages without
  overwriting their files.
- Leave the status description off failed spans under `NO_CONTENT` and
  `EVENT_ONLY`, since the error message can quote the prompt. The span keeps
  its ERROR status and `error.type`.

### Deprecated

- Deprecate ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` environment variable; it
  is now ignored. ``gen_ai.client.inference.operation.details`` events are now
  emitted by default whenever the logger is enabled and can be disabled via
  logger configuration, see
  https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/examples/suppress-inference-event/README.rst
  for examples.
  ([#836](https://github.com/open-telemetry/opentelemetry-python-genai/pull/836))
- Deprecate `record_stream_chunk()` on invocation types other than
  `InferenceInvocation`, where it is now a no-op.
  ([#880](https://github.com/open-telemetry/opentelemetry-python-genai/pull/880))

### Removed

- Drop support for Python 3.10.
  ([#992](https://github.com/open-telemetry/opentelemetry-python-genai/pull/992))

### Fixed

- Record the `gen_ai.client.operation.time_to_first_chunk` and
  `gen_ai.client.operation.time_per_output_chunk` metrics only for streamed
  inference invocations; streamed agent, workflow and fetch_response
  invocations no longer emit them.
  ([#880](https://github.com/open-telemetry/opentelemetry-python-genai/pull/880))
