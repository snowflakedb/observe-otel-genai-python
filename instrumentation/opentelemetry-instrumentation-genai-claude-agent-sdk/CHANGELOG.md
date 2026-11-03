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

First release.

### Added

- Instrument `query()` and `ClaudeSDKClient`, building the span tree from the
  SDK's message stream: one `invoke_agent` span per invocation, one `chat` span
  per API call, an `execute_tool` span per tool call, and nested `invoke_agent`
  spans for delegated subagents.
- Accept `ClaudeAgentSDKInstrumentor().instrument(agent_name="...")` to name the
  root `invoke_agent` span, for a caller whose own application is the agent
  being invoked rather than the SDK. It defaults to `claude-agent-sdk` and is
  never inferred from the caller's options.
