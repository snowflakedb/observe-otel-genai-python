# Recorded CLI sessions

Raw message streams from real `claude-agent-sdk` runs against Claude Code CLI
2.1.220, one JSON object per line exactly as the SDK received it. They are the
evidence behind the ordering rules in `_run.py` — a maintainer changing one can
grep here for what the CLI actually sends, rather than trusting a comment.

All nine run the same scenario: delegate `echo from-subagent` to a subagent and
ask for a written report. They differ in timing, which is the point — whether a
task's receipt beats its notification, whether a backgrounded task finishes
before the result, whether the session sends one result or two.

`test_recordings.py` replays them and asserts the invariants the span tree is
built on.

## Scrubbing

Session, message and tool ids are replaced with deterministic fakes derived from
the original, so the cross-references the stream depends on still line up. The
recording user's home directory is replaced with `/home/agent` and the project
directory with `/home/agent/project`, including the dash-encoded form the CLI uses
for project directories. The host's kernel version and distribution in the
agents' prose are made generic; streamed deltas are rewritten in place, so their
count and timing are unchanged. The `SessionStart` hook output is emptied, and
the `init` message's MCP servers and tools, skills, slash commands and plugins
are dropped. Nothing else is altered.

The content is synthetic — a shell `echo` and prose about it. No private code or
personal data was recorded.
