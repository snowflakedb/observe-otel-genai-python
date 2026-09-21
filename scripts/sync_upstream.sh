#!/bin/bash
# Merge upstream opentelemetry-python-genai into this fork.
#
# This fork keeps upstream's directory layout and import paths, so the only
# conflicts a sync produces are:
#
#   1. Upstream changed a package we pruned. Resolved here by keeping it deleted.
#      This is the bulk of them and needs no review.
#   2. Upstream changed a file we also changed. Left for you.
#
# The merge is never committed: review it, run the tests, then commit yourself.
#
# Usage: scripts/sync_upstream.sh [upstream-ref]     (default: main)

set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_REF="${1:-main}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  echo "No '$UPSTREAM_REMOTE' remote. Add it with:" >&2
  echo "  git remote add $UPSTREAM_REMOTE https://github.com/open-telemetry/opentelemetry-python-genai.git" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean. Commit or stash first." >&2
  exit 1
fi

echo "==> Fetching $UPSTREAM_REMOTE/$UPSTREAM_REF"
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_REF"

echo "==> Merging $UPSTREAM_REMOTE/$UPSTREAM_REF (no commit)"
# git reports every pruned package as a conflict before we resolve them, which
# buries the few that matter. Keep the full output in a log and summarise below.
merge_log="$(mktemp -t sync-upstream-merge.XXXXXX)"
git merge --no-commit --no-ff "$UPSTREAM_REMOTE/$UPSTREAM_REF" >"$merge_log" 2>&1 || true
echo "    full merge output: $merge_log"

# "DU" is git's code for deleted-by-us: a file we pruned that upstream still
# maintains. Keep it deleted. During a merge "us" is this branch, so DU is the
# right code to look for; it would be UD during a rebase.
deleted_by_us=$(git status --porcelain | sed -n 's/^DU //p')
if [ -n "$deleted_by_us" ]; then
  count=$(echo "$deleted_by_us" | wc -l | tr -d ' ')
  echo "==> Keeping $count pruned file(s) deleted"
  echo "$deleted_by_us" | xargs -r git rm -q -f --
fi

remaining=$(git diff --name-only --diff-filter=U || true)
if [ -n "$remaining" ]; then
  echo
  echo "Conflicts left for you to resolve:"
  echo "$remaining" | sed 's/^/    /'
else
  echo "==> No conflicts left to resolve by hand."
fi

cat <<'NEXT'

Next:
  uv lock                          # if any dependency moved
  uvx --with tox-uv tox            # run everything
  git status                       # review, then commit the merge yourself
NEXT
