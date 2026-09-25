#!/bin/bash
# Merge upstream opentelemetry-python-genai into this fork.
#
# This fork keeps upstream's directory layout and import paths, so a sync only
# has to deal with the prune:
#
#   1. Upstream changed a file we pruned. Git reports it as a conflict; resolved
#      here by keeping it deleted. This is the bulk of them and needs no review.
#   2. Upstream added a file inside a package we pruned. Git merges that cleanly,
#      with no conflict to notice, so the prune would silently rot. Also removed
#      here.
#   3. Upstream changed a file we also changed. Left for you.
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

# Paths this fork does not carry. Upstream additions under them merge cleanly, so
# the prune has to be re-applied rather than merely defended by conflicts.
KEEP_UNDER_INSTRUMENTATION='^instrumentation/(AGENTS\.md|CLAUDE\.md|opentelemetry-instrumentation-genai-claude-agent-sdk/)'
KEEP_WORKFLOWS='^\.github/workflows/(ci\.yml|test\.yml|misc\.yml|changelog\.yml|check-links\.yml|check_links_config\.json)$'
PRUNED_PATHS=(
  docs
  .readthedocs.yml
  docs-requirements.txt
  rtd-requirements.txt
  RELEASING.md
  .github/ISSUE_TEMPLATE
  .github/actions
  .github/scripts
  .github/renovate.json5
  .github/copilot-instructions.md
)

crept=$(
  {
    git ls-files -- instrumentation | grep -Ev "$KEEP_UNDER_INSTRUMENTATION" || true
    git ls-files -- .github/workflows | grep -Ev "$KEEP_WORKFLOWS" || true
    git ls-files -- "${PRUNED_PATHS[@]}" 2>/dev/null || true
  } | sort -u
)
if [ -n "$crept" ]; then
  count=$(echo "$crept" | wc -l | tr -d ' ')
  echo "==> Removing $count file(s) that crept back into pruned paths"
  echo "$crept" | xargs -r git rm -q -f --
  # An emptied directory still matches the uv workspace glob and breaks uv lock.
  find instrumentation -mindepth 1 -type d -empty -delete 2>/dev/null || true
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
  UV_DEFAULT_INDEX=https://pypi.org/simple uv lock   # if any dependency moved
  uvx --with tox-uv tox            # run everything
  git status                       # review, then commit the merge yourself

main only takes PRs. Push the merge to a branch and open a PR with the
"Skip Changelog" label (upstream releases edit CHANGELOG.md directly). Merge
it with "Create a merge commit": a squash or rebase drops upstream's history,
and the next sync would conflict on everything again.
NEXT
