#!/bin/bash
# Bring upstream opentelemetry-python-genai changes into this fork.
#
# This fork differs from upstream in two mechanical ways: it carries only some of
# upstream's packages (the prune), and it moves them under the `observeinc`
# import namespace (scripts/rename_namespace.py).
#
# UPSTREAM_COMMIT records the upstream commit we last synced to. This
# script prunes and renames a copy of that commit and of the new one, and
# applies the difference between the two copies to our tree as a three-way
# merge. Both copies already carry the prune and the rename, so the difference
# is only what upstream changed. What is left to resolve is:
#
#   1. Upstream changed a file we deleted. Kept deleted here, and listed so you
#      can check none of them matter.
#   2. Upstream changed a file we also changed. Left for you.
#
# Upstream is the source of truth: keep our patch set small, and drop a local
# change once its upstream equivalent ships.
#
# Nothing is committed: review the result, run the tests
# (uvx --with tox-uv tox), then commit yourself.
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
old_sha="$(cat UPSTREAM_COMMIT)"
new_sha="$(git rev-parse "$UPSTREAM_REMOTE/$UPSTREAM_REF")"

if [ "$old_sha" = "$new_sha" ]; then
  echo "Already at upstream ${new_sha:0:8}."
  exit 0
fi
echo "==> Upstream ${old_sha:0:8}..${new_sha:0:8}: $(git rev-list --count "$old_sha..$new_sha") commit(s)"

# Paths this fork does not carry.
KEEP_UNDER_INSTRUMENTATION='^instrumentation/(AGENTS\.md|CLAUDE\.md|opentelemetry-instrumentation-genai-(langchain|openai|openai-agents)/)'
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

old_dir="$(mktemp -d -t sync-upstream.XXXXXX)"
new_dir="$(mktemp -d -t sync-upstream.XXXXXX)"
trap 'git worktree remove --force "$old_dir"; git worktree remove --force "$new_dir"' EXIT

# Checks out upstream commit $1 in directory $2, prunes and renames it, and prints
# a commit holding the result. The commit has no parent: it only exists to give
# the merge below its inputs.
prepare() {
  git worktree add -q --detach "$2" "$1"
  (
    cd "$2"
    {
      git ls-files -- instrumentation | grep -Ev "$KEEP_UNDER_INSTRUMENTATION" || true
      git ls-files -- .github/workflows | grep -Ev "$KEEP_WORKFLOWS" || true
      git ls-files -- "${PRUNED_PATHS[@]}" 2>/dev/null || true
    } | sort -u | xargs -r git rm -q -f --
    python3 "$REPO_ROOT/scripts/rename_namespace.py" >/dev/null
    git add -A
    git commit-tree "$(git write-tree)" -m "upstream ${1:0:8}, pruned and renamed"
  )
}

echo "==> Pruning and renaming both upstream commits"
old_copy="$(prepare "$old_sha" "$old_dir")"
new_copy="$(prepare "$new_sha" "$new_dir")"

echo "==> Applying upstream's changes"
merge_log="$(mktemp -t sync-upstream-merge.XXXXXX)"
export "GITHEAD_$new_copy=upstream ${new_sha:0:8}"
git merge-recursive "$old_copy" -- HEAD "$new_copy" >"$merge_log" 2>&1 || true
echo "    full merge output: $merge_log"
echo "$new_sha" >UPSTREAM_COMMIT
git add UPSTREAM_COMMIT

# During the merge "us" is this branch: DU is a file we deleted that upstream
# changed.
deleted_by_us=$(git status --porcelain | sed -n 's/^DU //p')
if [ -n "$deleted_by_us" ]; then
  count=$(echo "$deleted_by_us" | wc -l | tr -d ' ')
  echo "==> Keeping $count file(s) we deleted, which upstream changed:"
  echo "$deleted_by_us" | sed 's/^/    /'
  echo "$deleted_by_us" | xargs -r git rm -q -f --
fi

# An emptied directory still matches the uv workspace glob and breaks uv lock.
find instrumentation -mindepth 1 -type d -empty -delete 2>/dev/null || true

remaining=$(git diff --name-only --diff-filter=U || true)
if [ -n "$remaining" ]; then
  echo
  echo "Conflicts left for you to resolve:"
  echo "$remaining" | sed 's/^/    /'
else
  echo "==> No conflicts left to resolve by hand."
fi

cat <<NEXT

Next:
  UV_DEFAULT_INDEX=https://pypi.org/simple uv lock   # if any dependency moved
  uvx --with tox-uv tox            # run everything
  git status                       # review, then commit, for example as
                                   # "Sync upstream ${old_sha:0:8}..${new_sha:0:8}"

main only takes PRs. Push the sync to a branch and open a PR with the
"Skip Changelog" label (upstream releases edit CHANGELOG.md directly).
To undo the sync instead: git reset --hard HEAD
NEXT
