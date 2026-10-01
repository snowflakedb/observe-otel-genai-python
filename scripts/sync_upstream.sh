#!/bin/bash
# Merge upstream opentelemetry-python-genai into this fork.
#
# This fork differs from upstream in two mechanical ways: it carries only some of
# upstream's packages (the prune), and it moves them under the `observeinc`
# import namespace (scripts/rename_namespace.py). Merging upstream as-is would
# make git reconcile both with every upstream change, so this script first
# applies the same prune and rename to a copy of upstream and merges that copy
# instead. Each copy is committed with two parents: the upstream commit it came
# from, and the copy the previous sync merged. Git then compares the new copy
# with the previous one, which differ only by what upstream changed, so the
# rename never conflicts. What is left to resolve is:
#
#   1. Upstream changed a file we deleted outside the prune. Kept deleted here,
#      and listed so you can check none of them matter.
#   2. A file we carry at a path the prune drops from upstream (for example our
#      own RELEASING.md). Kept as ours here.
#   3. Upstream changed a file we also changed. Left for you.
#
# Upstream is the source of truth: keep our patch set small, and drop a local
# change once its upstream equivalent ships.
#
# The merge is never committed: review it, run the tests
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
upstream_sha="$(git rev-parse "$UPSTREAM_REMOTE/$UPSTREAM_REF")"

PREPARED_SUBJECT="Prune and rename upstream"
previous_prepared="$(git log -1 --format=%H --grep="^$PREPARED_SUBJECT " HEAD)"
if [ -z "$previous_prepared" ]; then
  echo "No earlier '$PREPARED_SUBJECT' commit in this branch's history." >&2
  exit 1
fi

# Paths this fork does not carry.
KEEP_UNDER_INSTRUMENTATION='^instrumentation/(AGENTS\.md|CLAUDE\.md|opentelemetry-instrumentation-genai-(claude-agent-sdk|langchain|openai|openai-agents)/)'
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

echo "==> Preparing upstream ${upstream_sha:0:8}: prune, then rename to observeinc"
prep_dir="$(mktemp -d -t sync-upstream.XXXXXX)"
git worktree add -q --detach "$prep_dir" "$upstream_sha"
trap 'git worktree remove --force "$prep_dir"' EXIT
(
  cd "$prep_dir"
  {
    git ls-files -- instrumentation | grep -Ev "$KEEP_UNDER_INSTRUMENTATION" || true
    git ls-files -- .github/workflows | grep -Ev "$KEEP_WORKFLOWS" || true
    git ls-files -- "${PRUNED_PATHS[@]}" 2>/dev/null || true
  } | sort -u | xargs -r git rm -q -f --
  python3 "$REPO_ROOT/scripts/rename_namespace.py" >/dev/null
  git add -A
)
prepared_tree="$(git -C "$prep_dir" write-tree)"
prepared_sha="$(git commit-tree "$prepared_tree" -p "$upstream_sha" -p "$previous_prepared" \
  -m "$PREPARED_SUBJECT ${upstream_sha:0:8} for this fork")"

echo "==> Merging the prepared upstream (no commit)"
merge_log="$(mktemp -t sync-upstream-merge.XXXXXX)"
git merge --no-commit --no-ff "$prepared_sha" >"$merge_log" 2>&1 || true
echo "    full merge output: $merge_log"

# During a merge "us" is this branch: DU is a file we deleted that upstream
# changed, UD a file we have that the prepared upstream does not.
deleted_by_us=$(git status --porcelain | sed -n 's/^DU //p')
if [ -n "$deleted_by_us" ]; then
  count=$(echo "$deleted_by_us" | wc -l | tr -d ' ')
  echo "==> Keeping $count file(s) we deleted, which upstream changed:"
  echo "$deleted_by_us" | sed 's/^/    /'
  echo "$deleted_by_us" | xargs -r git rm -q -f --
fi

# A UD file still present in real upstream was dropped by the prune, so it is
# ours to keep. One upstream itself deleted is a real conflict.
deleted_by_them=$(git status --porcelain | sed -n 's/^UD //p')
for path in $deleted_by_them; do
  if git cat-file -e "$upstream_sha:$path" 2>/dev/null; then
    git checkout -q --ours -- "$path"
    git add -- "$path"
  fi
done

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
