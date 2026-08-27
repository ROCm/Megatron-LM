#!/usr/bin/env bash
# Gate G9 -- nothing outside kimi_k3/ may change.
#
# The Kimi K3 integration is fork-local so that IFU merges from upstream never
# conflict with it (rule R2.1). This guard is what makes that a fact rather than
# an intention.
#
# Usage:
#   kimi_k3/ci/no_core_diff_guard.sh                  # branch vs its merge-base with rocm_dev
#   kimi_k3/ci/no_core_diff_guard.sh --staged         # what is staged right now (pre-commit)
#   kimi_k3/ci/no_core_diff_guard.sh <base>..<head>   # an explicit range (CI)
#
# The default compares against the **merge-base**, not against rocm_dev's tip:
# after an IFU rebase the branch legitimately contains upstream's core commits,
# and a tip comparison would flag every one of them.
set -euo pipefail

ALLOWED_ROOT_FILES=(
  ".github/workflows/kimi_k3.yml"
  ".pre-commit-config.yaml"
)
UPSTREAM_BRANCH="${K3_UPSTREAM_BRANCH:-rocm_dev}"

cd "$(git rev-parse --show-toplevel)"

case "${1:-}" in
  --staged)
    mode="staged"
    changed=$(git diff --cached --name-only)
    ;;
  "")
    mode="merge-base vs ${UPSTREAM_BRANCH}"
    base=$(git merge-base HEAD "${UPSTREAM_BRANCH}")
    changed=$(git diff --name-only "${base}" HEAD)
    ;;
  *)
    mode="range $1"
    changed=$(git diff --name-only "$1")
    ;;
esac

offenders=""
while IFS= read -r path; do
  [ -z "$path" ] && continue
  case "$path" in
    kimi_k3/*) continue ;;
  esac
  allowed=false
  for f in "${ALLOWED_ROOT_FILES[@]}"; do
    [ "$path" = "$f" ] && allowed=true && break
  done
  $allowed || offenders="${offenders}${path}"$'\n'
done <<< "$changed"

n_total=$(printf '%s' "$changed" | grep -c . || true)

if [ -n "$offenders" ]; then
  echo "FAIL: changes outside kimi_k3/ (${mode})" >&2
  printf '%s' "$offenders" | sed 's/^/  /' >&2
  echo >&2
  echo "kimi_k3/ is fork-local so IFU merges never conflict with it (rule R2.1)." >&2
  echo "Allowed outside it: ${ALLOWED_ROOT_FILES[*]}" >&2
  echo "If a core change is genuinely required, it is an upstream_proposals/ patch" >&2
  echo "plus an issue draft plus human sign-off -- never a direct edit (rule R2.3)." >&2
  exit 1
fi

echo "OK: ${n_total} changed file(s), all within kimi_k3/ or the allowlist (${mode})"
