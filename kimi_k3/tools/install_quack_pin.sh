#!/bin/bash
# Install the pinned quack-flydsl + FlyDSL 0.2.4 for --muon-batch-syrk (G67).
#
#   kimi_k3/tools/install_quack_pin.sh [PREFIX]      # default /opt
#
# Reads kimi_k3/deps/quack-flydsl.pin; see PINS.md SS7. Idempotent.
#
# Deliberately a clone, not a pip install: quack requires nvidia-cutlass-dsl (CUDA)
# and its [amd] extra declares flydsl UNPINNED, which pip would resolve to 0.3.2+
# over the global 0.1.1.dev409 that amd-aiter pins and MoRI EP imports.
set -euo pipefail

PREFIX="${1:-/opt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_FILE="$HERE/../deps/quack-flydsl.pin"

[ -r "$PIN_FILE" ] || { echo "cannot read $PIN_FILE" >&2; exit 1; }
get () { sed -n "s/^$1=//p" "$PIN_FILE" | head -1; }
REPO=$(get REPO); SHA=$(get SHA); BRANCH=$(get BRANCH)
FLYDSL_VERSION=$(get FLYDSL_VERSION); FLYDSL_GLOBAL_KEEP=$(get FLYDSL_GLOBAL_KEEP)
MODULE=$(get MODULE)
QUACK_DIR="$PREFIX/quack-flydsl"
FLY_DIR="$PREFIX/flydsl-$FLYDSL_VERSION"

echo "pin  : $SHA ($BRANCH)"
echo "into : $QUACK_DIR + $FLY_DIR"

# --- quack at the pinned SHA -------------------------------------------------
if [ -d "$QUACK_DIR/.git" ]; then
  git -C "$QUACK_DIR" fetch --quiet origin "$SHA" 2>/dev/null || git -C "$QUACK_DIR" fetch --quiet origin
else
  git clone --quiet "$REPO" "$QUACK_DIR"
fi
git -C "$QUACK_DIR" checkout --quiet "$SHA"
got=$(git -C "$QUACK_DIR" rev-parse HEAD)
[ "$got" = "$SHA" ] || { echo "checkout is $got, wanted $SHA" >&2; exit 1; }

# --- FlyDSL 0.2.4 into its own tree, never over the global one ---------------
if [ ! -d "$FLY_DIR/flydsl" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  # --no-deps: this wheel is unpacked, not installed; nothing may touch the env.
  pip download "flydsl==$FLYDSL_VERSION" --no-deps -d "$tmp" >/dev/null
  mkdir -p "$FLY_DIR"
  python -m zipfile -e "$tmp"/flydsl-"$FLYDSL_VERSION"-*.whl "$FLY_DIR"
fi

# --- the global flydsl must be untouched ------------------------------------
# cd / first: Python puts the cwd on sys.path, so running this from inside the
# unpacked tree reports 0.2.4 and hides a real break. (Cost me a wrong answer.)
global=$(cd / && pip show flydsl 2>/dev/null | sed -n 's/^Version: //p')
if [ "$global" != "$FLYDSL_GLOBAL_KEEP" ]; then
  echo "WARNING: global flydsl is '$global', expected '$FLYDSL_GLOBAL_KEEP'." >&2
  echo "         amd-aiter pins it and MoRI EP imports it -- check the dispatcher." >&2
fi

# --- verify the import actually resolves -----------------------------------
PYTHONPATH="$FLY_DIR:$QUACK_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python -c "import importlib,sys; m=importlib.import_module('$MODULE'); \
    [getattr(m,s) for s in ('batched_tsyrk_ex','can_use_batched_tsyrk')]; \
    print('import ok:', m.__file__)" || {
  echo "the pinned checkout does not import; see PINS.md SS7" >&2; exit 1; }

cat <<EOF

done. Add to the environment (both trees, in this order):

  export PYTHONPATH=$FLY_DIR:$QUACK_DIR:\$PYTHONPATH

then --muon-batch-syrk works. Without it, --muon-batch-ns alone still gives
1.121x at B=48 and needs no external dependency.
EOF
