#!/bin/bash
# Replace the CI image TransformerEngine with TE_SHA from ROCm/TransformerEngine,
# then run the Megatron unit-test suite. Used by te-release-regression.yml.
set -euxo pipefail

: "${TE_BRANCH:?TE_BRANCH is required}"
: "${TE_SHA:?TE_SHA is required}"

echo "=== TE before swap ==="
pip show transformer-engine transformer-engine-torch 2>/dev/null || true
python3 -c "import transformer_engine as te; print(getattr(te, '__version__', '?'), te.__file__)" || true

# Drop every installed TE wheel so leftover HIP extensions cannot be imported.
pip freeze | grep -iE 'transformer.?engine' | cut -d= -f1 | xargs -r pip uninstall -y || true

python3 - <<'PY'
import sys

try:
    import transformer_engine
except ImportError:
    sys.exit(0)
print(
    "transformer_engine still importable after uninstall:",
    transformer_engine.__file__,
    file=sys.stderr,
)
sys.exit(1)
PY

STAGE_DIR="${STAGE_DIR:-/workspace/installs}"
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"
rm -rf TransformerEngine
git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine
git fetch origin "$TE_BRANCH"
git checkout --detach "$TE_SHA"
git submodule update --init --recursive
echo "TransformerEngine at commit: $(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "$TE_SHA"
pip install --no-build-isolation .

echo "=== TE after swap ==="
pip show transformer-engine transformer-engine-torch
python3 -c "import transformer_engine as te; print('TE', getattr(te, '__version__', '?'), te.__file__)"

cd /workspace/Megatron-LM
exec /workspace/Megatron-LM/run_unit_tests.sh
