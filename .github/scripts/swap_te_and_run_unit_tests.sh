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
# Do not parse `pip freeze`: editable/direct-URL lines look like
# `transformer-engine @ file:///...` and `cut -d=` feeds `@` to pip.
python3 - <<'PY'
import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path

names = []
for dist in md.distributions():
    name = dist.metadata["Name"]
    key = name.lower().replace("_", "-")
    if "transformer-engine" in key or key in {"transformer-engine", "transformer-engine-torch"}:
        names.append(name)

# Always try the published names too (pip freeze may not list them).
for extra in ("transformer-engine", "transformer-engine-torch", "transformer_engine"):
    if extra not in names:
        names.append(extra)

print("Uninstalling:", names)
subprocess.run(["pip", "uninstall", "-y", *names], check=False)

# Direct-URL / leftover .so trees can survive pip uninstall.
try:
    import transformer_engine
except ImportError:
    sys.exit(0)

pkg = Path(transformer_engine.__file__).resolve().parent
print("Removing leftover TE tree:", pkg, file=sys.stderr)
shutil.rmtree(pkg, ignore_errors=True)
for distinfo in pkg.parent.glob("transformer_engine*.dist-info"):
    shutil.rmtree(distinfo, ignore_errors=True)
for distinfo in pkg.parent.glob("transformer_engine*.egg-info"):
    shutil.rmtree(distinfo, ignore_errors=True)

sys.modules.pop("transformer_engine", None)
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
