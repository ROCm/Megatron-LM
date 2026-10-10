#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/megamoe_shim.sh PT_DIR SHIM: SHIM/primus_turbo, Primus-Turbo's python tree importable in KOSMOS_CT for the
# megamoe arm (PYTHONPATH=SHIM). A tree of symlinks to PT_DIR/primus_turbo; only the package inits of pytorch,
# pytorch/core, pytorch/ops and pytorch/ops/moe are replaced by empty ones, so the compiled _C (built for the Primus
# image's torch) and DeepEP are never imported. MegaMoE itself is FlyDSL kernels, torch and a ctypes HIP wrapper; no
# kernel or op file is changed. CPU, any host; rerun after PT_DIR changes.
set -euo pipefail
P=$(cd "${1:?usage: megamoe_shim.sh PT_DIR SHIM}" && pwd)
D=${2:?usage: megamoe_shim.sh PT_DIR SHIM}
[ -f "$P/primus_turbo/pytorch/ops/moe/fused_mega_moe.py" ] || { echo "megamoe_shim.sh: $P has no fused_mega_moe" >&2; exit 1; }
mkdir -p "$D"
rm -rf "$D/primus_turbo"
cp -as "$P/primus_turbo" "$D/"
find "$D/primus_turbo" -depth -name __pycache__ -exec rm -rf {} +
for d in pytorch pytorch/core pytorch/ops pytorch/ops/moe; do
    rm -f "$D/primus_turbo/$d/__init__.py"
    echo "# KOSMOS kit MegaMoE shim: package init without the compiled _C / DeepEP / other ops" > "$D/primus_turbo/$d/__init__.py"
done
echo "megamoe_shim.sh: $D/primus_turbo -> $P ($(git -C "$P" log -1 --format=%h 2>/dev/null))"
