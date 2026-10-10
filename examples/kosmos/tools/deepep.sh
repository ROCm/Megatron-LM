#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/deepep.sh tree PT_DIR OUT | build OUT: Primus-Turbo's intranode DeepEP as a standalone `deep_ep` package.
#   tree   copy its sources from a Primus-Turbo checkout at PT_COMMIT that has been built (setup.py hipifies the *.cu /
#          *.cpp into the *_hip / *.hip files used here), then apply deps/deepep.patch: torch 2.12's HIP stream type
#          (c10::cuda::CUDAStream), the DeepEP pybind block bound as deep_ep._C, get_rdma_buffer_size_hint 0 when
#          intranode (Megatron calls it unconditionally), the package's imports. No kernel changes. CPU, any host.
#   build  OUT/pylib/deep_ep/_C<ext>.so for GPU_ARCH (default gfx950) with -DDISABLE_ROCSHMEM (no internode, no low
#          latency) and Primus-Turbo's own flags. Run in KOSMOS_CT (hipcc on PATH, ROCM_PATH, torch). CPU only.
# Use: PYTHONPATH=OUT/pylib.
set -euo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case ${1:-} in
tree)
    P=${2:?usage: deepep.sh tree PT_DIR OUT}; D=${3:?usage: deepep.sh tree PT_DIR OUT}
    for f in csrc/kernels/deep_ep/intranode.hip csrc/pytorch/deep_ep/deep_ep_hip.cpp; do
        [ -f "$P/$f" ] || { echo "deepep.sh: no $P/$f: build Primus-Turbo first (setup.py hipifies it)" >&2; exit 1; }
    done
    rm -rf "$D/src" "$D/pylib"
    mkdir -p "$D/src/kernels" "$D/src/pytorch" "$D/src/include/primus_turbo/deep_ep" "$D/pylib/deep_ep"
    for f in intranode.hip layout.hip runtime.hip launch_hip.cuh utils_hip.cuh buffer.cuh; do
        cp "$P/csrc/kernels/deep_ep/$f" "$D/src/kernels/"
    done
    for f in deep_ep_hip.cpp deep_ep_hip.hpp event_hip.hpp; do cp "$P/csrc/pytorch/deep_ep/$f" "$D/src/pytorch/"; done
    for f in arch common dtype macros platform float4 float8 floating_point_utils; do
        cp "$P/csrc/include/primus_turbo/$f.h" "$D/src/include/primus_turbo/"
    done
    cp "$P"/csrc/include/primus_turbo/deep_ep/{api.h,configs.h,config.hpp} "$D/src/include/primus_turbo/deep_ep/"
    cp "$P"/primus_turbo/pytorch/deep_ep/{__init__.py,buffer.py,utils.py} "$D/pylib/deep_ep/"
    patch -s -d "$D" -p1 --no-backup-if-mismatch < "$KIT/deps/deepep.patch"
    echo "deepep.sh: $D/src, $D/pylib/deep_ep from $P ($(git -C "$P" log -1 --format=%h 2>/dev/null))" ;;
build)
    D=${2:?usage: deepep.sh build OUT}
    S=$D/src B=$D/build OUT=$D/pylib/deep_ep
    ROCM=${ROCM_PATH:?set ROCM_PATH}
    HIPCC=${HIPCC:-hipcc}
    TORCH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/torch
    PYINC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
    EXT=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
    ARCH="--offload-arch=${GPU_ARCH:-gfx950}"
    INC="-I$S/include -I$S/kernels -I$S/pytorch -I$ROCM/include"
    mkdir -p "$B" "$OUT"
    # Primus-Turbo setup.py kernel flags without -fgpu-rdc (no cross-TU device symbols once rocSHMEM is out).
    KFLAGS="-O3 $ARCH -fPIC -std=c++20 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
-U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_BFLOAT16_OPERATORS__ \
-U__HIP_NO_BFLOAT16_CONVERSIONS__ -U__HIP_NO_BFLOAT162_OPERATORS__ -U__HIP_NO_BFLOAT162_CONVERSIONS__ \
-fno-offload-uniform-block -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=0 \
-mllvm -amdgpu-early-inline-all=true -Wno-unknown-warning-option -DPRIMUS_TURBO_GFX950 -DDISABLE_ROCSHMEM"
    CFLAGS="-O3 $ARCH -fPIC -std=c++20 -fvisibility=hidden -Wno-unknown-warning-option \
-DPRIMUS_TURBO_GFX950 -DDISABLE_ROCSHMEM -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C \
-I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$PYINC"
    # Primus-Turbo adds this one only if the compiler accepts it.
    if echo 'int main(){return 0;}' | $HIPCC -O3 $ARCH -mllvm -amdgpu-coerce-illegal-types=1 -x hip -c - -o /dev/null 2>/dev/null; then
        KFLAGS="$KFLAGS -mllvm -amdgpu-coerce-illegal-types=1"
    fi
    pids=()
    for f in intranode layout runtime; do $HIPCC $KFLAGS $INC -c "$S/kernels/$f.hip" -o "$B/$f.o" & pids+=($!); done
    for f in deep_ep_hip bindings_deep_ep; do $HIPCC $CFLAGS $INC -c "$S/pytorch/$f.cpp" -o "$B/$f.o" & pids+=($!); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc -eq 0 ] || { echo "deepep.sh: compile failed" >&2; exit 1; }
    $HIPCC -O3 $ARCH -shared -fPIC --hip-link \
        "$B/intranode.o" "$B/layout.o" "$B/runtime.o" "$B/deep_ep_hip.o" "$B/bindings_deep_ep.o" \
        -L"$TORCH/lib" -lc10 -lc10_hip -ltorch -ltorch_cpu -ltorch_hip -ltorch_python -L"$ROCM/lib" -lamdhip64 \
        -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$ROCM/lib" -o "$OUT/_C$EXT"
    echo "deepep.sh: built $OUT/_C$EXT" ;;
*)
    echo "usage: deepep.sh tree PT_DIR OUT | build OUT" >&2; exit 2 ;;
esac
