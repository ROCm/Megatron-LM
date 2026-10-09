#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/setup.sh [--checkout] [--build] [--build-deps]: verifies the three branches, the builds and every dependency.
# Read-only by default. --checkout switches a clean tree to its branch; --build runs `make all python` in KOSMOS and the
# in-place TE build against it; --build-deps rebuilds DeepEP and Primus-Turbo. Builds run in the containers, CPU only.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
CHECKOUT=0; BUILD=0; DEPS=0
for a in "$@"; do
    case $a in
        --checkout)   CHECKOUT=1 ;;
        --build)      BUILD=1 ;;
        --build-deps) DEPS=1 ;;
        *)            die "unknown setup flag $a (--checkout --build --build-deps)" ;;
    esac
done
O=$OUT_ROOT/setup
run mkdir -p "$O"
BAD=0
bad() { warn "$*"; BAD=$((BAD + 1)); }

log "node: $GPU_NAME ($GPU_ARCH), $NGPU GPUs, power cap ${POWER_CAP_W:-?} W, HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
[ "$GPU_ARCH" = gfx950 ] || bad "GPU_ARCH=$GPU_ARCH: every KOSMOS kernel here is gfx950 only"
[ -f "$KOSMOS_SITE" ] && log "site file: $KOSMOS_SITE"

# repo DIR BRANCH COMMIT: the tree is on BRANCH at COMMIT (git log / diff only; checkout only with --checkout).
repo() {
    local d=$1 b=$2 c=$3 head refs dirty
    if [ ! -d "$d/.git" ] && [ ! -f "$d/.git" ]; then
        bad "$d is not a git checkout"; return
    fi
    head=$(git -C "$d" log -1 --format=%H)
    refs=$(git -C "$d" log -1 --format=%D)
    dirty=$(git -C "$d" diff HEAD --stat | tail -1)
    log "$d: ${head:0:12} [$refs]${dirty:+ uncommitted: $dirty}"
    if [[ $head != "$c"* ]] || [[ $refs != *"HEAD -> $b"* ]]; then
        if [ "$CHECKOUT" = 1 ]; then
            [ -z "$dirty" ] || die "$d has uncommitted changes: not switching it to $b"
            run git -C "$d" checkout "$b"
            dry || [[ $(git -C "$d" log -1 --format=%H) == "$c"* ]] || bad "$d: $b is not at $c"
        else
            bad "$d is at ${head:0:12} [$refs], expected $b at $c (--checkout switches a clean tree)"
        fi
    fi
}
repo "$KOSMOS_DIR" "$KOSMOS_BRANCH" "$KOSMOS_COMMIT"
repo "$TE_DIR" "$TE_BRANCH" "$TE_COMMIT"
repo "$MLM_DIR" "$MLM_BRANCH" "$MLM_COMMIT"
hk=$(git -C "$KOSMOS_DIR/3rdparty/hipkittens" log -1 --format=%H 2>/dev/null)
[[ $hk == "$HK_COMMIT"* ]] || bad "KOSMOS 3rdparty/hipkittens is at ${hk:-nothing}, expected $HK_COMMIT (git submodule update)"
[ -f "$KOSMOS_DIR/python/kosmos.cpp" ] && grep -q '^python:' "$KOSMOS_DIR/Makefile" ||
    bad "KOSMOS has no python/kosmos.cpp or no Makefile 'python' target (the pybind11 module)"
for f in kosmos_moe.py kosmos_gate.py megamoe_moe.py; do
    [ -f "$MLM_DIR/megatron/core/transformer/moe/$f" ] || bad "Megatron has no megatron/core/transformer/moe/$f"
done
grep -q MOE_ROUTING_STATS "$MLM_DIR/megatron/core/transformer/moe/moe_utils.py" &&
    grep -q record_routing_stats "$MLM_DIR/megatron/core/transformer/moe/router.py" &&
    grep -q _report_status "$MLM_DIR/megatron/core/transformer/moe/kosmos_moe.py" ||
    bad "Megatron lacks the skewed-routing hooks (moe_utils MOE_ROUTING_STATS, router.py) or the KOSMOS status report (kosmos_moe.py)"
[ -f "$KOSMOS_DIR/bench/moe/run.sh" ] && grep -q '^do_moe()' "$KOSMOS_DIR/bench/run_all.sh" ||
    bad "KOSMOS has no bench/moe/run.sh or bench/run_all.sh has no moe step (the MoE runner)"
[ -f "$KOSMOS_DIR/bench/moe/baseline.py" ] && [ -f "$KOSMOS_DIR/bench/tpsp/baseline.py" ] ||
    bad "KOSMOS has no bench/moe/baseline.py or bench/tpsp/baseline.py (the PyTorch baselines)"
MX=0; [[ " $A2A_PRECISIONS " == *" mxfp8 "* ]] && MX=1
if [ "$MX" = 1 ]; then   # the A2A MXFP8 runs
    grep -q KOSMOS_MXFP8 "$MLM_DIR/megatron/core/transformer/moe/kosmos_moe.py" ||
        bad "Megatron kosmos_moe.py has no MXFP8 experts (KOSMOS_MXFP8)"
    grep -q NVTE_USE_HIPKITTENS_GROUPED_GEMM "$TE_DIR/transformer_engine/pytorch/quantization.py" ||
        bad "TE quantization.py does not pad MXFP8 experts to 256 under NVTE_USE_HIPKITTENS_GROUPED_GEMM"
    grep -q 'moe-router-padding-for-quantization"$' "$MLM_DIR/examples/qwen3/train_qwen3.sh" &&
        grep -q 'moe-router-padding-for-quantization"$' "$MLM_DIR/examples/deepseek_v3/train_deepseekv3.sh" ||
        bad "train_qwen3.sh / train_deepseekv3.sh: no mxfp8 router-padding line for tools/gen_scripts.sh to gate"
fi

# ---- builds ---------------------------------------------------------------------------------------------------------
KENV="export ROCM_PATH=$CT_ROCM OMPI_HOME=$CT_OMPI GPU_ARCH=$GPU_ARCH KOSMOS_BUILD=$KOSMOS_BUILD"
need_ct "$KOSMOS_CT"
if [ "$BUILD" = 1 ]; then
    log "KOSMOS: make all python (KOSMOS_BUILD=$KOSMOS_BUILD) in ${KOSMOS_CT:-host} -> $O/kosmos_build.log"
    cexec "$KOSMOS_CT" "$KENV && cd $KOSMOS_DIR && make -j KOSMOS_BUILD=$KOSMOS_BUILD all python > $O/kosmos_build.log 2>&1 \
|| { tail -20 $O/kosmos_build.log; exit 1; }" || die "KOSMOS build failed ($O/kosmos_build.log)"
    printf '[safe]\n\tdirectory = *\n' | wfile "$O/gitconfig_safe"
    log "TE: in-place build against KOSMOS in ${KOSMOS_CT:-host} -> $O/te_build.log"
    cexec "$KOSMOS_CT" "cd $TE_DIR && env NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=$GPU_ARCH \
NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 NVTE_KOSMOS_ROOT=$KOSMOS_DIR NVTE_KOSMOS_LIB=$KOSMOS_BUILD/libkosmos_tpsp.a \
GIT_CONFIG_SYSTEM=$O/gitconfig_safe MAX_JOBS=$TE_MAX_JOBS python setup.py build_ext --inplace > $O/te_build.log 2>&1 \
|| { tail -20 $O/te_build.log; exit 1; }" || die "TE build failed ($O/te_build.log)"
    if [ "$TE_BASE_DIR" != "$TE_DIR" ]; then
        log "TE baseline: in-place build of $TE_BASE_DIR without KOSMOS -> $O/te_base_build.log"
        cexec "$KOSMOS_CT" "cd $TE_BASE_DIR && env NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=$GPU_ARCH \
NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 GIT_CONFIG_SYSTEM=$O/gitconfig_safe MAX_JOBS=$TE_MAX_JOBS \
python setup.py build_ext --inplace > $O/te_base_build.log 2>&1 || { tail -20 $O/te_base_build.log; exit 1; }" ||
            die "TE baseline build failed"
    fi
fi
if [ "$DEPS" = 1 ]; then
    log "DeepEP: $DEEPEP_DIR/build.sh in ${KOSMOS_CT:-host} -> $O/deepep_build.log"
    cexec "$KOSMOS_CT" "bash $DEEPEP_DIR/build.sh > $O/deepep_build.log 2>&1 || { tail -20 $O/deepep_build.log; exit 1; }" ||
        die "DeepEP build failed"
    need_ct "$PRIMUS_CT"
    log "Primus-Turbo: in-place build of $PT_DIR in $PRIMUS_CT -> $O/primus_turbo_build.log"
    cexec "$PRIMUS_CT" "cd $PT_DIR && env GPU_ARCHS=$GPU_ARCH PYTORCH_ROCM_ARCH=$GPU_ARCH MAX_JOBS=32 \
python3 setup.py build_ext --inplace > $O/primus_turbo_build.log 2>&1 || { tail -20 $O/primus_turbo_build.log; exit 1; }" ||
        die "Primus-Turbo build failed"
fi
[ "$BUILD" = 1 ] || [ "$DEPS" = 1 ] && fixown "$O" "$KOSMOS_BUILD" "$DEEPEP_DIR" "$PT_DIR"

# ---- build state (CPU checks; under DRYRUN only printed) ---------------------------------------------------------------
PYEXT=.cpython-312-x86_64-linux-gnu.so
CHECK="$KENV && cd $KOSMOS_DIR && \
for t in libkosmos_moe.a libkosmos_tpsp.a moe_fwd moe_bwd_dx moe_train gen_routing_ref kosmos_report; do \
  make -q KOSMOS_BUILD=$KOSMOS_BUILD $KOSMOS_BUILD/\$t 2>/dev/null || echo \"STALE KOSMOS $KOSMOS_BUILD/\$t (setup --build)\"; done; \
[ -x $KOSMOS_BUILD/overlap_bench ] || echo 'STALE KOSMOS no overlap_bench (setup --build)'; \
m=$KOSMOS_PYTHON/kosmos$PYEXT; \
{ [ -f \$m ] && [ \$m -nt $KOSMOS_BUILD/libkosmos_moe.a ] && [ \$m -nt python/kosmos.cpp ]; } || echo \"STALE KOSMOS python module \$m (setup --build)\"; \
so=$TE_DIR/libtransformer_engine.so; \
strings -n 6 \$so 2>/dev/null | grep -q 'KOSMOS\\] ub' || echo \"STALE TE \$so has no KOSMOS backend: built without NVTE_KOSMOS_ROOT (setup --build)\"; \
[ \$so -nt $KOSMOS_BUILD/libkosmos_tpsp.a ] || echo \"STALE TE \$so is older than libkosmos_tpsp.a (setup --build)\"; \
grep -q '^NVTE_KOSMOS_ROOT:PATH=$KOSMOS_DIR\$' $TE_DIR/build/cmake/CMakeCache.txt 2>/dev/null || echo \"STALE TE CMake cache NVTE_KOSMOS_ROOT is not $KOSMOS_DIR\"; \
python3 -c 'import pybind11' 2>/dev/null || echo 'MISSING pybind11 in the container python (make python)'; \
if [ $MX = 1 ]; then \
  strings -n 6 \$m 2>/dev/null | grep -qx quantize_mxfp8 || echo \"STALE KOSMOS python module \$m has no quantize_mxfp8 (setup --build)\"; \
  strings -n 6 \$so 2>/dev/null | grep -q 'HK-grouped' || echo \"STALE TE \$so has no HipKittens MXFP8 grouped GEMM (USE_HIPKITTENS_GEMM)\"; \
fi; \
command -v numactl > /dev/null || echo 'MISSING numactl (only for the numa ablation)'; true"
if dry; then
    cexec "$KOSMOS_CT" "$CHECK"
else
    while read -r l; do
        case $l in MISSING*numactl*) warn "$l" ;; STALE*|MISSING*) bad "$l" ;; *) [ -n "$l" ] && log "$l" ;; esac
    done < <(cexec "$KOSMOS_CT" "$CHECK")
fi

# ---- dependencies ---------------------------------------------------------------------------------------------------
dep() {  # dep PATH WHAT
    { [ -e "$1" ] || [ -L "$1" ]; } && { log "ok: $2 ($1)"; return; }
    bad "missing: $2 ($1)"
}
[[ " $TPSP_GA " == *" 1 "* ]] && dep "$APEX_SHIM_DIR/fused_weight_gradient_mlp_cuda.py" "APEX stand-in (TPSP_GA=1 only)"
f=$(ls "$DEEPEP_DIR"/pylib/deep_ep/_C*.so 2>/dev/null | head -1)
dep "${f:-$DEEPEP_DIR/pylib/deep_ep/_C.so}" "DeepEP build (setup --build-deps)"
dep "$PT_DIR/primus_turbo/lib/libprimus_turbo_kernels.so" "Primus-Turbo $PT_COMMIT build (setup --build-deps)"
dep "$MEGAMOE_SHIM/primus_turbo" "MegaMoE shim (Primus-Turbo python tree)"
dep "$MEGAMOE_PYENV/flydsl" "MegaMoE FlyDSL 0.2.4 copy"
dep "$MEGAMOE_PYENV/lib/libamdhip64.so" "MegaMoE libamdhip64 link"
dep "$NUMA_BIND_SCRIPT" "NUMA wrapper (numa ablation only)"
dep "$MOE_HF_HOME/hub/models--Qwen--Qwen3-235B-A22B" "Qwen3-235B tokenizer"
dep "$MOE_HF_HOME/hub/models--deepseek-ai--DeepSeek-V3" "DeepSeek-V3 tokenizer"
[ -n "$MICRO_FLYDSL_SEED" ] && dep "$MICRO_FLYDSL_SEED" "MegaMoE autotune seed for the microbenchmarks"
[ "$A2A_PRIMUS" = 1 ] && dep "$PRIMUS_DIR/primus-cli" "Primus 4969fd51 (workflow arm)"
c=$(git -C "$PT_DIR" log -1 --format=%h 2>/dev/null)
[[ $c == "$PT_COMMIT"* ]] || bad "Primus-Turbo at ${c:-?}, expected $PT_COMMIT"

if [ "$BAD" -gt 0 ]; then
    warn "setup: $BAD problem(s) above; run_all.sh manual lists the hand steps"
    dry || exit 1
fi
if dry; then log "setup: dry run, $BAD problem(s)"; else log "setup: OK"; fi
