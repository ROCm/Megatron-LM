#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/setup.sh [--checkout] [--build] [--build-deps]: verifies the three branches, the builds and every dependency.
# Read-only by default. --checkout switches a clean tree to its branch; --build runs `make all python` in KOSMOS, the
# in-place TE build against it and TE's editable install; --build-deps builds Primus-Turbo PT_DIR and PT_MX_DIR, DeepEP
# from PT_DIR (tools/deepep.sh), the MegaMoE shims (tools/megamoe_shim.sh) and copies FlyDSL 0.2.4 from the Primus
# image. Builds run in the containers, CPU only; clones and downloads are `run_all.sh manual`.
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

# repo DIR BRANCH COMMIT: the tree is on BRANCH at COMMIT or a descendant of it (git log / diff only; checkout only
# with --checkout).
repo() {
    local d=$1 b=$2 c=$3 head refs dirty
    if [ ! -d "$d/.git" ] && [ ! -f "$d/.git" ]; then
        bad "$d is not a git checkout"; return
    fi
    head=$(git -C "$d" log -1 --format=%H)
    refs=$(git -C "$d" log -1 --format=%D)
    dirty=$(git -C "$d" diff HEAD --stat | tail -1)
    log "$d: ${head:0:12} [$refs]${dirty:+ uncommitted: $dirty}"
    if ! git -C "$d" merge-base --is-ancestor "$c" HEAD 2>/dev/null || [[ $refs != *"HEAD -> $b"* ]]; then
        if [ "$CHECKOUT" = 1 ]; then
            [ -z "$dirty" ] || die "$d has uncommitted changes: not switching it to $b"
            run git -C "$d" checkout "$b"
            dry || git -C "$d" merge-base --is-ancestor "$c" HEAD 2>/dev/null || bad "$d: $b does not contain $c"
        else
            bad "$d is at ${head:0:12} [$refs], expected $b at or after $c (--checkout switches a clean tree)"
        fi
    elif [[ $head != "$c"* ]]; then
        log "$d: $b is ahead of $c"
    fi
}
repo "$KOSMOS_DIR" "$KOSMOS_BRANCH" "$KOSMOS_COMMIT"
repo "$TE_DIR" "$TE_BRANCH" "$TE_COMMIT"
repo "$TPSP_MLM_DIR" "$TPSP_MLM_BRANCH" "$TPSP_MLM_COMMIT"
[ "$A2A_MLM_DIR" = "$TPSP_MLM_DIR" ] || repo "$A2A_MLM_DIR" "$A2A_MLM_BRANCH" "$A2A_MLM_COMMIT"
hk=$(git -C "$KOSMOS_DIR/3rdparty/hipkittens" log -1 --format=%H 2>/dev/null)
[[ $hk == "$HK_COMMIT"* ]] || bad "KOSMOS 3rdparty/hipkittens is at ${hk:-nothing}, expected $HK_COMMIT (git submodule update)"
[ -f "$KOSMOS_DIR/python/kosmos.cpp" ] && grep -qs '^python:' "$KOSMOS_DIR/Makefile" ||
    bad "KOSMOS has no python/kosmos.cpp or no Makefile 'python' target (the pybind11 module)"
[ -f "$KOSMOS_DIR/bench/moe/src/moe_fwd_part.cu" ] ||
    bad "KOSMOS has no bench/moe/src/moe_fwd_part.cu (the Makefile builds moe_bench from it)"
[ -f "$KOSMOS_DIR/bench/moe/run.sh" ] && grep -qs '^do_moe()' "$KOSMOS_DIR/bench/run_all.sh" ||
    bad "KOSMOS has no bench/moe/run.sh or bench/run_all.sh has no moe step (the MoE runner)"
[ -f "$KOSMOS_DIR/bench/moe/baseline.py" ] && [ -f "$KOSMOS_DIR/bench/tpsp/baseline.py" ] ||
    bad "KOSMOS has no bench/moe/baseline.py or bench/tpsp/baseline.py (the PyTorch baselines)"
MX=0; [[ " $A2A_PRECISIONS " == *" mxfp8 "* ]] && MX=1
TMX=0; [[ " $TPSP_PRECISIONS " == *" mxfp8 "* ]] && TMX=1
# The TP+SP tree: the switches tools/gen_scripts.sh llama and steps/tpsp.sh use
L3=$TPSP_MLM_DIR/examples/llama/train_llama3.sh
grep -qs '^GROUP_SIZE=' "$L3" && grep -qs '^GRADIENT_ACCUMULATION_FUSION=' "$L3" &&
    grep -qs -- '--no-gradient-accumulation-fusion' "$L3" && grep -qs -- '--tp-comm-overlap' "$L3" &&
    grep -qs '^DDP_AVERAGE_IN_COLLECTIVE=' "$L3" ||
    bad "$L3 lacks the GA-fusion / TP-overlap switches or the fusion defaults (DDP_AVERAGE_IN_COLLECTIVE ...)"
[ "$TMX" = 1 ] && { grep -qs kosmos_mxfp8 "$TE_DIR/transformer_engine/common/comm_gemm_overlap/rocm_comm_gemm_overlap.cpp" ||
    bad "TE has no MXFP8 KOSMOS TP+SP dispatch (rocm_comm_gemm_overlap.cpp kosmos_mxfp8)"; }
# The A2A tree
A=$A2A_MLM_DIR/megatron/core/transformer/moe
for f in kosmos_moe.py kosmos_gate.py megamoe_moe.py; do
    [ -f "$A/$f" ] || bad "Megatron $A2A_MLM_DIR has no megatron/core/transformer/moe/$f"
done
grep -qs MOE_ROUTING_STATS "$A/moe_utils.py" && grep -qs record_routing_stats "$A/router.py" &&
    grep -qs _report_status "$A/kosmos_moe.py" ||
    bad "Megatron $A2A_MLM_DIR lacks the skewed-routing hooks (moe_utils MOE_ROUTING_STATS, router.py) or the KOSMOS status report (kosmos_moe.py)"
if [ "$MX" = 1 ]; then   # the A2A MXFP8 runs
    grep -qs KOSMOS_MXFP8 "$A/kosmos_moe.py" || bad "Megatron kosmos_moe.py has no MXFP8 experts (KOSMOS_MXFP8)"
    grep -qs fused_mega_moe_fp8 "$A/megamoe_moe.py" || bad "Megatron megamoe_moe.py has no MXFP8 MegaMoE"
    grep -qs NVTE_USE_HIPKITTENS_GROUPED_GEMM "$TE_DIR/transformer_engine/pytorch/quantization.py" ||
        bad "TE quantization.py does not pad MXFP8 experts to 256 under NVTE_USE_HIPKITTENS_GROUPED_GEMM"
    grep -qs 'moe-router-padding-for-quantization"$' "$A2A_MLM_DIR/examples/qwen3/train_qwen3.sh" &&
        grep -qs 'moe-router-padding-for-quantization"$' "$A2A_MLM_DIR/examples/deepseek_v3/train_deepseekv3.sh" ||
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
    # The container's transformer_engine metadata must be this tree's (Megatron reads its version): editable install,
    # once, with the same flags (CMake reuses the in-place build).
    log "TE: editable install of $TE_DIR in ${KOSMOS_CT:-host} unless already there -> $O/te_editable.log"
    cexec "$KOSMOS_CT" "pip show transformer_engine 2>/dev/null | grep -qx 'Editable project location: $TE_DIR' || \
{ cd $TE_DIR && env NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=$GPU_ARCH NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
NVTE_KOSMOS_ROOT=$KOSMOS_DIR NVTE_KOSMOS_LIB=$KOSMOS_BUILD/libkosmos_tpsp.a GIT_CONFIG_SYSTEM=$O/gitconfig_safe \
MAX_JOBS=$TE_MAX_JOBS pip install --no-build-isolation --no-deps -e . > $O/te_editable.log 2>&1 \
|| { tail -20 $O/te_editable.log; exit 1; }; }" || die "TE editable install failed ($O/te_editable.log)"
fi
if [ "$DEPS" = 1 ]; then
    need_ct "$PRIMUS_CT"
    # Primus-Turbo, in place in PRIMUS_CT: PT_DIR (MegaMoE bf16 microbenchmark, the Primus workflow, DeepEP's hipified
    # sources) and PT_MX_DIR (MegaMoE MXFP8).
    for p in "$PT_DIR" $([ "$PT_MX_DIR" = "$PT_DIR" ] || echo "$PT_MX_DIR"); do
        b=$O/primus_turbo_build_$(basename "$p").log
        log "Primus-Turbo: in-place build of $p in $PRIMUS_CT -> $b"
        cexec "$PRIMUS_CT" "cd $p && env GPU_ARCHS=$GPU_ARCH PYTORCH_ROCM_ARCH=$GPU_ARCH MAX_JOBS=32 \
python3 setup.py build_ext --inplace > $b 2>&1 || { tail -20 $b; exit 1; }" || die "Primus-Turbo build of $p failed"
    done
    fixown "$PT_DIR" "$PT_MX_DIR"
    log "DeepEP: tools/deepep.sh tree $PT_DIR -> $DEEPEP_DIR, then build in ${KOSMOS_CT:-host} -> $O/deepep_build.log"
    run bash "$KIT/tools/deepep.sh" tree "$PT_DIR" "$DEEPEP_DIR" || die "DeepEP sources failed"
    cexec "$KOSMOS_CT" "export ROCM_PATH=$CT_ROCM GPU_ARCH=$GPU_ARCH && bash $KIT/tools/deepep.sh build $DEEPEP_DIR \
> $O/deepep_build.log 2>&1 || { tail -20 $O/deepep_build.log; exit 1; }" || die "DeepEP build failed"
    log "MegaMoE shims: $MEGAMOE_SHIM from $PT_DIR, $MEGAMOE_SHIM_MX from $PT_MX_DIR"
    run bash "$KIT/tools/megamoe_shim.sh" "$PT_DIR" "$MEGAMOE_SHIM" || die "MegaMoE shim failed"
    run bash "$KIT/tools/megamoe_shim.sh" "$PT_MX_DIR" "$MEGAMOE_SHIM_MX" || die "MegaMoE MXFP8 shim failed"
    log "MegaMoE FlyDSL: $PRIMUS_CT_SITE/flydsl* -> $MEGAMOE_PYENV, libamdhip64.so -> the KOSMOS container's ROCm"
    cexec "$PRIMUS_CT" "mkdir -p $MEGAMOE_PYENV/lib && cp -a $PRIMUS_CT_SITE/flydsl $PRIMUS_CT_SITE/flydsl-*.dist-info \
$PRIMUS_CT_SITE/flydsl.libs $MEGAMOE_PYENV/" || die "FlyDSL copy failed"
    run ln -sfn "$CT_ROCM_CORE_LIB/libamdhip64.so.7" "$MEGAMOE_PYENV/lib/libamdhip64.so"
    run mkdir -p "$MEGAMOE_FLYDSL_CACHE"
fi
[ "$BUILD" = 1 ] || [ "$DEPS" = 1 ] && fixown "$O" "$KOSMOS_BUILD" "$DEEPEP_DIR" "$MEGAMOE_PYENV"

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
[ \$so -nt $TE_DIR/transformer_engine/common/comm_gemm_overlap/rocm_comm_gemm_overlap.cpp ] || echo \"STALE TE \$so is older than its KOSMOS dispatch source (setup --build)\"; \
python3 -c 'import transformers' 2>/dev/null || echo 'MISSING transformers in the container python (run_all.sh manual)'; \
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
# commit DIR COMMIT WHAT: a dependency checkout at COMMIT
commit() {
    local c
    c=$(git -C "$1" log -1 --format=%H 2>/dev/null)
    [[ $c == "$2"* ]] || bad "$3 at ${c:0:12}, expected $2 ($1)"
}
arms() { [[ " $A2A_ARMS " == *" $1 "* ]]; }
MEGA_MX=0
arms megamoe && [ "$MX" = 1 ] && [[ " $A2A_MXFP8_ARMS " == *" megamoe "* ]] && MEGA_MX=1
[ "$MICRO_MEGA" = 1 ] && [[ " $MICRO_PRECISIONS " == *" mxfp8 "* ]] && MEGA_MX=1
[[ " $TPSP_GA " == *" 1 "* ]] && dep "$APEX_SHIM_DIR/fused_weight_gradient_mlp_cuda.py" "APEX stand-in (TPSP_GA=1 only)"
if arms deepep; then
    f=$(ls "$DEEPEP_DIR"/pylib/deep_ep/_C*.so 2>/dev/null | head -1)
    dep "${f:-$DEEPEP_DIR/pylib/deep_ep/_C.so}" "DeepEP build (setup --build-deps)"
fi
dep "$PT_DIR/primus_turbo/lib/libprimus_turbo_kernels.so" "Primus-Turbo $PT_COMMIT build (setup --build-deps)"
commit "$PT_DIR" "$PT_COMMIT" "Primus-Turbo"
if [ "$MEGA_MX" = 1 ]; then
    dep "$PT_MX_DIR/primus_turbo/lib/libprimus_turbo_kernels.so" "Primus-Turbo $PT_MX_COMMIT build, MXFP8 MegaMoE (setup --build-deps)"
    commit "$PT_MX_DIR" "$PT_MX_COMMIT" "Primus-Turbo (MXFP8 MegaMoE)"
fi
if arms megamoe; then
    dep "$MEGAMOE_SHIM/primus_turbo/pytorch/ops/moe/fused_mega_moe.py" "MegaMoE shim of $PT_DIR (setup --build-deps)"
    [ "$MEGA_MX" = 1 ] && dep "$MEGAMOE_SHIM_MX/primus_turbo/pytorch/ops/moe/fused_mega_moe_fp8.py" \
        "MegaMoE MXFP8 shim of $PT_MX_DIR (setup --build-deps)"
    dep "$MEGAMOE_PYENV/flydsl-0.2.4.dist-info" "MegaMoE FlyDSL 0.2.4 copy (setup --build-deps)"
    dep "$MEGAMOE_PYENV/lib/libamdhip64.so" "MegaMoE libamdhip64 link (setup --build-deps)"
fi
[[ " $A2A_ABLATIONS " == *":numa "* ]] && dep "$NUMA_BIND_SCRIPT" "NUMA wrapper (numa ablation only)"
dep "$MOE_HF_HOME/hub/models--Qwen--Qwen3-235B-A22B" "Qwen3-235B tokenizer (run_all.sh manual)"
dep "$MOE_HF_HOME/hub/models--deepseek-ai--DeepSeek-V3" "DeepSeek-V3 tokenizer (run_all.sh manual)"
[ -n "$MICRO_FLYDSL_SEED" ] && dep "$MICRO_FLYDSL_SEED" "MegaMoE autotune seed for the microbenchmarks"
if [ "$A2A_PRIMUS" = 1 ]; then
    dep "$PRIMUS_DIR/primus-cli" "Primus $PRIMUS_COMMIT (workflow arm)"
    commit "$PRIMUS_DIR" "$PRIMUS_COMMIT" "Primus"
fi

if [ "$BAD" -gt 0 ]; then
    warn "setup: $BAD problem(s) above; run_all.sh manual lists the hand steps"
    dry || exit 1
fi
if dry; then log "setup: dry run, $BAD problem(s)"; else log "setup: OK"; fi
