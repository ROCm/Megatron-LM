#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/micro.sh: KOSMOS bench/run_all.sh (forward gate, backward checks, MoE vs the PyTorch baseline cold + warm, TP+SP
# every config vs the PyTorch baseline in TE's non-overlapped call order; MICRO_ABLATIONS (default unfused) adds the KOSMOS
# unfused ablation)
# -> $OUT_ROOT/micro/$GPU_NAME.md (MICRO_RESULTS_IN_TREE=1: KOSMOS results/$GPU_NAME.md).
#   Three GPU invocations in KOSMOS_CT: run_all.sh env md5 gate check, then moe (bench/moe/run.sh), then tpsp md5end
#   (bench/tpsp/run.sh); the report runs on the CPU. MICRO_MEGA=1 (internal only) adds run_all.sh mega in PRIMUS_CT,
#   where Primus-Turbo imports, before the report: one invocation per precision, bf16 on PT_DIR, mxfp8 on PT_MX_DIR.
# MICRO_PRECISIONS (bf16 mxfp8): run_all.sh PRECISIONS; MXFP8 = MXFP8 training (gate, checks, moe_bench and the TE
# MXFP8 grouped-MLP baseline, MegaMoE MXFP8, TP+SP MXFP8), reported next to bf16.
# MICRO_DISTS: routing distributions (run_all.sh DISTS): 0 balanced, 1 skewed (the first E/8 experts' logits raised by
# log 3, rank 0 receives 2.32-2.39x the mean rows); the report tables each apart.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
K=$KOSMOS_DIR
O=$OUT_ROOT/micro
RES=$O/$GPU_NAME.md
[ "$MICRO_RESULTS_IN_TREE" = 1 ] && RES=$K/results/$GPU_NAME.md
need_ct "$KOSMOS_CT"
run mkdir -p "$O"
dry || echo "SEED=$SEED MEGA=$MICRO_MEGA GPU=$GPU_NAME $(date '+%F %T')" >> "$O/RUN.txt"
sampler_start "$O/loadavg.log"
KENV="export KOSMOS_ROOT=$K KOSMOS_BUILD=$KOSMOS_BUILD KOSMOS_WORK=$O/work ROCM_PATH=$CT_ROCM OMPI_HOME=$CT_OMPI \
HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES GPU_ARCH=$GPU_ARCH PYTHON=$CT_PYTHON"
RA="$KENV OUT=$O RESULTS=$RES GPU=$GPU_NAME REPS=$MICRO_REPS SEED=$SEED CHECK_SHAPES='$MICRO_CHECK_SHAPES' \
DISTS='$MICRO_DISTS' ABLATIONS='$MICRO_ABLATIONS' PRECISIONS='$MICRO_PRECISIONS'"
ND=$(echo $MICRO_DISTS | wc -w)
NP=$(echo $MICRO_PRECISIONS | wc -w)
U=${MICRO_ABLATIONS:+1}

plan_add "$(awk -v a="$EST_MICRO_CHECK_MIN" -v p="$NP" 'BEGIN { print a * p }')"
gpu "$KOSMOS_CT" micro_checks "$RA && cd $K && bash bench/run_all.sh env md5 gate check"
plan_add "$(awk -v a="$EST_MICRO_MOE_MIN" -v b="$EST_MICRO_MOE_UNF_MIN" -v n="$ND" -v u="${U:-0}" -v p="$NP" \
    'BEGIN { print p * n * (a + u * b) }')"
gpu "$KOSMOS_CT" micro_moe "$RA && cd $K && bash bench/run_all.sh moe"
plan_add "$(awk -v a="$EST_MICRO_TPSP_MIN" -v b="$EST_MICRO_TPSP_UNF_MIN" -v u="${U:-0}" 'BEGIN { print a + u * b }')"
gpu "$KOSMOS_CT" micro_tpsp "$RA && cd $K && bash bench/run_all.sh tpsp md5end"
if [ "$MICRO_MEGA" = 1 ]; then
    need_ct "$PRIMUS_CT"
    FLYC=$O/flydsl_autotune
    ROUT=$O/routing
    cexec "$KOSMOS_CT" "$KENV MEGAMOE_ROUTING_DIR=$ROUT && cd $K && bash bench/megamoe/gen_routing.sh 8192 \
$(awk '!/^#/ && NF { printf "%s ", $1 }' "$K/bench/moe/shapes.txt" 2>/dev/null)"
    NS=$(awk '!/^#/ && NF' "$K/bench/moe/shapes.txt" 2>/dev/null | wc -l)
    # One invocation per precision: bf16 on PT_DIR, mxfp8 on PT_MX_DIR, each with its own autotune cache.
    for p in $MICRO_PRECISIONS; do
        pt=$PT_DIR fc=$FLYC; [ "$p" = mxfp8 ] && pt=$PT_MX_DIR fc=${FLYC}_mx
        [ -n "$MICRO_FLYDSL_SEED" ] && [ ! -d "$fc" ] && run cp -r "$MICRO_FLYDSL_SEED" "$fc"
        run mkdir -p "$fc"
        plan_add "$(awk -v m="$EST_MICRO_MEGA_SWEEP_MIN" -v n="$ND" -v s="${NS:-7}" -v r="$MICRO_REPS" \
            'BEGIN { print 2 * n * s * r * m }')"
        gpu "$PRIMUS_CT" "micro_mega_$p" "export KOSMOS_ROOT=$K KOSMOS_WORK=$O/work ROCM_PATH=$PRIMUS_CT_ROCM \
HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES GPU_ARCH=$GPU_ARCH PRIMUS_TURBO_ROOT=$pt MEGAMOE_ROUTING_DIR=$ROUT \
FLYDSL_AUTOTUNE_CACHE_DIR=$fc OUT=$O REPS=$MICRO_REPS SEED=$SEED DISTS='$MICRO_DISTS' PRECISIONS=$p && cd $K && \
bash bench/run_all.sh mega"
    done
fi
cexec "$KOSMOS_CT" "$RA && cd $K && bash bench/run_all.sh report"
fixown "$O"
dry || log "microbenchmarks: $RES (runs $O/moe, $O/tpsp; notes $O/NOTES.txt)"
plan_report micro
