#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/fsdp.sh: Llama 3 8B / 70B / 405B with PyTorch FSDP2 (FSDP=1: TP 1, data parallel 8 on one node) as the
# comparison point for KOSMOS TP+SP; BF16 and MXFP8, SEQ 8192, GBS 64, E2E_ITERS iterations, mock data, GA fusion off,
# one run per point. Same models as steps/tpsp.sh: the layer count of each (model, MBS) is the one the TP+SP runs use
# (TPSP_CONFIGS), so every FSDP point has a KOSMOS point of the same model.
# Sweep per (model, precision): recompute off at MBS 1, 2, 4, 8 until the first out-of-memory; then full recompute
# from the first MBS that ran out of memory, until it runs out of memory too. Any other failure also ends that sweep.
# Runs: $OUT_ROOT/fsdp/runs/<tag>/output_perf.log; list $OUT_ROOT/fsdp/runs.tsv; best point per (model, precision,
# layers) in $OUT_ROOT/fsdp/fsdp_best.csv (columns of figs/final/fsdp_best.csv).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
O=$OUT_ROOT/fsdp
FSDP_MODELS=${FSDP_MODELS:-"8 70 405"}
FSDP_MBS=${FSDP_MBS:-"1 2 4 8"}
FSDP_PRECISIONS=${FSDP_PRECISIONS:-"bf16 mxfp8"}
FSDP_TMO=${FSDP_TMO:-2400}
FSDP_STALL_S=${FSDP_STALL_S:-900}
FSDP_EST_MIN=${FSDP_EST_MIN:-6}       # plan only: minutes per run
need_ct "$KOSMOS_CT"
run mkdir -p "$O/runs"
dry || echo "SEED=$SEED GPU=$GPU_NAME $(date '+%F %T')" >> "$O/RUN.txt"
SCRIPT=$O/train_llama3_e2e.sh
log "Megatron: $TPSP_MLM_DIR ($TPSP_MLM_BRANCH)"
if dry; then
    echo "[dryrun] bash $KIT/tools/gen_scripts.sh llama $TPSP_MLM_DIR $O   # -> $SCRIPT"
else
    bash "$KIT/tools/gen_scripts.sh" llama "$TPSP_MLM_DIR" "$O" > /dev/null || die "cannot derive $SCRIPT"
fi
sampler_start "$O/loadavg.log"
OOM_RE="out of memory|OutOfMemoryError|hipErrorOutOfMemory|too many resources requested"

layers_of() {  # layers_of MODEL MBS: the layer count of TPSP_CONFIGS for that point
    local c M MBS L _
    for c in $TPSP_CONFIGS; do
        IFS=: read -r M MBS _ L _ <<< "$c"
        [ "$M" = "$1" ] && [ "$MBS" = "$2" ] && { echo "$L"; return; }
    done
}

# run_point MODEL MBS LAYERS PREC RECOMPUTE: one run; sets RES to its class (- ok, oom, hang, timeout, fail).
run_point() {
    local M=$1 MBS=$2 L=$3 pr=$4 rcp=$5 tag RUN rc cls
    tag=llama${M}B_fsdp_mbs${MBS}_gbs64_L${L}_rc${rcp}$([ "$pr" = mxfp8 ] && echo _mxfp8)
    RUN=$O/runs/$tag
    if done_log "$RUN/output_perf.log" '^elapsed time per iteration: [0-9]'; then
        log "$tag done already"; RES=-; return
    fi
    cls=$(awk -F'\t' -v t="$tag" '$2 == t { c = $10 } END { print c }' "$O/runs.tsv" 2>/dev/null)
    if [ -n "$cls" ]; then
        log "$tag $cls earlier (runs.tsv): not retried"; RES=$cls; return
    fi
    next_port
    local args="MODEL_SIZE=$M TP=1 PP=1 CP=1 FSDP=1 MBS=$MBS BS=64 SEQ_LENGTH=8192 TOTAL_ITERS=$E2E_ITERS RECOMPUTE=$rcp"
    case $pr in bf16) args="$args TE_FP8=0" ;; mxfp8) args="$args TE_FP8=1 TE_FP8_RECIPE=mxfp8" ;; esac
    args="$args TP_COMM_OVERLAP=0 NUM_LAYERS_OVERRIDE=$L GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES MASTER_PORT=$PORT"
    { echo "export PYTHONPATH=$TE_BASE_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
      echo "export GA_FUSION=0 HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES${LLAMA_HF_HOME:+ HF_HOME=$LLAMA_HF_HOME}"
      echo "export ROCM_PATH=$CT_ROCM HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 GEMM_TUNING=0"
      echo "MLM_DIR=$TPSP_MLM_DIR SCRIPT=$SCRIPT TMO=$FSDP_TMO STALL_S=$FSDP_STALL_S"
      echo "LLAMA_ARGS='$args'"; } | wfile "$RUN/run.env"
    plan_add "$FSDP_EST_MIN"
    gpu "$KOSMOS_CT" "$tag" "bash $KIT/tools/llama_run.sh $RUN > $RUN/runner.log 2>&1"
    rc=$?
    cls=-
    if dry; then
        RES=-; return
    fi
    case $rc in
        0) ;;
        86) cls=hang ;;
        124 | 137) cls=timeout ;;
        *) if grep -Eq "$OOM_RE" "$RUN/output_perf.log" 2>/dev/null; then cls=oom
           elif ! done_log "$RUN/output_perf.log" '^elapsed time per iteration: [0-9]'; then cls=fail; fi ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$tag" "$M" "$MBS" "$L" "$pr" "$rcp" \
        "$rc" "$LOAD0" "$cls" "$RUN/output_perf.log" >> "$O/runs.tsv"
    fixown "$RUN"
    dry || sleep "$POST_RUN_SLEEP_S"
    RES=$cls
}

for M in $FSDP_MODELS; do
    for pr in $FSDP_PRECISIONS; do
        first_oom=
        for MBS in $FSDP_MBS; do
            L=$(layers_of "$M" "$MBS"); [ -n "$L" ] || continue
            run_point "$M" "$MBS" "$L" "$pr" 0; cls=$RES
            [ "$cls" = - ] && continue
            first_oom=$MBS; log "llama${M}B $pr recompute off: $cls at MBS $MBS, ends the sweep"; break
        done
        [ -n "$first_oom" ] || continue
        for MBS in $FSDP_MBS; do
            [ "$MBS" -lt "$first_oom" ] && continue
            L=$(layers_of "$M" "$MBS"); [ -n "$L" ] || continue
            run_point "$M" "$MBS" "$L" "$pr" 1; cls=$RES
            [ "$cls" = - ] && continue
            log "llama${M}B $pr recompute on: $cls at MBS $MBS, ends the sweep"; break
        done
    done
done
plan_report fsdp
dry && exit 0

# fsdp_best.csv: per (model, precision, layers) the valid run (all E2E_ITERS iterations and the summary printed) with the
# most tokens/GPU/s; Megatron's printed end-of-run numbers.
{
    echo "gpu,model,prec,config,ms,tflops,tok,log"
    awk -F'\t' '$10 == "-" { print $11 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$O/runs.tsv" | while IFS=$'\t' read -r f M MBS L pr rcp; do
        it=$(grep -ac ' iteration *[0-9]*/ *[0-9]* |' "$f")
        [ "$it" -ge "$E2E_ITERS" ] || continue
        ms=$(awk '/^elapsed time per iteration:/ { v = $NF } END { print v }' "$f")
        tf=$(awk '/^throughput per GPU:/ { v = $NF } END { print v }' "$f")
        tok=$(awk '/^tokens\/GPU\/s:/ { v = $NF } END { print v }' "$f")
        [ -n "$ms" ] && echo "$M,$pr,$L,$MBS,$rcp,$ms,$tf,$tok,$f"
    done | sort -t, -k1,1n -k2,2 -k3,3n -k8,8gr | awk -F, -v g="$GPU_NAME" '!seen[$1 FS $2 FS $3]++ {
        printf "%s,%sB,%s,FSDP2 DP8 MBS%s GBS64 L%s recompute %s,%s,%s,%s,%s\n", g, $1, toupper($2), $4, $3, ($5 ? "on" : "off"), $6, $7, $8, $9 }'
} > "$O/fsdp_best.csv"
log "wrote $O/fsdp_best.csv ($(($(wc -l < "$O/fsdp_best.csv") - 1)) best points)"
