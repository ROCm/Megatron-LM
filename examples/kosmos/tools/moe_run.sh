#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/moe_run.sh RUNDIR: runs in KOSMOS_CT. One Megatron example-script run as run_*_perf_compare.sh's run_train does
# it: source the proxy, then env OUTPUT_BASEPATH=... bash <train script>. RUNDIR/run.env sets the runner (MLM_DIR,
# PROXY, TRAIN, TMO, STALL_S, PREC, MASTER_PORT, HF_HOME, ...); RUNDIR/arm.env is the arm and the precision (PR,
# FP8_RECIPE), applied AFTER the proxy. Log and output: RUNDIR/train_$PREC.log, RUNDIR/output_$PREC.
set -o pipefail
RUN=$1
export KIT_RUN=$RUN
# sweep: kill what is left of this run (every process carrying KIT_RUN=$RUN in its environment).
sweep() {
    local i p pids
    for i in 1 2 3 4 5 6; do
        pids=""
        for p in /proc/[0-9]*; do
            p=${p#/proc/}
            [ "$p" = $$ ] && continue
            tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "KIT_RUN=$RUN" && pids="$pids $p"
        done
        [ -z "$pids" ] && return 0
        echo "[moe_run] leftover processes:$pids"
        if [ $i -le 2 ]; then kill -TERM $pids 2>/dev/null; else kill -KILL $pids 2>/dev/null; fi
        sleep 10
    done
}
. "$RUN/run.env"
cd "$MLM_DIR" || exit 2
source "$PROXY"
. "$RUN/arm.env"
PREC=${PREC:-bf16}
case $PREC:${PR:-}:${FP8_RECIPE:-} in
    bf16:bf16:* | mxfp8:fp8:mxfp8) ;;
    *) echo "[moe_run] PREC=$PREC but PR=${PR:-} FP8_RECIPE=${FP8_RECIPE:-}" >&2; exit 2 ;;
esac
mkdir -p "$RUN/output_$PREC"
env | grep -E '^(PR|FP8_RECIPE|MOE_[A-Z_]*|FORCE_BALANCE|NUM_LAYERS|TRAIN_ITERS|RECOMPUTE_[A-Z_]*|DDP_OVERLAP|GA_FUSION[A-Z_]*|ENABLE_[A-Z_]*|NUMA_[A-Z_]*|PRETRAIN_SCRIPT|KOSMOS_[A-Z_]*|MORI_[A-Z_]*|FLYDSL_[A-Z_]*|MASTER_PORT|ROCM_PATH|PYTHONPATH|LD_LIBRARY_PATH|TE_[A-Z_]*|NVTE_[A-Z_]*|HSA_[A-Z_]*|HF_HOME|USE_GROUPED_GEMM|GEMM_TUNING)=' | sort > "$RUN/arm_env.txt"
LOG=$RUN/train_$PREC.log
date +%s > "$RUN/t_start"
setsid timeout -k 30 "$TMO" env OUTPUT_BASEPATH="$RUN/output_$PREC" bash "$TRAIN" > "$LOG" 2>&1 &
P=$!
last=0; idle=0; stalled=0
while kill -0 $P 2>/dev/null; do
    sleep 10
    sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$sz" = "$last" ]; then idle=$((idle + 10)); else idle=0; last=$sz; fi
    if [ $idle -ge "$STALL_S" ]; then
        stalled=1
        echo "[moe_run] STALL: log unchanged for ${idle}s, killing process group $P" >> "$LOG"
        kill -TERM -- -$P 2>/dev/null; sleep 20; kill -KILL -- -$P 2>/dev/null
        break
    fi
done
wait $P; rc=$?
[ $stalled = 1 ] && rc=86
[ $rc = 0 ] && ! grep -Eq '^elapsed time per iteration: [0-9]' "$LOG" && rc=97   # exited 0 without a summary
last=$(grep -aoE ' iteration +[0-9]+/' "$LOG" | tail -1 | grep -oE '[0-9]+')
[ $rc = 0 ] && [ -n "${TRAIN_ITERS:-}" ] && [ "${last:-0}" != "$TRAIN_ITERS" ] && rc=98   # a rank died before the last iteration
date +%s > "$RUN/t_end"
sweep >> "$LOG"
echo "[moe_run] rc=$rc" >> "$LOG"
echo "[moe_run] rc=$rc $(date '+%F %T')" | tee -a "$RUN/rc"
exit $rc
