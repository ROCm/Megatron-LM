#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/llama_run.sh RUNDIR: runs in KOSMOS_CT. One train_llama3.sh run from the Megatron tree with RUNDIR/run.env (the
# arm's exports, LLAMA_ARGS, SCRIPT, MLM_DIR, TMO, STALL_S), log RUNDIR/output_perf.log, a hard timeout, a no-log-growth
# watchdog (rc 86) and a sweep of leftover ranks.
set -u
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
        echo "[llama_run] leftover processes:$pids"
        if [ $i -le 2 ]; then kill -TERM $pids 2>/dev/null; else kill -KILL $pids 2>/dev/null; fi
        sleep 10
    done
}
. "$RUN/run.env"
cd "$MLM_DIR" || exit 2
LOG=$RUN/output_perf.log
env | grep -E '^(NVTE_|PYTHONPATH|GA_FUSION|HIP_VISIBLE|HF_HOME|LD_LIBRARY_PATH|ROCM_PATH|HSA_)' | sort > "$RUN/arm_env.txt"
echo "$LLAMA_ARGS LOG_DIR=$RUN" > "$RUN/args.txt"
echo "[llama_run] $(date '+%F %T') args: $LLAMA_ARGS LOG_DIR=$RUN"
sed 's/^/[llama_run] env /' "$RUN/arm_env.txt"
HIP_VISIBLE_DEVICES= python3 -c 'import transformer_engine as t; print("[llama_run] te", t.__file__)' 2>&1 | tail -1
date +%s > "$RUN/t_start"
setsid timeout -k 30 "$TMO" bash "$SCRIPT" $LLAMA_ARGS LOG_DIR="$RUN" &
P=$!
last=0; lastt=$SECONDS; stalled=0
while kill -0 $P 2>/dev/null; do
    sleep 15
    sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    [ "$sz" != "$last" ] && { last=$sz; lastt=$SECONDS; }
    if [ $((SECONDS - lastt)) -ge "$STALL_S" ]; then
        stalled=1
        echo "[llama_run] STALL: no log growth for ${STALL_S}s, $(grep -c ' iteration .*/' "$LOG") iterations"
        tail -3 "$LOG" | cut -c1-300 | sed 's/^/[llama_run]   /'
        kill -TERM -- -$P 2>/dev/null; sleep 20; kill -KILL -- -$P 2>/dev/null
        break
    fi
done
wait $P; rc=$?
[ $stalled = 1 ] && rc=86
[ $rc = 0 ] && ! grep -Eq '^elapsed time per iteration: [0-9]' "$LOG" && rc=97   # exited 0 without a summary
date +%s > "$RUN/t_end"
sweep
echo "[llama_run] rc=$rc $(date '+%F %T')" | tee -a "$RUN/rc"
exit $rc
