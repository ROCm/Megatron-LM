#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/primus_run.sh RUNDIR: runs in PRIMUS_CT. Primus's own workflow with MegaMoE (primus-cli direct -- train pretrain)
# from PRIMUS_DIR, with Primus-Turbo put on the path by --env RUNDIR/primus_turbo.env. RUNDIR/run.env sets PRIMUS_DIR,
# PRIMUS_ARGS, TMO, STALL_S, MASTER_PORT, HF_HOME, TAG. Log RUNDIR/train_primus.log.
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
        echo "[primus_run] leftover processes:$pids"
        if [ $i -le 2 ]; then kill -TERM $pids 2>/dev/null; else kill -KILL $pids 2>/dev/null; fi
        sleep 10
    done
}
. "$RUN/run.env"
cd "$PRIMUS_DIR" || exit 2
unset PYTHONPATH LD_LIBRARY_PATH_EXTRA
export PRIMUS_SKIP_PIP=1 PRIMUS_EXP_NAME=$TAG PRIMUS_WORKSPACE=$RUN/primus_output
echo "$PRIMUS_ARGS" > "$RUN/primus_args.txt"
LOG=$RUN/train_primus.log
date +%s > "$RUN/t_start"
setsid timeout -k 30 "$TMO" ./primus-cli direct --env "$RUN/primus_turbo.env" -- train pretrain $PRIMUS_ARGS \
    --disable_wandb True --disable_tensorboard True > "$LOG" 2>&1 &
P=$!
last=0; idle=0; stalled=0
while kill -0 $P 2>/dev/null; do
    sleep 10
    sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$sz" = "$last" ]; then idle=$((idle + 10)); else idle=0; last=$sz; fi
    if [ $idle -ge "$STALL_S" ]; then
        stalled=1
        echo "[primus_run] STALL: log unchanged for ${idle}s, killing process group $P" >> "$LOG"
        kill -TERM -- -$P 2>/dev/null; sleep 20; kill -KILL -- -$P 2>/dev/null
        break
    fi
done
wait $P; rc=$?
[ $stalled = 1 ] && rc=86
[ $rc = 0 ] && ! grep -aEq "iteration +$ITERS/ *$ITERS \\|" "$LOG" && rc=97   # no last iteration printed
date +%s > "$RUN/t_end"
sweep >> "$LOG"
echo "[primus_run] rc=$rc" >> "$LOG"
echo "[primus_run] rc=$rc $(date '+%F %T')" | tee -a "$RUN/rc"
exit $rc
