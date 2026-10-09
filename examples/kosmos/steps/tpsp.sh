#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/tpsp.sh: Llama 3 8B / 70B / 405B TP+SP E2E, BF16 and MXFP8 (TPSP_PRECISIONS; train_llama3.sh, SEQ 8192, TP8,
# E2E_ITERS (10) iterations, mock data; mxfp8: TE_FP8=1 TE_FP8_RECIPE=mxfp8, tag suffix _mxfp8), every published config
# x GA fusion x arm x precision, (arm, precision) order shuffled per (config, GA, rep). One TE build (TE_DIR) for every arm:
#   base_noovl  TP_COMM_OVERLAP=0 (hipBLASLt + RCCL)
#   kos_tpsp    NVTE_USE_KOSMOS=1 NVTE_KOSMOS_BULK=1 (every overlapped op on KOSMOS)
# GA fusion 1 (optional, TPSP_GA) = GA_FUSION=1 with the APEX stand-in first on PYTHONPATH. Runs: $OUT_ROOT/tpsp/runs/<tag>/output_perf.log; list $OUT_ROOT/tpsp/runs.tsv.
# Hang policy (TPSP_HANG_SKIP=1): a run that stalls (rc 86), hits the hard timeout or fails with an out-of-memory /
# launch-resource error stops its (model, arm, GA): its remaining repeats and every larger MBS are recorded as skipped
# rows; other arms go on. The state is rebuilt from runs.tsv on a rerun (delete a failed row there to retry it).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
O=$OUT_ROOT/tpsp
need_ct "$KOSMOS_CT"
run mkdir -p "$O/runs"
dry || echo "SEED=$SEED GPU=$GPU_NAME $(date '+%F %T')" >> "$O/RUN.txt"
SCRIPT=$O/train_llama3_e2e.sh
if dry; then
    echo "[dryrun] bash $KIT/tools/gen_scripts.sh llama $MLM_DIR $O   # -> $SCRIPT"
else
    bash "$KIT/tools/gen_scripts.sh" llama "$MLM_DIR" "$O" > /dev/null || die "cannot derive $SCRIPT"
    case " $TPSP_ARMS " in
        *" kos_tpsp "*)
            strings -n 6 "$TE_DIR/libtransformer_engine.so" | grep -q 'KOSMOS\] ub' ||
                die "$TE_DIR/libtransformer_engine.so has no KOSMOS backend: run_all.sh setup --build" ;;
    esac
fi
sampler_start "$O/loadavg.log"
OOM_RE="out of memory|OutOfMemoryError|hipErrorOutOfMemory|too many resources requested"
declare -A FAIL=()   # model:arm:ga -> MBS:rep:class of its smallest-MBS failure
fail_note() {  # fail_note KEY MBS REP CLASS: keep the failure if it is the smallest MBS so far
    local cur=${FAIL[$1]:-}
    if [ -z "$cur" ] || [ "$2" -lt "${cur%%:*}" ]; then FAIL[$1]=$2:$3:$4; fi
}
if [ "$TPSP_HANG_SKIP" = 1 ] && ! dry && [ -s "$O/runs.tsv" ]; then
    while IFS=$'\t' read -r _ _ M MBS _ _ arm ga r rc _ _ cls pr; do
        case ${cls:-} in hang | timeout | oom) fail_note "$M:$arm:$ga:${pr:-bf16}" "$MBS" "$r" "$cls" ;; esac
    done < "$O/runs.tsv"
fi

arm_env() {  # arm_env ARM GA: the arm's exports
    local te=$TE_BASE_DIR
    case $1 in kos_*) te=$TE_DIR ;; esac
    echo "export PYTHONPATH=$([ "$2" = 1 ] && echo "$APEX_SHIM_DIR:")$te\${PYTHONPATH:+:\$PYTHONPATH}"
    echo "export GA_FUSION=$2 HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES${LLAMA_HF_HOME:+ HF_HOME=$LLAMA_HF_HOME}"
    echo "export ROCM_PATH=$CT_ROCM HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 GEMM_TUNING=0"
    case $1 in
        base_noovl) ;;
        kos_tpsp)   echo "export NVTE_USE_KOSMOS=1 NVTE_KOSMOS_BULK=1 NVTE_KOSMOS_LOG=1" ;;
        *)          die "unknown TP+SP arm $1" ;;
    esac
}

for ga in $TPSP_GA; do
    reps=$TPSP_REPS_GA0; [ "$ga" = 1 ] && reps=$TPSP_REPS_GA1
    for c in $TPSP_CONFIGS; do
        IFS=: read -r M MBS GBS L TMO EST EST_NO <<< "$c"
        [ "$ga" = 1 ] && [[ " $TPSP_GA1_CONFIGS " != *" $M:$MBS "* ]] && continue
        for r in $(seq 1 "$reps"); do
            for ap in $(shuffle "$SEED-$c-$ga-$r" $(for a in $TPSP_ARMS; do for p in $TPSP_PRECISIONS; do echo "$a:$p"; done; done)); do
                arm=${ap%:*} pr=${ap#*:} ptag=; [ "$pr" = mxfp8 ] && ptag=_mxfp8
                stall=$TPSP_STALL_S; est=$EST
                [ "$M" = 405 ] && stall=$TPSP_STALL_S_405
                [ "$arm" = base_noovl ] && est=$EST_NO
                tag=llama${M}B_mbs${MBS}_gbs${GBS}_L${L}_${arm}${ptag}_ga${ga}_r${r}
                RUN=$O/runs/$tag
                if done_log "$RUN/output_perf.log" '^elapsed time per iteration: [0-9]'; then
                    log "$tag done already"; continue
                fi
                key=$M:$arm:$ga:$pr
                if [ "$TPSP_HANG_SKIP" = 1 ] && [ -n "${FAIL[$key]:-}" ]; then
                    IFS=: read -r fm fr fc <<< "${FAIL[$key]}"
                    if [ "$MBS" -gt "$fm" ]; then
                        why="skipped: smaller size hung/failed (MBS $fm $fc)"
                    elif [ "$r" = "$fr" ]; then
                        why=""
                        log "$tag $fc earlier (runs.tsv): not retried"
                    else
                        why="skipped: this config $fc in repeat $fr"
                    fi
                    if [ -n "$why" ]; then
                        log "$tag $why"
                        dry || awk -F'\t' -v t="$tag" '$2 == t && $10 == "skip" { f = 1 } END { exit !f }' \
                            "$O/runs.tsv" 2>/dev/null ||
                            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tskip\t-\t-\t%s\t%s\n' "$(date '+%F %T')" "$tag" \
                                "$M" "$MBS" "$GBS" "$L" "$arm" "$ga" "$r" "$why" "$pr" >> "$O/runs.tsv"
                    fi
                    continue
                fi
                next_port
                ovl=1; [ "$arm" = base_noovl ] && ovl=0
                hwq=$GPU_MAX_HW_QUEUES
                for x in $TPSP_HWQ; do [ "${x%:*}" = "$M:$arm" ] && hwq=${x##*:}; done
                rcp=0; [[ " $TPSP_RECOMPUTE_MODELS " == *" $M "* ]] && rcp=1
                args="MODEL_SIZE=$M TP=8 PP=1 CP=1 MBS=$MBS BS=$GBS SEQ_LENGTH=8192 TOTAL_ITERS=$E2E_ITERS RECOMPUTE=$rcp"
                case $pr in bf16) args="$args TE_FP8=0" ;; mxfp8) args="$args TE_FP8=1 TE_FP8_RECIPE=mxfp8" ;; esac
                args="$args TP_COMM_OVERLAP=$ovl NUM_LAYERS_OVERRIDE=$L GPU_MAX_HW_QUEUES=$hwq MASTER_PORT=$PORT"
                { arm_env "$arm" "$ga"
                  echo "MLM_DIR=$MLM_DIR SCRIPT=$SCRIPT TMO=$TMO STALL_S=$stall"
                  echo "LLAMA_ARGS='$args'"; } | wfile "$RUN/run.env"
                cls=-
                if dry; then  # the plan follows the published outcomes (TPSP_EST_FAIL)
                    for x in $TPSP_EST_FAIL; do
                        [ "${x%:*}" = "$M:$MBS:$arm:$ga" ] && { est=${x##*:}; cls=published; }
                    done
                fi
                plan_add "$est"
                gpu "$KOSMOS_CT" "$tag" "bash $KIT/tools/llama_run.sh $RUN > $RUN/runner.log 2>&1"
                rc=$?
                if ! dry; then
                    case $rc in
                        0) ;;
                        86) cls=hang ;;
                        124 | 137) cls=timeout ;;
                        *) grep -Eq "$OOM_RE" "$RUN/output_perf.log" 2>/dev/null && cls=oom ;;
                    esac
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$tag" "$M" "$MBS" \
                        "$GBS" "$L" "$arm" "$ga" "$r" "$rc" "$LOAD0" "$RUN/output_perf.log" "$cls" "$pr" >> "$O/runs.tsv"
                fi
                [ "$cls" != - ] && fail_note "$key" "$MBS" "$r" "$cls"
                fixown "$RUN"
                dry || sleep "$POST_RUN_SLEEP_S"
            done
        done
    done
done
plan_report tpsp
