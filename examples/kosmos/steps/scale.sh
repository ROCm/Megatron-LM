#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/scale.sh: Llama 3 70B TP+SP E2E weak and strong scaling over TP (2, 4, 8), BF16, DP = 1: a TP-T run uses GPUs
# 0..T-1 and the others sit idle. Same model (SCALE_LAYERS layers) at every TP; train_llama3.sh, SEQ 8192, E2E_ITERS
# iterations, mock data, GA fusion off, arms base_noovl and kos_tpsp as steps/tpsp.sh.
#   weak    MBS and GBS grow with TP (MBS = TP/2, GBS = 16 MBS: per-GPU work and micro-batch count constant)
#   strong  MBS and GBS fixed (MBS 2, GBS 32: 16 micro-batches), per-GPU work shrinks as 1/TP
# Runs: $OUT_ROOT/scale/runs/<tag>/output_perf.log; list $OUT_ROOT/scale/runs.tsv; table $OUT_ROOT/RESULTS_SCALE.md.
# A run that stalls, times out or runs out of memory is recorded and not retried; the other runs go on.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
O=$OUT_ROOT/scale
SCALE_LAYERS=${SCALE_LAYERS:-16}       # fits TP 2 (DP 1): ~7.7 GB of weights + optimizer per layer per GPU
SCALE_TMO=${SCALE_TMO:-1800}
SCALE_STALL_S=${SCALE_STALL_S:-600}
# SERIES:TP:MBS:GBS:EST (EST minutes per run, for the plan); a point shared by both series runs once.
SCALE_CONFIGS=${SCALE_CONFIGS:-"weak:2:1:16:4 weak:4:2:32:4 weak:8:4:64:4 strong:2:2:32:6 strong:4:2:32:4 strong:8:2:32:3"}
SCALE_ARMS=${SCALE_ARMS:-"base_noovl kos_tpsp"}
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

arm_env() {  # arm_env ARM TP: the arm's exports
    local te=$TE_BASE_DIR
    case $1 in kos_*) te=$TE_DIR ;; esac
    echo "export PYTHONPATH=$te\${PYTHONPATH:+:\$PYTHONPATH}"
    echo "export GA_FUSION=0 HIP_VISIBLE_DEVICES=$(seq -s, 0 $(($2 - 1)))${LLAMA_HF_HOME:+ HF_HOME=$LLAMA_HF_HOME}"
    echo "export ROCM_PATH=$CT_ROCM HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 GEMM_TUNING=0"
    case $1 in
        base_noovl) ;;
        kos_tpsp)   echo "export NVTE_USE_KOSMOS=1 NVTE_KOSMOS_BULK=1 NVTE_KOSMOS_LOG=1" ;;
        *)          die "unknown TP+SP arm $1" ;;
    esac
}

points=$(for c in $SCALE_CONFIGS; do IFS=: read -r _ T MBS GBS EST <<< "$c"; echo "$T:$MBS:$GBS:$EST"; done | sort -u)
for r in $(seq 1 "$E2E_REPS"); do
    for pa in $(shuffle "$SEED-scale-$r" $(for p in $points; do for a in $SCALE_ARMS; do echo "$p:$a"; done; done)); do
        IFS=: read -r T MBS GBS EST arm <<< "$pa"
        tag=llama70B_tp${T}_mbs${MBS}_gbs${GBS}_L${SCALE_LAYERS}_${arm}_r${r}
        RUN=$O/runs/$tag
        if done_log "$RUN/output_perf.log" '^elapsed time per iteration: [0-9]'; then
            log "$tag done already"; continue
        fi
        if awk -F'\t' -v t="$tag" '$2 == t { f = 1 } END { exit !f }' "$O/runs.tsv" 2>/dev/null; then
            log "$tag failed earlier (runs.tsv): not retried"; continue
        fi
        next_port
        ovl=1; [ "$arm" = base_noovl ] && ovl=0
        args="MODEL_SIZE=70 TP=$T PP=1 CP=1 MBS=$MBS BS=$GBS SEQ_LENGTH=8192 TOTAL_ITERS=$E2E_ITERS RECOMPUTE=0 TE_FP8=0"
        args="$args TP_COMM_OVERLAP=$ovl NUM_LAYERS_OVERRIDE=$SCALE_LAYERS GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES MASTER_PORT=$PORT"
        { arm_env "$arm" "$T"
          echo "MLM_DIR=$TPSP_MLM_DIR SCRIPT=$SCRIPT TMO=$SCALE_TMO STALL_S=$SCALE_STALL_S"
          echo "LLAMA_ARGS='$args'"; } | wfile "$RUN/run.env"
        plan_add "$EST"
        gpu "$KOSMOS_CT" "$tag" "bash $KIT/tools/llama_run.sh $RUN > $RUN/runner.log 2>&1"
        rc=$?
        if ! dry; then
            cls=-
            case $rc in
                0) ;;
                86) cls=hang ;;
                124 | 137) cls=timeout ;;
                *) grep -Eq "$OOM_RE" "$RUN/output_perf.log" 2>/dev/null && cls=oom ;;
            esac
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$tag" "$T" "$MBS" "$GBS" \
                "$SCALE_LAYERS" "$arm" "$r" "$rc" "$cls" "$RUN/output_perf.log" >> "$O/runs.tsv"
        fi
        fixown "$RUN"
        dry || sleep "$POST_RUN_SLEEP_S"
    done
done
plan_report scale
dry && exit 0

# RESULTS_SCALE.md: per series, TP and arm the median over valid runs (all E2E_ITERS iterations and the summary
# printed) of Megatron's end-of-run numbers; KOSMOS speedup = base_noovl / kos_tpsp ms/iter; weak-scaling efficiency =
# TFLOP/s/GPU at TP / at TP 2; strong-scaling speedup = ms/iter at TP 2 / at TP.
{
    echo "# Llama 3 70B TP+SP scaling over TP, BF16, $GPU_NAME"
    echo
    echo "$SCALE_LAYERS layers, SEQ 8192, $E2E_ITERS iterations, DP 1 (a TP-T run uses GPUs 0..T-1, the rest idle), GA fusion off."
    echo "Weak: MBS = TP/2, GBS = 16 MBS. Strong: MBS 2, GBS 32. Medians over the valid runs of Megatron's printed numbers."
    for c in $SCALE_CONFIGS; do echo "${c%%:*} ${c#*:}"; done | awk -v O="$O" -v L="$SCALE_LAYERS" -v ITERS="$E2E_ITERS" '
        function med(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } return n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
        function stat(T, MBS, GBS, arm, what,   f, cmd, line, n, x, k, it, ms, tf, tok) {
            n = 0
            cmd = "ls -d " O "/runs/llama70B_tp" T "_mbs" MBS "_gbs" GBS "_L" L "_" arm "_r* 2>/dev/null"
            while ((cmd | getline f) > 0) {
                it = 0; ms = tf = tok = ""
                while ((getline line < (f "/output_perf.log")) > 0) {
                    if (line ~ / iteration +[0-9]+\/ *[0-9]+ \|/) it++
                    if (line ~ /^throughput per GPU:/) { split(line, x, ": "); tf = x[2] }
                    if (line ~ /^elapsed time per iteration:/) { split(line, x, ": "); ms = x[2] }
                    if (line ~ /^tokens\/GPU\/s:/) { split(line, x, ": "); tok = x[2] }
                }
                close(f "/output_perf.log")
                if (it >= ITERS && ms != "") { n++; v["ms", n] = ms; v["tf", n] = tf; v["tok", n] = tok }
            }
            close(cmd)
            if (what == "n") return n
            if (n == 0) return ""
            for (k = 1; k <= n; k++) x[k] = v[what, k]
            return med(x, n)
        }
        { S[++ns] = $1; split($2, p, ":"); TP[ns] = p[1]; MB[ns] = p[2]; GB[ns] = p[3] }
        END {
            for (s = 1; s <= ns; s++) {
                if (S[s] != last) {
                    print "\n## " (S[s] == "weak" ? "Weak scaling" : "Strong scaling") "\n"
                    print "| TP | MBS | GBS | arm | ms/iter | TFLOP/s/GPU | tokens/GPU/s | valid runs | KOSMOS speedup | " (S[s] == "weak" ? "efficiency vs TP 2" : "speedup vs TP 2") " |"
                    print "|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|"
                    last = S[s]
                }
                mb = stat(TP[s], MB[s], GB[s], "base_noovl", "ms")
                for (a = 0; a < 2; a++) {
                    arm = a ? "kos_tpsp" : "base_noovl"
                    ms = stat(TP[s], MB[s], GB[s], arm, "ms"); tf = stat(TP[s], MB[s], GB[s], arm, "tf")
                    tok = stat(TP[s], MB[s], GB[s], arm, "tok"); n = stat(TP[s], MB[s], GB[s], arm, "n")
                    if (TP[s] == 2) ref[S[s], arm] = (S[s] == "weak" ? tf : ms)
                    sc = "-"
                    if (ms != "" && ref[S[s], arm] != "") sc = sprintf("%.3f", S[s] == "weak" ? tf / ref[S[s], arm] : ref[S[s], arm] / ms)
                    printf "| %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |\n", TP[s], MB[s], GB[s], arm == "kos_tpsp" ? "**KOSMOS**" : "no overlap",
                        ms == "" ? "-" : sprintf("%.1f", ms), tf == "" ? "-" : sprintf("%.1f", tf), tok == "" ? "-" : sprintf("%.0f", tok), n,
                        (a && ms != "" && mb != "") ? sprintf("%.3fx", mb / ms) : "-", sc
                }
            }
        }'
    echo
    echo "## Every run"
    echo
    echo "| run | TP | MBS | GBS | arm | rc | note | log |"
    echo "|---|--:|--:|--:|---|--:|---|---|"
    awk -F'\t' '{ printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", $2, $3, $4, $5, $7, $9, $10, $11 }' "$O/runs.tsv"
} > "$OUT_ROOT/RESULTS_SCALE.md"
log "wrote $OUT_ROOT/RESULTS_SCALE.md"
