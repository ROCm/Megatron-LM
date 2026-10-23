#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/collect.sh: Megatron's own printed numbers -> $OUT_ROOT/RESULTS_TPSP.md, RESULTS_A2A.md (public: no MegaMoE,
# no ablations) and RESULTS_A2A_internal.md (with MegaMoE, the Primus workflow and the ablations); A2A one section per
# precision (BF16, MXFP8) and routing, with each skewed run's measured routing (tools/routing_stats.sh) and KOSMOS's exit status report. Values
# are copied from the logs: the train script's end-of-run lines (TFLOP/s/GPU, ms/iter, tokens/GPU/s; Llama also mem
# usages), "mem usages" of the last iteration line (MoE), iteration-1 lm loss / grad norm; Primus prints no summary, so
# its last-iteration averages. A run is valid if it exited 0 (or, Llama 405B only, 1: its ranks' SIGSEGV at exit after
# the summary), printed E2E_ITERS iterations and a summary, its mean run-window host load was below COLLECT_LOAD_MAX,
# for kos_tpsp no op fell back ([KOSMOS] ub N op: non-overlapped / HipKittens, or "did not take") and, for KOSMOS A2A,
# every rank's exit status report said ok. Per (config, arm): the median of the valid runs, their min / max and count;
# fewer than E2E_REPS valid runs is flagged. Speedup = the arm's median ms/iter / KOSMOS's median ms/iter. The
# iteration-1 lm loss of every run is compared with the median of its (model, size, precision) group (LOSS_RTOL).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

# summ LOG RUN: TSV "tf ms tok mem iters loss1 gn1 load"
summ() {
    local load=-
    if [ -f "$2/t_start" ] && [ -f "$2/t_end" ]; then
        load=$(cat "$(dirname "$(dirname "$2")")/loadavg.log" 2>/dev/null |
            awk -v a="$(($(cat "$2/t_start") - 60))" -v b="$(cat "$2/t_end")" '$1 >= a && $1 <= b { s += $2; n++ }
                END { if (n) printf "%.1f (%d)", s / n, n; else print "-" }')
    fi
    [ -f "$1" ] || { printf -- '-\t-\t-\t-\t0\t-\t-\t%s\n' "$load"; return; }
    awk -v ld="$load" -v iters="$E2E_ITERS" '
        function grab(s, rx,   m) { return match(s, rx) ? substr(s, RSTART, RLENGTH) : "" }
        function val(s, key,   v) { v = grab(s, key "[^|]*"); sub(/^[^:]*: */, "", v); sub(/ *$/, "", v); return v }
        /^throughput per GPU:/         { tf = $4 }
        /^elapsed time per iteration:/ { ms = $5 }
        /^tokens\/GPU\/s:/             { tok = $2 }
        /^mem usages:/                 { mems = $3 }
        / iteration +[0-9]+\/ *[0-9]+ \|/ {
            it = grab($0, "iteration +[0-9]+"); sub(/iteration +/, "", it); n = it
            m = val($0, "mem usages"); if (m != "") mem = m
            if (it == 1) { l1 = val($0, "lm loss"); g1 = val($0, "grad norm") }
            if ($0 ~ /inst\/harmonic mean/) {    # Primus: the averages of the last printed iteration
                e = val($0, "elapsed time per iteration \\(ms\\)"); sub(/.*\//, "", e); pms = e
                t = grab($0, "\\(avg [0-9.]+\\)"); gsub(/[^0-9.]/, "", t); ptf = t
                k = val($0, "tokens/s/GPU inst/harmonic mean"); sub(/.*\//, "", k); ptok = k
                r = grab($0, "rocm mem usage[^|]*"); sub(/.*\//, "", r); sub(/ .*/, "", r); pmem = r
            }
        }
        END {
            if (mems != "") mem = mems
            if (ms == "" && pms != "" && n == iters) { tf = ptf; ms = pms; tok = ptok; mem = pmem }
            printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n", tf == "" ? "-" : tf, ms == "" ? "-" : ms, tok == "" ? "-" : tok,
                mem == "" ? "-" : mem, n, l1 == "" ? "-" : l1, g1 == "" ? "-" : g1, ld }' "$1"
}

# Relative deviation of a run's iteration-1 lm loss from the median of its group (same model, size, precision, GA,
# routing) that collect flags. The arms share the seed, initialization and mock data, so their first forward differs
# only in GEMM / collective rounding: about 2^-8 per bf16 element, averaged over >= 65536 tokens into the fp32 mean loss,
# i.e. far below 1e-4. 1e-3 is passed only by a systematic difference (other data, initialization, masking, or a wrong
# GEMM / collective), not by rounding; at initialization the loss is near ln(vocab) for any small logits, so the check
# catches gross errors only.
LOSS_RTOL=1e-3

# dedup RUNS_TSV LOG_COLUMN: the rows of RUNS_TSV, one per log path (a run repeated in place appends a second row with
# the same log): the last row of each, in file order.
dedup() {
    awk -F'\t' -v c="$2" '{ k = $c == "-" ? "-" NR : $c; R[NR] = $0; K[NR] = k; last[k] = NR }
        END { for (i = 1; i <= NR; i++) if (last[K[i]] == i) print R[i] }' "$1"
}

# runs_fp RUNS_TSV LOG_COLUMN: the build fingerprints (fingerprint.txt) of its runs, a count per build; runs moved
# aside as stale (RUN.stale.<epoch>, superseded by a re-run of another build) are left out.
runs_fp() {
    dedup "$1" "$2" | awk -F'\t' -v c="$2" '$c != "-" && $c !~ /[.]stale[.][0-9]+\// { print $c }' | while read -r lg; do
        cat "$(dirname "$lg")/fingerprint.txt" 2>/dev/null || echo "(none)"
    done | sort | uniq -c
}

header() {  # header MEGATRON_TREE RUNS_TSV LOG_COLUMN
    local d img=host
    [ -z "$KOSMOS_CT" ] || img=$("$DOCKER" inspect -f '{{.Config.Image}} ({{.Image}})' "$KOSMOS_CT" 2>/dev/null) || img=?
    echo "Node: $NGPU x $GPU_NAME ($GPU_ARCH), power cap ${POWER_CAP_W:-?} W per GPU; container ${KOSMOS_CT:-none}:" \
        "$img. Trees now:"
    for d in "$KOSMOS_DIR" "$TE_DIR" "$1"; do
        echo "- \`$(basename "$d")\` $(git -C "$d" log -1 --format='%h [%D] %s' 2>/dev/null)," \
            "\`$(git -C "$d" describe --always --dirty 2>/dev/null)\`, $(git -C "$d" status --short 2>/dev/null |
            wc -l) changed files"
    done
    echo
    echo "Builds of the runs (KOSMOS and TE git describe --dirty, md5 of libtransformer_engine.so), count per build:"
    runs_fp "$2" "$3" | sed 's/^ */- /'
    [ "$(runs_fp "$2" "$3" | wc -l)" -le 1 ] || echo "**WARNING: the runs were made by more than one build.**"
    dedup "$2" "$3" | grep -q '[.]stale[.][0-9]*/' &&
        echo "Runs superseded by a re-run with another build (moved to RUN.stale.<epoch>) are listed but not valid."
    echo
    echo "$E2E_ITERS iterations per run, $E2E_REPS runs per (config, arm). A run's ms/iter is the train script's own"
    echo "summary: the mean of Megatron's per-iteration elapsed-time prints over iterations 3-$((E2E_ITERS - 1)) (Llama:"
    echo "lines[2:-1]) or 3-$E2E_ITERS (MoE: --skip-first 2, profiling off)."
    echo "A run is valid if it exited 0 (Llama 405B: or 1, its known crash at process exit after the summary), printed"
    echo "all $E2E_ITERS iterations and the end-of-run summary, and the mean host"
    [ "$COLLECT_LOAD_MAX" = 0 ] && echo "1-min load is reported per run (not gated)." ||
        echo "1-min load during it was below $COLLECT_LOAD_MAX."
    echo "Values are the median over the valid runs of Megatron's printed numbers; min / max are of ms/iter; speedup ="
    echo "arm median ms/iter / KOSMOS median ms/iter. Fewer than $E2E_REPS valid runs is marked."
    echo "Iteration-1 lm loss: each run against the median of its group (same model, size, precision); a relative"
    echo "deviation above $LOSS_RTOL is flagged (it 1 loss dev); the arms share seed, initialization and data, so"
    echo "rounding alone stays far below it."
}

# Shared awk helpers: med(list) of a space-separated list, mm(list) "min / max".
AWK_LIB='
    function srt(s, a,   n, i, j, t) { n = split(s, a, " "); for (i = 2; i <= n; i++) { t = a[i] + 0; for (j = i - 1; j >= 1 && a[j] + 0 > t; j--) a[j + 1] = a[j]; a[j + 1] = t } return n }
    function med(s,   a, n) { n = srt(s, a); if (!n) return ""; return n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
    function mm(s,   a, n) { n = srt(s, a); return n ? sprintf("%.1f / %.1f", a[1], a[n]) : "" }
    function runs(c) { return c < reps ? sprintf("%d of %d (fewer than %d)", c, reps, reps) : c }
'

# kfb LOG: the KOSMOS TE backend's fallback lines (NVTE_KOSMOS_LOG=1, tp rank 0, once per buffer and op): an op run
# non-overlapped or on the in-tree HipKittens kernels instead of KOSMOS, or a fused call KOSMOS did not take.
kfb() {
    [ -f "$1" ] || { echo -; return; }
    grep -Ec '\[KOSMOS\] ub [0-9]+ [^:]*: (non-overlapped|HipKittens)|did not take' "$1"
}

tpsp() {
    local O=$OUT_ROOT/tpsp F=$OUT_ROOT/RESULTS_TPSP.md S=$OUT_ROOT/tpsp/summary.tsv fb
    : > "$S"
    while IFS=$'\t' read -r d tag M MBS GBS L arm ga r rc l0 lg note pr; do
        fb=-
        case $arm in kos_*) fb=$(kfb "$lg") ;; esac
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$M" "$MBS" "$GBS" "$L" "$arm" "$ga" \
            "$r" "$rc" "$l0" "$lg" "$(summ "$lg" "$(dirname "$lg")")" "${note:--}" "${pr:-bf16}" "$fb" \
            "$(cat "$(dirname "$lg")/fingerprint.txt" 2>/dev/null || echo -)" >> "$S"
    done < <(dedup "$O/runs.tsv" 12)
    {
        echo "# Llama 3 TP+SP E2E, BF16 and MXFP8 (train_llama3.sh, SEQ 8192, TP8; MXFP8: TE_FP8=1 TE_FP8_RECIPE=mxfp8)"
        echo
        header "$TPSP_MLM_DIR" "$O/runs.tsv" 12
        echo "Arms: base_noovl = no overlap (hipBLASLt + RCCL); kos_tpsp = KOSMOS for every overlapped op. A kos_tpsp run"
        echo "with a KOSMOS fallback line (KOSMOS fallbacks column) is not valid."
        echo "Layer-reduced proxies: a model whose layers column is below its full depth (8B 32, 70B 80, 405B 126 layers) is"
        echo "a proxy of that model, labelled \"proxy, L of N layers\"."
        echo "Speedup is vs kos_tpsp."
        echo "A run that stalled, timed out or ran out of memory / launch resources (note) stops its (model, arm, GA): its"
        echo "remaining repeats and every larger MBS are listed as skipped."
        awk -F'\t' -v lmax="$COLLECT_LOAD_MAX" -v iters="$E2E_ITERS" -v reps="$E2E_REPS" -v rtol="$LOSS_RTOL" "$AWK_LIB"'
            BEGIN { nar = split("base_noovl kos_tpsp", AO, " "); for (a = 1; a <= nar; a++) known[AO[a]] = 1
                    FULL[8] = 32; FULL[70] = 80; FULL[405] = 126 }
            function ok(i) { return (RC[i] == 0 || (RC[i] == 1 && M[i] == 405)) && IT[i] == iters && R[i, "ms"] ~ /^[0-9.]+$/ && (lmax == 0 || L[i] == "-" || L[i] + 0 < lmax) && (FB[i] == "-" || FB[i] == 0) && LG[i] !~ /[.]stale[.][0-9]+\// }
            function model(m, l) { return m "B" ((m in FULL) && l + 0 < FULL[m] ? sprintf(" (proxy, %d of %d layers)", l, FULL[m]) : "") }
            { n++; T[n] = $1; M[n] = $2; B[n] = $3; G[n] = $4; Y[n] = $5; A[n] = $6; GA[n] = $7; RC[n] = $9
              R[n, "tf"] = $12; R[n, "ms"] = $13; R[n, "tok"] = $14; R[n, "mem"] = $15; IT[n] = $16
              L1[n] = $17; G1[n] = $18; LD[n] = $19; L[n] = $19; sub(/ .*/, "", L[n]); if (L[n] == "-") L[n] = $10; LG[n] = $11
              NT[n] = $20; PR[n] = $21 == "" ? "bf16" : $21; hasp[PR[n]] = 1; FB[n] = $22; FPR[n] = $23
              k = $7 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP PR[n]
              if (!(k in seen)) { seen[k] = 1; nk++; K[nk] = k }
              has[k, $6] = 1
              if (!($6 in known)) { known[$6] = 1; nar++; AO[nar] = $6 } }
            END {
                for (i = 1; i <= n; i++) if (L1[i] ~ /^[0-9]/) { q = GA[i] SUBSEP M[i] SUBSEP B[i] SUBSEP G[i] SUBSEP Y[i] SUBSEP PR[i]; LS[q] = LS[q] " " L1[i] }
                for (i = 1; i <= n; i++) {
                    q = GA[i] SUBSEP M[i] SUBSEP B[i] SUBSEP G[i] SUBSEP Y[i] SUBSEP PR[i]; DV[i] = "-"
                    if (L1[i] ~ /^[0-9]/) { ref = med(LS[q]); d = ref ? (L1[i] - ref) / ref : 0; d = d < 0 ? -d : d; DV[i] = sprintf("%.1e%s", d, d > rtol ? " DEV" : "") }
                }
                np = split("bf16 mxfp8", PO, " ")
                for (pi = 1; pi <= np; pi++) for (g = 0; g <= 1; g++) {
                    pp = PO[pi]; if (!(pp in hasp)) continue
                    gany = 0; for (j = 1; j <= nk; j++) { split(K[j], f, SUBSEP); if (f[1] == g && f[6] == pp) gany = 1 }
                    if (!gany) continue
                    printf "\n## %s, gradient-accumulation fusion %s\n\n", pp == "mxfp8" ? "MXFP8" : "BF16", g ? "on" : "off"
                    print "| model | MBS | GBS | layers | arm | ms/iter (median) | min / max | TFLOP/s/GPU | tokens/GPU/s | mem usages | valid runs | speedup |"
                    print "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|"
                    for (j = 1; j <= nk; j++) {
                        split(K[j], f, SUBSEP); if (f[1] != g || f[6] != pp) continue
                        delete V; delete C; delete W
                        for (i = 1; i <= n; i++) if (GA[i] == f[1] && M[i] == f[2] && B[i] == f[3] && G[i] == f[4] && Y[i] == f[5] && PR[i] == f[6] && !ok(i) && NT[i] != "-" && !(A[i] in W)) W[A[i]] = NT[i]
                        for (i = 1; i <= n; i++) if (GA[i] == f[1] && M[i] == f[2] && B[i] == f[3] && G[i] == f[4] && Y[i] == f[5] && PR[i] == f[6] && ok(i)) {
                            V[A[i], "ms"] = V[A[i], "ms"] " " R[i, "ms"]; V[A[i], "tf"] = V[A[i], "tf"] " " R[i, "tf"]
                            V[A[i], "tok"] = V[A[i], "tok"] " " R[i, "tok"]; V[A[i], "mem"] = V[A[i], "mem"] " " R[i, "mem"]; C[A[i]]++ }
                        ref = C["kos_tpsp"] ? med(V["kos_tpsp", "ms"]) : 0
                        for (a = 1; a <= nar; a++) {
                            x = AO[a]; if (!((K[j], x) in has)) continue
                            if (!C[x]) { printf "| %s | %s | %s | %s | %s | %s | | | | | 0 of %d | |\n", model(f[2], f[5]), f[3], f[4], f[5], x, (x in W) && W[x] == "oom" ? "OOM" : "no valid run" ((x in W) ? " (" W[x] ")" : ""), reps; continue }
                            ms = med(V[x, "ms"])
                            printf "| %s | %s | %s | %s | %s | %.1f | %s | %.1f | %.1f | %.4f | %s | %s |\n", model(f[2], f[5]), f[3], f[4], f[5], x, ms,
                                mm(V[x, "ms"]), med(V[x, "tf"]), med(V[x, "tok"]), med(V[x, "mem"]), runs(C[x]),
                                (ref && x != "kos_tpsp") ? sprintf("%.3fx", ms / ref) : ""
                        }
                    }
                    dl = ""
                    for (i = 1; i <= n; i++) if (GA[i] == g && PR[i] == pp && DV[i] ~ /DEV/) dl = dl (dl == "" ? "" : ", ") T[i] " (" DV[i] ")"
                    printf "\nIteration-1 lm loss vs its group median: %s\n", dl == "" ? "every run within " rtol : "**deviation above " rtol ": " dl "**"
                }
                print "\n## Every run\n"
                print "| run | arm | precision | GA | rc | iters | TFLOP/s/GPU | ms/iter | tokens/GPU/s | mem usages | it 1 lm loss | it 1 loss dev | it 1 grad norm | load | KOSMOS fallbacks | note | valid | build | log |"
                print "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|---|"
                for (i = 1; i <= n; i++)
                    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", T[i], A[i], PR[i], GA[i], RC[i], IT[i],
                        R[i, "tf"], R[i, "ms"], R[i, "tok"], R[i, "mem"], L1[i], DV[i], G1[i], LD[i], FB[i], NT[i], ok(i) ? "yes" : "no", FPR[i], LG[i]
            }' "$S"
    } > "$F"
    log "wrote $F"
}

# kstatus LOG PREC: the KOSMOS adapter's exit report over the ranks: ok (every rank printed ok), OVERFLOW, query-failed,
# or missing(n/NGPU); an mxfp8 run whose layers did not all run MXFP8 experts: bf16-experts (the adapter's fallback) or
# no-mxfp8 (no layer reported MXFP8 experts).
kstatus() {
    local n
    [ -f "$1" ] || { echo "missing(0/$NGPU)"; return; }
    if [ "$2" = mxfp8 ]; then
        grep -q '\[KOSMOS\] layer [0-9]*: bf16 experts' "$1" && { echo bf16-experts; return; }
        grep -q '\[KOSMOS\] layer [0-9]*: MXFP8 experts' "$1" || { echo no-mxfp8; return; }
    fi
    if grep -q '\[KOSMOS\] plan status rank [0-9]*: OVERFLOW' "$1"; then echo OVERFLOW; return; fi
    if grep -q '\[KOSMOS\] plan status rank [0-9]*: QUERY FAILED' "$1"; then echo query-failed; return; fi
    n=$(grep -o '\[KOSMOS\] plan status rank [0-9]*: ok' "$1" | sort -u | wc -l)
    [ "$n" -ge "$NGPU" ] && echo ok || echo "missing($n/$NGPU)"
}

# gfb LOG: TE grouped-GEMM warning lines (NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1; every rank and call): [HK-grouped] /
# [HK-wgrad] (including the non-contiguous notice a wgrad call prints before HipKittens' grouped wgrad serves it; not
# the NT probe) and the CK -> hipBLASLt fallback. "-" where the arm has no TE grouped GEMM.
gfb() {
    [ -f "$1" ] || { echo -; return; }
    grep -E '\[HK-(grouped|wgrad)\]|Fallback to cuBLAS grouped GEMM' "$1" | grep -vc 'NT layout: use grouped_mxfp8_wgrad'
}

a2a() {  # a2a FILE INTERNAL(0|1)
    local O=$OUT_ROOT/a2a F=$1 S=$OUT_ROOT/a2a/summary.tsv kst cfs rs fb
    : > "$S"
    while IFS=$'\t' read -r d tag m arm var r rc l0 lg rt pr; do
        kst=-; cfs=-
        if [ "$arm" = kosmos ]; then
            kst=$(kstatus "$lg" "${pr:-bf16}")
            cfs=$(sed -n 's/^KOSMOS_CAPACITY_FACTOR=//p' "$(dirname "$lg")/arm_env.txt" 2>/dev/null)
            cfs=${cfs:-1.5}
        fi
        rs=$(bash "$KIT/tools/routing_stats.sh" "$(dirname "$lg")/routing")
        fb=-
        case $arm in grouped | deepep) fb=$(gfb "$lg") ;; esac
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$m" "$arm" "$var" "$r" "$rc" \
            "$l0" "$lg" "$(summ "$lg" "$(dirname "$lg")")" "${rt:-balanced}" "$kst" "$(echo "$rs" | tr ' ' '\t')" "$cfs" \
            "${pr:-bf16}" "$fb" "$(cat "$(dirname "$lg")/fingerprint.txt" 2>/dev/null || echo -)" >> "$S"
    done < <(dedup "$O/runs.tsv" 9)
    {
        echo "# KOSMOS MoE (A2A) end-to-end training, $NGPU x $GPU_NAME, BF16 / MXFP8$([ "$2" = 1 ] && echo " -- INTERNAL")"
        echo
        header "$A2A_MLM_DIR" "$O/runs.tsv" 9
        echo "Megatron-LM example scripts, EP=8, profiling off. KOSMOS runs Megatron's routing (balanced: its"
        echo "force-load-balancing draw; natural: from the router weights) in its fused router, BF16 and MXFP8 alike (its"
        echo "per-layer [KOSMOS] line: fused router; DeepSeek-V3 also fused shared expert); the other backends use"
        echo "Megatron's router.$([ "$2" = 1 ] && echo " * Primus's workflow runs its own stack and config (not like-for-like); its values are its printed last-iteration averages.")"
        echo "A KOSMOS run is valid only if every rank's exit report ([KOSMOS] plan status) says ok: no call exceeded the"
        echo "arena (capacity factor x T x k rows per rank) or had a bad routing; under MXFP8 also only if every MoE layer"
        echo "ran MXFP8 experts (no [KOSMOS] bf16-experts fallback)."
        echo "MXFP8: --fp8-recipe mxfp8 on every layer. KOSMOS runs its expert GEMMs in MXFP8 (KOSMOS_MXFP8=1) with its fused"
        echo "router; the TE grouped GEMM backends on the HipKittens MXFP8 grouped GEMM, experts padded to 256 rows by TE's"
        echo "Fp8Padding$([ "$A2A_MXFP8_ROUTER_PAD" = 1 ] && echo " (A2A_MXFP8_ROUTER_PAD=1: router padding to 128; HipKittens declines odd multiples of 128)"); MegaMoE on Primus-Turbo's staged MXFP8 op."
        echo "Both models are layer-reduced proxies: Qwen3-235B-A22B with 24 of its 94 layers, DeepSeek-V3 with 3 of its 61."
        echo "Speedups are against KOSMOS at the same precision. HK grouped-GEMM notices (every run; not a validity criterion)"
        echo "counts TE's grouped-GEMM warning lines (NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1), every rank and call:"
        echo "[HK-grouped] / [HK-wgrad] and \"Fallback to cuBLAS grouped GEMM\". TE libtransformer_engine.so"
        echo "a3b5b255ae1c487004f74400f033f01b prints \"[HK-grouped] non-contiguous activation/output at expert 1, falling"
        echo "back\" once per MXFP8 wgrad (NT) call and rank: its forward HipKittens check runs on the wgrad operands and"
        echo "declines, then HipKittens' grouped wgrad (try_kittens_grouped_mxfp8_wgrad) serves the call; build"
        echo "9a925162eba5855ab1500ccc6049bef1 skips that check for NT calls and does not print it."
        awk -F'\t' -v lmax="$COLLECT_LOAD_MAX" -v iters="$E2E_ITERS" -v reps="$E2E_REPS" -v internal="$2" -v rtol="$LOSS_RTOL" "$AWK_LIB"'
            BEGIN {
                no = split("kosmos:- grouped:- deepep:- megamoe:- primus:- kosmos:rtroff kosmos:gateoff kosmos:noovl kosmos:numa", ORD, " ")
                LB["kosmos:-"] = "**KOSMOS**"; LB["grouped:-"] = "Megatron base (TE grouped GEMM + all-to-all)"
                LB["deepep:-"] = "DeepEP"; LB["megamoe:-"] = "MegaMoE (Primus-Turbo) in Megatron"
                LB["primus:-"] = "MegaMoE, Primus'"'"'s own workflow *"; LB["kosmos:rtroff"] = "KOSMOS, router off"
                LB["kosmos:gateoff"] = "KOSMOS, DDP gate off"; LB["kosmos:noovl"] = "KOSMOS, DDP overlap off"; LB["kosmos:numa"] = "KOSMOS, NUMA binding"
                MT["qwen"] = "Qwen3-235B-A22B proxy (24 of 94 layers)"; MT["dsv3"] = "DeepSeek-V3 proxy (3 of 61 layers)"
                nr = split("balanced natural", RO, " ")
                np = split("bf16 mxfp8", PO, " "); PD["bf16"] = "BF16"; PD["mxfp8"] = "MXFP8"
                RTD["balanced"] = "Balanced routing (Megatron force-load-balancing)"
                RTD["natural"] = "Skewed routing, natural (FORCE_BALANCE=false: Megatron'"'"'s router with its aux loss, from initialization)"
            }
            function pub(k) { return internal || (k ~ /:-$/ && k !~ /^(megamoe|primus):/) }
            function ok(i) { return RC[i] == 0 && IT[i] == iters && MS[i] ~ /^[0-9.]+$/ && (lmax == 0 || L[i] == "-" || L[i] + 0 < lmax) && (KS[i] == "-" || KS[i] == "ok") && LG[i] !~ /[.]stale[.][0-9]+\// }
            { n++; T[n] = $1; MD[n] = $2; KEY[n] = $3 ":" $4; RC[n] = $6; LG[n] = $8; TF[n] = $9; MS[n] = $10; TK[n] = $11
              ME[n] = $12; IT[n] = $13; L1[n] = $14; G1[n] = $15; LD[n] = $16; L[n] = $16; sub(/ .*/, "", L[n]); if (L[n] == "-") L[n] = $7
              RTG[n] = $17; KS[n] = $18; NC[n] = $19; RK[n] = $20; HE[n] = $21; CN[n] = $22; CS[n] = $23
              PRC[n] = $24; FB[n] = $25; FPR[n] = $26
              if (!($2 in sm)) { sm[$2] = 1; nm++; MM[nm] = $2 }
              hasrt[$24 SUBSEP $17] = 1
              if (!(KEY[n] in LB)) { LB[KEY[n]] = KEY[n]; ORD[++no] = KEY[n] } }
            END {
                for (i = 1; i <= n; i++) if (L1[i] ~ /^[0-9]/ && pub(KEY[i]) && KEY[i] != "primus:-") { g = MD[i] SUBSEP RTG[i] SUBSEP PRC[i]; LS[g] = LS[g] " " L1[i] }
                for (i = 1; i <= n; i++) {
                    g = MD[i] SUBSEP RTG[i] SUBSEP PRC[i]; DV[i] = "-"
                    if (L1[i] ~ /^[0-9]/ && pub(KEY[i]) && KEY[i] != "primus:-") { ref = med(LS[g]); d = ref ? (L1[i] - ref) / ref : 0; d = d < 0 ? -d : d; DV[i] = sprintf("%.1e%s", d, d > rtol ? " DEV" : "") }
                }
                for (p = 1; p <= np; p++) for (q = 1; q <= nr; q++) {
                    pr = PO[p]; rt = RO[q]; if (!((pr, rt) in hasrt)) continue
                    printf "\n## %s: %s\n", PD[pr], RTD[rt]
                    for (j = 1; j <= nm; j++) {
                        m = MM[j]; delete V; delete C; rk = ""; he = ""; cmax = 0; cset = ""; any = 0; rlo = 0; rhi = 0
                        for (i = 1; i <= n; i++) if (MD[i] == m && RTG[i] == rt && PRC[i] == pr) {
                            any = 1
                            if (!ok(i)) continue
                            if (CS[i] != "-") cset = CS[i]
                            V[KEY[i], "ms"] = V[KEY[i], "ms"] " " MS[i]; V[KEY[i], "tf"] = V[KEY[i], "tf"] " " TF[i]
                            V[KEY[i], "tok"] = V[KEY[i], "tok"] " " TK[i]; C[KEY[i]]++
                            if (RK[i] != "-" && KEY[i] ~ /^kosmos:/) {
                                rk = rk " " RK[i]; he = he " " HE[i]; if (CN[i] + 0 > cmax) cmax = CN[i] + 0 }
                            if (RK[i] != "-") { if (!rlo || RK[i] + 0 < rlo) rlo = RK[i] + 0; if (RK[i] + 0 > rhi) rhi = RK[i] + 0 }
                        }
                        if (!any) continue
                        ref = C["kosmos:-"] ? med(V["kosmos:-", "ms"]) : 0
                        printf "\n**%s**\n\n", (m in MT) ? MT[m] : m
                        if (rk != "")
                            printf "Measured routing (MOE_ROUTING_STATS, max over each run'"'"'s router calls, median over the valid KOSMOS runs): busiest rank %.3fx the mean rows, hottest expert %.3fx the mean; KOSMOS capacity factor needed %.3f (max over its runs), set %s. Each arm trains its own router from initialization, so the skew differs by arm: busiest rank %.3fx to %.3fx over all valid runs.\n\n", med(rk), med(he), cmax, cset == "" ? "1.5" : cset, rlo, rhi
                        printf "| backend | ms/iter (median) | min / max | TFLOP/s/GPU | tokens/GPU/s | valid runs | KOSMOS speedup |\n|---|--:|--:|--:|--:|--:|--:|\n"
                        for (o = 1; o <= no; o++) {
                            k = ORD[o]; if (!pub(k)) continue
                            hk = 0; for (i = 1; i <= n; i++) if (MD[i] == m && KEY[i] == k && RTG[i] == rt && PRC[i] == pr) hk = 1
                            if (!hk) continue
                            lb = LB[k]
                            if (!C[k]) { printf "| %s | no valid run | | | | 0 of %d | |\n", lb, reps; continue }
                            ms = med(V[k, "ms"])
                            printf "| %s | %.1f | %s | %.0f | %.0f | %s | %s |\n", lb, ms, mm(V[k, "ms"]), med(V[k, "tf"]), med(V[k, "tok"]),
                                runs(C[k]), (k == "kosmos:-" || !ref) ? "-" : sprintf("%.2fx", ms / ref)
                        }
                    }
                    dl = ""
                    for (i = 1; i <= n; i++) if (RTG[i] == rt && PRC[i] == pr && pub(KEY[i]) && DV[i] ~ /DEV/) dl = dl (dl == "" ? "" : ", ") T[i] " (" DV[i] ")"
                    printf "\nIteration-1 lm loss vs its group median (model, routing, precision): %s\n", dl == "" ? "every run within " rtol : "**deviation above " rtol ": " dl "**"
                }
                print "\n## Every run\n"
                print "| run | backend | precision | routing | rc | iters | TFLOP/s/GPU | ms/iter | tokens/GPU/s | mem (last it) | it 1 lm loss | it 1 loss dev | it 1 grad norm | load | KOSMOS status | busiest rank / mean | KOSMOS CF needed / set | HK grouped-GEMM notices | valid | build | log |"
                print "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|---|---|---|"
                for (i = 1; i <= n; i++) if (pub(KEY[i]))
                    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", T[i], LB[KEY[i]], PD[PRC[i]], RTG[i], RC[i], IT[i], TF[i],
                        MS[i], TK[i], ME[i], L1[i], DV[i], G1[i], LD[i], KS[i], RK[i], CN[i] (CS[i] == "-" ? "" : " / " CS[i]), FB[i], ok(i) ? "yes" : "no", FPR[i], LG[i]
            }' "$S"
    } > "$F"
    log "wrote $F"
}

if dry; then
    echo "[dryrun] $OUT_ROOT/tpsp/runs.tsv -> $OUT_ROOT/RESULTS_TPSP.md"
    echo "[dryrun] $OUT_ROOT/a2a/runs.tsv -> $OUT_ROOT/RESULTS_A2A.md (public) and RESULTS_A2A_internal.md, one section per"
    echo "[dryrun]   precision ($A2A_PRECISIONS) and routing (balanced, skewed natural); KOSMOS runs without 8 ok status"
    echo "[dryrun]   lines are invalid"
    echo "[dryrun] microbenchmarks: $OUT_ROOT/micro/$GPU_NAME.md (written by KOSMOS bench/run_all.sh report; balanced and"
    echo "[dryrun]   skewed MoE tables per MICRO_DISTS)"
    exit 0
fi
[ -s "$OUT_ROOT/tpsp/runs.tsv" ] && tpsp
if [ -s "$OUT_ROOT/a2a/runs.tsv" ]; then
    a2a "$OUT_ROOT/RESULTS_A2A.md" 0
    a2a "$OUT_ROOT/RESULTS_A2A_internal.md" 1
fi
[ -f "$OUT_ROOT/micro/$GPU_NAME.md" ] && log "microbenchmarks: $OUT_ROOT/micro/$GPU_NAME.md"
exit 0
