#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# run_all.sh [STEP [--flag...]]...: regenerates every KOSMOS result on one 8-GPU gfx950 node. Steps run in the order
# given; default: setup micro tpsp a2a collect. DRYRUN=1 prints every command (and each step's run count and GPU time).
#   manual   print the hand steps (images, containers, clones, tokenizers)
#   setup    verify branches, builds and dependencies (read-only); --checkout, --build, --build-deps change things
#   micro    KOSMOS bench/run_all.sh -> OUT_ROOT/micro/<GPU>.md
#   tpsp     Llama 3 8B / 70B / 405B TP+SP E2E
#   a2a      Qwen3-235B / DeepSeek-V3 MoE E2E, every backend
#   collect  Megatron's printed numbers -> OUT_ROOT/RESULTS_TPSP.md, RESULTS_A2A.md, RESULTS_A2A_internal.md
#   scale    Llama 3 70B TP+SP weak and strong scaling over TP 2, 4, 8 (BF16) -> OUT_ROOT/RESULTS_SCALE.md
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$KIT/lib/common.sh"

steps=(); flags=()
for a in "${@:-setup micro tpsp a2a collect}"; do
    for w in $a; do
        if [[ $w == --* ]]; then
            [ ${#steps[@]} -gt 0 ] || die "flag $w before any step"
            flags[${#steps[@]} - 1]+=" $w"
        else
            steps+=("$w"); flags+=("")
        fi
    done
done
log "node $GPU_NAME ($GPU_ARCH, $NGPU GPUs, ${POWER_CAP_W:-?} W), out $OUT_ROOT, SEED=$SEED$(dry && echo ', DRYRUN')"
for i in "${!steps[@]}"; do
    s=${steps[$i]}
    case $s in
        manual | setup | micro | tpsp | a2a | collect | scale) ;;
        *) die "unknown step $s (manual setup micro tpsp a2a collect scale)" ;;
    esac
    log "=== step $s${flags[$i]} ==="
    # shellcheck disable=SC2086
    SEED=$SEED bash "$KIT/steps/$s.sh" ${flags[$i]}
    rc=$?
    [ $rc -eq 0 ] || die "step $s failed (rc=$rc)"
done
log "done"
