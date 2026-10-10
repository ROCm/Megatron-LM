#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/gen_scripts.sh KIND MLM_DIR OUTDIR: derived copies of this tree's train scripts, written to OUTDIR (outside the
# tree), for switches the published runs used and the scripts do not carry. Prints the generated path.
#   llama   examples/llama/train_llama3.sh + NUM_LAYERS_OVERRIDE, GRADIENT_ACCUMULATION_FUSION taken from GA_FUSION
#           (default 0: the container APEX has no fused_weight_gradient_mlp_cuda; 1 needs the APEX stand-in). The
#           script's other fusion / DP-comm toggles (DP-average in the collective, CE fusion, fused QKV+RoPE, BF16 grad
#           reduce) keep their defaults (on)
#   qwen | dsv3   train_qwen3.sh / train_deepseekv3.sh + DDP_OVERLAP=false (no --overlap-grad-reduce /
#           --overlap-param-gather; noovl ablation) and MOE_ROUTER_PADDING_FOR_QUANT=false (no
#           --moe-router-padding-for-quantization under mxfp8: TE's Fp8Padding pads the experts instead), CURRENT_DIR
#           pinned to the tree. With neither variable set the copy runs as the original
set -euo pipefail
KIND=$1; MLM=$2; OUT=$3
mkdir -p "$OUT"
case $KIND in
    llama)
        src=$MLM/examples/llama/train_llama3.sh; dst=$OUT/train_llama3_e2e.sh
        awk '/^GROUP_SIZE=/ && !n { print "NUM_LAYERS=\"${NUM_LAYERS_OVERRIDE:-$NUM_LAYERS}\""; print ""; n = 1 }
             /^GRADIENT_ACCUMULATION_FUSION=/ { print "GRADIENT_ACCUMULATION_FUSION=\"${GA_FUSION:-0}\""; next }
             { print }' "$src" > "$dst"
        grep -q 'NUM_LAYERS_OVERRIDE' "$dst" && grep -qx 'GRADIENT_ACCUMULATION_FUSION="${GA_FUSION:-0}"' "$dst" &&
            grep -q -- '--no-gradient-accumulation-fusion' "$dst" ||
            { echo "gen_scripts: anchors not found in $src" >&2; exit 1; } ;;
    qwen | dsv3)
        [ "$KIND" = qwen ] && sub=qwen3/train_qwen3.sh || sub=deepseek_v3/train_deepseekv3.sh
        src=$MLM/examples/$sub; dst=$OUT/e2e_$(basename "$sub")
        awk -v cur="$(dirname "$src")" '
             /^CURRENT_DIR=/ { print "CURRENT_DIR=\"" cur "\""; next }
             /^DISTRIBUTED_ARGS=/ { print "if [ \"${DDP_OVERLAP:-true}\" = false ]; then comm_overlap_option=\" --ddp-bucket-size 629145600\"; fi" }
             /^ *(moe_options=|echo ).*--moe-router-padding-for-quantization"$/ {
                 match($0, /^ */); $0 = substr($0, 1, RLENGTH) "[ \"${MOE_ROUTER_PADDING_FOR_QUANT:-true}\" = false ] || " substr($0, RLENGTH + 1) }
             { print }' "$src" > "$dst"
        [ "$(grep -c 'DDP_OVERLAP' "$dst")" = 1 ] && [ "$(grep -c 'MOE_ROUTER_PADDING_FOR_QUANT' "$dst")" = 2 ] &&
            grep -q "^CURRENT_DIR=\"$(dirname "$src")\"" "$dst" ||
            { echo "gen_scripts: anchors not found in $src" >&2; exit 1; } ;;
    *) echo "usage: gen_scripts.sh llama|qwen|dsv3 MLM_DIR OUTDIR" >&2; exit 2 ;;
esac
echo "$dst"
