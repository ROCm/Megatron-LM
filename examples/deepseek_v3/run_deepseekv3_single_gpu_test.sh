#!/bin/bash
###############################################################################
# Single-GPU DSV3 16B e2e test (gfx1250 validation)
# Runs 20 steps, profiles step 12 via PyTorch profiler.
#
# Usage:
#   cd /path/to/Megatron-LM
#   bash examples/deepseek_v3/run_deepseekv3_single_gpu_test.sh <dtype>
#
# Required argument:
#   bf16          — BF16 precision
#   fp8_delayed   — FP8 with delayed scaling
#   fp8_current   — FP8 with tensorwise (current) scaling
#   mxfp8         — FP8 with MXFP8 scaling
###############################################################################

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <bf16|fp8_delayed|fp8_current|mxfp8>" >&2
    exit 1
fi
DTYPE="$1"

# common config
export MODEL_SIZE=16B
export TRAIN_ITERS=20
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=4
export MOCK_DATA=1
export TP=1
export PP=1
export EP=1
export ETP=1
export CP=1
export AC=none
export USE_GROUPED_GEMM=false
export GA_FUSION=false
export PROFILE=true
export PROFILE_START=12
export PROFILE_END=13
export NVTE_FLASH_ATTN=1

# dtype-specific config
case "${DTYPE}" in
    bf16)
        export PR=bf16
        ;;
    fp8_delayed)
        export PR=fp8
        export FP8_RECIPE=delayed
        export NVTE_FP8_DPA_BWD=0
        export FP8_FORMAT=${FP8_FORMAT:-hybrid}
        ;;
    fp8_current)
        export PR=fp8
        export FP8_RECIPE=tensorwise
        export NVTE_FP8_DPA_BWD=0
        ;;
    mxfp8)
        export PR=fp8
        export FP8_RECIPE=mxfp8
        export NVTE_FP8_DPA_BWD=0
        ;;
    *)
        echo "ERROR: unknown dtype '${DTYPE}' (expected: bf16|fp8_delayed|fp8_current|mxfp8)" >&2
        exit 1
        ;;
esac

# Disable Megatron NaN-rerun so we can see the full trace
export RERUN_MODE=${RERUN_MODE:-disabled}

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${CURRENT_DIR}")")"
cd "${REPO_ROOT}"

echo "Running DSV3 16B single-GPU test with dtype=${DTYPE}"
bash examples/deepseek_v3/train_deepseekv3.sh
