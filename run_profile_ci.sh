#!/bin/bash
###############################################################################
# Short profiled TE E2E run for CI: PyTorch profiler + optional TraceLens report.
# Usage: bash run_profile_ci.sh
# Env:
#   TRACE_DIR   - output root (default: output/te_profile_run)
#   RUN_TRACELENS - if 1, run tools/run_tracelens_report.sh on exported trace
###############################################################################
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
TRACE_DIR="${TRACE_DIR:-${ROOT}/output/te_profile_run}"
mkdir -p "$TRACE_DIR"

export ENABLE_PROFILING=1
# train_llama3.sh defaults USE_TE=1; override with USE_TE=0 if needed
export TE_FP8="${TE_FP8:-0}"
# Short run: profiling defaults to a single *active* step (small trace). Override with PROFILE_TRAIN_ITERS.
export TOTAL_ITERS="${PROFILE_TRAIN_ITERS:-6}"
export LOG_DETAILED_TIMERS=1
export MBS=1
export BS=64
export SEQ_LENGTH=4096
export TP=1
export CP=1
export MODEL_SIZE=8
export LOG_DIR="${TRACE_DIR}/tb"

echo "Running profiled training; logs under $TRACE_DIR"
bash examples/llama/train_llama3.sh 2>&1 | tee "${TRACE_DIR}/train.log"

# Find most recent json trace under LOG_DIR (PyTorch tensorboard_trace_handler)
TRACE_JSON=$(find "$LOG_DIR" -name "*.json" -type f 2>/dev/null | head -1 || true)
if [ -n "$TRACE_JSON" ]; then
  echo "Found profiler trace: $TRACE_JSON"
  echo "$TRACE_JSON" > "${TRACE_DIR}/trace_path.txt"
  if [ "${RUN_TRACELENS:-0}" = "1" ]; then
    export TRACE_JSON
    export TRACELENS_OUT_DIR="${TRACE_DIR}/tracelens"
    bash tools/run_tracelens_report.sh || echo "TraceLens step failed (install AMD-AGI/TraceLens)"
  fi
else
  echo "No exported Chrome trace JSON found under $LOG_DIR; TensorBoard may use a different layout."
fi

# Perf breakdown from Megatron timer lines in the training log
if command -v python3 >/dev/null; then
  python3 tools/gen_perf_breakdown_charts.py "${TRACE_DIR}/train.log" \
    --output-json "${TRACE_DIR}/perf_breakdown.json" \
    --output-png "${TRACE_DIR}/perf_breakdown.png" \
    --label profile_te_llama3_8b || true
fi

echo "Profile artifacts in $TRACE_DIR"
