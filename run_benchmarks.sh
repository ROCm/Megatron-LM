#!/bin/bash
###############################################################################
# Run Megatron-LM benchmarks and generate a performance report.
# Single entry for CI/local: Llama uses Transformer Engine by default
# (examples/llama/train_llama3.sh, USE_TE=1); use USE_TE=0 on a row for local baseline.
###############################################################################
set -e

BENCHMARK_REPORT="${BENCHMARK_REPORT:-output/benchmark_report.txt}"
BENCHMARK_JSON="${BENCHMARK_JSON:-output/benchmark_report.json}"
RESULTS_FILE="${RESULTS_FILE:-output/benchmark_results.tmp}"
mkdir -p output

# Run benchmark and append metrics to RESULTS_FILE (caller sets RESULTS_FILE)
run_and_collect() {
    local name="$1"
    shift
    local log_file="output/bench_${name}.log"
    echo "=========================================="
    echo "Running benchmark: $name"
    echo "Command: $*"
    echo "=========================================="

    if eval "$@" 2>&1 | tee "$log_file"; then
        echo "[PASS] $name"
    else
        echo "[FAIL] $name (exit code: $?)"
    fi

    # Parse metrics from log file
    local throughput elapsed_ms tokens_per_gpu_s mem_gb
    throughput=$(grep -E "throughput per GPU:" "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*: ([0-9\.]+).*/\1/' || echo "N/A")
    elapsed_ms=$(grep -E "elapsed time per iteration:" "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*: ([0-9\.]+).*/\1/' || echo "N/A")
    tokens_per_gpu_s=$(grep -E "tokens/GPU/s:" "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*: ([0-9\.]+).*/\1/' || echo "N/A")
    mem_gb=$(grep -E "mem usages:" "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*: ([0-9\.]+).*/\1/' || echo "N/A")

    echo "$name|$throughput|$elapsed_ms|$tokens_per_gpu_s|$mem_gb" >> "$RESULTS_FILE"
    echo ""
}

# Main
# BENCHMARK_SUITE (optional, CI/local):
#   unset, empty, "full", or "all" → run the full matrix below (default).
#   "llama" → Llama train_llama3 rows only.
#   "deepseek" → DeepSeek v2/v3 rows only.
#   "llama,deepseek" → same as full for current matrix (both groups).
echo "=============================================="
echo "     Megatron-LM Benchmark Performance Report"
echo "     $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "=============================================="
echo ""
_SUITE_RAW="${BENCHMARK_SUITE:-}"
_SUITE=$(echo "${_SUITE_RAW}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
RUN_LLAMA=1
RUN_DEEPSEEK=1
if [ -n "${_SUITE}" ] && [ "${_SUITE}" != "full" ] && [ "${_SUITE}" != "all" ]; then
  RUN_LLAMA=0
  RUN_DEEPSEEK=0
  _LINE=$(echo "${BENCHMARK_SUITE}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
  IFS=',' read -r -a _PARTS <<< "$_LINE"
  for _p in "${_PARTS[@]}"; do
    case "$_p" in
      llama) RUN_LLAMA=1 ;;
      deepseek) RUN_DEEPSEEK=1 ;;
      *)
        echo "Unknown BENCHMARK_SUITE segment '${_p}' (use full, llama, deepseek, or llama,deepseek)"
        exit 1
        ;;
    esac
  done
fi
echo "BENCHMARK_SUITE=${_SUITE_RAW:-<empty>=full} RUN_LLAMA=$RUN_LLAMA RUN_DEEPSEEK=$RUN_DEEPSEEK"
echo ""

# Use TOTAL_ITERS for short runs (Llama defaults to 12)
export TOTAL_ITERS="${TOTAL_ITERS:-12}"
export TRAIN_ITERS="${TRAIN_ITERS:-12}"
export TEE_OUTPUT=1
# Richer timer lines for tools/gen_perf_breakdown_charts.py (optional)
if [ "${GENERATE_BENCHMARK_CHARTS:-0}" = "1" ]; then
  export LOG_DETAILED_TIMERS=1
fi

# Initialize results file with header
echo "Benchmark|Throughput (TFLOP/s/GPU)|Elapsed (ms/iter)|Tokens/GPU/s|Mem (GB)" > "$RESULTS_FILE"
echo "---------|------------------------|-----------------|-------------|--------" >> "$RESULTS_FILE"

# Llama3 benchmarks (train_llama3.sh defaults to USE_TE=1 / --transformer-impl=transformer_engine for BF16)
if [ "$RUN_LLAMA" -eq 1 ]; then
run_and_collect "llama3_8B_TP1_CP1_FP8" \
    "MBS=1 BS=128 SEQ_LENGTH=8192 TP=1 CP=1 MODEL_SIZE=8 TE_FP8=1 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_8B_TP1_CP1_BF16" \
    "MBS=1 BS=128 SEQ_LENGTH=8192 TP=1 CP=1 MODEL_SIZE=8 TE_FP8=0 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_70B_TP8_TE_BF16" \
    "MBS=1 BS=8 SEQ_LENGTH=8192 TP=8 TE_FP8=0 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_70B_PYTORCH_FSDP_RECOMPUTE" \
    "MBS=1 BS=8 FSDP=1 TP=1 TE_FP8=0 SEQ_LENGTH=8192 RECOMPUTE=1 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_70B_TP8" \
    "MBS=1 BS=8 TP=8 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_70B_TP4_PP2" \
    "MBS=1 BS=8 TP=4 PP=2 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true

# Optional non-TE baseline (local transformer impl) for A/B vs Transformer Engine
run_and_collect "llama3_70B_TP8_local" \
    "USE_TE=0 TE_FP8=0 MBS=1 BS=8 SEQ_LENGTH=8192 TP=8 bash examples/llama/train_llama3.sh" || true
fi

# DeepSeek benchmarks (use correct paths: deepseek_v2, deepseek_v3)
if [ "$RUN_DEEPSEEK" -eq 1 ]; then
run_and_collect "deepseek_v2" \
    "bash examples/deepseek_v2/train_deepseekv2.sh" || true

run_and_collect "deepseek_v3" \
    "bash examples/deepseek_v3/train_deepseekv3.sh" || true
fi

# Print results table
echo "=============================================="
echo "              Performance Summary"
echo "=============================================="
cat "$RESULTS_FILE"
echo ""
echo "Benchmark logs saved to output/bench_*.log"
echo "=============================================="

# Generate JSON report
python3 - <<PYEOF 2>/dev/null || true
import json
import re
import os

results_path = os.environ.get('RESULTS_FILE', 'output/benchmark_results.tmp')
json_path = os.environ.get('BENCHMARK_JSON', 'output/benchmark_report.json')

results = []
try:
    with open(results_path) as f:
        for line in f:
            m = re.match(r'^([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)$', line.strip())
            if m and m.group(1) not in ('Benchmark', '--------'):
                results.append({
                    'benchmark': m.group(1).strip(),
                    'throughput_tflop_s_per_gpu': m.group(2).strip() if m.group(2) != 'N/A' else None,
                    'elapsed_ms_per_iter': m.group(3).strip() if m.group(3) != 'N/A' else None,
                    'tokens_per_gpu_s': m.group(4).strip() if m.group(4) != 'N/A' else None,
                    'mem_gb': m.group(5).strip() if m.group(5) != 'N/A' else None,
                })
    with open(json_path, 'w') as f:
        json.dump({'benchmarks': results}, f, indent=2)
    print(f"JSON report saved to {json_path}")
except Exception as e:
    print(f"Could not generate JSON: {e}")
PYEOF

# Optional: theoretical memory + perf breakdown charts for CI artifacts
if [ "${GENERATE_BENCHMARK_CHARTS:-0}" = "1" ]; then
  echo "Generating memory footprint and perf breakdown charts..."
  python3 tools/gen_memory_footprint_charts.py --output-json output/memory_footprint.json --output-png output/memory_footprint.png || true
  shopt -s nullglob
  mapfile -t BENCH_LOGS < <(ls -1 output/bench_*.log 2>/dev/null || true)
  if [ "${#BENCH_LOGS[@]}" -gt 0 ]; then
    python3 tools/gen_perf_breakdown_charts.py "${BENCH_LOGS[@]}" \
      --output-json output/perf_breakdown.json --output-png output/perf_breakdown.png || true
  fi
  shopt -u nullglob
fi
