#!/bin/bash
###############################################################################
# Run Megatron-LM benchmarks and generate a performance report.
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
echo "=============================================="
echo "     Megatron-LM Benchmark Performance Report"
echo "     $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "=============================================="
echo ""

# Use TOTAL_ITERS for short runs (Llama defaults to 12)
export TOTAL_ITERS="${TOTAL_ITERS:-12}"
export TRAIN_ITERS="${TRAIN_ITERS:-12}"
export TEE_OUTPUT=1

# Initialize results file with header
echo "Benchmark|Throughput (TFLOP/s/GPU)|Elapsed (ms/iter)|Tokens/GPU/s|Mem (GB)" > "$RESULTS_FILE"
echo "---------|------------------------|-----------------|-------------|--------" >> "$RESULTS_FILE"

# Llama3 benchmarks
run_and_collect "llama3_8B_TP1_CP1_FP8" \
    "MBS=1 BS=128 SEQ_LENGTH=8192 TP=1 CP=1 MODEL_SIZE=8 TE_FP8=1 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_8B_TP8_BF16" \
    "MBS=1 BS=8 SEQ_LENGTH=8192 TP=8 TE_FP8=0 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_8B_PYTORCH_FSDP_RECOMPUTE" \
    "MBS=1 BS=8 FSDP=1 TP=1 TE_FP8=0 SEQ_LENGTH=8192 RECOMPUTE=1 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_8B_TP8" \
    "MBS=1 BS=8 TP=8 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true

run_and_collect "llama3_8B_TP4_PP2" \
    "MBS=1 BS=8 TP=4 PP=2 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true

# DeepSeek benchmarks (use correct paths: deepseek_v2, deepseek_v3)
run_and_collect "deepseek_v2" \
    "bash examples/deepseek_v2/train_deepseekv2.sh" || true

    run_and_collect "deepseek_v3" \
        "bash examples/deepseek_v3/train_deepseekv3.sh" || true

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
