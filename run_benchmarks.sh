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
# BENCHMARK_PRESET (optional): fine-grained row selection; when set, overrides BENCHMARK_SUITE.
#   all | full          → full matrix
#   llama | llama_all   → all Llama rows
#   deepseek            → DeepSeek v2 + v3
#   llama_fsdp          → Llama 70B PyTorch FSDP + recompute only
#   llama_8b            → Llama 8B FP8 + 8B BF16
#   llama_70b           → Llama 70B rows (excludes 8B): TE BF16, FSDP, TP8, TP4/PP2, local
#   deepseek_v2 | deepseek_v3 → single DeepSeek run
#
# BENCHMARK_SUITE (optional, used only when BENCHMARK_PRESET is unset):
#   unset, empty, "full", or "all" → full matrix (default).
#   "llama" → Llama rows only. "deepseek" → DeepSeek v2/v3 only. "llama,deepseek" → both groups.
echo "=============================================="
echo "     Megatron-LM Benchmark Performance Report"
echo "     $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "=============================================="
echo ""

L8_FP8=0
L8_BF16=0
L70_TE=0
L70_FSDP=0
L70_TP8=0
L70_PP=0
L70_LOC=0
D2=0
D3=0

_PRESET=$(echo "${BENCHMARK_PRESET:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')

if [ -n "${_PRESET}" ]; then
  case "${_PRESET}" in
    all|full)
      L8_FP8=1 L8_BF16=1 L70_TE=1 L70_FSDP=1 L70_TP8=1 L70_PP=1 L70_LOC=1 D2=1 D3=1
      ;;
    llama|llama_all)
      L8_FP8=1 L8_BF16=1 L70_TE=1 L70_FSDP=1 L70_TP8=1 L70_PP=1 L70_LOC=1
      ;;
    deepseek)
      D2=1 D3=1
      ;;
    llama_fsdp)
      L70_FSDP=1
      ;;
    llama_8b)
      L8_FP8=1 L8_BF16=1
      ;;
    llama_70b)
      L70_TE=1 L70_FSDP=1 L70_TP8=1 L70_PP=1 L70_LOC=1
      ;;
    deepseek_v2)
      D2=1
      ;;
    deepseek_v3)
      D3=1
      ;;
    *)
      echo "Unknown BENCHMARK_PRESET '${BENCHMARK_PRESET}' (see run_benchmarks.sh header for valid values)"
      exit 1
      ;;
  esac
  echo "BENCHMARK_PRESET=${BENCHMARK_PRESET} (row flags: 8b_fp8=$L8_FP8 8b_bf16=$L8_BF16 70_te=$L70_TE 70_fsdp=$L70_FSDP 70_tp8=$L70_TP8 70_pp=$L70_PP 70_loc=$L70_LOC deepseek_v2=$D2 deepseek_v3=$D3)"
else
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
  if [ "$RUN_LLAMA" -eq 1 ]; then
    L8_FP8=1 L8_BF16=1 L70_TE=1 L70_FSDP=1 L70_TP8=1 L70_PP=1 L70_LOC=1
  fi
  if [ "$RUN_DEEPSEEK" -eq 1 ]; then
    D2=1 D3=1
  fi
  echo "BENCHMARK_SUITE=${_SUITE_RAW:-<empty>=full} RUN_LLAMA=$RUN_LLAMA RUN_DEEPSEEK=$RUN_DEEPSEEK"
fi
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
if [ "$L8_FP8" -eq 1 ]; then
run_and_collect "llama3_8B_TP1_CP1_FP8" \
    "MBS=1 BS=128 SEQ_LENGTH=8192 TP=1 CP=1 MODEL_SIZE=8 TE_FP8=1 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L8_BF16" -eq 1 ]; then
run_and_collect "llama3_8B_TP1_CP1_BF16" \
    "MBS=1 BS=128 SEQ_LENGTH=8192 TP=1 CP=1 MODEL_SIZE=8 TE_FP8=0 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L70_TE" -eq 1 ]; then
run_and_collect "llama3_70B_TP8_TE_BF16" \
    "MBS=1 BS=8 SEQ_LENGTH=8192 TP=8 TE_FP8=0 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L70_FSDP" -eq 1 ]; then
run_and_collect "llama3_70B_PYTORCH_FSDP_RECOMPUTE" \
    "MBS=1 BS=8 FSDP=1 TP=1 TE_FP8=0 SEQ_LENGTH=8192 RECOMPUTE=1 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L70_TP8" -eq 1 ]; then
run_and_collect "llama3_70B_TP8" \
    "MBS=1 BS=8 TP=8 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L70_PP" -eq 1 ]; then
run_and_collect "llama3_70B_TP4_PP2" \
    "MBS=1 BS=8 TP=4 PP=2 TE_FP8=0 SEQ_LENGTH=8192 bash examples/llama/train_llama3.sh" || true
fi
if [ "$L70_LOC" -eq 1 ]; then
# Optional non-TE baseline (local transformer impl) for A/B vs Transformer Engine
run_and_collect "llama3_70B_TP8_local" \
    "USE_TE=0 TE_FP8=0 MBS=1 BS=8 SEQ_LENGTH=8192 TP=8 bash examples/llama/train_llama3.sh" || true
fi

# DeepSeek benchmarks (use correct paths: deepseek_v2, deepseek_v3)
if [ "$D2" -eq 1 ]; then
run_and_collect "deepseek_v2" \
    "bash examples/deepseek_v2/train_deepseekv2.sh" || true
fi
if [ "$D3" -eq 1 ]; then
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

# When a PyTorch Chrome trace exists under output/, generate TraceLens Excel + text summary (+ NCCL CSV if multi-rank traces).
TRACE_JSON=""
while IFS= read -r f; do
  TRACE_JSON="$f"
  break
done < <(find output -type f -path '*/tb/*.json' 2>/dev/null || true)
if [ -z "$TRACE_JSON" ]; then
  while IFS= read -r f; do
    case "$f" in
      *benchmark_report.json) continue ;;
    esac
    if head -c 800 "$f" 2>/dev/null | grep -q '"traceEvents"'; then
      TRACE_JSON="$f"
      break
    fi
  done < <(find output -type f -name '*.json' 2>/dev/null || true)
fi
if [ -n "$TRACE_JSON" ]; then
  echo "Chrome profiler trace found: $TRACE_JSON — running TraceLens + summarize_profiler_trace (+ optional NcclAnalyser)"
  export TRACE_JSON
  export TRACELENS_OUT_DIR="${TRACELENS_OUT_DIR:-output/tracelens_bench}"
  bash tools/run_tracelens_report.sh || echo "TraceLens step failed (see ${TRACELENS_OUT_DIR:-output/tracelens_bench}/tracelens.log)"
  python3 tools/summarize_profiler_trace.py "$TRACE_JSON" -o output/profiler_trace_summary.txt || true
  python3 tools/run_nccl_analyser_if_multirank.py --search-dir output --output-csv output/nccl_summary.csv || true
else
  echo "No Chrome trace JSON under output/ (enable profiling / tensorboard trace export to get TraceLens artifacts)."
fi
