#!/bin/bash
# Generate TraceLens Excel from a PyTorch profiler Chrome trace JSON.
# Docs: https://github.com/AMD-AGI/TraceLens/blob/main/docs/generate_perf_report.md
#
# Required env:
#   TRACE_JSON   Absolute path to profiler JSON (Chrome trace format).
#
# Optional env:
#   TRACELENS_OUT_DIR       Output directory (default: output/tracelens)
#   TRACELENS_GPU_ARCH_JSON Path to gpu_arch JSON for roofline / Pct Roofline
#                            (default: tools/tracelens/gpu_arch_mi325.json next to this script)
#   TRACELENS_TOPK_OPS      Default 80
#   TRACELENS_TOPK_ROOFLINE Default 30
#   TRACELENS_TOPK_SHORT    Default 50
#   TRACELENS_EXTRA_ARGS    Extra CLI args (quoted string)

set -euo pipefail
TRACE_JSON="${TRACE_JSON:?Set TRACE_JSON to profiler json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${TRACELENS_OUT_DIR:-${REPO_ROOT}/output/tracelens}"
mkdir -p "$OUT"

DEFAULT_ARCH="${SCRIPT_DIR}/tracelens/gpu_arch_mi325.json"
GPU_ARCH="${TRACELENS_GPU_ARCH_JSON:-$DEFAULT_ARCH}"
TOPK_OPS="${TRACELENS_TOPK_OPS:-80}"
TOPK_RF="${TRACELENS_TOPK_ROOFLINE:-30}"
TOPK_SK="${TRACELENS_TOPK_SHORT:-50}"

if ! command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
  echo "TraceLens CLI not found. Install with:"
  echo "  pip install 'git+https://github.com/AMD-AGI/TraceLens.git'"
  exit 1
fi

BASE_OUT="$(basename "$TRACE_JSON" .json)_perf_report.xlsx"
OUT_XLSX="${OUT}/${BASE_OUT}"

ARGS=(
  TraceLens_generate_perf_report_pytorch
  --profile_json_path "$TRACE_JSON"
  --output_xlsx_path "$OUT_XLSX"
  --short_kernel_study
  --topk_ops "$TOPK_OPS"
  --topk_roofline_ops "$TOPK_RF"
  --topk_short_kernels "$TOPK_SK"
)

if [ -f "$GPU_ARCH" ]; then
  ARGS+=(--gpu_arch_json_path "$GPU_ARCH")
else
  echo "Warning: GPU arch JSON not found at $GPU_ARCH — roofline ceilings omitted."
fi

if [ -n "${TRACELENS_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA=( $TRACELENS_EXTRA_ARGS )
  ARGS+=("${EXTRA[@]}")
fi

set +e
( cd "$OUT" && "${ARGS[@]}" ) 2>&1 | tee "$OUT/tracelens.log"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "TraceLens_generate_perf_report_pytorch exited with $RC (see $OUT/tracelens.log)"
  exit "$RC"
fi
echo "TraceLens Excel: $OUT_XLSX"
