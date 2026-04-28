#!/bin/bash
# Generate TraceLens Excel report from a PyTorch profiler Chrome trace JSON.
# Requires: pip install git+https://github.com/AMD-AGI/TraceLens.git
# Env: TRACE_JSON (path), TRACELENS_OUT_DIR (optional output directory)

set -euo pipefail
TRACE_JSON="${TRACE_JSON:?Set TRACE_JSON to profiler json}"
OUT="${TRACELENS_OUT_DIR:-output/tracelens}"
mkdir -p "$OUT"

if ! command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
  echo "TraceLens CLI not found. Install with:"
  echo "  pip install 'git+https://github.com/AMD-AGI/TraceLens.git'"
  exit 1
fi

( cd "$OUT" && TraceLens_generate_perf_report_pytorch --profile_json_path "$TRACE_JSON" ) 2>&1 | tee "$OUT/tracelens.log" || true
echo "TraceLens report step completed. See $OUT"
