#!/usr/bin/env bash
# Gate G7 -- AttnRes payload across a PP=2 boundary, gradients included.
# Reproduces develop/results/pp_payload.md. Needs 2 GPUs.
set -euo pipefail
cd "$(dirname "$0")/../../../.."          # repo root
PORT=${PORT:-29887}
OUT=${OUT:-kimi_k3/develop/results/pp_payload_raw.jsonl}

echo "== 1/3 reference (PP=1, also measures the run-to-run floor)"
torchrun --nproc_per_node=1 --master_port=$((PORT))   -m kimi_k3.tools.pp_payload_probe --mode reference

echo "== 2/3 pipelined (PP=2) -- must be MATCH"
torchrun --nproc_per_node=2 --master_port=$((PORT+1)) -m kimi_k3.tools.pp_payload_probe --mode pipeline --json "$OUT"

echo "== 3/3 negative control (slots detached) -- must be MISMATCH"
torchrun --nproc_per_node=2 --master_port=$((PORT+2)) -m kimi_k3.tools.pp_payload_probe --mode pipeline --detach-slots --json "$OUT"
