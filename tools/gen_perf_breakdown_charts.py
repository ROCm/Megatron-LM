#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc.
"""Parse Megatron training logs for timer blocks; emit stacked % bar chart + JSON.

Timer ratios are serial shares of summed categorized times (not GPU overlap).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Megatron log lines (see megatron/core/timers.py)
_RE_MAX = re.compile(
    r"^\s+([\w\-]+)\s*\.+:\s+([\d.]+)\s*$"
)
_RE_MINMAX = re.compile(
    r"^\s+([\w\-]+)\s*\.+:\s+\([\d.]+,\s*([\d.]+)\)\s*$"
)
_RE_ELAPSED = re.compile(
    r"elapsed time per iteration \(ms\):\s*([\d.]+)", re.I
)

# Bucket mapping (extend for MoE / custom timers)
COMPUTE_KEYS = {"forward-compute", "backward-compute"}
SEND_RECV_KEYS = {
    "forward-send",
    "backward-send",
    "forward-recv",
    "backward-recv",
    "forward-send-forward-recv",
    "forward-send-backward-recv",
    "backward-send-forward-recv",
    "backward-send-backward-recv",
    "forward-backward-send-forward-backward-recv",
}
COLLECTIVE_KEYS = {
    "layernorm-grads-all-reduce",
    "embedding-grads-all-reduce",
    "all-grads-sync",
    "params-all-gather",
}
OPT_KEYS = {
    "optimizer",
    "optimizer-copy-to-main-grad",
    "optimizer-unscale-and-check-inf",
    "optimizer-clip-main-grad",
    "optimizer-count-zeros",
    "optimizer-inner-step",
    "optimizer-copy-main-to-model-params",
}
DATA_KEYS = {"batch-generator"}
# Remainder of forward-backward pipeline comms
OTHER_FB_KEYS = {"forward-backward"}


def _parse_timer_blocks(text: str) -> List[Dict[str, float]]:
    """Extract timer dicts from each 'max time across ranks' section."""
    blocks: List[Dict[str, float]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "max time across ranks (ms):" in line or "(min, max) time across ranks (ms):" in line:
            i += 1
            timers: Dict[str, float] = {}
            while i < len(lines):
                ln = lines[i]
                m = _RE_MAX.match(ln) or _RE_MINMAX.match(ln)
                if m:
                    timers[m.group(1).strip()] = float(m.group(2))
                    i += 1
                    continue
                break
            if timers:
                blocks.append(timers)
            continue
        i += 1
    return blocks


def _aggregate_timers(blocks: List[Dict[str, float]]) -> Dict[str, float]:
    if not blocks:
        return {}
    keys = set()
    for b in blocks:
        keys |= set(b.keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [b[k] for b in blocks if k in b]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _bucket_ms(timers: Dict[str, float]) -> Dict[str, float]:
    buckets = {
        "compute_ms": 0.0,
        "comm_send_recv_ms": 0.0,
        "comm_collective_ms": 0.0,
        "optimizer_ms": 0.0,
        "data_ms": 0.0,
        "forward_backward_total_ms": 0.0,
        "bubble_other_ms": 0.0,
    }
    accounted = set()
    for k, v in timers.items():
        if k in COMPUTE_KEYS:
            buckets["compute_ms"] += v
            accounted.add(k)
        elif k in SEND_RECV_KEYS:
            buckets["comm_send_recv_ms"] += v
            accounted.add(k)
        elif k in COLLECTIVE_KEYS:
            buckets["comm_collective_ms"] += v
            accounted.add(k)
        elif k in OPT_KEYS:
            buckets["optimizer_ms"] += v
            accounted.add(k)
        elif k in DATA_KEYS:
            buckets["data_ms"] += v
            accounted.add(k)
        elif k in OTHER_FB_KEYS:
            buckets["forward_backward_total_ms"] += v
            accounted.add(k)
    fb = buckets["forward_backward_total_ms"]
    inner = (
        buckets["compute_ms"]
        + buckets["comm_send_recv_ms"]
        + buckets["comm_collective_ms"]
    )
    if fb > 0 and inner <= fb + 1e-6:
        buckets["bubble_other_ms"] = max(0.0, fb - inner)
    else:
        unacc = sum(v for k, v in timers.items() if k not in accounted)
        buckets["bubble_other_ms"] = max(0.0, unacc)
    return buckets


def _pct_row(buckets: Dict[str, float]) -> Dict[str, Any]:
    parts = {
        "compute": buckets.get("compute_ms", 0.0),
        "comm_send_recv": buckets.get("comm_send_recv_ms", 0.0),
        "comm_collective": buckets.get("comm_collective_ms", 0.0),
        "optimizer": buckets.get("optimizer_ms", 0.0),
        "data": buckets.get("data_ms", 0.0),
        "bubble_other": buckets.get("bubble_other_ms", 0.0),
    }
    total = sum(parts.values())
    if total <= 0:
        return {**{k: 0.0 for k in parts}, "total_ms": 0.0, "note": "no_timer_data"}
    pct = {f"{k}_pct": round(100.0 * v / total, 2) for k, v in parts.items()}
    pct["total_ms"] = round(total, 2)
    pct["accounting"] = "serial_timer_ratio"
    return pct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logs",
        nargs="+",
        help="Training log files (e.g. output/bench_*.log)",
    )
    ap.add_argument(
        "--output-json",
        default="output/perf_breakdown.json",
        help="Write aggregated JSON",
    )
    ap.add_argument(
        "--output-png",
        default="output/perf_breakdown.png",
        help="Write stacked bar chart",
    )
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label per log file (same order as logs); default stem names",
    )
    args = ap.parse_args()
    labels = args.label
    if labels and len(labels) != len(args.logs):
        raise SystemExit("--label count must match logs")
    if not labels:
        labels = [Path(p).stem for p in args.logs]

    series: List[Dict[str, Any]] = []
    for path, label in zip(args.logs, labels):
        text = Path(path).read_text(errors="replace")
        blocks = _parse_timer_blocks(text)
        agg = _aggregate_timers(blocks)
        buckets = _bucket_ms(agg)
        row = _pct_row(buckets)
        elapsed_m = None
        m = _RE_ELAPSED.search(text)
        if m:
            elapsed_m = float(m.group(1))
        series.append(
            {
                "label": label,
                "log_path": path,
                "timer_means_ms": {k: round(v, 3) for k, v in agg.items()},
                "buckets_ms": {k: round(v, 3) for k, v in buckets.items()},
                "percentages": row,
                "elapsed_time_per_iter_ms": elapsed_m,
            }
        )

    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    any_timers = any(s.get("timer_means_ms") for s in series)
    note = None if any_timers else "no_megatron_timer_blocks_found_in_logs"
    Path(args.output_json).write_text(json.dumps({"series": series, "note": note}, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; JSON written only.")
        return

    if not any_timers:
        print("No timer blocks found; JSON written, skipping PNG.")
        return

    keys_order = [
        "compute_pct",
        "comm_send_recv_pct",
        "comm_collective_pct",
        "optimizer_pct",
        "data_pct",
        "bubble_other_pct",
    ]
    labels_short = [s["label"] for s in series]
    data = []
    for s in series:
        p = s["percentages"]
        data.append([p.get(k, 0.0) for k in keys_order])
    arr = np.array(data).T if data else np.zeros((len(keys_order), 0))
    x = np.arange(len(labels_short))
    fig, ax = plt.subplots(figsize=(max(8, len(labels_short) * 1.2), 5))
    bottom = np.zeros(len(labels_short))
    colors = ["#1f4e79", "#c55a11", "#ffc000", "#92d050", "#00b0f0", "#5b9bd5"]
    pretty = [
        "compute",
        "comm send/recv",
        "comm collective",
        "optimizer",
        "data",
        "bubble / other",
    ]
    for i, row in enumerate(arr):
        ax.bar(x, row, bottom=bottom, label=pretty[i], color=colors[i % len(colors)])
        bottom += row
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=25, ha="right")
    ax.set_ylabel("Ratio (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_title("Training step time breakdown (Megatron timers, serial ratios)")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {args.output_json} and {args.output_png}")


if __name__ == "__main__":
    main()
