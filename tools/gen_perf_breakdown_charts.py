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


def _write_breakdown_markdown(series: List[Dict[str, Any]], md_path: str) -> None:
    """Human-readable table: ms and % per Megatron timer bucket per benchmark row."""
    lines = [
        "## Megatron timer bucket breakdown (serial ratio)",
        "",
        "*Not GPU kernel overlap; use TraceLens `gpu_timeline` / Excel for device-level comm and compute.*",
        "",
    ]
    for s in series:
        label = s["label"]
        p = s["percentages"]
        b = s["buckets_ms"]
        tot = p.get("total_ms", 0.0)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            "| Bucket | ms | % of step |"
        )
        lines.append("|--------|-----|-----------|")
        order = [
            ("compute", "compute_ms"),
            ("comm send/recv", "comm_send_recv_ms"),
            ("comm collective", "comm_collective_ms"),
            ("optimizer", "optimizer_ms"),
            ("data", "data_ms"),
            ("bubble / other", "bubble_other_ms"),
        ]
        pct_keys = [
            "compute_pct",
            "comm_send_recv_pct",
            "comm_collective_pct",
            "optimizer_pct",
            "data_pct",
            "bubble_other_pct",
        ]
        for (name, bk), pk in zip(order, pct_keys):
            ms = b.get(bk, 0.0)
            pc = p.get(pk, 0.0)
            lines.append(f"| {name} | {ms:.2f} | {pc:.2f}% |")
        lines.append(f"| **total (summed buckets)** | **{tot:.2f}** | **100%** |")
        lines.append("")
    Path(md_path).write_text("\n".join(lines))


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
    ap.add_argument(
        "--output-md",
        default="",
        help="Write markdown table of ms and %% per bucket (default: next to --output-json as perf_breakdown_table.md)",
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
    payload = {
        "series": series,
        "note": note,
        "interpretation": (
            "Megatron log timer categories; ratios are serial shares of summed bucket times, "
            "not GPU overlap or kernel time. For GPU/kernel/comm detail use TraceLens Excel from the profiler trace."
        ),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2))

    md_path = args.output_md or str(Path(args.output_json).with_name("perf_breakdown_table.md"))
    if any_timers:
        _write_breakdown_markdown(series, md_path)

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
    _ms_for_layer = [
        "compute_ms",
        "comm_send_recv_ms",
        "comm_collective_ms",
        "optimizer_ms",
        "data_ms",
        "bubble_other_ms",
    ]
    for i, row in enumerate(arr):
        ax.bar(x, row, bottom=bottom, label=pretty[i], color=colors[i % len(colors)])
        for xi, seg_pct in enumerate(row):
            if seg_pct < 1.5:
                continue
            ms_val = series[xi]["buckets_ms"].get(_ms_for_layer[i], 0.0)
            ypos = bottom[xi] + seg_pct / 2.0
            ax.text(
                float(xi),
                ypos,
                f"{seg_pct:.1f}%\n({ms_val:.0f}ms)",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if i < 2 else "black",
            )
        bottom += row
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=25, ha="right")
    ax.set_ylabel("Ratio (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_title(
        "Megatron timer buckets (serial ratio of summed categories; not GPU overlap)"
    )
    fig.tight_layout()
    # Second figure: non-compute buckets only (easier to read when compute dominates)
    fig2, ax2 = plt.subplots(figsize=(max(8, len(labels_short) * 1.2), 4))
    nc_keys = keys_order[1:]  # skip compute_pct
    nc_pretty = pretty[1:]
    nc_colors = colors[1:]
    nc_arr = arr[1:, :]
    bottom2 = np.zeros(len(labels_short))
    for i, row in enumerate(nc_arr):
        ax2.bar(x, row, bottom=bottom2, label=nc_pretty[i], color=nc_colors[i % len(nc_colors)])
        bottom2 += row
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, rotation=25, ha="right")
    ax2.set_ylabel("Ratio (%) (of full step)")
    ax2.set_ylim(0, min(100, float(np.max(bottom2)) * 1.25 + 1.0))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax2.set_title("Non-compute buckets only (same scale as top chart)")
    fig2.tight_layout()
    nc_png = Path(args.output_png).parent / (Path(args.output_png).stem + "_non_compute.png")
    fig2.savefig(str(nc_png), dpi=120)
    plt.close(fig2)
    fig.savefig(args.output_png, dpi=120)
    plt.close(fig)
    extra = f", {md_path}, {nc_png}" if any_timers else ""
    print(f"Wrote {args.output_json}, {args.output_png}{extra}")


if __name__ == "__main__":
    main()
