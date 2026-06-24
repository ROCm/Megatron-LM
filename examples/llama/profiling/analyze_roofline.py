#!/usr/bin/env python3
"""Roofline profiling analysis for Megatron-LM training runs.

Parses rocprof output to identify kernel optimization candidates based on
how far they are from the roofline (peak memory bandwidth or peak compute).

Usage:
    python analyze_roofline.py --profiling-dir <dir> [--gpu-arch <gfxNNN>]

Inputs:
    - rocprof stats CSV  (*_stats.csv)  — per-kernel aggregated timing
    - rocprof output CSV (rocprof_output.csv) — per-invocation data with
      FETCH_SIZE / WRITE_SIZE counters (optional, enables bandwidth analysis)

Outputs:
    - roofline_report.txt — actionable kernel ranking with roofline metrics
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# GPU specifications for roofline analysis
# ---------------------------------------------------------------------------
GPU_SPECS = {
    "gfx942": {
        "name": "AMD Instinct MI300X",
        "peak_hbm_bw_GBs": 5300,
        "peak_bf16_tflops": 1307,
        "peak_fp16_tflops": 1307,
        "peak_fp32_tflops": 653,
    },
    "gfx90a": {
        "name": "AMD Instinct MI250X (per GCD)",
        "peak_hbm_bw_GBs": 1638,
        "peak_bf16_tflops": 383,
        "peak_fp16_tflops": 383,
        "peak_fp32_tflops": 191,
    },
    "gfx950": {
        "name": "AMD Instinct MI350X",
        "peak_hbm_bw_GBs": 8000,
        "peak_bf16_tflops": 2300,
        "peak_fp16_tflops": 2300,
        "peak_fp32_tflops": 1150,
    },
}

# ---------------------------------------------------------------------------
# Kernel classification patterns (checked case-insensitively)
# ---------------------------------------------------------------------------
KERNEL_CATEGORIES = {
    "GEMM": [
        r"Cijk_",
        r"hipblaslt",
        r"rocblas",
        r"gemm",
        r"mfma",
        r"blas",
    ],
    "Attention": [
        r"flash_attn",
        r"fmha",
        r"attention",
        r"softmax",
        r"ck_tiled_fmha",
    ],
    "Communication": [
        r"nccl",
        r"rccl",
        r"allreduce",
        r"allgather",
        r"reducescatter",
        r"reduce_scatter",
        r"broadcast",
    ],
    "Normalization": [
        r"layernorm",
        r"rmsnorm",
        r"layer_norm",
        r"rms_norm",
    ],
    "Elementwise": [
        r"vectorized",
        r"elementwise",
        r"cast",
        r"silu",
        r"gelu",
        r"swiglu",
        r"dropout",
        r"fill",
    ],
    "Optimizer": [
        r"adam",
        r"multi_tensor",
        r"fused.*adam",
    ],
    "Transpose": [
        r"transpose",
        r"permute",
    ],
    "Memory": [
        r"memset",
        r"memcpy",
        r"copy_kernel",
    ],
}


@dataclass
class KernelStats:
    name: str
    calls: int
    total_duration_ns: int
    avg_duration_ns: float
    percentage: float
    category: str = "Other"
    # Counter-derived metrics (populated when FETCH_SIZE/WRITE_SIZE available)
    total_fetch_bytes: int = 0
    total_write_bytes: int = 0
    avg_fetch_bytes: float = 0.0
    avg_write_bytes: float = 0.0
    # Roofline metrics
    achieved_bw_GBs: float = 0.0
    bw_utilization_pct: float = 0.0
    roofline_bound: str = ""
    opportunity_score: float = 0.0


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def detect_gpu_arch() -> Optional[str]:
    """Detect GPU architecture from rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:") and "gfx" in stripped:
                match = re.search(r"(gfx\w+)", stripped)
                if match:
                    return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------
def classify_kernel(name: str) -> str:
    name_lower = name.lower()
    for category, patterns in KERNEL_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return category
    return "Other"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_stats_csv(filepath: str) -> List[KernelStats]:
    """Parse rocprof *_stats.csv (aggregated per-kernel timing)."""
    kernels = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("name", "")).strip('"')
            if not name:
                continue
            ks = KernelStats(
                name=name,
                calls=int(row.get("Calls", row.get("calls", 0))),
                total_duration_ns=int(
                    row.get("TotalDurationNs", row.get("total_duration_ns", 0))
                ),
                avg_duration_ns=float(
                    row.get("AverageNs", row.get("average_ns", 0))
                ),
                percentage=float(
                    row.get("Percentage", row.get("percentage", 0))
                ),
            )
            ks.category = classify_kernel(ks.name)
            kernels.append(ks)
    return kernels


def _find_column(fields: List[str], *candidates: str) -> Optional[str]:
    """Find a CSV column matching any candidate substring (case-insensitive)."""
    for f in fields:
        fl = f.strip('"').lower()
        for c in candidates:
            if c in fl:
                return f
    return None


def parse_counter_csv(filepath: str) -> Dict[str, Dict]:
    """Parse the per-invocation rocprof CSV and aggregate counters per kernel."""
    aggregated: Dict[str, Dict] = defaultdict(
        lambda: {"fetch_bytes": 0, "write_bytes": 0, "count": 0, "total_duration_ns": 0}
    )
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        name_col = _find_column(fields, "name")
        fetch_col = _find_column(fields, "fetch_size", "fetch")
        write_col = _find_column(fields, "write_size")
        begin_col = _find_column(fields, "beginns", "begin_ns")
        end_col = _find_column(fields, "endns", "end_ns")

        if not name_col:
            name_col = fields[0] if fields else None

        if not fetch_col and not write_col:
            return {}  # no counter columns found

        for row in reader:
            name = row.get(name_col, "").strip('"')
            if not name:
                continue
            entry = aggregated[name]
            entry["count"] += 1
            if fetch_col and row.get(fetch_col):
                try:
                    entry["fetch_bytes"] += int(float(row[fetch_col]))
                except (ValueError, TypeError):
                    pass
            if write_col and row.get(write_col):
                try:
                    entry["write_bytes"] += int(float(row[write_col]))
                except (ValueError, TypeError):
                    pass
            if begin_col and end_col:
                try:
                    duration = int(row[end_col]) - int(row[begin_col])
                    entry["total_duration_ns"] += max(0, duration)
                except (ValueError, TypeError):
                    pass
    return dict(aggregated)


# ---------------------------------------------------------------------------
# Roofline computation
# ---------------------------------------------------------------------------
def compute_roofline_metrics(
    kernels: List[KernelStats],
    counter_data: Optional[Dict[str, Dict]],
    gpu_spec: Dict,
) -> List[KernelStats]:
    peak_bw = gpu_spec["peak_hbm_bw_GBs"]  # GB/s

    for ks in kernels:
        if counter_data and ks.name in counter_data:
            cd = counter_data[ks.name]
            ks.total_fetch_bytes = cd["fetch_bytes"]
            ks.total_write_bytes = cd["write_bytes"]
            total_bytes = ks.total_fetch_bytes + ks.total_write_bytes

            if ks.calls > 0:
                ks.avg_fetch_bytes = ks.total_fetch_bytes / ks.calls
                ks.avg_write_bytes = ks.total_write_bytes / ks.calls

            duration_ns = cd.get("total_duration_ns", 0) or ks.total_duration_ns
            if duration_ns > 0 and total_bytes > 0:
                duration_s = duration_ns / 1e9
                ks.achieved_bw_GBs = (total_bytes / 1e9) / duration_s
                ks.bw_utilization_pct = min(
                    (ks.achieved_bw_GBs / peak_bw) * 100, 100.0
                )

        # Classify bound type
        if ks.category == "GEMM":
            ks.roofline_bound = "Compute"
        elif ks.category in ("Elementwise", "Normalization", "Transpose", "Memory"):
            ks.roofline_bound = "Memory"
        elif ks.category == "Communication":
            ks.roofline_bound = "Network"
        elif ks.category == "Attention":
            ks.roofline_bound = "Mixed"
        else:
            ks.roofline_bound = "Unknown"

        # Opportunity score = gap-to-peak × time-weight
        if ks.achieved_bw_GBs > 0:
            gap = max(0.0, 100.0 - ks.bw_utilization_pct)
            ks.opportunity_score = gap * ks.percentage / 100.0
        else:
            # Without counter data, use time percentage as proxy
            ks.opportunity_score = ks.percentage

    return kernels


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b/1e9:.1f} GB"
    if b >= 1e6:
        return f"{b/1e6:.1f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


def _fmt_time(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} us"
    return f"{ns:.0f} ns"


def _trunc(name: str, n: int = 60) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def _suggestion(k: KernelStats) -> str:
    if k.category == "GEMM":
        if k.bw_utilization_pct > 0 and k.bw_utilization_pct < 50:
            return "Tune GEMM / check dims"
        return "Check GEMM config"
    if k.category == "Communication":
        return "Overlap with compute"
    if k.category == "Attention":
        return "Check FA / fused impl"
    if k.category in ("Elementwise", "Normalization"):
        return "Fuse with neighbors"
    if k.category == "Transpose":
        return "Reduce / fuse transposes"
    if k.category == "Optimizer":
        return "Check fused optimizer"
    if k.category == "Memory":
        return "Reduce copies"
    return "Investigate"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    kernels: List[KernelStats],
    gpu_spec: Dict,
    gpu_arch: str,
    has_counters: bool,
    output_file: Optional[str] = None,
) -> str:
    W = 100
    lines: List[str] = []

    def p(s=""):
        lines.append(s)

    peak_bw = gpu_spec["peak_hbm_bw_GBs"]
    peak_compute = gpu_spec["peak_bf16_tflops"]
    ridge_point = (peak_compute * 1000) / peak_bw  # FLOP/Byte

    p("=" * W)
    p("ROOFLINE PROFILING REPORT")
    p("=" * W)
    p(f"GPU:                    {gpu_spec['name']} ({gpu_arch})")
    p(f"Peak HBM Bandwidth:     {peak_bw} GB/s")
    p(f"Peak BF16 Compute:      {peak_compute} TFLOP/s")
    p(f"Roofline Ridge Point:   {ridge_point:.1f} FLOP/Byte")
    p(f"Hardware Counters:      {'Yes (FETCH_SIZE, WRITE_SIZE)' if has_counters else 'No — timing only'}")
    p()

    kernels_by_time = sorted(kernels, key=lambda k: k.percentage, reverse=True)

    # ---- category breakdown ------------------------------------------------
    p("-" * W)
    p("CATEGORY BREAKDOWN")
    p("-" * W)
    cat_stats: Dict[str, Dict] = defaultdict(
        lambda: {"time_pct": 0.0, "calls": 0, "count": 0}
    )
    for ks in kernels:
        c = cat_stats[ks.category]
        c["time_pct"] += ks.percentage
        c["calls"] += ks.calls
        c["count"] += 1

    p(f"  {'Category':<20} {'Unique Kernels':>15} {'Total Calls':>12} {'Time %':>10}")
    p(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*10}")
    for cat, st in sorted(
        cat_stats.items(), key=lambda x: x[1]["time_pct"], reverse=True
    ):
        p(f"  {cat:<20} {st['count']:>15} {st['calls']:>12} {st['time_pct']:>9.1f}%")
    p()

    # ---- top kernels by time -----------------------------------------------
    top_n = min(30, len(kernels_by_time))
    p("-" * W)
    p(f"TOP {top_n} KERNELS BY TOTAL GPU TIME")
    p("-" * W)
    p(
        f"  {'#':>3}  {'Category':<15} {'Calls':>8} {'Total':>12} {'Avg':>12} "
        f"{'Time%':>7}  Kernel"
    )
    p(
        f"  {'---':>3}  {'-'*15} {'-'*8} {'-'*12} {'-'*12} "
        f"{'-'*7}  {'-'*30}"
    )
    for i, ks in enumerate(kernels_by_time[:top_n]):
        p(
            f"  {i+1:>3}  {ks.category:<15} {ks.calls:>8} "
            f"{_fmt_time(ks.total_duration_ns):>12} "
            f"{_fmt_time(ks.avg_duration_ns):>12} "
            f"{ks.percentage:>6.1f}%  {_trunc(ks.name, 60)}"
        )
    p()

    # ---- bandwidth analysis (counter-based) --------------------------------
    if has_counters:
        kernels_with_bw = [k for k in kernels_by_time if k.achieved_bw_GBs > 0]
        if kernels_with_bw:
            p("-" * W)
            p("MEMORY BANDWIDTH UTILIZATION (from HW counters)")
            p("-" * W)
            p(
                f"  {'#':>3}  {'Category':<15} {'Ach.BW GB/s':>12} {'Peak%':>7} "
                f"{'Data/call':>12} {'Bound':<12}  Kernel"
            )
            p(
                f"  {'---':>3}  {'-'*15} {'-'*12} {'-'*7} "
                f"{'-'*12} {'-'*12}  {'-'*30}"
            )
            for i, ks in enumerate(kernels_with_bw[:25]):
                avg_data = ks.avg_fetch_bytes + ks.avg_write_bytes
                p(
                    f"  {i+1:>3}  {ks.category:<15} "
                    f"{ks.achieved_bw_GBs:>10.0f}   "
                    f"{ks.bw_utilization_pct:>5.1f}% "
                    f"{_fmt_bytes(avg_data):>12} "
                    f"{ks.roofline_bound:<12}  "
                    f"{_trunc(ks.name, 50)}"
                )
            p()

    # ---- optimization candidates -------------------------------------------
    p("-" * W)
    p("OPTIMIZATION CANDIDATES (ranked by opportunity score)")
    p("-" * W)

    if has_counters:
        candidates = sorted(
            kernels, key=lambda k: k.opportunity_score, reverse=True
        )
        candidates = [c for c in candidates if c.opportunity_score > 0.5][:20]
        if candidates:
            p(
                f"  {'Pri':>3}  {'Category':<15} {'Time%':>7} {'BW%':>7} "
                f"{'Score':>7}  {'Suggestion':<24}  Kernel"
            )
            p(
                f"  {'---':>3}  {'-'*15} {'-'*7} {'-'*7} "
                f"{'-'*7}  {'-'*24}  {'-'*30}"
            )
            for i, c in enumerate(candidates):
                p(
                    f"  {i+1:>3}  {c.category:<15} "
                    f"{c.percentage:>6.1f}% "
                    f"{c.bw_utilization_pct:>5.1f}% "
                    f"{c.opportunity_score:>6.1f}  "
                    f"{_suggestion(c):<24}  "
                    f"{_trunc(c.name, 45)}"
                )
        else:
            p("  No significant optimization candidates identified.")
    else:
        p(
            "  (Without HW counters, candidates are ranked by time share only."
        )
        p(
            "   Re-run with FETCH_SIZE/WRITE_SIZE counters for bandwidth data.)"
        )
        p()
        candidates = [k for k in kernels_by_time if k.percentage > 1.0][:20]
        p(
            f"  {'Pri':>3}  {'Category':<15} {'Time%':>7} {'Bound':<15}  "
            f"{'Suggestion':<24}  Kernel"
        )
        p(
            f"  {'---':>3}  {'-'*15} {'-'*7} {'-'*15}  "
            f"{'-'*24}  {'-'*30}"
        )
        for i, c in enumerate(candidates):
            p(
                f"  {i+1:>3}  {c.category:<15} "
                f"{c.percentage:>6.1f}% "
                f"{c.roofline_bound:<15}  "
                f"{_suggestion(c):<24}  "
                f"{_trunc(c.name, 45)}"
            )

    p()
    p("=" * W)
    p("END OF REPORT")
    p("=" * W)

    report = "\n".join(lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\nReport written to: {output_file}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze rocprof output for roofline profiling of Megatron-LM runs."
    )
    parser.add_argument(
        "--profiling-dir",
        required=True,
        help="Directory containing rocprof output files.",
    )
    parser.add_argument(
        "--gpu-arch",
        default=None,
        help="GPU architecture override (e.g. gfx942). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Explicit path to the rocprof stats CSV.",
    )
    parser.add_argument(
        "--counter-file",
        default=None,
        help="Explicit path to the rocprof per-invocation CSV with counters.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path. Default: <profiling-dir>/roofline_report.txt",
    )
    args = parser.parse_args()

    profiling_dir = Path(args.profiling_dir)
    if not profiling_dir.exists():
        print(f"Error: profiling directory not found: {profiling_dir}", file=sys.stderr)
        sys.exit(1)

    # --- GPU detection ---
    gpu_arch = args.gpu_arch or detect_gpu_arch()
    if not gpu_arch:
        print(
            "Warning: could not detect GPU arch; use --gpu-arch. Defaulting to gfx942.",
            file=sys.stderr,
        )
        gpu_arch = "gfx942"

    gpu_arch_base = gpu_arch.split(":")[0]
    if gpu_arch_base not in GPU_SPECS:
        print(
            f"Warning: unknown arch '{gpu_arch_base}', using gfx942 specs.",
            file=sys.stderr,
        )
        gpu_arch_base = "gfx942"
    gpu_spec = GPU_SPECS[gpu_arch_base]

    # --- locate stats file ---
    stats_file = args.stats_file
    if not stats_file:
        candidates = sorted(profiling_dir.glob("*_stats.csv"))
        if candidates:
            stats_file = str(candidates[0])
    if not stats_file or not os.path.exists(stats_file):
        print(f"Error: no stats CSV found in {profiling_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Stats file : {stats_file}")
    kernels = parse_stats_csv(stats_file)
    if not kernels:
        print("Error: no kernels found in stats file.", file=sys.stderr)
        sys.exit(1)

    # --- locate counter file (optional) ---
    counter_data = None
    has_counters = False
    counter_file = args.counter_file

    if not counter_file:
        # Look for non-stats CSV that contains counter columns
        for candidate in sorted(profiling_dir.glob("*.csv")):
            if "stats" in candidate.name.lower():
                continue
            with open(candidate, "r") as f:
                header = f.readline().lower()
            if "fetch" in header or "write_size" in header:
                counter_file = str(candidate)
                break

    if counter_file and os.path.exists(counter_file):
        print(f"Counter file: {counter_file}")
        counter_data = parse_counter_csv(counter_file)
        has_counters = bool(counter_data)
    else:
        print("No counter data found — report will be timing-only.")

    # --- compute & report ---
    kernels = compute_roofline_metrics(kernels, counter_data, gpu_spec)
    output_file = args.output or str(profiling_dir / "roofline_report.txt")
    report = generate_report(kernels, gpu_spec, gpu_arch_base, has_counters, output_file)
    print()
    print(report)


if __name__ == "__main__":
    main()
