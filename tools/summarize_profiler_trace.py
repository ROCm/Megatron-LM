#!/usr/bin/env python3
"""Lightweight summary of a PyTorch Chrome trace JSON (top GPU + CPU ops by duration).

For full reports install TraceLens and use tools/run_tracelens_report.sh.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _dur_ms(ev: dict) -> float | None:
    d = ev.get("dur")
    if d is None:
        return None
    try:
        return float(d) / 1000.0  # μs -> ms
    except (TypeError, ValueError):
        return None


def _event_bucket(ev: dict) -> str | None:
    """Route Complete ('X') events into cuda | cpu | cuda_runtime_cpu | skip."""
    if ev.get("ph") != "X":
        return None
    cat = (ev.get("cat") or "").lower()
    name = (ev.get("name") or "").lower()

    # GPU kernels / device work (category varies by PyTorch / tool version)
    if (
        "cuda" in cat
        or "kernel" in cat
        or "gpu" in cat
        or "gpu_mem" in cat
        or "stream" in cat
        or "Memcpy" in (ev.get("name") or "")
    ):
        return "cuda"

    # CPU-side PyTorch / Python (profiler labels)
    if (
        "cpu_op" in cat
        or cat in ("python_function", "user_annotation", "python")
        or cat.startswith("cpu")
        or "torch" in cat
    ):
        return "cpu"

    # CUDA driver/runtime API time attributed to CPU thread (still useful for bottlenecks)
    if "cuda_runtime" in cat or "cuda api" in cat or cat == "cuda_driver":
        return "cuda_runtime_cpu"

    return None


def _aggregate(events: list) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    cuda_by_name: defaultdict[str, float] = defaultdict(float)
    cpu_by_name: defaultdict[str, float] = defaultdict(float)
    cuda_rt_by_name: defaultdict[str, float] = defaultdict(float)

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ms = _dur_ms(ev)
        if ms is None or ms <= 0:
            continue
        name = (ev.get("name") or "").strip()
        if not name:
            continue
        bucket = _event_bucket(ev)
        if bucket == "cuda":
            cuda_by_name[name] += ms
        elif bucket == "cpu":
            cpu_by_name[name] += ms
        elif bucket == "cuda_runtime_cpu":
            cuda_rt_by_name[name] += ms

    return cuda_by_name, cpu_by_name, cuda_rt_by_name


def _fmt_section(title: str, totals: dict[str, float], top_n: int) -> list[str]:
    lines = [title, ""]
    top = sorted(totals.items(), key=lambda x: -x[1])[:top_n]
    if not top:
        lines.append("  (no events matched)")
        lines.append("")
        return lines
    for name, ms in top:
        lines.append(f"  {ms:10.3f}  {name}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_json", help="Chrome trace .json from PyTorch profiler")
    ap.add_argument("-o", "--output", default="", help="Optional text summary path")
    ap.add_argument(
        "-n",
        "--top",
        type=int,
        default=40,
        help="Max lines per section (default 40)",
    )
    ap.add_argument(
        "--no-cuda-runtime-cpu",
        action="store_true",
        help="Omit CUDA runtime (CPU-thread) section",
    )
    args = ap.parse_args()
    path = Path(args.trace_json)
    data = json.loads(path.read_text())
    events = data.get("traceEvents") or data.get("events") or []

    cuda_by_name, cpu_by_name, cuda_rt_by_name = _aggregate(events)

    lines: list[str] = []
    lines.extend(
        _fmt_section(
            "Top GPU / CUDA-named regions by inclusive duration (ms):",
            cuda_by_name,
            args.top,
        )
    )
    lines.extend(
        _fmt_section(
            "Top CPU-side regions (cpu_op / Python / annotations, ms):",
            cpu_by_name,
            args.top,
        )
    )
    if not args.no_cuda_runtime_cpu:
        lines.extend(
            _fmt_section(
                "Top CUDA runtime API on CPU thread (ms, launch/sync overhead):",
                cuda_rt_by_name,
                args.top,
            )
        )

    sum_cuda = sum(cuda_by_name.values())
    sum_cpu = sum(cpu_by_name.values())
    sum_rt = sum(cuda_rt_by_name.values())
    lines.append("Bucket totals (sum of matched Complete events; overlapping GPU work is not deduplicated):")
    lines.append("")
    lines.append(f"  GPU / CUDA-named regions: {sum_cuda:,.3f} ms")
    lines.append(f"  CPU-side regions:         {sum_cpu:,.3f} ms")
    if not args.no_cuda_runtime_cpu:
        lines.append(f"  CUDA runtime on CPU:      {sum_rt:,.3f} ms")
    lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
