#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc.
"""If multiple rank trace JSONs exist, run TraceLens NcclAnalyser and write CSV.

See https://github.com/AMD-AGI/TraceLens/blob/main/docs/NcclAnalyser.md

Looks for rank*_trace.json or rank_*_trace.json under SEARCH_DIR (default: cwd).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _discover_traces(search_dir: Path) -> list[Path] | None:
    patterns = [
        re.compile(r"^rank(\d+)_trace\.json$", re.I),
        re.compile(r"^rank(\d+).*\.json$", re.I),
        re.compile(r"^trace_rank(\d+)\.json$", re.I),
    ]
    by_rank: dict[int, Path] = {}
    for p in search_dir.rglob("*.json"):
        name = p.name
        for pat in patterns:
            m = pat.match(name)
            if m:
                by_rank[int(m.group(1))] = p
                break
    if len(by_rank) < 2:
        return None
    ranks = sorted(by_rank.keys())
    expected = list(range(len(ranks)))
    if ranks != expected:
        # still try world_size = len(by_rank) with sorted paths
        paths = [by_rank[r] for r in ranks]
    else:
        paths = [by_rank[r] for r in ranks]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search-dir",
        type=Path,
        default=Path(os.environ.get("NCCL_TRACE_SEARCH_DIR", ".")),
        help="Directory to search for rank*.json traces",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/nccl_summary.csv"),
        help="Write NCCL summary CSV here",
    )
    args = ap.parse_args()

    paths = _discover_traces(args.search_dir)
    if not paths:
        print(
            "NcclAnalyser: fewer than 2 rank trace JSONs found; skipping (single-rank trace).",
            file=sys.stderr,
        )
        return 0

    try:
        from TraceLens import NcclAnalyser
    except ImportError:
        print("TraceLens not installed; skipping NcclAnalyser.", file=sys.stderr)
        return 0

    world_size = len(paths)
    str_paths = [str(p) for p in paths]
    analyser = NcclAnalyser(str_paths, world_size)
    df = analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=["mean"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"NcclAnalyser summary written to {args.output_csv} (world_size={world_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
