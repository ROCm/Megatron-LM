#!/usr/bin/env python3
"""
Convert benchmark results to markdown table.
Reads from output/benchmark_report.json or output/benchmark_results.tmp
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_results():
    rows = []
    json_path = os.environ.get("BENCHMARK_JSON", "output/benchmark_report.json")
    results_path = os.environ.get("BENCHMARK_RESULTS", "output/benchmark_results.tmp")

    if Path(json_path).exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            for b in data.get("benchmarks", []):
                rows.append({
                    "benchmark": b.get("benchmark", ""),
                    "throughput": b.get("throughput_tflop_s_per_gpu") or "N/A",
                    "elapsed": b.get("elapsed_ms_per_iter") or "N/A",
                    "tokens": b.get("tokens_per_gpu_s") or "N/A",
                    "mem": b.get("mem_gb") or "N/A",
                })
        except Exception as e:
            print(f"Warning: Could not parse JSON: {e}", file=os.sys.stderr)

    if not rows and Path(results_path).exists():
        with open(results_path) as f:
            for line in f:
                m = re.match(r"^([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)$", line.strip())
                first = m.group(1).strip()
                if m and first != "Benchmark" and not first.startswith("-"):
                    rows.append({
                        "benchmark": m.group(1).strip(),
                        "throughput": m.group(2).strip() or "N/A",
                        "elapsed": m.group(3).strip() or "N/A",
                        "tokens": m.group(4).strip() or "N/A",
                        "mem": m.group(5).strip() or "N/A",
                    })
    return rows


def to_markdown_table(rows):
    if not rows:
        return "No benchmark results available."
    header = "| Benchmark | Throughput (TFLOP/s/GPU) | Elapsed (ms/iter) | Tokens/GPU/s | Mem (GB) |"
    sep = "|-----------|--------------------------|-------------------|--------------|----------|"
    data_rows = [
        f"| {r['benchmark']} | {r['throughput']} | {r['elapsed']} | {r['tokens']} | {r['mem']} |"
        for r in rows
    ]
    return "\n".join([header, sep] + data_rows)


def main():
    try:
        rows = parse_results()
    except Exception as e:
        print(f"Warning: Could not parse benchmark results: {e}", file=os.sys.stderr)
        rows = []
    table = to_markdown_table(rows)
    report = f"""## Megatron-LM Benchmark Performance Report

{table}

*Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*"""
    out_path = os.environ.get("BENCHMARK_MARKDOWN_OUT", "output/benchmark_summary.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
