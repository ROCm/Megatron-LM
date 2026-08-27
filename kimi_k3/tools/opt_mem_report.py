"""Render the G5 measurement table from opt_mem_probe output.

    python -m kimi_k3.tools.opt_mem_report \
        kimi_k3/develop/results/opt_mem_raw.jsonl > kimi_k3/develop/results/opt_mem.md
"""

import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load(sys.argv[1])
    groups = defaultdict(list)
    for r in rows:
        key = (r["optimizer"] + ("+precaware" if r.get("precision_aware") else ""), r["world"])
        groups[key].append(r)

    print("# G5 — measured optimizer memory\n")
    print(
        "> Produced by `kimi_k3/tools/opt_mem_probe.py`; regenerate with\n"
        "> `kimi_k3/tools/opt_mem_report.py`. Bytes are per parameter **resident on\n"
        "> one rank**, measured as the CUDA allocator delta from before model\n"
        "> construction to after two optimizer steps, so masters, moments, gradient\n"
        "> buffers and any all-gather scratch are all included.\n"
    )
    first = rows[0]
    print(
        f"Probe model: {first['params'] / 1e6:.1f} M parameters "
        f"(MLA + MoE + norms, bf16 weights, fp32 grad reduce), TP=1 PP=1 EP=1, "
        f"world size = DP.\n"
    )

    print("| recipe | DP | B/param (mean) | spread across ranks | param MB | grad MB | opt state MB | peak MB |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for (name, world) in sorted(groups, key=lambda k: (k[0], k[1])):
        rs = groups[(name, world)]
        bpp = [r["bytes_per_param"] for r in rs]
        mean = sum(bpp) / len(bpp)
        spread = f"{min(bpp):.2f} – {max(bpp):.2f}" if len(bpp) > 1 else "—"
        print(
            f"| `{name}` | {world} | **{mean:.2f}** | {spread} "
            f"| {sum(r['param_mb'] for r in rs) / len(rs):.0f} "
            f"| {sum(r['grad_mb'] for r in rs) / len(rs):.0f} "
            f"| {sum(r['opt_total_mb'] for r in rs) / len(rs):.0f} "
            f"| {sum(r['peak_mb'] for r in rs) / len(rs):.0f} |"
        )

    print("\n## Optimizer state split (mean per rank, MB)\n")
    print(
        "> The 2-D column is empty for the distributed optimizer by construction: it\n"
        "> keeps state in flattened shard buffers, so the parameters its `state` dict\n"
        "> is keyed on are 1-D views rather than the original matrices. Read the\n"
        "> split only for the non-distributed recipes.\n"
    )
    print("| recipe | DP | fp32 masters | state on 2-D params | state on other params |")
    print("|---|---:|---:|---:|---:|")
    for (name, world) in sorted(groups, key=lambda k: (k[0], k[1])):
        rs = groups[(name, world)]
        n = len(rs)
        print(
            f"| `{name}` | {world} "
            f"| {sum(r['master_mb'] for r in rs) / n:.0f} "
            f"| {sum(r['state_2d_mb'] for r in rs) / n:.0f} "
            f"| {sum(r['state_other_mb'] for r in rs) / n:.1f} |"
        )


if __name__ == "__main__":
    main()
