#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc.
"""Theoretical per-GPU memory stacked bars (params / grads / optimizer / activations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from megatron.training.theoretical_memory_usage import (  # noqa: E402
    build_llama_dense_args_for_theoretical_memory,
    compute_theoretical_memory_breakdown_megabytes,
)


def _series_from_benchmark_matrix() -> list:
    """Preset sweep roughly matching run_benchmarks.sh Llama rows."""
    rows = []
    # name, model_size, tp, pp, cp, seq, mbs, bs, world_hint (TP*PP*CP implied by tp*pp for single node)
    specs = [
        ("llama3_8B_TP1_CP1_FP8", 8, 1, 1, 1, 8192, 1, 128, 1),
        ("llama3_8B_TP1_CP1_BF16", 8, 1, 1, 1, 8192, 1, 128, 1),
        ("llama3_70B_TP8", 70, 8, 1, 1, 8192, 1, 8, 8),
        ("llama3_70B_TP4_PP2", 70, 4, 2, 1, 8192, 1, 8, 8),
        ("llama3_70B_TP8_local", 70, 8, 1, 1, 8192, 1, 8, 8),
    ]
    for name, ms, tp, pp, cp, sl, mbs, bs, ws in specs:
        args = build_llama_dense_args_for_theoretical_memory(
            model_size=ms,
            tensor_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            seq_length=sl,
            micro_batch_size=mbs,
            global_batch_size=bs,
            world_size=ws,
        )
        br = compute_theoretical_memory_breakdown_megabytes(args, num_microbatches=None, verbose=False)
        if br is None:
            continue
        rows.append(
            {
                "label": name,
                "spec": {
                    "model_size": ms,
                    "tp": tp,
                    "pp": pp,
                    "cp": cp,
                    "seq_length": sl,
                    "micro_batch_size": mbs,
                    "world_size": ws,
                },
                "gigabytes": {k: round(v, 3) for k, v in br.items() if k.endswith("_gigabytes")},
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", default="output/memory_footprint.json")
    ap.add_argument("--output-png", default="output/memory_footprint.png")
    ap.add_argument(
        "--input-json",
        help="Optional JSON: { 'series': [ { 'label', 'model_size', 'tp', 'pp', 'cp', 'seq_length', 'micro_batch_size', 'global_batch_size', 'world_size' } ] }",
    )
    args = ap.parse_args()

    if args.input_json:
        data = json.loads(Path(args.input_json).read_text())
        series = []
        for ent in data.get("series", []):
            a = build_llama_dense_args_for_theoretical_memory(
                model_size=int(ent["model_size"]),
                tensor_parallel_size=int(ent.get("tp", 1)),
                pipeline_model_parallel_size=int(ent.get("pp", 1)),
                context_parallel_size=int(ent.get("cp", 1)),
                seq_length=int(ent["seq_length"]),
                micro_batch_size=int(ent["micro_batch_size"]),
                global_batch_size=int(ent.get("global_batch_size", 8)),
                world_size=int(ent["world_size"]),
            )
            br = compute_theoretical_memory_breakdown_megabytes(a, num_microbatches=None)
            series.append(
                {
                    "label": ent["label"],
                    "spec": ent,
                    "gigabytes": {k: round(v, 3) for k, v in br.items() if k.endswith("_gigabytes")},
                }
            )
    else:
        series = _series_from_benchmark_matrix()

    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"series": series}, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; JSON only.")
        return

    labels = [s["label"] for s in series]
    pg = [s["gigabytes"].get("param_gigabytes", 0) for s in series]
    gg = [s["gigabytes"].get("grad_gigabytes", 0) for s in series]
    og = [s["gigabytes"].get("optimizer_gigabytes", 0) for s in series]
    ag = [s["gigabytes"].get("activation_gigabytes", 0) for s in series]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax.bar(x, pg, label="param (bf16)")
    ax.bar(x, gg, bottom=np.array(pg), label="main grad (bf16)")
    ax.bar(x, og, bottom=np.array(pg) + np.array(gg), label="optimizer states")
    ax.bar(x, ag, bottom=np.array(pg) + np.array(gg) + np.array(og), label="activation (bf16)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Memory (GB, theoretical per GPU)")
    ax.set_title("Theoretical memory footprint breakdown")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {args.output_json} and {args.output_png}")


if __name__ == "__main__":
    main()
