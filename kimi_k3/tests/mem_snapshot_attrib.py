"""Attribute a torch CUDA memory snapshot to K3 activation-model components.

Consumes the `snap_*.pickle` files written by mem_probe.py. Rather than replaying
the ring-buffered `device_traces` (whose oldest events -- the persistent param /
optimizer / grad allocations -- are evicted once `max_entries` is exceeded and so
cannot be reconstructed), this buckets the snapshot's `segments`: the allocator's
true, complete live set at dump time. Every `active_allocated` block carries its
python allocation frames, so the dump-time resident set attributes exactly.

Dump happens after one full step (backward done, activations freed), so the
segment view is the RESIDENT set: params, bf16 casts, grad buffers, and optimizer
state -- including any state allocated lazily on the first `step()`. The transient
activation peak is `max_memory_allocated - resident`, reported from memstats.

Sweeping seq/mbs and regressing each bucket vs S*B tokens separates FIXED bytes
(seq-invariant resident: the real headroom floor) from LINEAR-in-(S*B) activation.

Usage::

    python -m kimi_k3.tests.mem_snapshot_attrib mem_snap_4L
    python -m kimi_k3.tests.mem_snapshot_attrib <dir> --dump-unattributed 25
"""

import argparse
import glob
import json
import os
import pickle
import re

GIB = 2 ** 30

# Bucket rules, specific -> generic. First matching keyword (substring, lower
# case) in the joined "filename:function" stack text wins. Derived from the real
# 4L-proxy frame signatures (see PERF_4L_mem_attribution.md).
RULES = [
    ("muon_state", ["orthogonalized_optimizer"]),          # Muon master+momentum (LAZY on step 1)
    ("opt_state", ["optimizer.py:__init__", "fused_adam", "layer_wise_optimizer",
                   "distrib_optimizer", "_init_group"]),
    ("param_grad_buf", ["param_and_grad_buffer"]),          # Megatron contiguous param+grad
    ("bf16_cast", ["tensor_cast_func"]),                    # .bfloat16() param copies
    ("attn_res", ["attn_res"]),
    ("mla", ["gated_mla", "k3gatedmla", "mla.py"]),
    ("kda", ["kimi_delta", "kda.py", "/fla/", "delta_attention"]),
    ("moe_dispatch", ["token_dispatcher", "moe_utils", "grouped_gemm",
                      "grouped_linear", "permute", "k3_moe", "/moe/"]),
    ("router", ["router.py"]),
    ("logits", ["cross_entropy", "lm_head", "output_layer", "loss_func",
                "vocab_parallel_cross"]),
    ("embed", ["word_embedding", "embedding.py", "language_model_embedding"]),
    ("layer_misc", ["k3_transformer_layer", "transformer_layer", "layers.py",
                    "random.py", "rmsnorm", "layernorm"]),
    ("nccl", ["reduce_scatter", "all_gather", "all_reduce", "c10d",
              "torch/distributed", "nccl"]),
    ("workspace", ["get_cublas_workspace", "cublas", "hipblaslt", "aiter",
                   "cudnn", "workspace"]),
]


def _frame_text(frames):
    return " ".join(f"{fr.get('filename', '')}:{fr.get('name', '')}"
                    for fr in frames).lower()


def classify(frames):
    text = _frame_text(frames)
    for bucket, keys in RULES:
        if any(k in text for k in keys):
            return bucket
    return "unattributed"


def bucket_snapshot(path):
    """Bucket the dump-time active_allocated set. Returns (buckets, totals, live)."""
    with open(path, "rb") as fh:
        snapshot = pickle.load(fh)
    buckets = {}
    reserved = 0
    inactive = 0
    active = 0
    live = []
    for seg in snapshot.get("segments", []):
        reserved += seg.get("total_size", 0)
        for b in seg.get("blocks", []):
            size = b.get("size", 0)
            if b.get("state") == "active_allocated":
                active += size
                frames = b.get("frames") or []
                buckets[classify(frames)] = buckets.get(classify(frames), 0) + size
                live.append((size, frames))
            else:
                inactive += size
    totals = {"reserved": reserved, "active_allocated": active, "inactive": inactive}
    return buckets, totals, live


def _parse_tag(path):
    m = re.search(r"seq(\d+)_mbs(\d+)_rc(\w+)", os.path.basename(path))
    if not m:
        return None
    return {"seq": int(m.group(1)), "mbs": int(m.group(2)), "rc": m.group(3)}


def _peak_from_memstats(path):
    """peak_alloc_gib recorded alongside the snapshot, if the json is present."""
    base = os.path.basename(path).replace("snap_", "memstats_").replace(".pickle", ".json")
    ms = os.path.join(os.path.dirname(path), base)
    if not os.path.exists(ms):
        return None
    with open(ms) as fh:
        row = json.load(fh).get("row", {})
    return row.get("peak_alloc_gib")


def _lstsq(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), (ys[0] if ys else float("nan"))
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), sy / n
    slope = (n * sxy - sx * sy) / denom
    return slope, (sy - slope * sx) / n


ORDER = ["param_grad_buf", "opt_state", "muon_state", "bf16_cast", "layer_misc",
         "attn_res", "mla", "kda", "moe_dispatch", "router", "logits", "embed",
         "nccl", "workspace", "unattributed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a snap_*.pickle file or a directory of them")
    ap.add_argument("--dump-unattributed", type=int, default=0,
                    help="print the top-N unattributed frame signatures and exit")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "snap_*.pickle")))
    else:
        files = [args.path]
    if not files:
        raise SystemExit(f"no snap_*.pickle under {args.path}")

    if args.dump_unattributed:
        sig = {}
        for path in files:
            _, _, live = bucket_snapshot(path)
            for size, frames in live:
                if classify(frames) == "unattributed":
                    top = " <- ".join(
                        f"{os.path.basename(fr.get('filename', '?'))}:{fr.get('name', '?')}"
                        for fr in frames[:4])
                    sig[top] = sig.get(top, 0) + size
        for top, size in sorted(sig.items(), key=lambda kv: -kv[1])[:args.dump_unattributed]:
            print(f"  {size / GIB:7.3f} GiB  {top}")
        return

    rows = []
    hdr = ["tag"] + [b[:8] for b in ORDER] + ["ACTIVE", "peak", "transient"]
    print("".join(f"{h:>10s}" for h in hdr))
    for path in files:
        meta = _parse_tag(path)
        buckets, totals, _ = bucket_snapshot(path)
        peak = _peak_from_memstats(path)
        active = totals["active_allocated"] / GIB
        transient = (peak - active) if peak is not None else float("nan")
        tag = os.path.basename(path).replace("snap_", "").replace(".pickle", "")
        cells = [f"{tag[:9]:>10s}"]
        cells += [f"{buckets.get(b, 0) / GIB:10.2f}" for b in ORDER]
        cells += [f"{active:10.2f}",
                  f"{(peak if peak else 0):10.2f}",
                  f"{transient:10.2f}"]
        print("".join(cells))
        if meta:
            rows.append((meta, buckets, active, peak, transient))

    lin = [r for r in rows if r[0]["rc"] == "full"]
    if len(lin) >= 2:
        print("\nregression vs S*B (recompute=full):  bytes = slope*tokens + intercept")
        print(f"{'bucket':16s} {'slope MB/tok':>14s} {'intercept GiB':>16s}")
        for b in ORDER + ["_ACTIVE", "_transient"]:
            xs = [m["seq"] * m["mbs"] for (m, _, _, _, _) in lin]
            if b == "_ACTIVE":
                ys = [a * GIB for (_, _, a, _, _) in lin]
            elif b == "_transient":
                ys = [t * GIB for (_, _, _, _, t) in lin if t == t]
                xs = [m["seq"] * m["mbs"] for (m, _, _, _, t) in lin if t == t]
            else:
                ys = [bk.get(b, 0) for (_, bk, _, _, _) in lin]
            if len(xs) < 2 or not any(ys):
                continue
            slope, intercept = _lstsq(xs, ys)
            print(f"{b:16s} {slope / 1e6:14.4f} {intercept / GIB:16.2f}")
        print("\n  intercept = seq-invariant RESIDENT floor (params+opt+grad+cast);")
        print("  _transient slope = real per-token activation to cross-check vs model.")


if __name__ == "__main__":
    main()
