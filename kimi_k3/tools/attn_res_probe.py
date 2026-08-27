"""Gate G6 -- size the AttnRes payload and the mixer's fp32 temporaries.

The incoming plan budgeted nothing for AttnRes memory (review finding A4). Two
costs need numbers before the pipeline design freezes:

1. **Payload on the wire.** The packed tensor is ``(1 + K) x S x B x H``, with
   ``K`` growing to 8 over a 93-layer model, so the last stage boundary carries
   9x a normal pipeline tensor.
2. **Mixer temporaries.** ``_apply_attn_res`` upcasts the whole ``[T, K+1, H]``
   stack to fp32 and does it twice per layer. This is the dominant non-GEMM
   memory cost in the model and P11's headline target.

Run::

    python -m kimi_k3.tools.attn_res_probe --width production
    python -m kimi_k3.tools.attn_res_probe --width tiny --json out.jsonl
"""

import argparse
import json
from typing import Dict, List

import torch

from kimi_k3.block.attn_res import attn_res_mix
from kimi_k3.block.attn_res_pp import pack, payload_bytes, slots_before, unpack

WIDTHS = {
    # name: (hidden, seq_len, micro_batch, num_layers, block_size)
    "tiny": (512, 128, 1, 4, 2),
    "mid": (2048, 2048, 1, 24, 12),
    "production": (7168, 8192, 1, 93, 12),
}


def payload_table(hidden: int, seq: int, mbs: int, num_layers: int, block: int, pp: int) -> List[Dict]:
    """Per-stage payload for an even-ish PP split aligned to block boundaries."""
    per_stage = [num_layers // pp] * pp
    for i in range(num_layers - sum(per_stage)):
        per_stage[i] += 1
    # prefer stages of exactly `block` layers where the layer count allows it
    if num_layers >= pp * block:
        per_stage = [block] * (pp - 1) + [num_layers - block * (pp - 1)]

    rows, first = [], 0
    for stage, count in enumerate(per_stage):
        last = first + count - 1
        send_slots = slots_before(last + 1, block)
        rows.append(
            {
                "stage": stage,
                "layers": f"{first}-{last}",
                "count": count,
                "recv_mult": 1 + slots_before(first, block),
                "send_mult": 1 + send_slots,
                "send_bytes": payload_bytes(seq, mbs, hidden, send_slots),
                # 1F1B holds (pp - stage) microbatches in flight during warmup
                "inflight_bytes": (pp - stage) * payload_bytes(seq, mbs, hidden, send_slots),
            }
        )
        first = last + 1
    rows[-1]["send_mult"] = 0
    rows[-1]["send_bytes"] = 0
    rows[-1]["inflight_bytes"] = 0
    return rows


def mixer_cost(hidden: int, tokens: int, slots: int, dtype, device, fp32: bool, iters: int = 5) -> Dict:
    """Peak allocator delta and wall-clock for one mix, forward and fwd+bwd."""
    prefix = torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(tokens, slots, hidden, device=device, dtype=dtype, requires_grad=True)
    norm_w = torch.ones(hidden, device=device, dtype=dtype)
    proj_w = torch.randn(1, hidden, device=device, dtype=dtype) * 0.02

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5, fp32=fp32)
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() - before
    del out

    def timed(fn) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    fwd_ms = timed(lambda: attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5, fp32=fp32))

    def fwd_bwd():
        o = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5, fp32=fp32)
        o.sum().backward()

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fwd_bwd()
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated() - before
    bwd_ms = timed(fwd_bwd)

    # bytes the mix must read at minimum: the [T, K+1, H] stack, once
    stack_elems = tokens * (slots + 1) * hidden
    return {
        "slots_plus_one": slots + 1,
        "tokens": tokens,
        "fwd_peak_mb": fwd_peak / 2**20,
        "fwdbwd_peak_mb": bwd_peak / 2**20,
        "fwd_ms": fwd_ms,
        "fwdbwd_ms": bwd_ms,
        "stack_bytes_in_dtype": stack_elems * torch.finfo(dtype).bits // 8,
        "stack_bytes_fp32": stack_elems * 4,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", default="production", choices=sorted(WIDTHS))
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--no-fp32", action="store_true",
                    help="run the mix in the input dtype -- measures the fp32 tax, never parity")
    args = ap.parse_args()

    hidden, seq, mbs, num_layers, block = WIDTHS[args.width]
    dtype = getattr(torch, args.dtype)
    device = "cuda"
    out = {"fp32": not args.no_fp32, "width": args.width, "hidden": hidden, "seq": seq, "mbs": mbs,
           "num_layers": num_layers, "block_size": block, "pp": args.pp}

    print(f"# AttnRes sizing -- {args.width} (H={hidden}, S={seq}, B={mbs}, "
          f"{num_layers} layers, block {block}, PP={args.pp}, {args.dtype})\n")

    rows = payload_table(hidden, seq, mbs, num_layers, block, args.pp)
    out["payload"] = rows
    print("| stage | layers | count | recv x | send x | send MB | in-flight MB |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['stage']} | {r['layers']} | {r['count']} | {r['recv_mult']} | "
              f"{r['send_mult'] or '—'} | {r['send_bytes'] / 2**20:.0f} | "
              f"{r['inflight_bytes'] / 2**20:.0f} |")
    peak_stage = max(rows, key=lambda r: r["inflight_bytes"])
    print(f"\nWorst in-flight payload: stage {peak_stage['stage']} at "
          f"{peak_stage['inflight_bytes'] / 2**20:.0f} MB "
          f"(x2 counting saved input and output tensors).")

    # verify pack/unpack round-trips and that the analytic byte count is right
    ps = torch.randn(8, 1, hidden, device=device, dtype=dtype)
    br = torch.randn(8, 1, 3, hidden, device=device, dtype=dtype)
    packed = pack(ps, br)
    assert packed.shape == (4 * 8, 1, hidden)
    assert packed.numel() * packed.element_size() == payload_bytes(8, 1, hidden, 3)
    a, b = unpack(packed, 8, 3)
    assert torch.equal(a, ps) and torch.equal(b, br)
    print("pack/unpack round-trip: OK\n")

    if not args.skip_timing:
        print("| K+1 | fwd peak MB | fwd+bwd peak MB | fwd ms | fwd+bwd ms | stack MB (bf16 / fp32) |")
        print("|---:|---:|---:|---:|---:|---|")
        costs = []
        for slots in range(1, num_layers // block + 2):
            c = mixer_cost(hidden, seq * mbs, slots, dtype, device, fp32=not args.no_fp32)
            costs.append(c)
            print(f"| {c['slots_plus_one']} | {c['fwd_peak_mb']:.0f} | {c['fwdbwd_peak_mb']:.0f} "
                  f"| {c['fwd_ms']:.2f} | {c['fwdbwd_ms']:.2f} "
                  f"| {c['stack_bytes_in_dtype'] / 2**20:.0f} / {c['stack_bytes_fp32'] / 2**20:.0f} |")
            torch.cuda.empty_cache()
        out["mixer"] = costs

        # what the whole model pays: two mixes per layer, K+1 growing per block
        per_layer = [1 + slots_before(l + 1, block) for l in range(num_layers)]
        mean_k1 = sum(per_layer) / len(per_layer)
        total_reads = 2 * sum(
            seq * mbs * k1 * hidden * (torch.finfo(dtype).bits // 8) for k1 in per_layer
        )
        by_k1 = {c["slots_plus_one"]: c["fwd_ms"] for c in costs}
        model_fwd_ms = 2 * sum(by_k1[min(k1, max(by_k1))] for k1 in per_layer)
        out["model"] = {"mean_k_plus_one": mean_k1, "read_bytes_per_forward": total_reads,
                        "eager_mixer_fwd_ms_per_microbatch": model_fwd_ms}
        print(f"\nWhole model, one microbatch: {2 * num_layers} mixes, mean K+1 = {mean_k1:.2f}, "
              f"{total_reads / 2**30:.1f} GiB read per forward, "
              f"eager mixer forward ~{model_fwd_ms:.0f} ms.")

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
