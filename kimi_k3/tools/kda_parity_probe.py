"""Gate G15 -- measure KDA backend agreement, and the floor to read it against.

Rule R4.4: a tolerance is established by measuring, not chosen. The order matters:

1. **the floor** -- the same eager code in bf16 against itself in fp32. That is
   what dtype alone costs, before any kernel is involved;
2. **the signal** -- fla against the fp32 oracle, at the same dtype.

A bound is then set above the floor with recorded margin. Quoting a fla-vs-oracle
number without the floor beside it says nothing: at 64 k tokens a recurrent bf16
model drifts by far more than any kernel bug would.

    python -m kimi_k3.tools.kda_parity_probe --seq 1024 4096
    python -m kimi_k3.tools.kda_parity_probe --production --seq 1024 --json out.jsonl
"""

import argparse
import json
import time
from typing import Dict

import torch

from kimi_k3.attention.kda_backends import kda_forward

GEOMETRIES = {
    # name: (heads, head_dim)
    "tiny": (2, 64),
    "mid": (8, 128),
    "production": (96, 128),
}


def _inputs(batch, seq, heads, head_dim, dtype, device="cuda", seed=0, requires_grad=False):
    torch.manual_seed(seed)
    mk = lambda *s, d=dtype: torch.randn(*s, device=device, dtype=d, requires_grad=requires_grad)
    return dict(
        q=mk(batch, seq, heads, head_dim),
        k=mk(batch, seq, heads, head_dim),
        v=mk(batch, seq, heads, head_dim),
        g=mk(batch, seq, heads, head_dim),
        beta=mk(batch, seq, heads, d=torch.float32),
        A_log=torch.rand(heads, device=device, dtype=torch.float32).log_().requires_grad_(requires_grad),
        dt_bias=mk(heads * head_dim, d=torch.float32),
    )


def compare(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """The three statistics every parity gate reports (rule R4.4)."""
    af, bf = a.float().flatten(), b.float().flatten()
    denom = bf.norm().clamp_min(1e-20)
    return {
        "rel_l2": ((af - bf).norm() / denom).item(),
        "max_abs": (af - bf).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(af, bf, dim=0).item(),
    }


def run_case(batch, seq, heads, head_dim, dtype, backend, grad=False):
    args = _inputs(batch, seq, heads, head_dim, dtype, requires_grad=grad)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, state = kda_forward(**args, backend=backend, output_final_state=True)
    torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    grads = None
    if grad:
        out.float().pow(2).sum().backward()
        grads = {n: args[n].grad.detach().clone() for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")}
    return out.detach(), state.detach(), grads, fwd_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="tiny", choices=sorted(GEOMETRIES))
    ap.add_argument("--production", action="store_true", help="shorthand for --geometry production")
    ap.add_argument("--seq", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad", action="store_true", help="also compare backward")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    name = "production" if args.production else args.geometry
    heads, head_dim = GEOMETRIES[name]
    print(f"# KDA parity -- geometry {name} (H={heads}, K={head_dim}), batch {args.batch}\n")
    print("| seq | comparison | rel-L2 | max-abs | cosine | note |")
    print("|---:|---|---:|---:|---:|---|")

    rows = []
    for seq in args.seq:
        # the floor: identical code, bf16 vs fp32
        out32, st32, gr32, t32 = run_case(args.batch, seq, heads, head_dim, torch.float32, "eager", args.grad)
        out16, st16, gr16, _ = run_case(args.batch, seq, heads, head_dim, torch.bfloat16, "eager", args.grad)
        floor = compare(out16, out32)
        rows.append({"seq": seq, "geometry": name, "comparison": "floor: eager bf16 vs fp32", **floor})
        print(f"| {seq} | eager bf16 vs eager fp32 | {floor['rel_l2']:.3e} | {floor['max_abs']:.3e} "
              f"| {floor['cosine']:.6f} | the floor |")

        # the signal, at each dtype
        for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            ref = out32 if dtype is torch.float32 else out16
            try:
                out_f, st_f, gr_f, tf = run_case(args.batch, seq, heads, head_dim, dtype, "fla", args.grad)
            except Exception as exc:  # noqa: BLE001 -- reporting, not handling
                print(f"| {seq} | fla {label} | — | — | — | FAILED: {type(exc).__name__} |")
                continue
            stats = compare(out_f, ref)
            rows.append({"seq": seq, "geometry": name, "comparison": f"fla {label} vs eager {label}", **stats})
            note = "same dtype as its reference"
            print(f"| {seq} | fla {label} vs eager {label} | {stats['rel_l2']:.3e} | {stats['max_abs']:.3e} "
                  f"| {stats['cosine']:.6f} | {note} |")

            if args.grad and gr_f is not None:
                ref_g = gr32 if dtype is torch.float32 else gr16
                worst = max(((n, compare(gr_f[n], ref_g[n])["rel_l2"]) for n in gr_f), key=lambda x: x[1])
                rows.append({"seq": seq, "geometry": name,
                             "comparison": f"fla {label} backward (worst grad)", "rel_l2": worst[1],
                             "which": worst[0]})
                print(f"| {seq} | fla {label} backward, worst grad ({worst[0]}) | {worst[1]:.3e} | — | — | |")

        print(f"| {seq} | — | — | — | — | eager fp32 forward took {t32:.2f} s |")

    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
