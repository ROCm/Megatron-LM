"""Achieved TFLOP/s and GB/s for the kernels K3 actually runs.

    python -m kimi_k3.tools.roofline

An iteration time says a run is slow; it does not say whether the GEMMs are
underperforming, the bandwidth-bound work is dominating, or neither. This
measures both sides against **this machine's own measured ceilings** rather than
a datasheet figure, so the comparison is honest on a part whose vendor number may
assume sparsity or a clock this container never reaches.

Shapes are K3's real ones at seq 8192, taken from `develop/architecture`:
projections at hidden 7168, the 3584-latent expert matmuls, and the
vocab-163840 output layer.
"""

import argparse
import json
import time
from typing import Dict, List

import torch


def timed(fn, iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def gemm(m: int, k: int, n: int, dtype=torch.bfloat16) -> Dict:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    seconds = timed(lambda: a @ b)
    flops = 2 * m * k * n
    bytes_moved = (a.numel() + b.numel() + m * n) * a.element_size()
    return {
        "m": m, "k": k, "n": n,
        "ms": seconds * 1e3,
        "tflops": flops / seconds / 1e12,
        "gbs": bytes_moved / seconds / 1e9,
        "arithmetic_intensity": flops / bytes_moved,
    }


def bandwidth(fn, bytes_moved: int, label: str) -> Dict:
    seconds = timed(fn)
    return {"op": label, "ms": seconds * 1e3, "gbs": bytes_moved / seconds / 1e9}


def measure_ceilings() -> Dict:
    """The machine's own peaks, so every ratio below has a real denominator."""
    best_gemm = max(gemm(n, n, n)["tflops"] for n in (4096, 8192, 12288))
    x = torch.empty(1 << 28, device="cuda", dtype=torch.bfloat16)  # 512 MiB
    y = torch.empty_like(x)
    copy = bandwidth(lambda: y.copy_(x), 2 * x.numel() * x.element_size(), "copy")
    return {"peak_gemm_tflops": best_gemm, "peak_copy_gbs": copy["gbs"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--json")
    args = ap.parse_args()
    S, H, V, LAT, INT = args.seq, 7168, 163840, 3584, 3072

    out: Dict = {"seq": S, "ceilings": measure_ceilings()}
    c = out["ceilings"]
    print(f"measured ceilings on this device:")
    print(f"  peak bf16 GEMM   {c['peak_gemm_tflops']:8.1f} TFLOP/s")
    print(f"  peak copy        {c['peak_copy_gbs']:8.1f} GB/s\n")

    shapes = [
        ("KDA q/k/v proj",   S, H, 12288),
        ("KDA o_proj",       S, 12288, H),
        ("MLA q_b_proj",     S, 1536, 18432),
        ("MLA kv_b_proj",    S, 512, 24576),
        ("MoE latent down",  S, H, LAT),
        ("MoE latent up",    S, LAT, H),
        ("expert w1/w3",     S * 16 // 896 * 8, LAT, 2 * INT),
        ("expert w2",        S * 16 // 896 * 8, INT, LAT),
        ("output layer",     S, H, V),
    ]
    print(f"{'GEMM (compute-bound)':22} {'m':>7} {'k':>6} {'n':>7} {'ms':>8} {'TFLOP/s':>9} {'% peak':>7} {'AI':>7}")
    out["gemms"] = []
    for label, m, k, n in shapes:
        r = gemm(m, k, n); r["label"] = label; out["gemms"].append(r)
        print(f"{label:22} {m:7d} {k:6d} {n:7d} {r['ms']:8.3f} {r['tflops']:9.1f} "
              f"{r['tflops']/c['peak_gemm_tflops']*100:6.1f}% {r['arithmetic_intensity']:7.1f}")

    print(f"\n{'memory-bound op':30} {'ms':>9} {'GB/s':>9} {'% peak':>7}")
    out["memory"] = []
    hid = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    eps = 1e-5
    cases = [
        ("RMSNorm [S, 7168]", lambda: hid * torch.rsqrt(hid.float().pow(2).mean(-1, keepdim=True) + eps).to(hid.dtype) * w,
         hid.numel() * 2 * 2),
        ("sigmoid gate [S, 7168]", lambda: hid * torch.sigmoid(hid), hid.numel() * 2 * 3),
    ]
    logits = torch.randn(S, V, device="cuda", dtype=torch.bfloat16)
    cases.append(("logits .float() [S, 163840]", lambda: logits.float(), logits.numel() * (2 + 4)))
    labels = torch.randint(0, V, (S,), device="cuda")
    cases.append(("cross_entropy fp32 [S, 163840]",
                  lambda: torch.nn.functional.cross_entropy(logits.float(), labels),
                  logits.numel() * (2 + 4 + 4)))
    slots = torch.randn(S, 9, H, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
    from kimi_k3.block.attn_res import attn_res_mix
    nw = torch.randn(H, device="cuda"); pw = torch.randn(1, H, device="cuda")
    cases.append(("AttnRes mix [S, 9, 7168] fp32",
                  lambda: attn_res_mix(prefix, slots, nw, pw, eps),
                  (slots.numel() + prefix.numel()) * 2 * 3))
    for label, fn, nbytes in cases:
        r = bandwidth(fn, nbytes, label); out["memory"].append(r)
        print(f"{label:30} {r['ms']:9.3f} {r['gbs']:9.1f} {r['gbs']/c['peak_copy_gbs']*100:6.1f}%")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=2)


if __name__ == "__main__":
    main()
