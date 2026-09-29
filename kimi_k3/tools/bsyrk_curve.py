"""Is the 48-vs-56 inversion in the quack kernel, or in the batching around it?

Times batched_tsyrk_ex alone at K3's two Gram shapes across the same B values the
e2e sweep used. If the inversion reproduces here, it is kernel config selection;
if not, it is something in the Muon step.
"""
import sys, time
import torch
sys.path.insert(0, "/tmp/quack")
from quack.amd.gemm_gfx950_nt_pingpong import batched_tsyrk_ex, can_use_batched_tsyrk

def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

for (M, K) in [(3584, 6144), (3072, 3584)]:
    print(f"\n=== Gram ({M},{M}) from (B,{M},{K}) ; 336 matrices total ===")
    print(f"{'B':>5} {'elig':>5} {'quack ms':>10} {'baddbmm ms':>11} {'ratio':>7} "
          f"{'chunks':>7} {'total quack':>12} {'total bad':>10}")
    for B in [16, 24, 32, 48, 56, 112, 168]:
        a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16) * 0.02
        elig = can_use_batched_tsyrk(a)
        tq = bench(lambda: batched_tsyrk_ex(a)) if elig else float("nan")
        tb = bench(lambda: torch.bmm(a, a.mT))
        n = 336 / B
        print(f"{B:5d} {str(elig):>5} {tq:10.3f} {tb:11.3f} {tb/tq:7.3f} "
              f"{n:7.1f} {tq*n:12.1f} {tb*n:10.1f}")
        del a
        torch.cuda.empty_cache()
