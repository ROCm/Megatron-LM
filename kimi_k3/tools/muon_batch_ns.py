"""Serial vs batched Newton-Schulz for Muon, at K3's two expert shapes.

`OrthogonalizedOptimizer.step` orthogonalises one parameter at a time. K3's Muon
population is 93 % expert weights -- 336 of (6144, 3584) and 336 of (3584, 3072)
per rank (G65) -- so that is 672 independent 5-step iterations, each three GEMMs
on a matrix too small to fill the GPU.

`install_muon_batched_ns` stacks same-shaped parameters and runs one batched
Newton-Schulz instead; `--syrk` additionally routes the two *symmetric* matmuls
of each step through quack-flydsl's triangular `batched_tsyrk_ex`.

This tool measures both against the serial path and checks how far the parameters
drift after several steps. The patch is class-global and install-once, so each arm
must run in its own process:

    python -m kimi_k3.tools.muon_batch_ns serial   --n 48
    python -m kimi_k3.tools.muon_batch_ns batch    --n 48 --batch 16
    PYTHONPATH=/path/to/flydsl-0.2.4:/path/to/quack \\
        python -m kimi_k3.tools.muon_batch_ns syrk --n 48 --batch 16
    python -m kimi_k3.tools.muon_batch_ns compare

Single rank, `mode="blockwise"` (K3's default), so `partition_dim` is None and no
collective runs inside Newton-Schulz -- the same condition the batching requires.
"""

import argparse
import pathlib
import time

import torch

# The third shape is deliberately alone: it has no batch partner, so it exercises
# the serial fallback inside the batched step and must come out bit-identical.
SHAPES = ((6144, 3584), (3584, 3072))
ODD_SHAPE = (896, 7168)
OUT_DIR = pathlib.Path("/tmp/muon_batch_ns")


def _build(n: int, device: str):
    shapes = [s for s in SHAPES for _ in range(n)] + [ODD_SHAPE]
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(s, device=device, dtype=torch.float32) * 0.02)
        for s in shapes
    ]
    torch.manual_seed(1)
    grads = [
        [torch.randn(s, device=device, dtype=torch.float32) * 1e-3 for s in shapes]
        for _ in range(4)
    ]
    return params, grads


def run(arm: str, n: int, batch: int, device: str) -> None:
    from megatron.core.optimizer.muon import TensorParallelMuon

    if arm != "serial":
        from kimi_k3.model.core_patch import install_muon_batched_ns

        install_muon_batched_ns(batch_size=batch, use_quack_syrk=(arm == "syrk"))

    params, grads = _build(n, device)
    opt = TensorParallelMuon(
        params,
        lr=1e-3,
        momentum_beta=0.95,
        use_nesterov=True,
        weight_decay=0.01,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        mode="blockwise",
    )

    times = []
    for it, step_grads in enumerate(grads):
        for p, g in zip(params, step_grads):
            p.grad = g.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)

    # times[0] carries kernel selection / autotune and is not a steady iteration.
    steady = sum(times[1:]) / len(times[1:])
    print(
        f"{arm:8s} n={n:4d} batch={batch:3d}  cold={times[0]:9.1f} ms  "
        f"steady={steady:8.2f} ms"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"steady_ms": steady, "params": [p.detach().cpu() for p in params]},
        OUT_DIR / f"{arm}.pt",
    )


def compare() -> None:
    ref = torch.load(OUT_DIR / "serial.pt", weights_only=False)
    print(f"{'arm':8s} {'steady ms':>10s} {'vs serial':>10s}   max rel-L2 vs serial")
    print(f"{'serial':8s} {ref['steady_ms']:10.2f} {1.0:9.3f}x")
    for arm in ("batch", "syrk"):
        path = OUT_DIR / f"{arm}.pt"
        if not path.exists():
            continue
        got = torch.load(path, weights_only=False)
        worst = max(
            ((a - b).norm() / a.norm()).item()
            for a, b in zip(ref["params"], got["params"])
        )
        odd = (ref["params"][-1] - got["params"][-1]).abs().max().item()
        print(
            f"{arm:8s} {got['steady_ms']:10.2f} "
            f"{ref['steady_ms'] / got['steady_ms']:9.3f}x   {worst:.3e}"
            f"   (unbatched param exact: {odd == 0.0})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("serial", "batch", "syrk", "compare"))
    ap.add_argument("--n", type=int, default=48, help="parameters per expert shape")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.arm == "compare":
        compare()
        return
    run(args.arm, args.n, args.batch, "cuda")


if __name__ == "__main__":
    main()
