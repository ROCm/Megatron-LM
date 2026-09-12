"""Which phase issues the aten::mm and aten::addmm calls?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.mm_attribution --preset 4L --ep 8

The P11 trace shows 3,523 aten::mm and 6,800 aten::addmm per iteration and does
not say who calls them. Rather than infer it from arithmetic, this profiles
forward+backward and the optimizer step as separate profiler sessions and counts
each phase's calls directly.
"""
import argparse, json, os
from collections import defaultdict

import torch


def kernel_classes(prof):
    """GPU kernel time by class, so the model's GEMMs can be told from Muon's.

    The P11 trace showed 67.5% of kernel time in Tensile GEMMs, which reads as
    "the model is GEMM-bound" until you count the calls: 10,526 against
    Newton-Schulz's known 10,200. Splitting by phase is the only way to see the
    model's own GEMM time.
    """
    from collections import defaultdict
    out = defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        ms = (getattr(e, "self_device_time_total", 0) or 0) / 1000.0
        if ms <= 0 or e.key.startswith("Optimizer.step#"):
            continue
        n = e.key
        if n.startswith("Cijk_") or "gemm" in n.lower():        g = "GEMM"
        elif "ck_tile" in n or "kentry" in n:                    g = "GEMM (CK grouped)"
        elif "fmha" in n.lower():                                g = "attention"
        elif "nccl" in n.lower():                                g = "collectives"
        elif n.startswith("aten::") or n.startswith("void "):    g = "other/elementwise"
        else:                                                    g = "other/elementwise"
        out[g][0] += ms; out[g][1] += e.count
    return {k: {"ms": round(v[0], 2), "calls": v[1]} for k, v in out.items()}


def peak_gemm_tflops() -> float:
    """The device's own bf16 GEMM peak, so every ratio has a real denominator."""
    import time
    best = 0.0
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(5): a @ b
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 10 * 1000
        best = max(best, 2 * n ** 3 / (ms / 1000) / 1e12)
        del a, b
    torch.cuda.empty_cache()
    return best


def counts(prof):
    out = defaultdict(lambda: [0, 0.0])
    for e in prof.key_averages():
        if e.key in ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"):
            out[e.key][0] += e.count
            out[e.key][1] += (getattr(e, "self_device_time_total", 0) or 0) / 1000.0
    return {k: {"calls": v[0], "ms": round(v[1], 2)} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--record-shapes", action="store_true",
                    help="also report achieved TFLOP/s per GEMM shape. Costs profiler "
                         "overhead, so the wall-clock numbers from this run are not "
                         "comparable to an untraced one.")
    ap.add_argument("--json")
    args = ap.parse_args()

    # The image exports NVTE_FLASH_ATTN=0, which core's `auto` backend rejects
    # outright. proxy_ep8 clears these for the same reason (:231).
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    from megatron.core import parallel_state, tensor_parallel
    parallel_state.initialize_model_parallel(expert_model_parallel_size=args.ep)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    from kimi_k3.config.presets import preset
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, loss_func, mock_batch

    model = build_k3_model(args.preset, allow_official=True, expert_model_parallel_size=args.ep,
                           recompute_granularity="full", recompute_method="uniform",
                           recompute_num_layers=1).bfloat16()
    ddp, opt = build_optimizer(model, optimizer="dist_muon", lr=1e-5, bf16=True)
    vocab = preset(args.preset)["model"]["vocab_size"]

    def fwd_bwd(step):
        tok, lab = mock_batch(vocab, args.seq, 1, seed=step)
        ddp.zero_grad_buffer(); opt.zero_grad()
        loss, _ = loss_func(lab)(ddp(input_ids=tok, position_ids=None, attention_mask=None))
        loss.backward(); ddp.finish_grad_sync()

    for s in range(3):          # warm; G49 -- a cold pass measures cache fill
        fwd_bwd(s); opt.step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    prof_kw = dict(record_shapes=True, with_flops=True) if args.record_shapes else {}

    def per_shape(prof):
        """Achieved TFLOP/s for every GEMM shape the profiler saw.

        `with_flops` only populates `flops` for ops torch knows how to count --
        aten matmuls. TE's own GEMMs go through its C++ extension and surface as
        raw Tensile kernels with no aten parent, so they carry neither shape nor
        FLOPs. Anything TE-dispatched is therefore absent here by construction,
        not missing by accident.
        """
        rows = []
        for e in prof.key_averages(group_by_input_shape=True):
            ms = (getattr(e, "self_device_time_total", 0) or 0) / 1000.0
            fl = getattr(e, "flops", 0) or 0
            if ms <= 0 or fl <= 0:
                continue
            rows.append({"op": e.key, "shapes": str(e.input_shapes)[:52],
                         "calls": e.count, "ms": round(ms, 2),
                         "tflops": round(fl / (ms / 1000) / 1e12, 1)})
        return sorted(rows, key=lambda r: -r["ms"])

    with profile(activities=acts, **prof_kw) as p_fb:   # forward + backward only
        fwd_bwd(100); torch.cuda.synchronize()
    fb, fb_k = counts(p_fb), kernel_classes(p_fb)

    with profile(activities=acts, **prof_kw) as p_opt:  # optimizer step only
        opt.step(); torch.cuda.synchronize()
    st, st_k = counts(p_opt), kernel_classes(p_opt)

    if rank == 0:
        print(f"\n{'op':16} {'fwd+bwd':>18} {'optimizer step':>22}")
        for op in ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"):
            a, b = fb.get(op, {"calls": 0, "ms": 0}), st.get(op, {"calls": 0, "ms": 0})
            print(f"{op:16} {a['calls']:7,} / {a['ms']:8.1f} ms {b['calls']:9,} / {b['ms']:8.1f} ms")
        print(f"\n{'kernel class':22} {'fwd+bwd':>20} {'optimizer step':>22}")
        for g in sorted(set(fb_k) | set(st_k)):
            a = fb_k.get(g, {"ms": 0, "calls": 0}); b = st_k.get(g, {"ms": 0, "calls": 0})
            print(f"{g:22} {a['ms']:9.1f} ms /{a['calls']:7,} {b['ms']:11.1f} ms /{b['calls']:7,}")
        if args.record_shapes:
            peak = peak_gemm_tflops()
            print(f"\npeak bf16 GEMM measured on this device: {peak:.1f} TFLOP/s")
            for tag, prof in (("forward+backward", p_fb), ("optimizer step", p_opt)):
                rows = per_shape(prof)
                print(f"\n{tag} — per-shape GEMM efficiency")
                if not rows:
                    print("  (nothing with both shapes and FLOPs — all TE-dispatched)")
                print(f"  {'op':13} {'ms':>8} {'calls':>7} {'TFLOP/s':>9} {'%peak':>6}  shapes")
                for r in rows[:12]:
                    print(f"  {r['op']:13} {r['ms']:8.1f} {r['calls']:7,} {r['tflops']:9.1f} "
                          f"{100*r['tflops']/peak:5.1f}%  {r['shapes']}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"fwd_bwd": fb, "optimizer_step": st,
                           "fwd_bwd_kernels": fb_k, "optimizer_kernels": st_k,
                           "preset": args.preset, "ep": args.ep, "seq": args.seq}, f, indent=2)
    torch.distributed.destroy_process_group()


main()
