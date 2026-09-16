"""Memory-attribution probe for the K3 proxy (Phase A1, PR #162 follow-up).

Captures a PyTorch CUDA memory-history snapshot of one real training step so the
bottom-up activation model can be cross-referenced allocation-by-allocation.
Unlike proxy_ep8.py this does NOT run the torch profiler during the measured
step -- the profiler inflates peak memory (proxy_ep8.py:250) and would corrupt
the very number we are attributing.

Sweeping seq/mbs and regressing each measured bucket against S*B separates:
  * FIXED bytes  (CUDA context + hipBLASLt/aiter/NCCL workspaces) -- the +43 GiB
    the flat 82 GiB headroom hides,
  * LINEAR-in-(S*B) bytes (activations) -- validates/corrects the analytic slope,
  * QUADRATIC-in-S bytes (a stored score) -- would contradict flash-style MLA.

Run (8 GPUs, EP8) inside the megatron container::

    PYTHONPATH=<my-branch-worktree> torchrun --nproc_per_node=8 \
        -m kimi_k3.tests.mem_probe --preset 4L --ep 8 \
        --seq 8192 --mbs 1 --recompute full \
        --out-dir develop/profile/mem_snap_4L

Emits (rank 0 only): a `_record_memory_history` snapshot pickle and a
memory-stats JSON per (seq, mbs, recompute). Analyze with mem_snapshot_attrib.py.
"""

import argparse
import json
import os
import time

import torch

from kimi_k3.tools.proxy_ep8 import instrument


def build_model(args, rank, world):
    """Replicates proxy_ep8.build() with a configurable recompute regime.

    proxy_ep8.build() hardcodes recompute_granularity="full"; we expose it so the
    same geometry can be measured under full recompute (the G28 anchor regime)
    and under no recompute (all layers resident) to settle which regime the
    measured 82 GiB headroom corresponds to.
    """
    from megatron.core import parallel_state, tensor_parallel

    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    overrides = {}
    if args.kda_backend:
        overrides["k3_kda_backend"] = args.kda_backend
    if args.mla_backend:
        overrides["k3_mla_backend"] = args.mla_backend
    if args.layers:
        from kimi_k3.config.presets import preset as get_preset

        kda = get_preset(args.preset)["config"]["k3_kda_layers"]
        overrides.update(num_layers=args.layers,
                         k3_kda_layers=tuple(n for n in kda if n <= args.layers))

    if args.recompute == "full":
        recompute = dict(recompute_granularity="full",
                         recompute_method="uniform",
                         recompute_num_layers=1)
    else:
        recompute = dict(recompute_granularity=None)

    model = build_k3_model(
        args.preset, allow_official=args.preset != "tiny",
        expert_model_parallel_size=args.ep,
        **recompute, **overrides,
    ).bfloat16()
    after_model = torch.cuda.memory_allocated()
    ddp, opt = build_optimizer(model, optimizer=args.optimizer, lr=1e-5, bf16=True)
    resident = {
        "after_model_gib": round(after_model / 2 ** 30, 2),
        "after_optimizer_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
    }
    return model, ddp, opt, resident


def one_step(ddp, opt, vocab, seq, mbs, step):
    from kimi_k3.training.pretrain_kimi_k3 import loss_func, mock_batch

    tokens, labels = mock_batch(vocab, seq, mbs, seed=step)
    ddp.zero_grad_buffer()
    opt.zero_grad()
    loss, _ = loss_func(labels)(ddp(input_ids=tokens, position_ids=None,
                                    attention_mask=None))
    loss.backward()
    ddp.finish_grad_sync()
    opt.step()
    return float(loss.detach())


def enable_memory_history(max_entries):
    """Enable CUDA allocator history with python stacks + record_function context.

    Tries the full modern signature, falling back to the minimal one so the probe
    works across torch builds.
    """
    rec = torch.cuda.memory._record_memory_history
    try:
        rec(enabled="all", context="all", stacks="python", max_entries=max_entries)
    except TypeError:
        rec(max_entries=max_entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--recompute", choices=("full", "off"), default="full")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--kda-backend", choices=("eager", "fla"), default="eager")
    ap.add_argument("--mla-backend", choices=("eager", "sdpa", "te"), default="sdpa",
                    help="sdpa dodges the mismatched aiter-v3 FMHA prebuilt while keeping "
                         "a flash-style (non-quadratic) MLA memory footprint; te is production")
    ap.add_argument("--warmup", type=int, default=2,
                    help="steady steps before the recorded step; workspaces/tuning settle here")
    ap.add_argument("--max-entries", type=int, default=800000)
    ap.add_argument("--out-dir", default="develop/profile/mem_snap_4L")
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)

    instrument()
    from kimi_k3.config.presets import preset as get_preset

    tag = f"seq{args.seq}_mbs{args.mbs}_rc{args.recompute}"
    row = {"tag": tag, "preset": args.preset, "ep": args.ep, "seq": args.seq,
           "mbs": args.mbs, "recompute": args.recompute, "world": world, "rank": rank,
           "kda_backend": args.kda_backend, "optimizer": args.optimizer}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # Record from the very start (rank 0): persistent params / optimizer state /
    # grad buffers are allocated at build, so history MUST be on before build or
    # those blocks carry no alloc frames and cannot be attributed. Warmup allocs
    # are freed and drop out of the peak-live set, so recording through them is
    # harmless -- only the frame budget grows (hence max_entries default 800k).
    if rank == 0:
        enable_memory_history(args.max_entries)
    try:
        model, ddp, opt, resident = build_model(args, rank, world)
        row.update(resident)
        vocab = get_preset(args.preset)["model"]["vocab_size"]

        # Warm to the steady allocator state (workspaces, tuning) before the
        # step we treat as representative.
        for step in range(args.warmup):
            one_step(ddp, opt, vocab, args.seq, args.mbs, step)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.cuda.reset_peak_memory_stats()
        one_step(ddp, opt, vocab, args.seq, args.mbs, args.warmup)
        torch.cuda.synchronize()

        row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        row["headroom_gib"] = round(row["peak_alloc_gib"] - row["after_optimizer_gib"], 2)
        row["params_this_rank"] = sum(p.numel() for p in model.parameters())
        row["status"] = "ok"

        if rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            snap = os.path.join(args.out_dir, f"snap_{tag}.pickle")
            torch.cuda.memory._dump_snapshot(snap)
            torch.cuda.memory._record_memory_history(enabled=None)
            row["snapshot"] = snap
            stats = os.path.join(args.out_dir, f"memstats_{tag}.json")
            with open(stats, "w") as fh:
                json.dump({"row": row,
                           "memory_stats": torch.cuda.memory_stats()}, fh, indent=2,
                          default=str)
            row["memstats"] = stats
    except Exception as exc:  # noqa: BLE001 -- an OOM here IS a measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:400]
        row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        print(json.dumps(gathered[0], indent=2))
        for other in gathered[1:]:
            print(json.dumps({k: other.get(k) for k in
                              ("rank", "status", "peak_alloc_gib", "headroom_gib")}))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
