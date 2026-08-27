"""Gate G28 -- run an official-width config and measure what it costs.

The 4 L official preset is ~94 B parameters. With EP it fits on one node in
principle; whether it fits in practice is the measurement. The plan's documented
fallbacks, in order, are: shorter sequence, higher gradient accumulation, then
precision-aware Adam instead of Muon.

    torchrun --nproc_per_node=8 -m kimi_k3.tools.official_smoke --preset 4L --ep 8 --seq 512
"""

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    from megatron.core import parallel_state, tensor_parallel

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    from kimi_k3.config.presets import preset as get_preset
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, loss_func, mock_batch

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "world": world,
           "optimizer": args.optimizer, "rank": rank}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        model = build_k3_model(
            args.preset, allow_official=True,
            expert_model_parallel_size=args.ep,
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1,
        ).bfloat16()
        row["build_s"] = round(time.perf_counter() - t0, 1)
        row["params_per_rank"] = sum(p.numel() for p in model.parameters())
        row["after_model_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)

        ddp, opt = build_optimizer(model, optimizer=args.optimizer, lr=1e-5, bf16=True)
        row["after_optimizer_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)

        vocab = get_preset(args.preset)["model"]["vocab_size"]
        losses = []
        for step in range(args.iterations):
            tokens, labels = mock_batch(vocab, args.seq, 1, seed=step)
            ddp.zero_grad_buffer()
            opt.zero_grad()
            out = ddp(input_ids=tokens, position_ids=None, attention_mask=None)
            loss, _ = loss_func(labels)(out)
            loss.backward()
            ddp.finish_grad_sync()
            opt.step()
            losses.append(round(float(loss.detach()), 4))
        row["losses"] = losses
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:300]
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        for r in gathered:
            print(json.dumps(r))
        if args.json:
            with open(args.json, "a") as f:
                for r in gathered:
                    f.write(json.dumps(r) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
