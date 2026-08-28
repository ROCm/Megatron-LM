"""Gate G26 -- expert parallelism across 8 ranks, and routing determinism.

    torchrun --nproc_per_node=8 -m kimi_k3.tools.ep_smoke --preset 4L --ep 8

P6 proved the single-rank paths. What only a multi-rank run can show is that the
experts really are *split* — that each rank owns its share, that tokens reach all
of them, and that the ranks agree on the loss after the dispatch round-trip.

The check that earns its place is **starvation**. A router that collapsed onto a
few experts still trains, still descends, and still produces a plausible loss;
the only visible symptom is that some ranks do almost no work. So the run reports
the per-rank token counts and the max/mean ratio rather than just asserting the
step completed.
"""

import argparse
import json
import os
from typing import Dict, List

import torch


def local_expert_count(model) -> int:
    from kimi_k3.moe.k3_qat_wiring import EXPERT_CONTAINER, SHARED

    # The membership test padded the name with dots on both sides, so a module
    # named `...mlp.experts` matched while `name.split(".experts.")` returned a
    # single element -- IndexError on every rank. Split first, then check.
    names = set()
    for name, _ in model.named_modules():
        if SHARED in name:
            continue
        parts = name.split(f".{EXPERT_CONTAINER}.")
        if len(parts) > 1:
            names.add(parts[1].split(".")[0])
    numeric = {n for n in names if n.isdigit()}
    if numeric:
        return len(numeric)
    # grouped GEMM: one module holding weight0..weightN
    for module in model.modules():
        if getattr(module, "num_gemms", None):
            return module.num_gemms
    return 0


def routing_load(model) -> Dict:
    """How evenly the tokens actually landed, from the router's own counter.

    `local_tokens_per_expert` is a buffer core keeps whenever
    `moe_router_enable_expert_bias` is on, which K3 sets. It is the router's view
    of all `num_moe_experts`, so this is the load the dispatcher was asked to
    carry, summed over the MoE layers on this rank.
    """
    total = None
    for module in model.modules():
        counts = getattr(module, "local_tokens_per_expert", None)
        if counts is None:
            continue
        counts = counts.detach().float()
        total = counts.clone() if total is None else total + counts
    if total is None or float(total.sum()) == 0:
        return {"available": False}

    mean = float(total.mean())
    return {
        "available": True,
        "experts": int(total.numel()),
        "max_over_mean": float(total.max()) / mean,
        "min_over_mean": float(total.min()) / mean,
        "starved": int((total == 0).sum()),
        "total_assignments": float(total.sum()),
    }


def main() -> None:
    from megatron.core import parallel_state, tensor_parallel

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--json")
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    from kimi_k3.config.presets import preset as get_preset
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, loss_func, mock_batch

    row: Dict = {"preset": args.preset, "ep": args.ep, "world": world, "rank": rank}
    try:
        model = build_k3_model(
            args.preset, allow_official=args.preset != "tiny",
            expert_model_parallel_size=args.ep,
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1,
        ).bfloat16()
        config = model.config
        row["num_moe_experts"] = config.num_moe_experts
        row["local_experts"] = local_expert_count(model)
        row["expected_local"] = config.num_moe_experts // args.ep
        row["params_this_rank"] = sum(p.numel() for p in model.parameters())

        ddp, opt = build_optimizer(model, optimizer="dist_muon", lr=1e-5, bf16=True)
        vocab = get_preset(args.preset)["model"]["vocab_size"]

        losses = []
        for step in range(args.steps):
            tokens, labels = mock_batch(vocab, args.seq, 1, seed=step)
            ddp.zero_grad_buffer()
            opt.zero_grad()
            loss, _ = loss_func(labels)(ddp(input_ids=tokens, position_ids=None, attention_mask=None))
            loss.backward()
            ddp.finish_grad_sync()
            opt.step()
            losses.append(float(loss.detach()))
        row["losses"] = [round(v, 5) for v in losses]
        # A router that collapsed onto a few experts still trains and still
        # descends. Without this the smoke test cannot tell that apart from a
        # healthy run -- which is the only failure worth running 8 ranks to find.
        row["routing"] = routing_load(model)

        # determinism: the same tokens twice, in eval, must route identically
        tokens, _ = mock_batch(vocab, args.seq, 1, seed=99)
        model.eval()
        with torch.no_grad():
            first = model(input_ids=tokens, position_ids=None, attention_mask=None)
            second = model(input_ids=tokens, position_ids=None, attention_mask=None)
        row["repeat_forward_bitwise"] = bool(torch.equal(first, second))
        row["repeat_max_abs"] = (first.float() - second.float()).abs().max().item()
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:400]

    gathered: List = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        ok = [r for r in gathered if r.get("status") == "ok"]
        summary = {
            "ranks_ok": len(ok),
            "world": world,
            "local_experts_per_rank": sorted({r["local_experts"] for r in ok}) if ok else [],
            "expected_local": ok[0]["expected_local"] if ok else None,
            "params_per_rank_spread": [min(r["params_this_rank"] for r in ok),
                                       max(r["params_this_rank"] for r in ok)] if ok else [],
            "losses_agree_across_ranks": len({tuple(r["losses"]) for r in ok}) == 1 if ok else False,
            "loss_spread": [min(r["losses"][-1] for r in ok), max(r["losses"][-1] for r in ok)]
            if ok else [],
            "repeat_forward_bitwise": all(r["repeat_forward_bitwise"] for r in ok) if ok else False,
            "routing": ok[0].get("routing") if ok else None,
            "worst_max_over_mean": max(
                (r["routing"]["max_over_mean"] for r in ok if r.get("routing", {}).get("available")),
                default=None,
            ),
            "total_starved": sum(
                r["routing"]["starved"] for r in ok if r.get("routing", {}).get("available")
            ),
        }
        print(json.dumps(summary, indent=2))
        for r in gathered:
            if r.get("status") != "ok":
                print(json.dumps({k: r[k] for k in ("rank", "status", "error") if k in r}))
        if args.json:
            with open(args.json, "a") as handle:
                handle.write(json.dumps({"summary": summary, "rows": gathered}) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
