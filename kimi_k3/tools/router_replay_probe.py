"""Gate G26 (second half) -- does `router_replay` pin routing at bf16 + EP?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.router_replay_probe --preset 4L --ep 8

G26's first half is green: expert parallelism shards correctly, no expert starves.
Its second half was owed, and `results/ep_smoke.md` says why it matters -- two
identical `eval()`/`no_grad` forwards differ by max-abs 0.53-0.79 on logits of std
1.69, because **4.0 % of tokens route to different experts**. Finding A18 located
the source inside the MoE accumulation (the first MoE router is bitwise identical
on every rank; the noise enters after it and compounds through routing).

`router_replay` is core's answer: RECORD the top-k indices on one forward, then
REPLAY_FORWARD them on the next so routing cannot drift. The question this probe
answers is whether replaying removes the divergence, and how much of it is
routing versus the accumulation underneath.

Three forwards on identical input:

  A  plain            -- the baseline
  B  plain again      -- the 4.0 % re-routing, reproduced
  C  replay of A      -- same tokens, routing pinned to A's decisions

If A-vs-C is much tighter than A-vs-B, routing was the amplifier and replay pins
it. If A-vs-C is still large, the accumulation noise reaches the output by paths
replay does not cover, and that is worth knowing before anyone relies on it.
"""

import argparse
import json
import os
from typing import Dict

import torch


def compare(a: torch.Tensor, b: torch.Tensor) -> Dict:
    x, y = a.float(), b.float()
    return {
        "bitwise": bool(torch.equal(a, b)),
        "max_abs": (x - y).abs().max().item(),
        "rel_l2": ((x - y).norm() / y.norm()).item(),
        "argmax_agreement": (x.argmax(-1) == y.argmax(-1)).float().mean().item(),
    }


def main() -> None:
    from megatron.core import parallel_state, tensor_parallel

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
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

    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

    from kimi_k3.config.presets import preset as get_preset
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import mock_batch

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "rank": rank}
    try:
        RouterReplay.clear_global_router_replay_instances()
        model = build_k3_model(
            args.preset, allow_official=args.preset != "tiny",
            expert_model_parallel_size=args.ep,
            moe_enable_routing_replay=True,
        ).bfloat16().eval()
        row["replay_instances"] = len(RouterReplay.global_router_replay_instances)

        vocab = get_preset(args.preset)["model"]["vocab_size"]
        tokens, _ = mock_batch(vocab, args.seq, 1, seed=99)
        run = lambda: model(input_ids=tokens, position_ids=None, attention_mask=None)

        with torch.no_grad():
            # A: record the routing decisions
            RouterReplay.clear_global_indices()
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            first = run()
            recorded = RouterReplay.get_recorded_data()
            RouterReplay.clear_global_router_replay_action()

            # B: plain repeat -- the divergence ep_smoke measured
            second = run()

            # C: same input, routing pinned to A's decisions
            RouterReplay.set_replay_data(recorded)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            replayed = run()
            RouterReplay.clear_global_router_replay_action()

        row["layers_recorded"] = len(recorded)
        row["plain_repeat"] = compare(first, second)
        row["replayed"] = compare(first, replayed)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:400]

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        print(json.dumps(gathered[0], indent=2))
        bad = [r for r in gathered if r.get("status") != "ok"]
        if bad:
            print(json.dumps({"failing_ranks": [(r["rank"], r["status"]) for r in bad]}))
        if args.json:
            with open(args.json, "a") as handle:
                for r in gathered:
                    handle.write(json.dumps(r) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
