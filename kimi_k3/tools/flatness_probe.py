"""T12.5 — is continued pretraining from the released checkpoint *flat*?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.flatness_probe \\
        --preset 4L --ep 8 --checkpoint /path/to/converted --steps 50

A correctly converted checkpoint resumed into a correctly built model should
produce a loss that starts where the released model left off and **stays there**.
The failure this catches is the one that looks like success: a run that spikes at
step 0 and re-converges over a few hundred steps looks like "training is working"
on a loss curve, and is actually the model relearning something the conversion
broke. By the time it flattens the evidence is gone.

## The statistic

Flatness is a *shape*, not a level, and this reports it as three numbers so one
cannot hide another:

* `arrival` — loss at step 0. Compared against `--expect-loss` when the released
  model's own figure is known; on synthetic tokens it should sit near
  `ln(vocab_size)`, which is chance, and anything far below means the "held-out"
  data is not held out.
* `drift` — mean of the last quarter minus mean of the first. Reported, but **not**
  the test: continued pretraining is supposed to descend, so negative drift alone
  says nothing.
* `deceleration` — how much faster the first half falls than the second. This is
  the recovery signature. A model climbing back from a conversion defect drops
  hard and then flattens; ordinary continued pretraining descends roughly
  linearly, so its two halves fall by the same amount and this is near zero.
  Testing `drift` instead would fail every healthy run that actually learns.
* `spike` — max loss in the window over the arrival loss. A single bad step shows
  here and nowhere else.

## What C6 means for the threshold

Review finding **C6**: the released routed experts are MXFP4, and the original
BF16 weights were never published. `dequantize_on_import` therefore starts a
continued-pretrain run from **quantised-then-dequantised** weights, not from the
weights the release trained. So a small arrival bump is *expected*, and calling it
a bug would be wrong.

`--quantisation-baseline` measures that bump instead of assuming it: it evaluates
the same batch with the expert weights fake-quantised and again without, and
reports the loss difference. Any arrival offset within that is attributable to C6;
anything beyond it is not.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch


def flatness(losses: List[float]) -> Dict:
    """Shape of a loss window: where it starts, whether it moves, and worst step."""
    quarter = max(1, len(losses) // 4)
    first = sum(losses[:quarter]) / quarter
    last = sum(losses[-quarter:]) / quarter
    half = len(losses) // 2
    early_drop = losses[0] - losses[max(0, half - 1)]
    late_drop = losses[half] - losses[-1] if half < len(losses) else 0.0
    return {
        "arrival": losses[0],
        "first_quarter_mean": first,
        "last_quarter_mean": last,
        "drift": last - first,
        "early_drop": early_drop,
        "late_drop": late_drop,
        "deceleration": early_drop - late_drop,
        "spike": max(losses) - losses[0],
        "min": min(losses),
        "max": max(losses),
    }


def verdict(stats: Dict, tolerance: float) -> Dict:
    """Flat, recovering, or spiking -- named, so a curve is not read by eye."""
    problems = []
    if stats["deceleration"] > tolerance:
        problems.append(
            f"deceleration {stats['deceleration']:.4f} above {tolerance}: the window "
            f"falls {stats['early_drop']:.4f} in its first half and only "
            f"{stats['late_drop']:.4f} in its second. That is *recovery* -- what a "
            "conversion defect looks like once the model has trained through it -- "
            "not continued pretraining, which descends at a steady rate"
        )
    if stats["spike"] > tolerance:
        problems.append(f"spike {stats['spike']:.4f} above {tolerance}")
    return {"flat": not problems, "problems": problems}


def evaluate(model, tokens, labels) -> float:
    from ..training.pretrain_kimi_k3 import loss_func

    with torch.no_grad():
        out = model(input_ids=tokens, position_ids=None, attention_mask=None)
    return float(loss_func(labels)(out)[0])


def quantisation_baseline(model, tokens, labels) -> Dict:
    """How much of any arrival offset is C6 rather than a defect.

    Evaluates the same batch twice -- expert weights as loaded, then fake-quantised
    through the same MXFP4 path the release shipped. The gap is the cost of the
    round trip the conversion already paid, and it bounds what an arrival bump is
    allowed to be before it means something.
    """
    from ..moe.k3_qat_wiring import disable_activation_quantisation, enable_qat_experts

    plain = evaluate(model, tokens, labels)
    touched = enable_qat_experts(model, quantize_activations=False)
    quantised = evaluate(model, tokens, labels)
    disable_activation_quantisation(model)
    return {
        "loss_as_loaded": plain,
        "loss_requantised": quantised,
        "c6_offset": quantised - plain,
        "expert_weights_touched": touched["weights"],
    }


def main() -> None:
    from megatron.core import parallel_state, tensor_parallel

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6, help="continued pretrain, not from scratch")
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--checkpoint", default=None,
                    help="converted release checkpoint; omitted means a fresh model, which "
                         "measures the harness rather than the checkpoint")
    ap.add_argument("--expect-loss", type=float, default=None,
                    help="the released model's own loss on this data, when known")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--quantisation-baseline", action="store_true")
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

    from ..config.presets import preset as get_preset
    from ..model.build import build_k3_model
    from ..training.pretrain_kimi_k3 import build_optimizer, loss_func, mock_batch

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "steps": args.steps,
           "rank": rank, "checkpoint": args.checkpoint, "lr": args.lr}
    try:
        model = build_k3_model(
            args.preset, allow_official=args.preset != "tiny",
            expert_model_parallel_size=args.ep,
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1,
        ).bfloat16()
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
            missing, unexpected = model.load_state_dict(state, strict=False)
            row["load"] = {"unexpected": unexpected[:5],
                           "missing_tensors": [m for m in missing
                                               if torch.is_tensor(model.state_dict().get(m))][:5]}

        vocab = get_preset(args.preset)["model"]["vocab_size"]
        tokens, labels = mock_batch(vocab, args.seq, 1, seed=7)

        if args.quantisation_baseline:
            row["c6"] = quantisation_baseline(model, tokens, labels)

        ddp, opt = build_optimizer(model, optimizer=args.optimizer, lr=args.lr, bf16=True)
        losses = []
        for step in range(args.steps):
            batch, target = mock_batch(vocab, args.seq, 1, seed=1000 + step)
            ddp.zero_grad_buffer()
            opt.zero_grad()
            loss, _ = loss_func(target)(ddp(input_ids=batch, position_ids=None, attention_mask=None))
            loss.backward()
            ddp.finish_grad_sync()
            opt.step()
            losses.append(float(loss.detach()))

        row["losses"] = [round(v, 5) for v in losses]
        row["flatness"] = flatness(losses)
        row["verdict"] = verdict(row["flatness"], args.tolerance)
        if args.expect_loss is not None:
            row["arrival_vs_expected"] = row["flatness"]["arrival"] - args.expect_loss
        row["chance_loss"] = float(torch.log(torch.tensor(float(vocab))))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:400]

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        print(json.dumps(gathered[0], indent=2))
        if args.json:
            with open(args.json, "a") as handle:
                for r in gathered:
                    handle.write(json.dumps(r) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
