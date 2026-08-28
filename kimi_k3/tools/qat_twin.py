"""Gate G41 -- does QAT train, and how far does it sit from BF16?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.qat_twin --preset 4L --ep 8 --steps 30

Read against the P9 twin protocol, with one difference that matters: the other
twins compare things that *should* be identical, so the question is whether they
sit inside the seed noise. QAT is not one of those. Quantising the routed experts
to MXFP4 **is** expected to move the loss, so the question is different:

* is the offset **stable**, or does it grow? A constant gap is quantisation noise
  the model trains through; a growing one is divergence.
* does QAT still *train* — does the loss fall at all?

**The batch pool is not a detail.** A single fixed batch is what P9's twins use,
but here it destroys the measurement: at 4 L official the model memorises one
batch completely and both curves reach *exactly* 0.0 by step 20. The offset then
reports `second_half = 2e-22` and "stable", which is true and vacuous — both
sides are sitting on zero and a divergence could not show up if it existed.
Cycling a small pool of distinct batches makes the loss fall without bottoming
out, which is the regime where an offset means something.

So the statistic is the offset's **trend**, not its size: the first-half mean
delta against the second-half mean delta. Reporting only "QAT is 0.02 above BF16"
would hide exactly the failure this gate exists to catch.

Serving parity is the second half: the QAT forward against the same weights run
the way a server would run them (dequantised MXFP4, no activation quantisation).

**One phase per process.** Each phase builds an official-width model that peaks
near 193 GiB per rank, and HIP does not hand that back to the next build quickly
enough -- running two phases in one process fails with 270 GiB held outside
PyTorch's allocator while it reports 17 GiB allocated, which reads like a leak and
is really a lifetime problem. So the phases are separate launches appending to one
jsonl, and `--mode report` compares them. `pp_payload_probe.py` is laid out the
same way for the same reason.

    for mode in bf16 qat; do
        torchrun --nproc_per_node=8 -m kimi_k3.tools.qat_twin --mode $mode --json out.jsonl
    done
    python -m kimi_k3.tools.qat_twin --mode report --json out.jsonl
"""

import argparse
import gc
import json
import os
from typing import Dict, List

import torch


def curves(args, qat: bool) -> Dict:
    """`args.steps` real training steps, on the same data either way.

    When `qat` is set this also measures serving parity **on the trained model**,
    before it is torn down. Measuring it on a freshly initialised one — as the
    first version did — compares a 2.6 % activation perturbation against a network
    with no signal in it, and reports a relative error near 1.0 and 1 % argmax
    agreement that look like catastrophic failure and mean nothing.
    """
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.moe.k3_qat_wiring import enable_qat_experts
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, loss_func, mock_batch
    from kimi_k3.config.presets import preset as get_preset
    from megatron.core import tensor_parallel

    torch.manual_seed(args.seed)
    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
    model = build_k3_model(
        args.preset, allow_official=args.preset != "tiny",
        expert_model_parallel_size=args.ep,
        recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1,
    ).bfloat16()
    if qat:
        enable_qat_experts(model, quantize_activations=args.quantize_activations)

    ddp, opt = build_optimizer(model, optimizer=args.optimizer, lr=args.lr, bf16=True)
    vocab = get_preset(args.preset)["model"]["vocab_size"]

    torch.distributed.barrier()
    losses = []
    for step in range(args.steps):
        tokens, labels = mock_batch(
            vocab, args.seq, 1, seed=args.seed + (step % max(1, args.batch_pool))
        )
        ddp.zero_grad_buffer()
        opt.zero_grad()
        loss, _ = loss_func(labels)(ddp(input_ids=tokens, position_ids=None, attention_mask=None))
        loss.backward()
        ddp.finish_grad_sync()
        opt.step()
        losses.append(float(loss.detach()))
    # Both curves run in one process, so the second model is built while the
    # first is being torn down. DDP allocates its buckets collectively, and if
    # ranks reach that point after different amounts of garbage collection they
    # enqueue collectives in different orders -- which shows up as an NCCL
    # timeout with the ranks one work item apart, not as an error anyone can read.
    stats = {"losses": [round(v, 5) for v in losses], "fell": losses[-1] < losses[0]}
    if qat:
        stats["serving"] = serving_parity_of(model, args, vocab)

    del model, ddp, opt
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    return stats


def offset_trend(bf16: List[float], qat: List[float]) -> Dict:
    """Is the gap stable? Halves, not endpoints, so one noisy step cannot decide."""
    deltas = [q - b for b, q in zip(bf16, qat)]
    half = len(deltas) // 2
    first, second = deltas[:half], deltas[half:]
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "mean_offset": mean(deltas),
        "first_half": mean(first),
        "second_half": mean(second),
        "drift": mean(second) - mean(first),
        "max_abs_offset": max(abs(d) for d in deltas),
    }


def serving_parity_of(model, args, vocab: int) -> Dict:
    """The trained QAT model's forward, against the same weights served.

    Both forwards run the *same* quantised weights; the only difference is that
    serving does not fake-quantise activations. So this isolates what QAT's
    activation path costs at serve time, which is the number the gate wants.

    The hooks come off by handle, not by clearing `_forward_pre_hooks` — that
    would also remove anything core had registered and quietly measure a broken
    module instead.
    """
    from kimi_k3.moe.k3_qat_wiring import disable_activation_quantisation
    from kimi_k3.training.pretrain_kimi_k3 import mock_batch

    tokens, _ = mock_batch(vocab, args.seq, 1, seed=args.seed + 10_000)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        training = model(input_ids=tokens, position_ids=None, attention_mask=None).float()
        removed = disable_activation_quantisation(model)
        serving = model(input_ids=tokens, position_ids=None, attention_mask=None).float()
    if was_training:
        model.train()

    diff = training - serving
    stats = {
        "hooks_removed": removed,
        "rel_l2": (diff.norm() / serving.norm()).item(),
        "max_abs": diff.abs().max().item(),
        "logit_std": serving.std().item(),
        "argmax_agreement": (training.argmax(-1) == serving.argmax(-1)).float().mean().item(),
    }
    del training, serving, diff
    return stats


def report(args) -> None:
    """Compare the phases already written to `--json`, and print the verdict."""
    rows = [json.loads(line) for line in open(args.json)]
    by_mode = {}
    for row in rows:
        if row.get("status") == "ok":
            by_mode[row["mode"]] = row
    missing = {"bf16", "qat"} - set(by_mode)
    if missing:
        raise SystemExit(f"missing phases: {sorted(missing)} -- run them first")

    bf16, qat = by_mode["bf16"]["losses"], by_mode["qat"]["losses"]
    out = {
        "steps": len(bf16),
        "bf16": [bf16[0], bf16[-1]],
        "qat": [qat[0], qat[-1]],
        "bf16_fell": by_mode["bf16"]["fell"],
        "qat_fell": by_mode["qat"]["fell"],
        "offset": offset_trend(bf16, qat),
    }
    if "serving" in by_mode["qat"]:
        out["serving"] = by_mode["qat"]["serving"]
    print(json.dumps(out, indent=2))


def main() -> None:
    from megatron.core import parallel_state

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-pool", type=int, default=8,
                    help="distinct batches cycled through; 1 reproduces the degenerate "
                         "fixed-batch run that bottoms out at 0.0")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--no-quantize-activations", dest="quantize_activations",
                    action="store_false", default=True)
    ap.add_argument("--mode", choices=("bf16", "qat", "report"), default="bf16")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.mode == "report":
        report(args)
        return

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "steps": args.steps,
           "rank": rank, "mode": args.mode}
    try:
        row.update(curves(args, qat=args.mode == "qat"), status="ok")
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row.update(status=type(exc).__name__, error=str(exc)[:400])

    # The gather allocates NCCL buffers, and by this point three official-width
    # models have been built and torn down. Without reclaiming first it fails
    # inside NCCL with "Failed to CUDA calloc", which reads like a comms bug.
    gc.collect()
    torch.cuda.empty_cache()
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        print(json.dumps(gathered[0], indent=2))
        if args.json:
            with open(args.json, "a") as handle:
                handle.write(json.dumps(gathered[0]) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
