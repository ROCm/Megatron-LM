"""Gate G41 -- does QAT train, and how far does it sit from BF16?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.qat_twin --preset 4L --ep 8 --steps 30

Read against the P9 twin protocol, with one difference that matters: the other
twins compare things that *should* be identical, so the question is whether they
sit inside the seed noise. QAT is not one of those. Quantising the routed experts
to MXFP4 **is** expected to move the loss, so the question is different:

* is the offset **stable**, or does it grow? A constant gap is quantisation noise
  the model trains through; a growing one is divergence.
* does QAT still *train* — does the loss fall on a fixed batch at all?

So the statistic is the offset's **trend**, not its size: the first-half mean
delta against the second-half mean delta. Reporting only "QAT is 0.02 above BF16"
would hide exactly the failure this gate exists to catch.

Serving parity is the second half: the QAT forward against the same weights run
the way a server would run them (dequantised MXFP4, no activation quantisation).
"""

import argparse
import gc
import json
import os
from typing import Dict, List

import torch


def curves(args, qat: bool) -> List[float]:
    """`args.steps` real training steps, on the same data either way."""
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
        tokens, labels = mock_batch(vocab, args.seq, 1, seed=args.seed)  # fixed batch: it must fall
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
    del model, ddp, opt
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    return losses


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


def serving_parity(args) -> Dict:
    """QAT forward vs the same weights served: dequantised weights, no fake-quant inputs.

    Both run the *same* quantised weights. The only difference is the activation
    path, so this isolates what QAT's activation quantisation costs at serve time.
    """
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.moe.k3_qat_wiring import enable_qat_experts
    from kimi_k3.training.pretrain_kimi_k3 import mock_batch
    from kimi_k3.config.presets import preset as get_preset
    from megatron.core import tensor_parallel

    torch.manual_seed(args.seed)
    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
    model = build_k3_model(
        args.preset, allow_official=args.preset != "tiny",
        expert_model_parallel_size=args.ep,
    ).bfloat16()
    enable_qat_experts(model, quantize_activations=True)

    vocab = get_preset(args.preset)["model"]["vocab_size"]
    tokens, _ = mock_batch(vocab, args.seq, 1, seed=args.seed + 1)
    with torch.no_grad():
        training = model(input_ids=tokens, position_ids=None, attention_mask=None).float()
        for module in model.modules():
            if getattr(module, "_k3_qat_hook", False):
                module._forward_pre_hooks.clear()
                module._k3_qat_hook = False
        serving = model(input_ids=tokens, position_ids=None, attention_mask=None).float()

    diff = training - serving
    stats = {
        "rel_l2": (diff.norm() / serving.norm()).item(),
        "max_abs": diff.abs().max().item(),
        "logit_std": serving.std().item(),
        "argmax_agreement": (training.argmax(-1) == serving.argmax(-1)).float().mean().item(),
    }
    del model, training, serving, diff
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def main() -> None:
    from megatron.core import parallel_state

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--no-quantize-activations", dest="quantize_activations",
                    action="store_false", default=True)
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

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "steps": args.steps, "rank": rank}
    try:
        bf16 = curves(args, qat=False)
        qat = curves(args, qat=True)
        torch.distributed.barrier()
        row.update(
            bf16=[round(v, 4) for v in bf16],
            qat=[round(v, 4) for v in qat],
            bf16_fell=bf16[-1] < bf16[0],
            qat_fell=qat[-1] < qat[0],
            offset=offset_trend(bf16, qat),
            serving=serving_parity(args),
            status="ok",
        )
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
