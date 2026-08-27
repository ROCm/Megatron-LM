"""Gate G37 -- save, resume, and get the same losses, under PP=2.

    torchrun --nproc_per_node=2 -m kimi_k3.tools.pp_resume_probe --steps 20 --save-at 10

Resume is where the AttnRes payload can quietly break. The block residual is
*activation* state and rightly absent from the checkpoint, but the optimizer, the
per-expert routing bias and the RNG are not, and a resume that drops any of them
still runs, still descends, and produces a subtly different model. So the claim
here is the strict one -- **bitwise** identical losses after the resume point --
and anything weaker is reported as the failure it is.

The comparison is against the tail of an uninterrupted run of the same length,
not against a fresh run from the checkpoint alone, because only the uninterrupted
run knows what those steps *should* have produced.
"""

import argparse
import json
import os
from typing import Dict, List

import torch

CKPT_PATH = "/tmp/k3_pp_resume.pt"


def build(pp_size: int, dropout: float = 0.0):
    from megatron.core import parallel_state, tensor_parallel

    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.model.build import build_k3_model

    p = preset("tiny")
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    model = build_k3_model(
        "tiny",
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
        deallocate_pipeline_outputs=False,
        sequence_parallel=False,
        hidden_dropout=dropout,
        attention_dropout=dropout,
        # the routing bias is a *training* statistic that must survive a resume;
        # it is left on deliberately, because switching it off would remove the
        # most likely thing for a resume to lose
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    return model.cuda().float(), p


def rng_state() -> Dict:
    from megatron.core import tensor_parallel

    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }


def load_rng(state: Dict) -> None:
    from megatron.core import tensor_parallel

    # torch.load(map_location="cuda") moves these; both setters want ByteTensors
    # on the host, and the error they raise is about the *type*, not the device.
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
    tensor_parallel.get_cuda_rng_tracker().set_states(state["tracker"])


def step(ddp, optimizer, model, tokens, labels, seq: int):
    """One pipelined training step; returns the loss on the last stage else None."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    from kimi_k3.pipeline.k3_schedule import resolve
    from kimi_k3.training.pretrain_kimi_k3 import loss_func

    def forward_step(_iterator, module):
        out = module(input_ids=tokens, position_ids=None, attention_mask=None)
        return out, loss_func(labels)

    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    losses = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=None,
        model=[ddp],
        num_microbatches=1,
        seq_length=seq,
        micro_batch_size=1,
        decoder_seq_length=seq,
        forward_only=False,
        adjust_tensor_shapes_fn=resolve(model, torch.distributed.get_world_size()),
    )
    ddp.finish_grad_sync()
    optimizer.step()
    for module in model.modules():
        update = getattr(module, "update_expert_bias", None)
        if callable(update):
            update()
    if not losses:
        return None
    return float(next(iter(losses[0].values())))


def run(args, start: int, resume_from=None) -> List[float]:
    """Steps `start`..`args.steps`, optionally restored from a checkpoint first."""
    from megatron.core import parallel_state

    from kimi_k3.optim.resume import load_optimizer_state_dict
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, mock_batch

    model, p = build(torch.distributed.get_world_size(), args.dropout)
    ddp, optimizer = build_optimizer(model, optimizer=args.optimizer, lr=1e-4, bf16=False)

    if resume_from is not None:
        model.load_state_dict(resume_from["model"], strict=False)
        if not args.drop_optimizer_state:
            load_optimizer_state_dict(optimizer, resume_from["optimizer"])
        if not args.drop_rng:
            load_rng(resume_from["rng"])

    seq, vocab = args.seq, p["model"]["vocab_size"]
    losses = []
    for index in range(start, args.steps):
        tokens, labels = mock_batch(vocab, seq, 1, seed=1000 + (0 if args.fixed_batch else index))
        losses.append(step(ddp, optimizer, model, tokens, labels, seq))
        if index + 1 == args.save_at and resume_from is None:
            torch.save(
                {
                    "model": {
                        k: v.cpu() for k, v in model.state_dict().items() if torch.is_tensor(v)
                    },
                    "optimizer": optimizer.state_dict(),
                    "rng": rng_state(),
                },
                f"{CKPT_PATH}.{torch.distributed.get_rank()}",
            )
    parallel_state.destroy_model_parallel()
    return losses


def main() -> None:
    from megatron.core import parallel_state

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--save-at", type=int, default=10)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="same tokens every step, so the loss descends and drift is visible")
    ap.add_argument("--drop-optimizer-state", action="store_true",
                    help="negative control: resume with a fresh optimizer")
    ap.add_argument("--drop-rng", action="store_true",
                    help="negative control: resume without restoring the RNG")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="turn dropout on so the RNG actually has a consumer")
    ap.add_argument("--json")
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=world
    )

    torch.manual_seed(0)
    continuous = run(args, start=0)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=world
    )
    torch.manual_seed(0)
    checkpoint = torch.load(f"{CKPT_PATH}.{rank}", map_location="cuda", weights_only=False)
    resumed = run(args, start=args.save_at, resume_from=checkpoint)

    result = {
        "rank": rank, "steps": args.steps, "save_at": args.save_at,
        "fixed_batch": args.fixed_batch, "dropout": args.dropout,
        "control": "drop_optimizer_state" if args.drop_optimizer_state
        else "drop_rng" if args.drop_rng else None,
    }
    if continuous[0] is None:  # not the last stage: no loss to compare
        result["has_loss"] = False
    else:
        tail = continuous[args.save_at :]
        deltas = [abs(a - b) for a, b in zip(tail, resumed)]
        result.update(
            has_loss=True,
            continuous_tail=tail,
            resumed=resumed,
            max_delta=max(deltas),
            bitwise=all(d == 0.0 for d in deltas),
        )
    print(json.dumps(result))
    if args.json and rank == world - 1:
        with open(args.json, "a") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
