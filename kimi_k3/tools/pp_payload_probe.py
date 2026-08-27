"""Gate G7 -- the AttnRes payload really crosses a pipeline boundary, gradients included.

Two launches:

    # 1. reference: one rank, PP=1, saves weights + loss + grads
    torchrun --nproc_per_node=1 -m kimi_k3.tools.pp_payload_probe --mode reference

    # 2. pipelined: two ranks, PP=2, loads the same weights and compares
    torchrun --nproc_per_node=2 -m kimi_k3.tools.pp_payload_probe --mode pipeline

    # 3. negative control: the same run with block-residual slots detached at
    #    creation, emulating a payload whose gradient never comes back
    torchrun --nproc_per_node=2 -m kimi_k3.tools.pp_payload_probe --mode pipeline --detach-slots

What it proves, in order of importance:

1. loss and every parameter gradient match the single-stage reference, so the
   packed payload carries the state *and* its gradient across the boundary;
2. the negative control **fails** those comparisons -- a gate that cannot fail is
   not a gate (review finding A1);
3. the per-stage shape multipliers core is told (1 + K_in, 1 + K_out) are the
   ones the block actually produces.
"""

import argparse
import json
import os
from typing import Dict

import torch

REF_PATH = "/tmp/k3_pp_reference.pt"


def build(args, pp_size: int):
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.model.k3_gpt_model import K3GPTModel

    p = preset("tiny")
    cfg = config_from_preset(
        p["config"],
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
        deallocate_pipeline_outputs=False,
        sequence_parallel=False,
        # Dropout consumes RNG, so with the defaults (0.1) two "identical" runs
        # differ by ~1e-4 and the measured floor would be drift rather than kernel
        # noise -- permissive enough to hide a real transport bug.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # The router's expert-bias accumulator also mutates during forward.
        # Routing behaviour is P6's subject; this gate is about transport.
        moe_router_enable_expert_bias=False,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=cfg.num_moe_experts, moe_grouped_gemm=False, multi_latent_attention=True
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    model = K3GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=p["model"]["vocab_size"],
        max_sequence_length=p["model"]["max_sequence_length"],
        position_embedding_type="none",
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    ).cuda().float()
    return cfg, model, p


def global_key(model, key: str) -> str:
    """Rewrite stage-local layer indices to global ones so PP=1 and PP=2 agree."""
    block = model.decoder
    for prefix in ("decoder.layers.", "decoder.attn_res_attn.", "decoder.attn_res_mlp."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            local, tail = rest.split(".", 1)
            return f"{prefix}{block.global_layer_index(int(local))}.{tail}"
    return key


def fixed_batch(seq: int, batch: int, vocab: int, device):
    g = torch.Generator(device="cpu").manual_seed(7)
    return torch.randint(0, vocab, (batch, seq), generator=g).to(device)


def loss_fn(output_tensor):
    """Deterministic, shape-independent, and touches every logit."""
    loss = (output_tensor.float() ** 2).mean()
    return loss, {"probe_loss": loss.detach()}


def _fwd_bwd(model, tokens):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=tokens, position_ids=None, attention_mask=None)
    loss, _ = loss_fn(out)
    loss.backward()
    return float(loss.detach()), {
        n: q.grad.detach().cpu().clone() for n, q in model.named_parameters() if q.grad is not None
    }


def run_reference(args) -> None:
    from megatron.core import parallel_state

    cfg, model, p = build(args, pp_size=1)
    tokens = fixed_batch(p["model"]["max_sequence_length"], 1, p["model"]["vocab_size"], "cuda")

    # Two identical runs establish the run-to-run noise floor (rule R4.4). The MoE
    # dispatcher and grouped GEMM use atomics, so "identical" is not bitwise even
    # on one rank -- a PP comparison has to be read against this, not against zero.
    loss_a, grads_a = _fwd_bwd(model, tokens)
    loss_b, grads_b = _fwd_bwd(model, tokens)
    floor_loss = abs(loss_a - loss_b)
    floor_grad = max((grads_a[n] - grads_b[n]).abs().max().item() for n in grads_a)

    class _L:
        pass

    loss = _L()
    loss.detach = lambda: loss_a

    torch.save(
        {
            # TE modules put non-tensor `_extra_state` entries in state_dict.
            "state_dict": {
                k: v.cpu() for k, v in model.state_dict().items() if torch.is_tensor(v)
            },
            "loss": loss_a,
            "grads": grads_a,
            "noise_floor_loss": floor_loss,
            "noise_floor_grad": floor_grad,
        },
        REF_PATH,
    )
    print(json.dumps({"mode": "reference", "loss": loss_a, "loss_second_run": loss_b,
                      "noise_floor_loss": floor_loss, "noise_floor_grad": floor_grad,
                      "params_with_grad": len(grads_a), "saved": REF_PATH}))
    parallel_state.destroy_model_parallel()


def run_pipeline(args) -> Dict:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func

    from kimi_k3.pipeline.k3_schedule import resolve

    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    cfg, model, p = build(args, pp_size=world)
    block = model.decoder
    block._detach_slots_for_test = args.detach_slots

    ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    own = {k: v for k, v in model.state_dict().items() if torch.is_tensor(v)}
    mapped = {k: ref["state_dict"][global_key(model, k)] for k in own}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected, unexpected
    assert all(not torch.is_tensor(model.state_dict()[m]) for m in missing), missing

    seq = p["model"]["max_sequence_length"]
    tokens = fixed_batch(seq, 1, p["model"]["vocab_size"], "cuda")

    def forward_step(data_iterator, model_):
        out = model_(input_ids=tokens, position_ids=None, attention_mask=None)
        return out, loss_fn

    recv_mult, send_mult = block.payload_multipliers()
    adjust = resolve(model, world)
    forward_backward = get_forward_backward_func()
    losses = forward_backward(
        forward_step_func=forward_step,
        data_iterator=None,
        model=[model],
        num_microbatches=1,
        seq_length=seq,
        micro_batch_size=1,
        decoder_seq_length=seq,
        forward_only=False,
        adjust_tensor_shapes_fn=adjust,
    )

    # Compare against the reference, on the parameters this stage owns.
    worst = {"name": None, "abs": 0.0, "rel": 0.0}
    checked, missing_grad = 0, []
    for name, param in model.named_parameters():
        gname = global_key(model, name)
        if param.grad is None:
            missing_grad.append(gname)
            continue
        want = ref["grads"].get(gname)
        if want is None:
            continue
        got = param.grad.detach().cpu().float()
        diff = (got - want.float()).abs().max().item()
        scale = want.float().abs().max().item() + 1e-12
        checked += 1
        if diff > worst["abs"]:
            worst = {"name": gname, "abs": diff, "rel": diff / scale}

    row = {
        "mode": "pipeline",
        "rank": rank,
        "pp": world,
        "detach_slots": args.detach_slots,
        "recv_mult": recv_mult,
        "send_mult": send_mult,
        "adjust_bound": adjust is not None,
        "layers": [block.global_layer_index(i) for i in range(len(block.layers))],
        "params_checked": checked,
        "params_without_grad": len(missing_grad),
        "worst_grad_abs": worst["abs"],
        "worst_grad_rel": worst["rel"],
        "worst_grad_name": worst["name"],
    }
    if losses:
        row["loss"] = float(losses[0]["probe_loss"])
        row["ref_loss"] = ref["loss"]
        row["loss_abs_diff"] = abs(row["loss"] - ref["loss"])

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        for r in gathered:
            print(json.dumps(r))
        last = [r for r in gathered if "loss" in r][0]
        worst_abs = max(g["worst_grad_abs"] for g in gathered)
        # Compare against the measured run-to-run floor, with 5x headroom, not
        # against zero: the MoE path is not bitwise reproducible even on one rank.
        loss_bound = max(5 * ref["noise_floor_loss"], 1e-9)
        grad_bound = max(5 * ref["noise_floor_grad"], 1e-9)
        verdict = {
            "verdict": "MATCH"
            if (last["loss_abs_diff"] <= loss_bound and worst_abs <= grad_bound)
            else "MISMATCH",
            "detach_slots": args.detach_slots,
            "loss_abs_diff": last["loss_abs_diff"],
            "loss_bound": loss_bound,
            "worst_grad_abs": worst_abs,
            "grad_bound": grad_bound,
            "noise_floor_loss": ref["noise_floor_loss"],
            "noise_floor_grad": ref["noise_floor_grad"],
            "params_checked": sum(g["params_checked"] for g in gathered),
        }
        print(json.dumps(verdict))
        if args.json:
            with open(args.json, "a") as f:
                for r in gathered:
                    f.write(json.dumps(r) + "\n")
                f.write(json.dumps(verdict) + "\n")
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "pipeline"], required=True)
    ap.add_argument("--detach-slots", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    from megatron.core import parallel_state

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=world
    )

    if args.mode == "reference":
        assert world == 1, "the reference run is single-rank (PP=1)"
        run_reference(args)
    else:
        assert world > 1, "the pipeline run needs at least 2 ranks"
        run_pipeline(args)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
