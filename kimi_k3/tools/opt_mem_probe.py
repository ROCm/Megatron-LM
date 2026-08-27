"""Gate G5 -- measure optimizer memory per parameter, per group, against DP.

The incoming plan asserted a flat 14 B/param for Muon because
``--use-distributed-optimizer`` is rejected for every Muon variant
(arguments.py:1552). But ``--optimizer dist_muon`` has its *own* sharding:
``LayerWiseDistributedOptimizer`` assigns whole tensors to DP ranks and keeps
only that rank's master weights and momentum (layer_wise_optimizer.py:105-158).
This probe measures what is actually resident.

Run (one recipe per launch, world size = DP)::

    torchrun --nproc_per_node=8 -m kimi_k3.tools.opt_mem_probe --optimizer dist_muon
    torchrun --nproc_per_node=4 -m kimi_k3.tools.opt_mem_probe --optimizer adam --dist-opt

Reported per rank and reduced across ranks:

* ``param``      -- model parameters resident on this rank (bf16 weights)
* ``grad``       -- DDP gradient buffers
* ``opt_state``  -- everything the optimizer holds: fp32 masters, momentum,
                    Adam moments, split into the Muon (2-D) and scalar groups
* ``peak``       -- ``torch.cuda.max_memory_allocated`` after two steps
"""

import argparse
import json
import os
from typing import Dict

import torch


def _mb(x) -> float:
    return x / 2**20


def build_probe_config(hidden: int, layers: int, experts: int):
    """A small model with K3's *shape mix*: MLA + MoE + norms, not just linears."""
    from kimi_k3.config.k3_config_builder import config_from_preset

    cfg = dict(
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=8,
        num_query_groups=8,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        gated_linear_unit=True,
        ffn_hidden_size=2 * hidden,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        num_moe_experts=experts,
        moe_router_topk=2,
        moe_ffn_hidden_size=hidden // 2,
        moe_shared_expert_intermediate_size=hidden,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_grouped_gemm=False,
        k3_kda_layers=tuple(n for n in range(1, layers + 1) if n % 4 != 0),
        k3_kda_num_heads=4,
        k3_kda_head_dim=64,
        k3_attn_res_block_size=4,
        k3_routed_expert_hidden_size=hidden // 2,
    )
    return config_from_preset(cfg, bf16=True, params_dtype=torch.bfloat16)


def optimizer_state_bytes(optimizer) -> Dict[str, float]:
    """Walk the (possibly chained) optimizer and total every tensor it owns.

    Counts fp32 master weights and per-parameter state (momentum / Adam moments)
    separately so the Muon 2-D group and the scalar group can be told apart.
    """
    seen = set()
    totals = {"master": 0, "state_2d": 0, "state_other": 0}

    def account(opt):
        inner = getattr(opt, "optimizer", None)
        # fp32 master weights held by the Float16 wrapper
        for attr in ("shard_fp32_from_float16_groups", "fp32_from_float16_groups"):
            for group in getattr(opt, attr, []) or []:
                for t in group:
                    if t is not None and id(t) not in seen:
                        seen.add(id(t))
                        totals["master"] += t.numel() * t.element_size()
        if inner is not None and hasattr(inner, "state"):
            for p, st in inner.state.items():
                bucket = "state_2d" if p.dim() == 2 else "state_other"
                for v in st.values():
                    if torch.is_tensor(v) and id(v) not in seen:
                        seen.add(id(v))
                        totals[bucket] += v.numel() * v.element_size()

    for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
        account(opt)
        for sub in getattr(opt, "chained_optimizers", []):
            account(sub)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="adam", choices=["adam", "muon", "dist_muon"])
    ap.add_argument("--dist-opt", action="store_true", help="--use-distributed-optimizer")
    ap.add_argument("--precision-aware", action="store_true")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.muon import get_megatron_muon_optimizer

    from kimi_k3.model.k3_gpt_model import K3GPTModel

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    cfg = build_probe_config(args.hidden, args.layers, args.experts)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=cfg.num_moe_experts, moe_grouped_gemm=False, multi_latent_attention=True
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    model = K3GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=4096,
        max_sequence_length=256,
        position_embedding_type="none",
    ).cuda().bfloat16()
    n_params = sum(p.numel() for p in model.parameters())
    after_model = torch.cuda.memory_allocated()

    ddp = DistributedDataParallel(
        cfg,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            # The distributed optimizer needs DDP to allocate param buffers; without
            # this the first step dies in _copy_main_params_to_model_params.
            use_distributed_optimizer=args.dist_opt,
        ),
        model,
    )
    after_ddp = torch.cuda.memory_allocated()

    opt_cfg = OptimizerConfig(
        optimizer=args.optimizer,
        lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=args.dist_opt,
        use_precision_aware_optimizer=args.precision_aware,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    if "muon" in args.optimizer:
        optimizer = get_megatron_muon_optimizer(
            opt_cfg, [ddp], layer_wise_distributed_optimizer="dist" in args.optimizer
        )
    else:
        optimizer = get_megatron_optimizer(opt_cfg, [ddp])
    after_opt = torch.cuda.memory_allocated()

    # Two steps so lazily-allocated state (Adam moments) actually materialises.
    for _ in range(2):
        for p in model.parameters():
            if p.requires_grad:
                if getattr(p, "main_grad", None) is not None:
                    p.main_grad.copy_(torch.randn_like(p.main_grad) * 1e-3)
                else:
                    p.grad = torch.randn_like(p) * 1e-3
        optimizer.step()
    after_step = torch.cuda.memory_allocated()

    state = optimizer_state_bytes(optimizer)
    row = {
        "rank": rank,
        "world": world,
        "optimizer": args.optimizer + ("+distopt" if args.dist_opt else ""),
        "precision_aware": args.precision_aware,
        "params": n_params,
        "param_mb": _mb(after_model - base),
        "grad_mb": _mb(after_ddp - after_model),
        "opt_build_mb": _mb(after_opt - after_ddp),
        "opt_total_mb": _mb(after_step - after_ddp),
        "master_mb": _mb(state["master"]),
        "state_2d_mb": _mb(state["state_2d"]),
        "state_other_mb": _mb(state["state_other"]),
        "peak_mb": _mb(torch.cuda.max_memory_allocated() - base),
    }
    row["bytes_per_param"] = (after_step - base) / n_params

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        for r in gathered:
            print(json.dumps(r))
        b = [r["bytes_per_param"] for r in gathered]
        print(
            json.dumps(
                {
                    "summary": row["optimizer"],
                    "world": world,
                    "bytes_per_param_min": round(min(b), 2),
                    "bytes_per_param_max": round(max(b), 2),
                    "bytes_per_param_mean": round(sum(b) / len(b), 2),
                    "params": row["params"],
                }
            )
        )
        if args.out:
            with open(args.out, "a") as f:
                for r in gathered:
                    f.write(json.dumps(r) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
