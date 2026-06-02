# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Builder for DeepSeek-V4 model.

Bypasses core_transformer_config_from_args because V4-specific fields
(hc_mult, compress_ratios, num_hash_layers, etc.) have no registered CLI args.
Config is built directly from a dict that mirrors the training script variables.
"""
import torch
from megatron.training import get_args, print_rank_0
from megatron.core.models.deepseekv4 import DeepSeekV4Model
from megatron.core.models.deepseekv4.config import DeepSeekV4Config


# V4-specific defaults; overridden by values passed from the training script.
_V4_DEFAULTS = dict(
    # mHC
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    # HCA compressor
    compress_ratios=(128,),
    compress_rope_theta=160000.0,
    attn_sliding_window=128,
    attn_sink=False,
    # Indexer (CSA path — disabled for HCA)
    index_topk=0,
    index_head_dim=128,
    index_n_heads=64,
    # Grouped output projection
    o_groups=8,
    o_lora_rank=0,
    # Hash routing
    num_hash_layers=3,
    hash_routing_seed=0,
    # MoE gating
    swiglu_limit=10.0,
)


def build_deepseekv4_config(overrides: dict) -> DeepSeekV4Config:
    """Construct DeepSeekV4Config from standard Megatron args + V4 overrides.

    Args:
        overrides: dict of V4-specific fields (and any standard field overrides).

    Returns:
        DeepSeekV4Config ready for model construction.
    """
    args = get_args()

    # --- Standard fields from args ---
    cfg = dict(
        # Transformer core
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        max_position_embeddings=args.max_position_embeddings,
        layernorm_epsilon=args.norm_epsilon,
        # MLA
        multi_latent_attention=True,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_head_dim=args.qk_head_dim,
        qk_pos_emb_head_dim=args.qk_pos_emb_head_dim,
        v_head_dim=args.v_head_dim,
        # MoE
        num_moe_experts=args.num_experts,
        moe_router_topk=args.moe_router_topk,
        moe_intermediate_size=args.moe_ffn_hidden_size,
        # Parallelism
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
        # Precision
        bf16=args.bf16,
        fp16=args.fp16,
        params_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        # Init
        init_method_std=args.init_method_std,
        # Dropout
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        # Rotary
        rotary_base=args.rotary_base,
        # Vocab
        vocab_size=args.padded_vocab_size,
        padded_vocab_size=args.padded_vocab_size,
        # Gradient
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        # Recompute
        recompute_granularity=args.recompute_granularity,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
    )

    # Merge V4 defaults, then caller overrides.
    cfg.update(_V4_DEFAULTS)
    cfg.update(overrides)

    return DeepSeekV4Config(**cfg)


def deepseekv4_model_builder(
    v4_config_overrides: dict,
    args,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    config=None,
    pg_collection=None,
) -> DeepSeekV4Model:
    """Model builder compatible with model_provider() signature.

    Pass this via functools.partial with v4_config_overrides bound.
    """
    print_rank_0('building DeepSeek-V4 model ...')

    if config is None:
        config = build_deepseekv4_config(v4_config_overrides)

    model = DeepSeekV4Model(
        config=config,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    )
    return model
