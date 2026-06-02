# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Layer specs for DeepSeek V4.

Unlike deepseekv2/layer_specs.py, V4's custom modules (HCASelfAttention,
DeepSeekV4TransformerLayer, mHC) don't plug into the ModuleSpec system —
they're wired directly in transformer_layer.py.

This file exposes a single factory function for building the full model,
and a smoke-test helper for verifying shapes without a cluster.
"""
from .config import DeepSeekV4Config
from .model import DeepSeekV4Model


def build_model(config: DeepSeekV4Config, **kwargs) -> DeepSeekV4Model:
    """Construct a DeepSeekV4Model from config.

    All parallelism (TP/PP/EP) must already be initialised by the caller via
    megatron.core.parallel_state before calling this.
    """
    return DeepSeekV4Model(config, **kwargs)


def get_v4_smoke_test_config() -> DeepSeekV4Config:
    """Tiny config for shape/forward-pass smoke tests (CPU-runnable)."""
    return DeepSeekV4Config(
        # Core transformer dims (shrunk for test).
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=512,
        # MLA dims.
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        # HCA.
        compress_ratios=(128,),
        attn_sliding_window=4,
        # mHC.
        hc_mult=4,
        hc_sinkhorn_iters=5,
        # MoE (tiny).
        num_moe_experts=4,
        moe_router_topk=2,
        moe_intermediate_size=128,
        swiglu_limit=10.0,
        num_hash_layers=1,
        # Grouped O.
        o_groups=2,
        # Vocab.
        vocab_size=512,
        padded_vocab_size=512,
        max_position_embeddings=64,
        # Norms.
        layernorm_epsilon=1e-5,
        # BF16.
        bf16=True,
        params_dtype=__import__("torch").bfloat16,
    )
