"""Kimi K3 presets.

``tiny`` is the default test geometry (rule R4.2). The official presets are for
meta-device construction and analytic parameter counting only (R4.3) -- 8L
official is ~215 B parameters and 93L is ~2.78 T; never instantiate them with
real weights in a test.

Official values are the released config.json (see
kimi_k3/develop/architecture/01-kimi-k3-architecture-deep-dive.md §2).
"""

from typing import Dict

OFFICIAL_LAYERS = 93
OFFICIAL_FULL_ATTN_1IDX = tuple(list(range(4, 93, 4)) + [93])
"""1-indexed gated-MLA layers, verbatim from config.json: 4, 8, ..., 88, 92, 93.
Note the tail breaks the 3:1 stride once -- 92 and 93 are both MLA."""


def kda_layers_1idx(num_layers: int = OFFICIAL_LAYERS) -> tuple:
    """Complement of the full-attention list, 1-indexed (69 entries at 93L)."""
    full = set(OFFICIAL_FULL_ATTN_1IDX)
    return tuple(n for n in range(1, num_layers + 1) if n not in full)


def _official_config(num_layers: int) -> dict:
    return dict(
        num_layers=num_layers,
        hidden_size=7168,
        num_attention_heads=96,
        num_query_groups=96,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        gated_linear_unit=True,
        # dense FFN of layer 1 (first_k_dense_replace = 1)
        ffn_hidden_size=33792,
        # MLA
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        # MoE
        num_moe_experts=896,
        moe_router_topk=16,
        moe_ffn_hidden_size=3072,
        moe_shared_expert_intermediate_size=6144,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=1.0,
        moe_grouped_gemm=True,
        # K3
        k3_kda_layers=kda_layers_1idx(num_layers),
        k3_kda_num_heads=96,
        k3_kda_head_dim=128,
        k3_attn_res_block_size=12,
        k3_routed_expert_hidden_size=3584,
    )


PRESETS: Dict[str, dict] = {
    "tiny": {
        "config": dict(
            num_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_query_groups=4,
            normalization="RMSNorm",
            layernorm_epsilon=1e-5,
            add_bias_linear=False,
            gated_linear_unit=True,
            # Megatron defaults both to 0.1. With dropout on, two identical
            # forwards differ by ~1e-4 and every determinism check becomes a
            # measurement of the RNG instead (see the G7 record).
            hidden_dropout=0.0,
            attention_dropout=0.0,
            ffn_hidden_size=512,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=256,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_grouped_gemm=False,
            k3_kda_layers=(1, 2, 3),  # true [K, K, K, M] pattern
            k3_kda_num_heads=2,
            k3_kda_head_dim=64,
            k3_attn_res_block_size=2,
            k3_routed_expert_hidden_size=256,
        ),
        "model": dict(vocab_size=4096, max_sequence_length=128),
    },
    "4L": {"config": _official_config(4), "model": dict(vocab_size=163840, max_sequence_length=8192)},
    "8L": {"config": _official_config(8), "model": dict(vocab_size=163840, max_sequence_length=8192)},
    "93L": {
        "config": _official_config(93),
        "model": dict(vocab_size=163840, max_sequence_length=8192),
    },
}


def preset(name: str) -> dict:
    assert name in PRESETS, f"unknown preset {name!r}; have {sorted(PRESETS)}"
    return PRESETS[name]
