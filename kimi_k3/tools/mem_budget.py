"""Analytic parameter and memory model for Kimi K3.

This is the oracle for the parameter-count gate (G13) and the input to the
capacity tables in develop/plan-0/06-capacity-and-parallelism.md. Every formula
below was checked against tensor shapes read out of the released safetensors
headers (develop/notes/2026-08-27-release-audit.md), not inferred from the paper.

Run directly for the 93 L breakdown::

    python -m kimi_k3.tools.mem_budget
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ParamBreakdown:
    kda: int = 0
    mla: int = 0
    moe: int = 0
    dense_ffn: int = 0
    layer_norms: int = 0
    embedding: int = 0

    @property
    def total(self) -> int:
        return self.kda + self.mla + self.moe + self.dense_ffn + self.layer_norms + self.embedding

    def as_dict(self) -> Dict[str, int]:
        d = {
            "kda_layers": self.kda,
            "mla_layers": self.mla,
            "moe_layers": self.moe,
            "dense_ffn": self.dense_ffn,
            "norms_and_attn_res": self.layer_norms,
            "embedding_and_head": self.embedding,
        }
        d["total"] = self.total
        return d


def kda_layer_params(cfg) -> int:
    """q/k/v/g/o projections dominate; the rest is gate low-rank + conv + scalars."""
    h = cfg.hidden_size
    hd = cfg.k3_kda_head_dim
    nh = cfg.k3_kda_num_heads
    p = nh * hd
    return (
        5 * h * p  # q_proj, k_proj, v_proj, g_proj (full-rank output gate), o_proj
        + h * hd  # f_a_proj  (low-rank decay gate)
        + hd * p  # f_b_proj
        + h * nh  # b_proj -> beta
        + 3 * p * cfg.k3_kda_conv_size  # q/k/v short convolutions
        + nh  # A_log   (checkpoint stores [128] zero-padded; we keep [96])
        + p  # dt_bias
        + hd  # o_norm
    )


def mla_layer_params(cfg) -> int:
    h = cfg.hidden_size
    nh = cfg.num_attention_heads
    q_head_dim = cfg.qk_head_dim + cfg.qk_pos_emb_head_dim
    return (
        h * cfg.q_lora_rank
        + cfg.q_lora_rank  # q_a_layernorm
        + cfg.q_lora_rank * nh * q_head_dim
        + h * (cfg.kv_lora_rank + cfg.qk_pos_emb_head_dim)
        + cfg.kv_lora_rank  # kv_a_layernorm
        + cfg.kv_lora_rank * nh * (cfg.qk_head_dim + cfg.v_head_dim)
        + nh * cfg.v_head_dim * h  # o_proj
        + h * nh * cfg.v_head_dim  # g_proj (full-rank sigmoid output gate)
    )


def moe_layer_params(cfg) -> int:
    """Routed experts run at the latent width; shared experts run at hidden."""
    h = cfg.hidden_size
    latent = cfg.k3_routed_expert_hidden_size
    inter = cfg.moe_ffn_hidden_size
    e = cfg.num_moe_experts
    shared = cfg.moe_shared_expert_intermediate_size or 0
    return (
        e * 3 * latent * inter  # routed experts (w1, w2, w3)
        + 3 * h * shared  # shared experts, on hidden
        + h * latent  # routed_expert_down_proj
        + latent * h  # routed_expert_up_proj
        + latent  # routed_expert_norm
        + e * h  # router gate
        + e  # e_score_correction_bias
    )


def dense_ffn_params(cfg) -> int:
    return 3 * cfg.hidden_size * cfg.ffn_hidden_size


def per_layer_norm_params(cfg) -> int:
    """input_layernorm + post_attention_layernorm + 2 AttnRes norms + 2 AttnRes projections."""
    return 6 * cfg.hidden_size


def model_level_params(cfg, vocab_size: int, tied: bool = False) -> int:
    h = cfg.hidden_size
    embed = vocab_size * h * (1 if tied else 2)
    return embed + h + 2 * h  # final norm + output AttnRes norm/proj


def breakdown(cfg, vocab_size: int, tied: bool = False) -> ParamBreakdown:
    b = ParamBreakdown()
    for layer_idx in range(cfg.num_layers):
        if cfg.is_kda_layer(layer_idx):
            b.kda += kda_layer_params(cfg)
        else:
            b.mla += mla_layer_params(cfg)
        if layer_idx == 0:
            b.dense_ffn += dense_ffn_params(cfg)
        else:
            b.moe += moe_layer_params(cfg)
        b.layer_norms += per_layer_norm_params(cfg)
    b.embedding = model_level_params(cfg, vocab_size, tied)
    return b


def active_params(cfg, vocab_size: int) -> int:
    """Parameters touched per token (top-k of the routed experts only)."""
    b = breakdown(cfg, vocab_size)
    latent = cfg.k3_routed_expert_hidden_size
    inter = cfg.moe_ffn_hidden_size
    routed_per_layer = cfg.num_moe_experts * 3 * latent * inter
    active_per_layer = cfg.moe_router_topk * 3 * latent * inter
    moe_layers = max(cfg.num_layers - 1, 0)
    inactive = moe_layers * (routed_per_layer - active_per_layer)
    return b.total - inactive - vocab_size * cfg.hidden_size  # embedding lookup is not a matmul


# --- optimizer memory -------------------------------------------------------
# Bytes per parameter held on one GPU, for the parameters resident on that GPU.
# dist_muon shards master weights + momentum across DP via
# LayerWiseDistributedOptimizer; plain muon does not. Measured values replace
# these in develop/results/opt_mem.md (gate G5).

OPTIMIZER_BYTES_PER_PARAM = {
    "adam": lambda dp: 2 + 4 + 4 + 8,
    "adam_dist": lambda dp: 2 + 4 + 12 / dp,
    "muon": lambda dp: 2 + 4 + 4 + 4,
    "dist_muon": lambda dp: 2 + 4 + 8 / dp,
}


def params_per_gpu(cfg, vocab_size: int, tp: int = 1, pp: int = 1, ep: int = 1) -> float:
    """Expert parallelism shards the routed experts; TP/PP shard everything."""
    b = breakdown(cfg, vocab_size)
    latent = cfg.k3_routed_expert_hidden_size
    routed = max(cfg.num_layers - 1, 0) * cfg.num_moe_experts * 3 * latent * cfg.moe_ffn_hidden_size
    non_expert = b.total - routed
    return non_expert / (tp * pp) + routed / (tp * pp * ep)


def _fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:,.3f} {unit}"
    return f"{n:,.0f}"


def main() -> None:
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    for name in ("tiny", "4L", "8L", "93L"):
        p = preset(name)
        cfg = config_from_preset(p["config"])
        vocab = p["model"]["vocab_size"]
        b = breakdown(cfg, vocab)
        print(f"\n=== preset {name}  ({cfg.num_layers} layers)")
        for k, v in b.as_dict().items():
            print(f"    {k:24s} {_fmt(v):>14s}")
        print(f"    {'active / token':24s} {_fmt(active_params(cfg, vocab)):>14s}")


if __name__ == "__main__":
    main()
