"""Kimi K3 transformer config.

Derives from ``MLATransformerConfig`` because the gated-MLA layers read
``q_lora_rank`` / ``kv_lora_rank`` / head-dim fields from it.

The config is built exclusively by ``k3_config_builder.k3_config_from_args``.
It must never be routed through ``core_transformer_config_from_args``: at the
pinned SHA that function replaces any caller-supplied ``config_class`` with
``MLATransformerConfig`` whenever ``args.multi_latent_attention`` is set
(megatron/training/arguments.py:1230-1232), silently discarding this subclass.

Defaults below are the released Kimi-K3 values unless marked otherwise; see
kimi_k3/develop/architecture/01-kimi-k3-architecture-deep-dive.md for evidence.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class KimiK3TransformerConfig(MLATransformerConfig):
    """Configuration for Kimi K3 (text tower)."""

    # ---- KDA (linear-attention) layers -------------------------------------
    k3_kda_layers: Tuple[int, ...] = ()
    """1-indexed layer numbers that use KDA. Matches config.json
    linear_attn_config.kda_layers; a layer is KDA iff (layer_idx + 1) is in
    this tuple. Empty means 'derive from k3_kda_pattern'."""

    k3_kda_num_heads: int = 96
    k3_kda_head_dim: int = 128
    k3_kda_conv_size: int = 4
    k3_kda_gate_lower_bound: float = -5.0
    k3_kda_use_full_rank_gate: bool = True
    k3_kda_backend: str = "fla"
    """eager | fla. The eager FP32 oracle stays in tree permanently and remains
    selectable, but `fla` is the default since 2026-08-30: G15 ran at production
    geometry (H=96, K=128) at seq 1024/4096/8192 and the kernel agrees with the
    oracle to **6.4-6.8e-07** rel-L2 in fp32, and to 4.31e-03 in bf16 against a
    measured bf16 floor of 3.31e-03 -- the same shape as at the smaller geometries,
    with no growth in sequence length. Backward: worst gradient 1.5-2.3e-05 fp32,
    6.1e-03 bf16 (rule R5.3, `results/kda_parity.md`).

    The flip is not cosmetic. The oracle keeps the recurrent state at every
    timestep for autograd, so at seq 8192 it costs **109 GiB per layer** against
    fla's **10.2** -- the difference between the 93 L model fitting on 28 nodes and
    not fitting at all (`results/scaleout_93l.md`)."""

    # ---- Gated MLA layers ---------------------------------------------------
    k3_mla_use_nope: bool = True
    """The 64 'rope' dims exist but are never rotated; k_rot is produced once
    MQA-style and expanded across heads."""

    k3_mla_use_output_gate: bool = True
    k3_mla_lora_norm_eps: float = 1e-6
    """q_a_layernorm / kv_a_layernorm are constructed WITHOUT an eps argument
    in the release, so they take KimiRMSNorm's class default of 1e-6 -- not
    rms_norm_eps (1e-5). Verified in modeling_kimi_linear.py:227."""

    k3_mla_fp32_attn_output: bool = True
    """[report] attention output is kept in fp32 during training."""

    k3_max_logit_chunk: int = 1024
    """Query-block size when recomputing the max attention logit for QK-clip.
    Purely a memory knob -- the statistic is identical at any value (P9/G36)."""

    # ---- Attention residuals ------------------------------------------------
    k3_attn_res_block_size: int = 12
    """A block-residual slot is appended by every layer whose 0-indexed
    position satisfies layer_idx % block_size == 0. For 93 layers: 8 slots."""

    k3_attn_res_fp32: bool = True
    k3_attn_res_fused: bool = False
    """Chunked mixer (P11). Off by default: the eager path is the oracle, and this
    turns on once G44 records a measured win (R5.3). Parity is G43."""

    k3_attn_res_chunk: int = 4096
    """Rows per chunk in the fused mixer. A memory knob only -- the forward is
    bit-identical at any value."""

    # ---- MoE ----------------------------------------------------------------
    k3_routed_expert_hidden_size: int = 3584
    """Latent width the routed experts run at; mirrored into moe_latent_size."""

    k3_first_k_dense_replace: int = 1
    """Leading layers that use a dense FFN (`ffn_hidden_size`) instead of MoE.
    The release sets 1: layer 1 is KDA + dense FFN at intermediate 33792."""

    k3_latent_moe_use_norm: bool = True
    """RMSNorm(latent) applied to the combined expert output BEFORE the
    up-projection. Core's MoELayer.postprocess has no such norm, so K3MoELayer
    overrides it (moe/moe_layer.py:505-515)."""

    k3_situ_beta: float = 4.0
    k3_situ_linear_beta: float = 25.0
    k3_router_quantile_balancing: bool = True
    k3_qb_num_bins: int = 1024

    # ---- QAT ----------------------------------------------------------------
    k3_qat_experts: bool = False
    k3_qat_stochastic_rounding: bool = False

    # ---- Derived ------------------------------------------------------------
    k3_kda_pattern: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    """Optional (kda_stride, total) shorthand used by the tiny preset."""

    def __post_init__(self):
        super().__post_init__()

        if self.k3_routed_expert_hidden_size and self.num_moe_experts:
            # Core owns the 7168<->3584 projections through this field.
            self.moe_latent_size = self.k3_routed_expert_hidden_size

        if not self.k3_kda_layers and self.k3_kda_pattern:
            stride, total = self.k3_kda_pattern
            self.k3_kda_layers = tuple(
                n for n in range(1, total + 1) if n % stride != 0
            )

        if self.k3_kda_layers:
            bad = [n for n in self.k3_kda_layers if not 1 <= n <= self.num_layers]
            assert not bad, f"k3_kda_layers out of range for {self.num_layers} layers: {bad}"

        assert self.k3_kda_backend in ("eager", "fla"), (
            f"unknown KDA backend {self.k3_kda_backend!r}; expected 'eager' or 'fla'"
        )
        assert self.k3_attn_res_block_size >= 1

    # ---- Layer-kind helpers -------------------------------------------------

    def is_kda_layer(self, layer_idx: int) -> bool:
        """layer_idx is 0-indexed; the released lists are 1-indexed."""
        return (layer_idx + 1) in self.k3_kda_layers

    def appends_attn_res_slot(self, layer_idx: int) -> bool:
        """0-indexed. True for layers 0, 12, 24, ... at the official block size."""
        return layer_idx % self.k3_attn_res_block_size == 0

    def attn_res_slots_before(self, layer_idx: int) -> int:
        """Slots visible on entry to 0-indexed ``layer_idx`` (= ceil(l / block))."""
        block = self.k3_attn_res_block_size
        return -(-layer_idx // block)
