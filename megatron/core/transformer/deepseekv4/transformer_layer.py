# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""DeepSeek V4 Transformer Layer.

Wires together:
  - mHC (HyperConnection) residual streams
  - HCASelfAttention (Hyper-Compressed Attention)
  - RMSNorm before attention and FFN
  - DeepSeekV4Router + SwiGLUExpert (MoE) or dense FFN for non-MoE layers
  - Shared expert (always-active dense FFN added to MoE output)

Layer numbering: layer_number is 1-based (Megatron convention).
moe_layer_idx is 0-based count among MoE layers (for hash router).
"""
import torch
import torch.nn as nn
from typing import Optional

from megatron.legacy.model.rms_norm import RMSNorm

from .hca_attention import HCASelfAttention
from .hyper_connection import build_hyper_connection
from .v4_router import DeepSeekV4Router
from .swiglu_expert import SwiGLUExpert


class DeepSeekV4MoELayer(nn.Module):
    """Token-dispatching MoE layer using DeepSeekV4Router + SwiGLUExpert.

    Shared expert is always active and its output added after routing.
    This uses simple sequential dispatch (no grouped GEMM) for functional correctness.
    """

    def __init__(self, config, moe_layer_idx: int):
        super().__init__()
        self.router = DeepSeekV4Router(config, moe_layer_idx)
        self.experts = nn.ModuleList([SwiGLUExpert(config) for _ in range(config.num_moe_experts)])
        self.shared_expert = SwiGLUExpert(config)
        self.n_experts = config.num_moe_experts
        self.topk = config.moe_router_topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (s, b, d) -> (s, b, d)"""
        s, b, d = x.shape
        x_2d = x.view(s * b, d)

        indices, weights = self.router(x_2d)   # (tokens, topk)
        tokens = x_2d.shape[0]

        # Sequential dispatch: iterate over selected experts.
        out = torch.zeros_like(x_2d)
        for k in range(self.topk):
            expert_ids = indices[:, k]    # (tokens,)
            w = weights[:, k]             # (tokens,)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                expert_in = x_2d[mask]
                expert_out = self.experts[eid](expert_in)
                out[mask] += w[mask].unsqueeze(-1) * expert_out

        # Update bias outside gradient (for learned router only).
        if hasattr(self.router.router, "update_bias"):
            self.router.router.update_bias(indices.detach())

        # Shared expert (always active).
        out += self.shared_expert(x_2d)

        return out.view(s, b, d)


class DeepSeekV4TransformerLayer(nn.Module):
    """One V4 transformer layer with mHC + HCA + MoE.

    Args:
        config: DeepSeekV4Config
        layer_number: 1-based layer index (Megatron convention)
        moe_layer_idx: 0-based index among MoE layers (None for dense layers)
    """

    def __init__(self, config, layer_number: int, moe_layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_number = layer_number
        self.hc_mult = config.hc_mult

        # mHC wrappers: one per sub-layer (attention + FFN).
        self.hc_attn = build_hyper_connection(config, layer_number)
        self.hc_ffn = build_hyper_connection(config, layer_number)

        # Norms.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # Attention.
        self.self_attention = HCASelfAttention(config, layer_number)

        # FFN: MoE or dense (dense uses SwiGLU with moe_intermediate_size=ffn_hidden_size).
        if moe_layer_idx is not None:
            self.mlp = DeepSeekV4MoELayer(config, moe_layer_idx)
        else:
            self.mlp = SwiGLUExpert(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        """
        Args:
            x: (s, b, hc_mult, d) if hc_mult > 1, else (s, b, d)

        Returns:
            x: same shape as input
        """
        # Ensure x has stream dimension.
        has_stream = x.dim() == 4
        if not has_stream:
            x = x.unsqueeze(2)  # (s, b, 1, d)

        # Transpose to (b, s, N, d) for HyperConnection.
        x = x.permute(1, 0, 2, 3)  # (b, s, N, d)

        # --- Attention sub-layer ---
        y, state = self.hc_attn.pre(x)            # y: (b, s, d)
        # Norm + attention (Megatron expects (s, b, d)).
        y_t = y.transpose(0, 1)                    # (s, b, d)
        y_t = self.input_layernorm(y_t)
        attn_out = self.self_attention(y_t, attention_mask, inference_params)  # (s, b, d)
        attn_out = attn_out.transpose(0, 1)        # (b, s, d)
        x = self.hc_attn.post(attn_out, state)    # (b, s, N, d)

        # --- FFN sub-layer ---
        y, state = self.hc_ffn.pre(x)             # y: (b, s, d)
        y_t = y.transpose(0, 1)                    # (s, b, d)
        y_t = self.pre_mlp_layernorm(y_t)
        ffn_out = self.mlp(y_t)                    # (s, b, d)
        ffn_out = ffn_out.transpose(0, 1)          # (b, s, d)
        x = self.hc_ffn.post(ffn_out, state)      # (b, s, N, d)

        # Transpose back to Megatron convention.
        x = x.permute(1, 0, 2, 3)  # (s, b, N, d)
        if not has_stream or self.hc_mult == 1:
            x = x.squeeze(2)       # (s, b, d)

        return x
