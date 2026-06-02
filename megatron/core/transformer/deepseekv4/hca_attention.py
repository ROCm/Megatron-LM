# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Hyper-Compressed Attention (HCA) for DeepSeek V4.

Architecture (per V4 paper §3.1):
  - Q from full hidden state via MLA down/up projection.
  - Compressed KV from Compressor (128-token pooling) via shared W_KV.
  - Local sliding-window attention (window=128) over uncompressed tokens.
  - Optional attention sink (first 1 token always attended).
  - DualRoPE: standard rope_theta for Q/full-K; compress_rope_theta for compressed-K positions.
  - No lightning indexer in HCA (compress_ratio=128 path).

The module plugs into MLASelfAttention's submodule system via the layer spec.
We inherit from torch.nn.Module directly (not from Megatron's Attention) to keep
it self-contained and avoid the MLA-specific gather_output=True TE restriction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from megatron.core.transformer.enums import AttnMaskType

from .compressor import Compressor
from .parallel_utils import column_parallel, row_parallel


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard rotary position embedding applied to last dimension."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def build_rope_cache(seq_len: int, head_dim: int, base: float, device) -> tuple:
    """Precompute (cos, sin) for RoPE at given base frequency."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)  # (seq_len, half)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, half)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


class LocalRMSNorm(nn.Module):
    """Parameter-free per-head RMS norm applied after q_up_proj (LocalRMSNorm in V4)."""

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_heads, head_dim)
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)


class HCASelfAttention(nn.Module):
    """Hyper-Compressed Attention.

    Inputs/outputs are (seq, batch, hidden) as per Megatron convention.
    """

    def __init__(self, config, layer_number: int, attn_mask_type=AttnMaskType.causal):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        H = config.num_attention_heads
        d = config.hidden_size
        q_lora_rank = config.q_lora_rank          # 512
        kv_lora_rank = config.kv_lora_rank        # 512
        qk_head_dim = config.qk_head_dim          # 128
        qk_pos_emb = config.qk_pos_emb_head_dim   # 64
        v_head_dim = config.v_head_dim             # 128

        self.n_heads = H
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb = qk_pos_emb
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_head_dim + qk_pos_emb  # 192

        # --- Q projection (MLA style) ---
        # Down: d -> q_lora_rank  (split across TP, no gather)
        self.linear_q_down = column_parallel(d, q_lora_rank, config)
        # Up: q_lora_rank -> H * (qk_head_dim + qk_pos_emb)
        # Must gather_output=True (TE cannot do this — use plain ColumnParallelLinear)
        self.linear_q_up = column_parallel(q_lora_rank, H * self.q_head_dim, config, gather_output=True)
        # Per-head RMS norm after up-projection (LocalRMSNorm).
        self.q_rms = LocalRMSNorm(self.q_head_dim)

        # --- KV projection (shared compressed KV) ---
        # Down: d -> kv_lora_rank + qk_pos_emb  (RoPE dims for K)
        self.linear_kv_down = column_parallel(d, kv_lora_rank + qk_pos_emb, config)
        # Up: kv_lora_rank -> H * (qk_head_dim + v_head_dim)
        self.linear_kv_up = column_parallel(kv_lora_rank, H * (qk_head_dim + v_head_dim), config, gather_output=True)

        # --- Output projection (grouped low-rank if o_groups > 1) ---
        o_groups = config.o_groups
        self.o_groups = o_groups
        if o_groups > 1:
            inter = config.o_lora_rank if config.o_lora_rank > 0 else d // o_groups
            self.linear_o_a = column_parallel(H * v_head_dim, o_groups * inter, config, gather_output=True)
            # wo_b: input_is_parallel=False not supported by TE — use plain RowParallelLinear
            self.linear_o_b = row_parallel(o_groups * inter, d, config, input_is_parallel=False)
        else:
            self.linear_o = row_parallel(H * v_head_dim, d, config, input_is_parallel=False)

        # --- Compressor ---
        self.compressor = Compressor(config)

        # --- Sliding window + sink config ---
        self.window_size = config.attn_sliding_window  # 128
        self.use_sink = config.attn_sink
        self.sink_size = config.attn_sink_size if config.attn_sink else 0

        # --- RoPE bases ---
        self.rope_theta = config.rotary_base          # standard base (10000 or user-set)
        self.compress_rope_theta = config.compress_rope_theta  # 160000 for V4

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (s, b, d) — Megatron seq-first convention
            attention_mask: ignored; we build our own causal + sliding-window mask

        Returns:
            output: (s, b, d)
        """
        s, b, d = hidden_states.shape
        device = hidden_states.device

        # Transpose to (b, s, d) for easier manipulation.
        h = hidden_states.transpose(0, 1)

        # --- Q projection ---
        q_down, _ = self.linear_q_down(hidden_states)   # (s, b, q_lora_rank/TP)
        q_down = q_down.transpose(0, 1)                 # (b, s, q_lora_rank/TP)
        q_up, _ = self.linear_q_up(q_down.transpose(0, 1))  # needs (s,b) input
        q_up = q_up.transpose(0, 1)                     # (b, s, H*q_head_dim)
        q_up = q_up.view(b, s, self.n_heads, self.q_head_dim)
        q_up = self.q_rms(q_up)

        # Split into non-RoPE and RoPE parts.
        q_nrope = q_up[..., :self.qk_head_dim]          # (b, s, H, qk_head_dim)
        q_rope = q_up[..., self.qk_head_dim:]            # (b, s, H, qk_pos_emb)

        # --- Compress hidden states -> compressed KV ---
        c = self.compressor(h)                           # (b, s_c, d)
        s_c = c.shape[1]

        # KV down on compressed tokens.
        c_t = c.transpose(0, 1)                          # (s_c, b, d)
        kv_down, _ = self.linear_kv_down(c_t)            # (s_c, b, kv_lora_rank+qk_pos_emb)
        kv_down = kv_down.transpose(0, 1)                # (b, s_c, kv_lora_rank+qk_pos_emb)
        kv_latent = kv_down[..., :self.config.kv_lora_rank]
        k_rope_c = kv_down[..., self.config.kv_lora_rank:]   # (b, s_c, qk_pos_emb)

        # KV up.
        kv_up, _ = self.linear_kv_up(kv_latent.transpose(0, 1))
        kv_up = kv_up.transpose(0, 1)                   # (b, s_c, H*(qk_head_dim+v_head_dim))
        kv_up = kv_up.view(b, s_c, self.n_heads, self.qk_head_dim + self.v_head_dim)
        k_nrope = kv_up[..., :self.qk_head_dim]          # (b, s_c, H, qk_head_dim)
        v = kv_up[..., self.qk_head_dim:]                # (b, s_c, H, v_head_dim)

        # --- DualRoPE ---
        # Standard RoPE for Q (full token positions 0..s-1).
        cos_q, sin_q = build_rope_cache(s, self.qk_pos_emb, self.rope_theta, device)
        q_rope = apply_rope(q_rope, cos_q[:, :, :s], sin_q[:, :, :s])

        # Compressed RoPE for K (compressed token positions 0..s_c-1).
        cos_kc, sin_kc = build_rope_cache(s_c, self.qk_pos_emb, self.compress_rope_theta, device)
        k_rope_c = apply_rope(k_rope_c, cos_kc[:, :, :s_c], sin_kc[:, :, :s_c])

        # Assemble full Q and K_compressed.
        Q = torch.cat([q_nrope, q_rope], dim=-1)         # (b, s, H, q_head_dim)
        K_c = torch.cat([k_nrope, k_rope_c.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        # (b, s_c, H, q_head_dim)

        # --- Sliding-window + sink attention over compressed KV ---
        # For simplicity in this functional prototype we compute full attention
        # then mask. Production path should use flash-attn with window support.
        scale = 1.0 / math.sqrt(self.q_head_dim)

        # Q: (b, H, s, q_head_dim), K_c: (b, H, s_c, q_head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K_c = K_c.permute(0, 2, 1, 3)
        V = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K_c.transpose(-2, -1)) * scale  # (b, H, s, s_c)

        # Build sliding-window causal mask on compressed indices.
        # Token i (full) attends to compressed token j if:
        #   j <= floor(i / compress_ratio)   (causal)
        #   j >= floor(i / compress_ratio) - window + 1  (window)
        # Sink tokens (j < sink_size) always attended.
        r = self.compressor.compress_ratio
        q_idx = torch.arange(s, device=device)           # (s,)
        k_idx = torch.arange(s_c, device=device)         # (s_c,)
        q_grp = q_idx // r                               # (s,) which compressed group
        causal_mask = k_idx.unsqueeze(0) <= q_grp.unsqueeze(1)       # (s, s_c)
        window_mask = k_idx.unsqueeze(0) >= (q_grp - self.window_size + 1).unsqueeze(1)  # (s, s_c)
        local_mask = causal_mask & window_mask
        if self.use_sink:
            sink_mask = k_idx.unsqueeze(0) < self.sink_size           # (1, s_c)
            local_mask = local_mask | sink_mask

        # Apply mask: (1, 1, s, s_c)
        mask_val = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), mask_val)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Context: (b, H, s, v_head_dim)
        context = torch.matmul(attn_weights, V)

        # Reshape to (b, s, H*v_head_dim).
        context = context.permute(0, 2, 1, 3).contiguous().view(b, s, self.n_heads * self.v_head_dim)

        # --- Output projection ---
        ctx_t = context.transpose(0, 1)   # (s, b, H*v_head_dim)
        if self.o_groups > 1:
            out_a, _ = self.linear_o_a(ctx_t)
            out, _ = self.linear_o_b(out_a)
        else:
            out, _ = self.linear_o(ctx_t)

        return out  # (s, b, d)
