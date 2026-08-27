"""FP32 eager oracle for Kimi K3's gated MLA.

Transcribed from `KimiMLAAttention.forward` in the release. Four things differ
from stock DeepSeek-style MLA, and every one of them is a silent-wrong-answer
risk rather than a crash:

1. **NoPE.** `use_nope` is asserted and `rotary_emb` is `None`. The 64
   "rope" dimensions still exist, but they are never rotated: `k_rot` is produced
   once, MQA-style, and *expanded* across heads as extra shared content. Someone
   "fixing" this by enabling rotary would destroy parity and nothing would fail.
2. **The scale is `q_head_dim ** -0.5`**, i.e. `192 ** -0.5` -- not
   `qk_head_dim ** -0.5 = 128 ** -0.5`. Core computes it from `q_head_dim` too,
   but only after a YaRN mscale that K3 does not use.
3. **A full-rank sigmoid output gate** multiplies the attention output *before*
   `o_proj`, elementwise over `num_heads * v_head_dim`.
4. **The two LoRA norms use eps 1e-6**, not `rms_norm_eps = 1e-5`: the release
   constructs them without an `eps` argument, so they take `KimiRMSNorm`'s class
   default (review finding B7, measured in P0).
"""

import math
from typing import Optional

import torch

from ..numerics import to_hi


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`KimiRMSNorm`: normalise in high precision, scale, cast back."""
    xf = to_hi(x)
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.to(xf.dtype) * normed).to(x.dtype)


def gated_mla_eager_fp32(
    hidden_states: torch.Tensor,
    weights: dict,
    *,
    num_heads: int,
    qk_head_dim: int = 128,
    qk_pos_emb_head_dim: int = 64,
    v_head_dim: int = 128,
    lora_norm_eps: float = 1e-6,
    use_output_gate: bool = True,
    fp32_attention_output: bool = True,
) -> torch.Tensor:
    """One gated-MLA layer, batch-first ``[B, S, hidden]``.

    ``weights`` holds the released parameter names: ``q_a_proj``, ``q_a_layernorm``,
    ``q_b_proj``, ``kv_a_proj_with_mqa``, ``kv_a_layernorm``, ``kv_b_proj``,
    ``o_proj`` and (optionally) ``g_proj``.
    """
    b, s, _ = hidden_states.shape
    q_head_dim = qk_head_dim + qk_pos_emb_head_dim
    scaling = q_head_dim**-0.5

    # --- query: low-rank down, norm, up ---
    q = hidden_states @ weights["q_a_proj"].T
    q = rms_norm(q, weights["q_a_layernorm"], lora_norm_eps)
    q = (q @ weights["q_b_proj"].T).view(b, s, num_heads, q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [qk_head_dim, qk_pos_emb_head_dim], dim=-1)

    # --- key/value: one MQA latent carrying both the compressed kv and k_rot ---
    compressed = hidden_states @ weights["kv_a_proj_with_mqa"].T
    k_pass, k_rot = torch.split(
        compressed, [weights["kv_a_layernorm"].shape[0], qk_pos_emb_head_dim], dim=-1
    )
    k_pass = rms_norm(k_pass, weights["kv_a_layernorm"], lora_norm_eps)
    k_pass = (k_pass @ weights["kv_b_proj"].T).view(
        b, s, num_heads, qk_head_dim + v_head_dim
    ).transpose(1, 2)
    k_pass, value = torch.split(k_pass, [qk_head_dim, v_head_dim], dim=-1)

    # NoPE: k_rot is shared across heads and never rotated
    k_rot = k_rot.view(b, 1, s, qk_pos_emb_head_dim).expand(*k_pass.shape[:-1], -1)

    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_pass, k_rot), dim=-1)

    # --- attention, in fp32 ---
    qf, kf, vf = to_hi(query), to_hi(key), to_hi(value)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scaling
    causal = torch.triu(torch.full((s, s), float("-inf"), device=scores.device, dtype=scores.dtype), 1)
    probs = torch.softmax(scores + causal, dim=-1)
    attn = torch.matmul(probs, vf)  # [B, H, S, v_head_dim]

    out = attn.transpose(1, 2).reshape(b, s, num_heads * v_head_dim)
    if not fp32_attention_output:
        out = out.to(hidden_states.dtype)

    if use_output_gate:
        gate = torch.sigmoid(to_hi(hidden_states @ weights["g_proj"].T))
        out = out * gate

    return (out.to(weights["o_proj"].dtype) @ weights["o_proj"].T).to(hidden_states.dtype)


def softmax_scale(qk_head_dim: int = 128, qk_pos_emb_head_dim: int = 64) -> float:
    """`q_head_dim ** -0.5`. Written once so no caller reinvents `128 ** -0.5`."""
    return (qk_head_dim + qk_pos_emb_head_dim) ** -0.5
