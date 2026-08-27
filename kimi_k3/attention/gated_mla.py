"""Gated multi-head latent attention for Kimi K3 (NoPE).

Built as its own module rather than as a thin subclass of core's
`MLASelfAttention`, because once rotary is removed, the softmax scale changes and
a gate is inserted before the output projection, almost nothing of core's MLA
forward survives -- and finding A9 showed core supports no NoPE mode at all
(`rope_type` accepts only `"rope"` and `"yarn"`).

What it *does* reuse is the shape contract, so the converter is a rename:
`q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`,
`kv_b_proj`, `g_proj`, `o_proj`.

The attention op itself goes through `scaled_dot_product_attention` when it can,
which on this stack is a fused kernel. The 192/128 head-dim asymmetry is handled
the way the release handles it for FlashAttention: pad V to `q_head_dim`, then
slice the output back.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ..numerics import to_hi
from .gated_mla_eager_fp32 import gated_mla_eager_fp32, rms_norm, softmax_scale

EAGER = "eager"
SDPA = "sdpa"
BACKENDS = (EAGER, SDPA)


class K3GatedMLA(torch.nn.Module):
    """One gated-MLA layer, batch-first."""

    def __init__(self, config, layer_number: Optional[int] = None, linear_cls=torch.nn.Linear):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = config.qk_head_dim
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.lora_norm_eps = config.k3_mla_lora_norm_eps
        self.use_output_gate = config.k3_mla_use_output_gate
        self.fp32_attention_output = config.k3_mla_fp32_attn_output
        self.scale = softmax_scale(self.qk_head_dim, self.qk_pos_emb_head_dim)
        assert config.k3_mla_use_nope, "K3 is NoPE; a rotary variant is not implemented"

        hidden = config.hidden_size
        self.q_a_proj = linear_cls(hidden, self.q_lora_rank, bias=False)
        self.q_b_proj = linear_cls(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = linear_cls(
            hidden, self.kv_lora_rank + self.qk_pos_emb_head_dim, bias=False
        )
        self.kv_b_proj = linear_cls(
            self.kv_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = linear_cls(self.num_heads * self.v_head_dim, hidden, bias=False)
        if self.use_output_gate:
            self.g_proj = linear_cls(hidden, self.num_heads * self.v_head_dim, bias=False)

        self.q_a_layernorm = torch.nn.Parameter(torch.ones(self.q_lora_rank))
        self.kv_a_layernorm = torch.nn.Parameter(torch.ones(self.kv_lora_rank))

        #: Read by core's `optimizer/qk_clip.py:clip_qk`, which skips modules
        #: without it. KDA layers are skipped for free; MLA layers are not.
        self.clip_qk = self._clip_qk
        #: Core's helper reads the max logit off `self_attention.core_attention`.
        #: We have no such submodule -- SDPA is called directly -- so this is the
        #: one attribute of it that core actually touches, and nothing more.
        self.core_attention = torch.nn.Module()
        self.core_attention.current_max_attn_logits = None
        self.track_max_attn_logits = bool(
            getattr(config, "qk_clip", False) or getattr(config, "log_max_attn_logit", False)
        )
        self.max_logit_chunk = getattr(config, "k3_max_logit_chunk", 1024)

    def weights(self) -> dict:
        w = {
            "q_a_proj": self.q_a_proj.weight,
            "q_a_layernorm": self.q_a_layernorm,
            "q_b_proj": self.q_b_proj.weight,
            "kv_a_proj_with_mqa": self.kv_a_proj_with_mqa.weight,
            "kv_a_layernorm": self.kv_a_layernorm,
            "kv_b_proj": self.kv_b_proj.weight,
            "o_proj": self.o_proj.weight,
        }
        if self.use_output_gate:
            w["g_proj"] = self.g_proj.weight
        return w

    # --- QK-clip (P9 / T9.4) ------------------------------------------------

    @torch.no_grad()
    def _record_max_attn_logits(self, query: torch.Tensor, key: torch.Tensor) -> None:
        """Per-head max attention logit over the causal region.

        Core reads this off `self.core_attention.current_max_attn_logits`, which
        on its own path is filled in by the attention kernel. Ours comes from
        `scaled_dot_product_attention`, which returns no logits, so it is
        recomputed here -- and that is a real cost: another `[b, h, s, s]` score
        matmul. It only runs when `qk_clip` or `log_max_attn_logit` is on.

        ponytail: query-chunked to bound memory at `[b, h, chunk, s]`; get it from
        the kernel instead if this ever shows up in a profile.
        """
        b, h, s, _ = query.shape
        running = query.new_zeros(h, dtype=torch.float32)
        for start in range(0, s, self.max_logit_chunk):
            stop = min(start + self.max_logit_chunk, s)
            scores = torch.matmul(query[:, :, start:stop].float(), key.float().transpose(-1, -2))
            scores = scores * self.scale
            # causal: query t may only see keys <= t
            rows = torch.arange(start, stop, device=scores.device).unsqueeze(1)
            scores = scores.masked_fill(rows < torch.arange(s, device=scores.device), float("-inf"))
            running = torch.maximum(running, scores.amax(dim=(0, 2, 3)))
        previous = self.core_attention.current_max_attn_logits
        self.core_attention.current_max_attn_logits = (
            running if previous is None else torch.maximum(previous, running)
        )

    @torch.no_grad()
    def _clip_qk(self) -> None:
        """Rescale the per-head q/k weights so the max logit falls to the threshold.

        The head's logit is `q . k`, so splitting the correction between the two
        as `eta**alpha` and `eta**(1 - alpha)` scales the logit by exactly `eta`.

        The `qk_pos_emb_head_dim` slice is the exception. K3 is NoPE, but `k_rot`
        is still produced by `kv_a_proj_with_mqa` and **shared across heads**, so
        there is no per-head k weight to scale -- the whole correction for that
        slice has to go on the query side, at full `eta`. This is the released
        MLA form of QK-clip, and getting it wrong silently under-clips.
        """
        if not self.config.qk_clip:
            raise ValueError("qk_clip option needs to be enabled")
        logits = self.core_attention.current_max_attn_logits
        if logits is None:
            raise ValueError("current_max_attn_logits is None")
        assert logits.shape == (self.num_heads,), f"expected [{self.num_heads}], got {list(logits.shape)}"

        eta = torch.clamp(self.config.qk_clip_threshold / logits, max=1.0)
        if bool((eta >= 1.0).all()):
            self.core_attention.current_max_attn_logits = None
            return

        alpha = self.config.qk_clip_alpha
        for weight in (self.q_b_proj.weight, getattr(self.q_b_proj.weight, "main_param", None)):
            if weight is None:
                continue
            view = weight.data.view(self.num_heads, self.q_head_dim, -1)
            view[:, : self.qk_head_dim].mul_(eta.to(view.dtype).pow(alpha).view(-1, 1, 1))
            view[:, self.qk_head_dim :].mul_(eta.to(view.dtype).view(-1, 1, 1))
        for weight in (self.kv_b_proj.weight, getattr(self.kv_b_proj.weight, "main_param", None)):
            if weight is None:
                continue
            view = weight.data.view(self.num_heads, self.qk_head_dim + self.v_head_dim, -1)
            view[:, : self.qk_head_dim].mul_(eta.to(view.dtype).pow(1 - alpha).view(-1, 1, 1))

        self.core_attention.current_max_attn_logits = None

    # --- attention ----------------------------------------------------------

    def _sdpa(self, query, key, value):
        """Fused attention with the release's own 192/128 workaround.

        `scaled_dot_product_attention` wants one head dim; K3's queries and keys
        are 192 wide while values are 128. The release pads V to `q_head_dim` for
        FlashAttention and slices the output back, so we do the same.
        """
        pad = self.q_head_dim - self.v_head_dim
        v = F.pad(value, (0, pad)) if pad else value
        out = F.scaled_dot_product_attention(query, key, v, is_causal=True, scale=self.scale)
        return out[..., : self.v_head_dim] if pad else out

    def forward(
        self, hidden_states: torch.Tensor, backend: str = SDPA
    ) -> torch.Tensor:
        """``hidden_states`` is ``[B, S, hidden]``."""
        assert backend in BACKENDS, f"unknown MLA backend {backend!r}"
        if backend == EAGER:
            return gated_mla_eager_fp32(
                hidden_states, self.weights(),
                num_heads=self.num_heads, qk_head_dim=self.qk_head_dim,
                qk_pos_emb_head_dim=self.qk_pos_emb_head_dim, v_head_dim=self.v_head_dim,
                lora_norm_eps=self.lora_norm_eps, use_output_gate=self.use_output_gate,
                fp32_attention_output=self.fp32_attention_output,
            )

        b, s, _ = hidden_states.shape
        q = rms_norm(self.q_a_proj(hidden_states), self.q_a_layernorm, self.lora_norm_eps)
        q = self.q_b_proj(q).view(b, s, self.num_heads, self.q_head_dim).transpose(1, 2)

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed, [self.kv_lora_rank, self.qk_pos_emb_head_dim], dim=-1
        )
        k_pass = rms_norm(k_pass, self.kv_a_layernorm, self.lora_norm_eps)
        k_pass = self.kv_b_proj(k_pass).view(
            b, s, self.num_heads, self.qk_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_pass, value = torch.split(k_pass, [self.qk_head_dim, self.v_head_dim], dim=-1)
        # NoPE: shared across heads, never rotated
        k_rot = k_rot.view(b, 1, s, self.qk_pos_emb_head_dim).expand(*k_pass.shape[:-1], -1)
        key = torch.cat((k_pass, k_rot), dim=-1)

        if self.track_max_attn_logits:
            self._record_max_attn_logits(q, key)
        attn = self._sdpa(q, key, value)
        out = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.v_head_dim)
        if self.fp32_attention_output:
            out = to_hi(out)
        if self.use_output_gate:
            out = out * torch.sigmoid(to_hi(self.g_proj(hidden_states)))
        return self.o_proj(out.to(self.o_proj.weight.dtype))


class K3GatedMLASelfAttention(torch.nn.Module):
    """`K3GatedMLA` wearing Megatron's sequence-first self-attention interface."""

    def __init__(self, config, submodules=None, layer_number: int = 1, attn_mask_type=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.mla = K3GatedMLA(config, layer_number=layer_number)

    @property
    def clip_qk(self):
        return self.mla.clip_qk

    @property
    def core_attention(self):
        """Core's `clip_qk` helper reaches through `self_attention.core_attention`."""
        return self.mla.core_attention

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        out = self.mla(hidden_states.transpose(0, 1).contiguous())
        return out.transpose(0, 1).contiguous(), None
