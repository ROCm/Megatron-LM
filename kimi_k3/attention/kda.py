"""Kimi Delta Attention.

Parameter names and shapes are the released checkpoint's, so the converter is a
rename rather than a reshape (`develop/notes/2026-08-27-release-audit.md`):

    q_proj, k_proj, v_proj : Linear(hidden, num_heads * head_dim)
    q_conv1d, k_conv1d, v_conv1d : causal depthwise conv, width 4, then SiLU
    f_a_proj (hidden -> head_dim), f_b_proj (head_dim -> P)   the DECAY gate
    g_proj  (hidden -> P)                                     the OUTPUT gate
    b_proj  (hidden -> num_heads)                             beta
    A_log [num_heads], dt_bias [P]                            fp32 in the checkpoint
    o_norm  gated RMSNorm over head_dim, sigmoid gate
    o_proj  Linear(P, hidden)

**Two gates, and they are not interchangeable.** `f_a/f_b` is low-rank and feeds
the *in-kernel decay*; `g_proj` is full-rank and gates the *output* through
`o_norm`. Conflating them was one of the review findings (B3).

Scope: this ships the math and the parameter layout. The projections are plain
`nn.Linear` behind a `linear_cls` hook -- tensor-parallel / TE linears are wired
when the trainer needs them (P7), and no numerical gate here depends on that.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .kda_backends import EAGER, kda_forward


def causal_short_conv(x: torch.Tensor, weight: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """Depthwise causal convolution over time, then SiLU.

    ``x`` is ``[B, T, D]`` and ``weight`` is ``[D, 1, W]`` -- the checkpoint's
    layout (`[12288, 1, 4]`). Left-padding by ``W - 1`` is what makes it causal:
    token *t* sees only *t-W+1 .. t*.
    """
    b, t, d = x.shape
    w = weight.shape[-1]
    y = F.conv1d(
        F.pad(x.transpose(1, 2), (w - 1, 0)),
        weight.to(x.dtype),
        groups=d,
    ).transpose(1, 2)
    if activation == "silu":
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"unsupported short-conv activation {activation!r}")
    return y


def gated_rms_norm(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """`FusedRMSNormGated(head_dim, activation='sigmoid')`: normalise, then gate."""
    hi = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(hi)
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.to(hi) * torch.sigmoid(gate.to(hi))).to(x.dtype)


class KimiDeltaAttention(torch.nn.Module):
    """One KDA layer."""

    def __init__(self, config, layer_number: Optional[int] = None, linear_cls=torch.nn.Linear):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden = config.hidden_size
        self.num_heads = config.k3_kda_num_heads
        self.head_dim = config.k3_kda_head_dim
        self.conv_size = config.k3_kda_conv_size
        self.eps = config.layernorm_epsilon
        self.backend = config.k3_kda_backend
        self.lower_bound = config.k3_kda_gate_lower_bound
        p = self.num_heads * self.head_dim

        self.q_proj = linear_cls(self.hidden, p, bias=False)
        self.k_proj = linear_cls(self.hidden, p, bias=False)
        self.v_proj = linear_cls(self.hidden, p, bias=False)
        self.o_proj = linear_cls(p, self.hidden, bias=False)

        for name in ("q", "k", "v"):
            self.register_parameter(
                f"{name}_conv1d_weight",
                torch.nn.Parameter(torch.zeros(p, 1, self.conv_size)),
            )

        # decay gate (low-rank) -> fed to the kernel
        self.f_a_proj = linear_cls(self.hidden, self.head_dim, bias=False)
        self.f_b_proj = linear_cls(self.head_dim, p, bias=False)
        # output gate
        if config.k3_kda_use_full_rank_gate:
            self.g_proj = linear_cls(self.hidden, p, bias=False)
        else:
            self.g_a_proj = linear_cls(self.hidden, self.head_dim, bias=False)
            self.g_b_proj = linear_cls(self.head_dim, p, bias=False)
        self.b_proj = linear_cls(self.hidden, self.num_heads, bias=False)

        # fp32 in the released checkpoint, and kept fp32 here (rule R7.3)
        self.A_log = torch.nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16).log()
        )
        self.dt_bias = torch.nn.Parameter(torch.zeros(p, dtype=torch.float32))
        self.o_norm_weight = torch.nn.Parameter(torch.ones(self.head_dim))

        self._init_convs()

    def _init_convs(self) -> None:
        with torch.no_grad():
            for name in ("q", "k", "v"):
                w = getattr(self, f"{name}_conv1d_weight")
                w.zero_()
                w[:, :, -1] = 1.0  # identity at t, i.e. a no-op convolution

    # --- pieces, exposed so tests can drive them in isolation ---------------

    def output_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "g_proj"):
            return self.g_proj(hidden_states)
        return self.g_b_proj(self.g_a_proj(hidden_states))

    def decay_gate_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.f_b_proj(self.f_a_proj(hidden_states))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        backend: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``hidden_states`` is ``[B, T, hidden]``; returns ``(out, final_state)``."""
        q = causal_short_conv(self.q_proj(hidden_states), self.q_conv1d_weight)
        k = causal_short_conv(self.k_proj(hidden_states), self.k_conv1d_weight)
        v = causal_short_conv(self.v_proj(hidden_states), self.v_conv1d_weight)

        g = self._heads(self.decay_gate_input(hidden_states))
        beta = self.b_proj(hidden_states).float()

        o, state = kda_forward(
            self._heads(q), self._heads(k), self._heads(v), g, beta,
            A_log=self.A_log, dt_bias=self.dt_bias,
            initial_state=initial_state,
            backend=backend or self.backend,
            config=self.config,
            output_final_state=output_final_state,
        )

        o = gated_rms_norm(o, self.o_norm_weight, self._heads(self.output_gate(hidden_states)), self.eps)
        return self.o_proj(o.flatten(-2)), state

    # --- checkpointing -------------------------------------------------------

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        """No tensor-parallel sharding yet, so every tensor is replicated.

        Written explicitly rather than inherited so that when TP linears land
        (P7) the change is visible here instead of silently absent.
        """
        from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor  # noqa: F401
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True), prefix, {}, sharded_offsets
        )


class K3KDASelfAttention(torch.nn.Module):
    """`KimiDeltaAttention` wearing Megatron's self-attention interface.

    Two adaptations, both mechanical and both easy to get wrong silently:

    * **Megatron is sequence-first.** A layer hands its attention `[s, b, h]`,
      while KDA (and the release) think in `[b, t, h]`. The transpose happens
      here, once, rather than inside the maths.
    * A layer expects `(output, bias)` back and passes attention-specific kwargs
      (masks, rotary, packed-seq) that a linear-attention module has no use for.
      They are accepted and ignored deliberately -- K3 is NoPE, and KDA's
      causality comes from its recurrence, not from a mask.
    """

    def __init__(self, config, submodules=None, layer_number: int = 1, attn_mask_type=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.kda = KimiDeltaAttention(config, layer_number=layer_number)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        out, _ = self.kda(hidden_states.transpose(0, 1).contiguous())
        return out.transpose(0, 1).contiguous(), None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        return self.kda.sharded_state_dict(f"{prefix}kda.", sharded_offsets, metadata)
