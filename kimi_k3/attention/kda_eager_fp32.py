"""FP32 eager oracle for Kimi Delta Attention.

This is the contract every KDA backend is measured against, and it stays in the
tree permanently (rule R8.1) -- fla's KDA backward has a bug history (#807, #785)
and, at our pin, only compiles at all on triton 3.7.1.

It implements the **released call's** semantics, not fla's defaults: the gate,
the beta activation and the q/k normalisation all happen where
`KimiDeltaAttention.forward` asks for them to happen
(`use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
use_qk_l2norm_in_kernel=True, safe_gate=True, lower_bound=-5`).

Sources, both read rather than inferred:

* gate activation -- `fla/ops/kda/chunk.py` docstring and
  `fla/ops/kda/fused_recurrent.py:163-176`;
* recurrence -- `fla/ops/kda/naive.py:naive_recurrent_kda`.

Two details that are easy to get subtly wrong and are pinned by tests:

* the q/k L2 normalisation puts its epsilon **inside** the square root
  (`x / sqrt(sum(x^2) + 1e-6)`), which is not the usual `x / (norm + eps)`;
* `q` is scaled by `K ** -0.5` *after* normalisation, not before.
"""

import math
from typing import Optional, Tuple

import torch

from ..numerics import hi_dtype, to_hi

L2_EPS = 1e-6


def kda_gate(
    g: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    lower_bound: Optional[float] = -5.0,
    safe_gate: bool = True,
) -> torch.Tensor:
    """Raw gate input -> log-space decay, in fp32.

    ``g`` is ``[B, T, HV, K]``; ``A_log`` is ``[HV]``; ``dt_bias`` is ``[HV * K]``.

    With ``safe_gate`` and a ``lower_bound`` (what the release uses) the decay is
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, which lands in
    ``[lower_bound, 0)`` by construction -- no clamp needed. Otherwise it is
    ``-exp(A_log) * softplus(g + dt_bias)``.
    """
    x = to_hi(g)
    hv, k = x.shape[-2], x.shape[-1]
    if dt_bias is not None:
        x = x + dt_bias.to(x.dtype).view(hv, k)
    a = A_log.to(x.dtype).view(1, 1, hv, 1).exp() if A_log is not None else torch.ones_like(x)
    if safe_gate and lower_bound is not None:
        return lower_bound * torch.sigmoid(a * x)
    return -a * torch.nn.functional.softplus(x)


def l2norm_last(x: torch.Tensor, eps: float = L2_EPS) -> torch.Tensor:
    """The kernel's normalisation: the epsilon sits inside the square root."""
    xf = to_hi(x)
    return (x / torch.sqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(xf.dtype)


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_log: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The gated delta rule, in fp32, one token at a time.

    Shapes: ``q, k`` ``[B, T, H, K]``; ``v`` ``[B, T, HV, V]``;
    ``decay_log`` ``[B, T, HV, K]``; ``beta`` ``[B, T, HV]`` (post-sigmoid);
    state ``[B, HV, K, V]``.

    Per step, with ``S`` the ``[K, V]`` state::

        S = S * exp(decay)[:, None]
        S = S + (beta * k) (x) (v - k^T S)
        o = q^T S
    """
    b, t, h, kdim = q.shape
    hv, vdim = v.shape[2], v.shape[-1]
    assert hv % h == 0, (hv, h)
    group = hv // h
    scale = kdim**-0.5 if scale is None else scale

    hi = hi_dtype(torch.promote_types(q.dtype, v.dtype))
    q = q.to(hi).repeat_interleave(group, dim=2) * scale
    k = k.to(hi).repeat_interleave(group, dim=2)
    v = v.to(hi)
    decay_log = decay_log.to(hi)
    beta = beta.to(hi)

    state = q.new_zeros(b, hv, kdim, vdim)
    if initial_state is not None:
        state = state + initial_state.to(hi)

    out = torch.zeros_like(v)
    for i in range(t):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        state = state * decay_log[:, i][..., None].exp()
        # the delta-rule correction: how much of v is already predicted by k
        delta = v_i - (k_i[..., None] * state).sum(-2)
        state = state + torch.einsum(
            "bhk,bhv->bhkv", beta[:, i][..., None] * k_i, delta
        )
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    return out, (state if output_final_state else None)


def chunk_kda_eager_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: Optional[float] = -5.0,
    transpose_state_layout: bool = True,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in oracle for the released ``chunk_kda`` call.

    Same signature and same defaults as `KimiDeltaAttention.forward` uses, so the
    two can be swapped by a backend switch with nothing else changing.
    """
    assert cu_seqlens is None, "variable-length batches are a long-context item, not v1"

    if use_qk_l2norm_in_kernel:
        q, k = l2norm_last(q), l2norm_last(k)
    decay_log = (
        kda_gate(g, A_log, dt_bias, lower_bound=lower_bound, safe_gate=safe_gate)
        if use_gate_in_kernel
        else to_hi(g)
    )
    beta_act = torch.sigmoid(to_hi(beta)) if use_beta_sigmoid_in_kernel else to_hi(beta)

    if initial_state is not None and transpose_state_layout:
        initial_state = initial_state.transpose(-1, -2)

    out, state = kda_recurrence(
        q, k, v, decay_log, beta_act,
        scale=scale, initial_state=initial_state, output_final_state=output_final_state,
    )
    if state is not None and transpose_state_layout:
        # the release asks for the V-first [V, K] layout
        state = state.transpose(-1, -2)
    return out.to(v.dtype), state


def released_call_kwargs(config=None) -> dict:
    """Exactly the kwargs `KimiDeltaAttention.forward` passes, as a dict.

    One definition, used by the oracle, the fla backend and the contract test, so
    the three cannot drift apart.
    """
    lower = getattr(config, "k3_kda_gate_lower_bound", -5.0) if config else -5.0
    return dict(
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=lower is not None,
        lower_bound=lower,
        transpose_state_layout=True,
    )
