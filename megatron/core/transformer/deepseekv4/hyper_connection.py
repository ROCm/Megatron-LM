# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Manifold-constrained Hyper-Connections (mHC).

Reference: arXiv:2512.24880 (DeepSeek mHC paper).
Original HC: arXiv:2409.19606 (ByteDance, ICLR 2025).

hc_mult=N means N residual streams of shape (b, s, N, d).
The mixing matrix (comb block) is projected to a doubly-stochastic matrix via
Sinkhorn-Knopp in every forward pass. Parameters (hc_fn, hc_base, hc_scale) are
unconstrained; the constraint lives in the forward computation, not the weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def sinkhorn_normalize(logits: torch.Tensor, n_iters: int, eps: float) -> torch.Tensor:
    """Project (N, N) logits to a doubly-stochastic matrix via Sinkhorn-Knopp.

    Args:
        logits: (..., N, N) unnormalized scores
        n_iters: number of alternating normalisation steps
        eps: numerical stability floor

    Returns:
        (..., N, N) doubly-stochastic matrix M (rows and cols sum to 1).
    """
    # Always run in FP32 for numerical stability.
    orig_dtype = logits.dtype
    M = logits.float().exp()
    M = M + eps
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)  # row normalize
        M = M / (M.sum(dim=-2, keepdim=True) + eps)  # col normalize
    return M.to(orig_dtype)


class HyperConnection(nn.Module):
    """Single mHC block wrapping one sub-layer (attention or FFN).

    Usage pattern mirrors the residual wrapper in the V4 reference implementation:
        hc = HyperConnection(config, layer_number)
        x_pre, residuals = hc.pre(x)           # pre-mix: produce single stream for sub-layer
        x_post = sub_layer(x_pre)
        x = hc.post(x_post, residuals)          # post-mix: update all streams

    x shape throughout: (b, s, hc_mult, hidden_size)

    The logits for pre/post/comb are computed from the concatenated streams each forward,
    making the doubly-stochastic comb matrix input-dependent (ephemeral, not stored).
    """

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.n_iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps

        N = self.hc_mult
        # Logit counts: pre(N) + post(N) + comb(N*N) = 2N + N^2
        n_logits = 2 * N + N * N

        # hc_fn: projects concatenated streams (N*hidden) -> n_logits
        # Keep in FP32 for stability; matches Primus implementation.
        self.hc_fn = nn.Linear(N * self.hidden_size, n_logits, bias=False, dtype=torch.float32)

        # Scalar base shift and scale for the logits (shape: n_logits and 3 respectively).
        self.hc_base = nn.Parameter(torch.zeros(n_logits, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

        self._N = N

    def _compute_logits(self, x: torch.Tensor):
        """x: (b, s, N, d) -> pre (b,s,N), post (b,s,N), comb_ds (b,s,N,N)"""
        b, s, N, d = x.shape
        # Flatten streams: (b, s, N*d)
        x_flat = x.reshape(b, s, N * d).float()
        # Project to logits.
        raw = self.hc_fn(x_flat) + self.hc_base  # (b, s, 2N+N^2)

        pre_logits = raw[..., :N] * self.hc_scale[0]              # (b, s, N)
        post_logits = raw[..., N: 2 * N] * self.hc_scale[1]       # (b, s, N)
        comb_logits = raw[..., 2 * N:].reshape(b, s, N, N) * self.hc_scale[2]  # (b, s, N, N)

        pre = F.softmax(pre_logits, dim=-1)    # (b, s, N)
        post = F.softmax(post_logits, dim=-1)  # (b, s, N)
        comb = sinkhorn_normalize(comb_logits, self.n_iters, self.eps)  # (b, s, N, N)
        return pre, post, comb

    def pre(self, x: torch.Tensor):
        """Produce the single-stream input for the sub-layer.

        Returns:
            y: (b, s, d) - collapsed single stream for the sub-layer
            state: tuple of (x, pre, post, comb) for use in post()
        """
        pre, post, comb = self._compute_logits(x)
        # y = sum_i pre_i * stream_i   (weighted combination -> single stream)
        # x: (b, s, N, d), pre: (b, s, N) -> (b, s, N, 1)
        y = (x * pre.unsqueeze(-1)).sum(dim=2)  # (b, s, d)
        return y.to(x.dtype), (x, pre, post, comb)

    def post(self, sub_out: torch.Tensor, state) -> torch.Tensor:
        """Update all streams with sub-layer output.

        Args:
            sub_out: (b, s, d) - sub-layer output
            state: tuple returned by pre()

        Returns:
            x_new: (b, s, N, d)
        """
        x, pre, post, comb = state
        sub_out = sub_out.float()
        # x_new_i = post_i * F(y) + sum_j comb_{ij} * x_j
        # post: (b, s, N) -> (b, s, N, 1)
        # sub_out: (b, s, d) -> (b, s, 1, d)
        post_contrib = post.unsqueeze(-1) * sub_out.unsqueeze(2)      # (b, s, N, d)
        # comb: (b, s, N, N), x: (b, s, N, d)
        comb_contrib = torch.einsum("bsnm,bsmd->bsnd", comb, x.float())  # (b, s, N, d)
        return (post_contrib + comb_contrib).to(x.dtype)


class HyperConnectionIdentity(nn.Module):
    """Drop-in when hc_mult == 1 (standard residual, no extra streams)."""

    def pre(self, x: torch.Tensor):
        # x: (b, s, 1, d) -> (b, s, d)
        return x.squeeze(2), x

    def post(self, sub_out: torch.Tensor, state) -> torch.Tensor:
        x = state
        # Standard residual: add sub_out back to the single stream.
        return (x.squeeze(2) + sub_out).unsqueeze(2)


def build_hyper_connection(config, layer_number: int) -> nn.Module:
    if config.hc_mult <= 1:
        return HyperConnectionIdentity()
    return HyperConnection(config, layer_number)
