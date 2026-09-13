"""A genuinely fused AttnRes mixer, in Triton.

`attn_res_mix_chunked` is not a kernel -- it is a Python loop over row slices
(G56). This is the kernel that was promised in P11 and never written.

What the eager path costs. For each token it materialises `[T, K+1, H]` three
times: the `cat`, the fp32 upcast, and the normalised copy `k`; then reads two of
them back for the score reduction and the output matmul. At production shape that
is the dominant non-GEMM memory cost in the model -- 109.6 GiB per forward
(`results/attn_res.md`).

What this does instead. Per token, two passes over the candidates straight out of
their original tensors:

    pass 1   s2[j]  = sum_h v[j,h]^2          -- for the RMS norm
             d[j]   = sum_h v[j,h] * w[h]     -- w = norm_weight * proj_weight
             score  = d * rsqrt(s2/H + eps)   -- the norm folds into a scalar
             p      = softmax(score)
    pass 2   out[h] = sum_j p[j] * v[j,h]

Nothing of size `[T, K+1, H]` is ever written. The `cat` never happens: `j < K`
reads `block_residual` and `j == K` reads `prefix_sum` in place.

The fold that makes this cheap is the one the release's own code makes obvious:
the RMSNorm gain and the `[1, H]` projection only ever appear multiplied together,
so they collapse to a single `[H]` vector, and the normalisation itself is a
per-candidate scalar applied *after* the dot product rather than to `H` elements
before it.

**This is not bit-identical to the eager oracle and cannot be.** The eager path
reduces over `H` with torch's ordering; this one reduces in tiles. Every other
AttnRes gate asserts bit-equality, so this one is gated on a tolerance instead,
measured rather than assumed (`tests/test_k3_p11_triton_attn_res.py`).
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton is a hard dep of fla, so present
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _attn_res_kernel(
        PREFIX, BLOCK_RES, W, OUT,
        T, K, H, EPS,
        BJ: tl.constexpr, BH: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        H64 = H.to(tl.int64)
        K64 = K.to(tl.int64)
        j = tl.arange(0, BJ).to(tl.int64)
        jm = j <= K64                       # K+1 candidates: K slots plus the prefix
        is_prefix = j == K64

        s2 = tl.zeros((BJ,), dtype=tl.float32)
        dot = tl.zeros((BJ,), dtype=tl.float32)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H64
            w = tl.load(W + h, mask=hm, other=0.0).to(tl.float32)
            br = tl.load(BLOCK_RES + t * K64 * H64 + j[:, None] * H64 + h[None, :],
                         mask=(j[:, None] < K64) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H64 + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)
            s2 += tl.sum(v * v, axis=1)
            dot += tl.sum(v * w[None, :], axis=1)

        scale = 1.0 / tl.sqrt(s2 / H.to(tl.float32) + EPS)
        score = tl.where(jm, dot * scale, float("-inf"))
        p = tl.exp(score - tl.max(score, axis=0))
        p = p / tl.sum(p, axis=0)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H64
            br = tl.load(BLOCK_RES + t * K64 * H64 + j[:, None] * H64 + h[None, :],
                         mask=(j[:, None] < K64) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H64 + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)
            out = tl.sum(v * p[:, None], axis=0)
            tl.store(OUT + t * H64 + h, out.to(OUT.dtype.element_ty), mask=hm)


def attn_res_mix_triton(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    block_h: int = 2048,
) -> torch.Tensor:
    """Fused mix. `prefix_sum` is `[T, H]`, `block_residual` is `[T, K, H]`."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is unavailable; use attn_res_mix")
    if block_residual.shape[1] == 0:
        return prefix_sum
    from .attn_res import score_vector

    T, K, H = block_residual.shape[0], block_residual.shape[1], prefix_sum.shape[-1]
    w = score_vector(norm_weight, proj_weight).reshape(H).float().contiguous()
    prefix_sum = prefix_sum.contiguous()
    block_residual = block_residual.contiguous()
    out = torch.empty_like(prefix_sum)
    bj = max(16, triton.next_power_of_2(K + 1))
    bh = min(block_h, triton.next_power_of_2(H))
    # Tuned at production shape (T 8192, K 8, H 7168): 0.522 ms at 4276 GB/s, 81%
    # of measured copy bandwidth. num_warps is the sensitive knob and the intuitive
    # choice is wrong -- 8 warps costs 1.83x and 16 costs 3.17x against 4, because
    # each program already has a [BJ, BH] tile's worth of work and more warps split
    # it below the point where the loads coalesce.
    _attn_res_kernel[(T,)](
        prefix_sum, block_residual, w, out,
        T, K, H, eps, BJ=bj, BH=bh, num_warps=4, num_stages=1,
    )
    return out


class _FusedAttnResMix(torch.autograd.Function):
    """Fast fused forward, eager recompute for the backward.

    A raw Triton kernel returns a tensor with no `grad_fn`, so calling
    `attn_res_mix_triton` directly inside a training step silently produces no
    gradients -- the loss still falls, because every *other* path still trains.
    That is the failure mode this wrapper exists to prevent.

    The backward recomputes the mix through the eager oracle under
    `enable_grad` and differentiates that. Consequences, stated rather than
    discovered later:

    * gradients are the **eager** gradients, exactly, so every existing AttnRes
      backward gate still applies unchanged;
    * the backward is *not* accelerated, and it re-materialises the
      `[T, K+1, H]` fp32 temporaries the forward avoids -- so the peak-memory
      win is much smaller than the forward-only figure suggests.

    A real backward kernel is the remaining work. It is tractable -- the
    gradient of a softmax over a scalar-scaled dot product is closed-form -- but
    it is a second kernel, not a tweak to this one.
    """

    @staticmethod
    def forward(ctx, prefix_sum, block_residual, norm_weight, proj_weight, eps, block_h):
        ctx.save_for_backward(prefix_sum, block_residual, norm_weight, proj_weight)
        ctx.eps = eps
        return attn_res_mix_triton(
            prefix_sum, block_residual, norm_weight, proj_weight, eps, block_h
        )

    @staticmethod
    def backward(ctx, grad_out):
        from .attn_res import attn_res_mix

        saved = ctx.saved_tensors
        leaves = [t.detach().requires_grad_(t.requires_grad) for t in saved]
        with torch.enable_grad():
            out = attn_res_mix(leaves[0], leaves[1], leaves[2], leaves[3], ctx.eps)
        wanted = [t for t in leaves if t.requires_grad]
        grads = (
            torch.autograd.grad(out, wanted, grad_out, allow_unused=True) if wanted else ()
        )
        it = iter(grads)
        return (*(next(it) if t.requires_grad else None for t in leaves), None, None)


def fused_attn_res_mix(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    block_h: int = 2048,
) -> torch.Tensor:
    """Differentiable entry point. Use this, not `attn_res_mix_triton`."""
    if block_residual.shape[1] == 0:
        return prefix_sum
    return _FusedAttnResMix.apply(
        prefix_sum, block_residual, norm_weight, proj_weight, eps, block_h
    )
