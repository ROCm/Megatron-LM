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
        # NB: do *not* call .to() on H/K. Triton specializes integer kernel args
        # equal to 1 into constexpr ints, which have no .to() -- and K == 1 is the
        # normal case early in a block (attn_res_block_size 12 means K grows from
        # 0). Starting the offset arithmetic from the int64 program id is enough
        # to keep the products 64-bit.
        j = tl.arange(0, BJ).to(tl.int64)
        jm = j <= K                       # K+1 candidates: K slots plus the prefix
        is_prefix = j == K

        s2 = tl.zeros((BJ,), dtype=tl.float32)
        dot = tl.zeros((BJ,), dtype=tl.float32)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H
            w = tl.load(W + h, mask=hm, other=0.0).to(tl.float32)
            br = tl.load(BLOCK_RES + t * K * H + j[:, None] * H + h[None, :],
                         mask=(j[:, None] < K) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)
            s2 += tl.sum(v * v, axis=1)
            dot += tl.sum(v * w[None, :], axis=1)

        scale = 1.0 / tl.sqrt(s2 / (1.0 * H) + EPS)
        score = tl.where(jm, dot * scale, float("-inf"))
        p = tl.exp(score - tl.max(score, axis=0))
        p = p / tl.sum(p, axis=0)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H
            br = tl.load(BLOCK_RES + t * K * H + j[:, None] * H + h[None, :],
                         mask=(j[:, None] < K) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)
            out = tl.sum(v * p[:, None], axis=0)
            tl.store(OUT + t * H + h, out.to(OUT.dtype.element_ty), mask=hm)


    @triton.jit
    def _attn_res_bwd_kernel(
        PREFIX, BLOCK_RES, W, GOUT, DPREFIX, DBLOCK_RES, DD,
        T, K, H, EPS,
        BJ: tl.constexpr, BH: tl.constexpr,
    ):
        """Gradients of the fused mix.

        With `p = softmax(d * r)`, `r = rsqrt(s2/H + eps)`, `d = <v, w>` and
        `s2 = <v, v>`, everything collapses to three per-candidate scalars and one
        elementwise expression:

            gp[j]  = <g, v[j]>
            ds[j]  = p[j] * (gp[j] - sum_i p[i] gp[i])        # softmax
            dd[j]  = ds[j] * r[j]                             # through d
            ds2[j] = ds[j] * d[j] * (-0.5 * r[j]^3 / H)       # through the rsqrt
            dv[j,h] = p[j]*g[h] + dd[j]*w[h] + 2*ds2[j]*v[j,h]
            dw[h]   = sum_t sum_j dd[j] * v[j,h]

        So the backward is the same two passes as the forward, not a bigger
        computation: pass 1 recomputes `s2`, `d` and adds `gp`; pass 2 writes `dv`
        and accumulates `dw`. Nothing of size [T, K+1, H] is materialised here
        either, which is what the eager-recompute wrapper could not avoid.

        `dw` is the one *cross-token* reduction. An earlier version did it with
        `atomic_add` into an fp32 [H] buffer and that dominated the kernel: 8192
        programs x 7 chunks x 1024 lanes is ~58M atomics contending on 7168
        addresses, and the backward ran at ~640 GB/s, an eighth of what its
        traffic justifies. Instead the kernel stores the per-candidate scalar
        `dd` ([T, BJ], a few hundred KB) and the host reduces
        `dw = sum_t sum_j dd[t,j] * v[t,j,h]` with two ordinary reductions. No
        atomics anywhere; `dprefix` and `dblock_res` are per-token already.
        """
        t = tl.program_id(0).to(tl.int64)
        # NB: do *not* call .to() on H/K. Triton specializes integer kernel args
        # equal to 1 into constexpr ints, which have no .to() -- and K == 1 is the
        # normal case early in a block (attn_res_block_size 12 means K grows from
        # 0). Starting the offset arithmetic from the int64 program id is enough
        # to keep the products 64-bit.
        j = tl.arange(0, BJ).to(tl.int64)
        jm = j <= K
        is_prefix = j == K

        s2 = tl.zeros((BJ,), dtype=tl.float32)
        dot = tl.zeros((BJ,), dtype=tl.float32)
        gp = tl.zeros((BJ,), dtype=tl.float32)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H
            w = tl.load(W + h, mask=hm, other=0.0).to(tl.float32)
            g = tl.load(GOUT + t * H + h, mask=hm, other=0.0).to(tl.float32)
            br = tl.load(BLOCK_RES + t * K * H + j[:, None] * H + h[None, :],
                         mask=(j[:, None] < K) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)
            s2 += tl.sum(v * v, axis=1)
            dot += tl.sum(v * w[None, :], axis=1)
            gp += tl.sum(v * g[None, :], axis=1)

        Hf = 1.0 * H
        r = 1.0 / tl.sqrt(s2 / Hf + EPS)
        score = tl.where(jm, dot * r, float("-inf"))
        p = tl.exp(score - tl.max(score, axis=0))
        p = p / tl.sum(p, axis=0)

        ds = p * (gp - tl.sum(p * gp, axis=0))
        dd = tl.where(jm, ds * r, 0.0)
        ds2 = tl.where(jm, ds * dot * (-0.5 * r * r * r / Hf), 0.0)

        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH).to(tl.int64)
            hm = h < H
            w = tl.load(W + h, mask=hm, other=0.0).to(tl.float32)
            g = tl.load(GOUT + t * H + h, mask=hm, other=0.0).to(tl.float32)
            br = tl.load(BLOCK_RES + t * K * H + j[:, None] * H + h[None, :],
                         mask=(j[:, None] < K) & hm[None, :], other=0.0).to(tl.float32)
            ps = tl.load(PREFIX + t * H + h, mask=hm, other=0.0).to(tl.float32)
            v = tl.where(is_prefix[:, None], ps[None, :], br)

            dv = p[:, None] * g[None, :] + dd[:, None] * w[None, :] + 2.0 * ds2[:, None] * v
            tl.store(DBLOCK_RES + t * K * H + j[:, None] * H + h[None, :],
                     dv.to(DBLOCK_RES.dtype.element_ty),
                     mask=(j[:, None] < K) & hm[None, :])
            dprefix = tl.sum(tl.where(is_prefix[:, None], dv, 0.0), axis=0)
            tl.store(DPREFIX + t * H + h, dprefix.to(DPREFIX.dtype.element_ty), mask=hm)
        tl.store(DD + t * BJ + j, dd, mask=jm)


    @triton.jit
    def _dw_reduce_kernel(
        PREFIX, BLOCK_RES, DD, DWPART,
        T, K, H, TB: tl.constexpr, BJ: tl.constexpr, BH: tl.constexpr,
    ):
        """`dw[h] = sum_t sum_j dd[t,j] * v[t,j,h]`, in fp32, without upcasting v.

        The cross-token reduction, as its own kernel. Two earlier attempts were
        worse for opposite reasons: `atomic_add` from the main backward kernel
        (58M atomics on 7168 addresses, ~640 GB/s), and a single bf16 GEMV, which
        rounds the fp32 `dd` before summing and cost 3.05e-03 relative error on
        the norm/proj *parameter* gradients. Here `dd` stays fp32 and only `v` is
        read in its native dtype, so neither problem arises.

        Grid is (H blocks, T blocks): each program owns a column strip and a slice
        of tokens, writing one partial row. The partials are summed by the caller.
        """
        hb = tl.program_id(0).to(tl.int64)
        tb = tl.program_id(1).to(tl.int64)
        # NB: do *not* call .to() on H/K. Triton specializes integer kernel args
        # equal to 1 into constexpr ints, which have no .to() -- and K == 1 is the
        # normal case early in a block (attn_res_block_size 12 means K grows from
        # 0). Starting the offset arithmetic from the int64 program id is enough
        # to keep the products 64-bit.
        h = hb * BH + tl.arange(0, BH).to(tl.int64)
        hm = h < H
        j = tl.arange(0, BJ).to(tl.int64)
        acc = tl.zeros((BH,), dtype=tl.float32)
        for i in range(TB):
            t = tb * TB + i
            if t < T:
                t64 = t.to(tl.int64)
                dd = tl.load(DD + t64 * BJ + j, mask=j <= K, other=0.0)
                br = tl.load(
                    BLOCK_RES + t64 * K * H + j[:, None] * H + h[None, :],
                    mask=(j[:, None] < K) & hm[None, :], other=0.0).to(tl.float32)
                ps = tl.load(PREFIX + t64 * H + h, mask=hm, other=0.0).to(tl.float32)
                v = tl.where((j == K)[:, None], ps[None, :], br)
                acc += tl.sum(dd[:, None] * v, axis=0)
        tl.store(DWPART + tb * H + h, acc, mask=hm)


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


def attn_res_mix_triton_bwd(
    grad_out: torch.Tensor,
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    block_h: int = 2048,
):
    """Returns (d_prefix_sum, d_block_residual, d_norm_weight, d_proj_weight)."""
    from .attn_res import score_vector

    T, K, H = block_residual.shape[0], block_residual.shape[1], prefix_sum.shape[-1]
    w = score_vector(norm_weight, proj_weight).reshape(H).float().contiguous()
    grad_out = grad_out.contiguous()
    prefix_sum = prefix_sum.contiguous()
    block_residual = block_residual.contiguous()

    d_prefix = torch.empty_like(prefix_sum)
    d_block = torch.empty_like(block_residual)
    bj = max(16, triton.next_power_of_2(K + 1))
    bh = min(block_h, triton.next_power_of_2(H))
    d_d = torch.zeros(T, bj, device=prefix_sum.device, dtype=torch.float32)

    _attn_res_bwd_kernel[(T,)](
        prefix_sum, block_residual, w, grad_out, d_prefix, d_block, d_d,
        T, K, H, eps, BJ=bj, BH=bh, num_warps=2, num_stages=1,
    )
    # dw = sum_t sum_j dd[t,j] * v[t,j,h]; slots for j < K, prefix for j == K.
    # As a GEMV against the original bf16 tensors, not an einsum over an upcast
    # copy: `block_residual.float()` is [T, K, H] fp32, 1.88 GiB at production
    # shape, which cost more peak memory than the whole kernel saves.
    tb_size = 64
    n_tb = (T + tb_size - 1) // tb_size
    bh_red = min(512, triton.next_power_of_2(H))
    d_w_part = torch.empty(n_tb, H, device=prefix_sum.device, dtype=torch.float32)
    _dw_reduce_kernel[((H + bh_red - 1) // bh_red, n_tb)](
        prefix_sum, block_residual, d_d, d_w_part,
        T, K, H, TB=tb_size, BJ=bj, BH=bh_red, num_warps=4, num_stages=1,
    )
    d_w = d_w_part.sum(0)
    # w = norm_weight * proj_weight, elementwise, so the two factors split it.
    d_norm = (d_w * proj_weight.reshape(H).float()).to(norm_weight.dtype)
    d_proj = (d_w * norm_weight.reshape(H).float()).to(proj_weight.dtype).reshape(
        proj_weight.shape
    )
    return d_prefix, d_block, d_norm, d_proj


class _FusedAttnResMix(torch.autograd.Function):
    """Fused forward and fused backward.

    A raw Triton kernel returns a tensor with no `grad_fn`, so calling
    `attn_res_mix_triton` directly inside a training step silently produces no
    gradients -- the loss still falls, because every *other* path still trains.
    That is the failure mode this wrapper exists to prevent.

    The backward is now its own kernel (G59) rather than an eager recompute, so
    it avoids the `[T, K+1, H]` fp32 temporaries too and the memory win is no
    longer forward-only. Gradients are therefore no longer bit-identical to the
    eager path -- they reduce over `H` in tiles -- and are gated on a measured
    tolerance like the forward.
    """

    @staticmethod
    def forward(ctx, prefix_sum, block_residual, norm_weight, proj_weight, eps, block_h):
        ctx.save_for_backward(prefix_sum, block_residual, norm_weight, proj_weight)
        ctx.eps, ctx.block_h = eps, block_h
        return attn_res_mix_triton(
            prefix_sum, block_residual, norm_weight, proj_weight, eps, block_h
        )

    @staticmethod
    def backward(ctx, grad_out):
        prefix_sum, block_residual, norm_weight, proj_weight = ctx.saved_tensors
        dp, db, dn, dq = attn_res_mix_triton_bwd(
            grad_out, prefix_sum, block_residual, norm_weight, proj_weight,
            ctx.eps, ctx.block_h,
        )
        need = (prefix_sum, block_residual, norm_weight, proj_weight)
        got = (dp, db, dn, dq)
        return (*(g if t.requires_grad else None for t, g in zip(need, got)), None, None)


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
