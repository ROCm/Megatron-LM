"""Fused KDA elementwise kernels: gated RMSNorm and the causal short conv.

Both are transcriptions of things the release ships *fused* and we wrote out
eagerly -- `gated_rms_norm`'s own docstring names its counterpart
`FusedRMSNormGated`. Measured at production shape (B 1, T 8192, D 12288, H 96,
head_dim 128, W 4), per call:

    gated_rms_norm     1.200 ms fwd, 3.924 ms fwd+bwd   (~8 aten ops)
    causal_short_conv  1.012 ms fwd, 3.287 ms fwd+bwd   (pad, 2 transposes,
                                                         miopen depthwise, silu)

They run in 69 of 93 layers, so this is not a rounding error at full depth.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _grn_fwd(X, W, G, Y, RSTD, N, D, EPS, BD: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, BD)
        m = d < D
        x = tl.load(X + row * D + d, mask=m, other=0.0).to(tl.float32)
        r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + EPS)
        w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + row * D + d, mask=m, other=0.0).to(tl.float32)
        tl.store(RSTD + row, r)
        tl.store(Y + row * D + d, (x * r * w * tl.sigmoid(g)).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _grn_bwd(X, W, G, DY, RSTD, DX, DG, DWPART, N, D, BD: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, BD)
        m = d < D
        x = tl.load(X + row * D + d, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + row * D + d, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * D + d, mask=m, other=0.0).to(tl.float32)
        r = tl.load(RSTD + row)

        s = tl.sigmoid(g)
        n = x * r
        # y = n * w * s
        dn = dy * w * s
        tl.store(DG + row * D + d, (dy * n * w * s * (1.0 - s)).to(DG.dtype.element_ty), mask=m)
        # n = x * rstd(x): the reduction couples every element of the row
        dot = tl.sum(dn * x, axis=0)
        dx = dn * r - (r * r * r / D) * x * dot
        tl.store(DX + row * D + d, dx.to(DX.dtype.element_ty), mask=m)
        # dw is the cross-row reduction; partials here, summed by the caller.
        tl.store(DWPART + tl.program_id(0) * D + d, dy * n * s, mask=m)


class _FusedGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, gate, eps):
        shape = x.shape
        d = shape[-1]
        xf = x.reshape(-1, d).contiguous()
        gf = gate.reshape(-1, d).contiguous()
        n = xf.shape[0]
        y = torch.empty_like(xf)
        rstd = torch.empty(n, device=x.device, dtype=torch.float32)
        bd = triton.next_power_of_2(d)
        _grn_fwd[(n,)](xf, weight, gf, y, rstd, n, d, eps, BD=bd, num_warps=4, num_stages=1)
        ctx.save_for_backward(xf, weight, gf, rstd)
        ctx.shape, ctx.d = shape, d
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, grad_out):
        xf, weight, gf, rstd = ctx.saved_tensors
        d = ctx.d
        n = xf.shape[0]
        dy = grad_out.reshape(-1, d).contiguous()
        dx = torch.empty_like(xf)
        dg = torch.empty_like(gf)
        dwpart = torch.empty(n, d, device=xf.device, dtype=torch.float32)
        bd = triton.next_power_of_2(d)
        _grn_bwd[(n,)](xf, weight, gf, dy, rstd, dx, dg, dwpart, n, d,
                       BD=bd, num_warps=4, num_stages=1)
        dw = dwpart.sum(0).to(weight.dtype)
        return dx.reshape(ctx.shape), dw, dg.reshape(ctx.shape), None


def fused_gated_rms_norm(x, weight, gate, eps):
    """Drop-in for `kda.gated_rms_norm`."""
    if not HAVE_TRITON or not x.is_cuda:
        from .kda import gated_rms_norm

        return gated_rms_norm(x, weight, gate, eps)
    return _FusedGatedRMSNorm.apply(x, weight, gate, eps)


if HAVE_TRITON:

    @triton.jit
    def _silu(z):
        return z * tl.sigmoid(z)

    @triton.jit
    def _conv_fwd(X, WT, Y, B, T, D, W: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
        """y[b,t,d] = silu(sum_k x[b, t-W+1+k, d] * wt[d,k]).

        The left pad is arithmetic, not a materialised copy: taps whose source
        index is negative are masked to zero. That removes the `F.pad` temporary
        and both transposes that `F.conv1d` needs, which is most of the traffic.
        """
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        bi = tl.program_id(2).to(tl.int64)
        t = pid_t * BT + tl.arange(0, BT).to(tl.int64)
        d = pid_d * BD + tl.arange(0, BD).to(tl.int64)
        tm = t < T
        dm = d < D
        acc = tl.zeros((BT, BD), dtype=tl.float32)
        for k in tl.static_range(W):
            src = t - (W - 1) + k
            ok = (src >= 0) & tm
            xv = tl.load(X + bi * T * D + src[:, None] * D + d[None, :],
                         mask=ok[:, None] & dm[None, :], other=0.0).to(tl.float32)
            wv = tl.load(WT + d * W + k, mask=dm, other=0.0).to(tl.float32)
            acc += xv * wv[None, :]
        tl.store(Y + bi * T * D + t[:, None] * D + d[None, :],
                 _silu(acc).to(Y.dtype.element_ty), mask=tm[:, None] & dm[None, :])

    @triton.jit
    def _conv_bwd(X, WT, DY, DX, DWPART, B, T, D, NT,
                  W: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
        """dx and the per-tile dw partials.

        `pre` is recomputed rather than saved: it is the same W loads the forward
        did, against an extra [B, T, D] fp32 tensor to stash. Recompute is the
        cheaper trade at these widths.
        """
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        bi = tl.program_id(2).to(tl.int64)
        t = pid_t * BT + tl.arange(0, BT).to(tl.int64)
        d = pid_d * BD + tl.arange(0, BD).to(tl.int64)
        tm = t < T
        dm = d < D

        # dpre[t] = dy[t] * silu'(pre[t]); needed at t and at shifted positions.
        dx = tl.zeros((BT, BD), dtype=tl.float32)
        for j in tl.static_range(W):
            # position that consumes x[t] with tap (W-1-j) is t + j
            tt = t + j
            okt = (tt < T) & tm
            pre = tl.zeros((BT, BD), dtype=tl.float32)
            for k in tl.static_range(W):
                src = tt - (W - 1) + k
                ok = (src >= 0) & okt
                xv = tl.load(X + bi * T * D + src[:, None] * D + d[None, :],
                             mask=ok[:, None] & dm[None, :], other=0.0).to(tl.float32)
                wv = tl.load(WT + d * W + k, mask=dm, other=0.0).to(tl.float32)
                pre += xv * wv[None, :]
            s = tl.sigmoid(pre)
            dsilu = s + pre * s * (1.0 - s)
            dyv = tl.load(DY + bi * T * D + tt[:, None] * D + d[None, :],
                          mask=okt[:, None] & dm[None, :], other=0.0).to(tl.float32)
            dpre = dyv * dsilu
            wv = tl.load(WT + d * W + (W - 1 - j), mask=dm, other=0.0).to(tl.float32)
            dx += dpre * wv[None, :]
            # dw[d, W-1-j] += sum_t dpre[t+j] * x[t]
            xv = tl.load(X + bi * T * D + t[:, None] * D + d[None, :],
                         mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            part = tl.sum(dpre * xv, axis=0)
            # Indexed by batch as well as time-tile: without the `bi` term every
            # batch writes the same slot and the last one wins, which showed up as
            # a correct dx and a dw off by 5e-01.
            slot = bi * NT + tl.program_id(0)
            tl.store(DWPART + (slot * D + d) * W + (W - 1 - j), part, mask=dm)
        tl.store(DX + bi * T * D + t[:, None] * D + d[None, :],
                 dx.to(DX.dtype.element_ty), mask=tm[:, None] & dm[None, :])


class _FusedCausalShortConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        b, t, d = x.shape
        w = weight.shape[-1]
        wt = weight.reshape(d, w).contiguous()
        x = x.contiguous()
        y = torch.empty_like(x)
        bt, bd = 64, min(128, triton.next_power_of_2(d))
        grid = ((t + bt - 1) // bt, (d + bd - 1) // bd, b)
        _conv_fwd[grid](x, wt, y, b, t, d, W=w, BT=bt, BD=bd, num_warps=4, num_stages=1)
        ctx.save_for_backward(x, wt)
        ctx.w, ctx.bt, ctx.bd = w, bt, bd
        ctx.weight_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, wt = ctx.saved_tensors
        b, t, d = x.shape
        w, bt, bd = ctx.w, ctx.bt, ctx.bd
        nt = (t + bt - 1) // bt
        dx = torch.empty_like(x)
        dwpart = torch.zeros(b * nt, d, w, device=x.device, dtype=torch.float32)
        grid = (nt, (d + bd - 1) // bd, b)
        _conv_bwd[grid](x, wt, grad_out.contiguous(), dx, dwpart, b, t, d, nt,
                        W=w, BT=bt, BD=bd, num_warps=4, num_stages=1)
        dw = dwpart.sum(0).reshape(ctx.weight_shape)
        return dx, dw.to(wt.dtype)


def fused_causal_short_conv(x, weight, activation="silu"):
    """Drop-in for `kda.causal_short_conv` (silu only, which is all K3 uses)."""
    if not HAVE_TRITON or not x.is_cuda or activation != "silu":
        from .kda import causal_short_conv

        return causal_short_conv(x, weight, activation)
    return _FusedCausalShortConv.apply(x, weight.contiguous())
