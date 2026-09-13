"""Fused SiTU-GLU, forward and backward.

`situ_glu` is plain PyTorch, and the trace says what that costs: **10 distinct
aten ops, 350 launches, 5.05 ms** against a `k3.moe` region of 8.97 ms. More than
half the MoE region's time, for two transcendentals and a few multiplies, because
every step is a separate pass over an `[~8000, 3072]` tensor --

    gate/beta -> tanh -> *beta -> sigmoid(gate) -> * -> ... -> * up

Core would normally fuse an activation, but `bias_activation_fusion` only knows
SwiGLU and GeGLU, and the seam SiTU has to use (`use_te_activation_func`, because
core's GLU cannot express SiTU's tanh-limited *up* branch) explicitly disables
fusion. So G52 bought the right math at the cost of an unfused activation; this
gets the speed back.

One kernel: read `[gate | up]` once, compute in registers, write once.

    situ_a = beta * tanh(gate/beta) * sigmoid(gate)
    up_t   = linear_beta * tanh(up/linear_beta)
    y      = situ_a * up_t

Backward is the same shape. With `t_g = tanh(gate/beta)` and `t_u = tanh(up/lb)`:

    d(situ_a)/dgate = (1 - t_g^2) * s + beta * t_g * s * (1 - s)
    dL/dgate = dy * up_t * d(situ_a)/dgate
    dL/dup   = dy * situ_a * (1 - t_u^2)

so it is one read of `[gate|up]` plus one of `dy`, and one write of `[dgate|dup]`.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from .situ import SITU_BETA, SITU_LINEAR_BETA


if HAVE_TRITON:

    @triton.jit
    def _tanh(x):
        """`tl.math.tanh` is absent in this Triton build; 2*sigmoid(2x) - 1 is the
        same function and is stable for large |x| because sigmoid saturates."""
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _situ_fwd(X, Y, N, D, BETA, LB, HAS_LB: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        d0 = tl.program_id(1) * BD
        d = d0 + tl.arange(0, BD).to(tl.int64)
        m = d < D
        base = n * 2 * D
        g = tl.load(X + base + d, mask=m, other=0.0).to(tl.float32)
        u = tl.load(X + base + D + d, mask=m, other=0.0).to(tl.float32)
        tg = _tanh(g / BETA)
        s = tl.sigmoid(g)
        situ_a = BETA * tg * s
        up_t = LB * _tanh(u / LB) if HAS_LB else u
        tl.store(Y + n * D + d, (situ_a * up_t).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _situ_bwd(X, DY, DX, N, D, BETA, LB, HAS_LB: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        d0 = tl.program_id(1) * BD
        d = d0 + tl.arange(0, BD).to(tl.int64)
        m = d < D
        base = n * 2 * D
        g = tl.load(X + base + d, mask=m, other=0.0).to(tl.float32)
        u = tl.load(X + base + D + d, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(DY + n * D + d, mask=m, other=0.0).to(tl.float32)

        tg = _tanh(g / BETA)
        s = tl.sigmoid(g)
        situ_a = BETA * tg * s
        if HAS_LB:
            tu = _tanh(u / LB)
            up_t = LB * tu
            dup = dy * situ_a * (1.0 - tu * tu)
        else:
            up_t = u
            dup = dy * situ_a
        dsitu = (1.0 - tg * tg) * s + BETA * tg * s * (1.0 - s)
        dgate = dy * up_t * dsitu

        tl.store(DX + base + d, dgate.to(DX.dtype.element_ty), mask=m)
        tl.store(DX + base + D + d, dup.to(DX.dtype.element_ty), mask=m)


class _FusedSituGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, beta, linear_beta):
        ctx.save_for_backward(gate_up)
        ctx.beta, ctx.linear_beta = beta, linear_beta
        flat = gate_up.reshape(-1, gate_up.shape[-1])
        n, d = flat.shape[0], flat.shape[-1] // 2
        out = torch.empty(n, d, device=gate_up.device, dtype=gate_up.dtype)
        bd = min(1024, triton.next_power_of_2(d))
        _situ_fwd[(n, (d + bd - 1) // bd)](
            flat, out, n, d, beta, linear_beta or 1.0,
            HAS_LB=linear_beta is not None, BD=bd, num_warps=4, num_stages=1,
        )
        return out.reshape(*gate_up.shape[:-1], d)

    @staticmethod
    def backward(ctx, grad_out):
        (gate_up,) = ctx.saved_tensors
        flat = gate_up.reshape(-1, gate_up.shape[-1])
        n, d = flat.shape[0], flat.shape[-1] // 2
        dx = torch.empty_like(flat)
        bd = min(1024, triton.next_power_of_2(d))
        _situ_bwd[(n, (d + bd - 1) // bd)](
            flat, grad_out.reshape(-1, d).contiguous(), dx, n, d,
            ctx.beta, ctx.linear_beta or 1.0,
            HAS_LB=ctx.linear_beta is not None, BD=bd, num_warps=4, num_stages=1,
        )
        return dx.reshape(gate_up.shape), None, None


def fused_situ_glu(gate_up, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA):
    """Drop-in for `situ_glu`. Falls back when Triton or CUDA is unavailable."""
    if not HAVE_TRITON or not gate_up.is_cuda:
        from .situ import situ_glu

        return situ_glu(gate_up, beta=beta, linear_beta=linear_beta)
    return _FusedSituGLU.apply(gate_up.contiguous(), beta, linear_beta)
