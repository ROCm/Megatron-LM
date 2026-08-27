"""The AITER a8w4 forward for routed experts, with an STE backward.

MXFP8 activations against MXFP4 weights, which is what the release serves. The
kernel is not differentiable and never will be, so the backward is a
straight-through estimator: gradients are taken against the **dequantised**
weights and passed to the fp32 masters. That is the same contract
`k3_qat_experts.py` already proves at G40; this module only swaps in the fast
forward.

Two things about the kernel's interface are easy to get wrong and silent:

* **weights are `[experts, K, N]`**, not `[N, K]` -- the transpose of a
  `torch.nn.Linear` weight;
* **the weight scales must be swizzled** (`swizzle_scales`, and then
  `swizzle_mx_scale="CDNA4_SCALE"`). Passing unswizzled scales does not fail: it
  returns a plausible tensor that is wrong by **rel-L2 0.53**, because the kernel
  reads each scale from the wrong position. `results/a8w4.md` records this;
  `assert_swizzle_matters` is the guard.

The plan named `aiter/ops/opus/moe_stage{1,2}_a8w4.py`. That module does not exist
in the pinned AITER checkout (P0's finding D1) -- `aiter.ops.triton.moe.moe_op_gemm_a8w4`
does, it needs no rebuild because it is Triton, and it is what is used here.
"""

from typing import Optional, Tuple

import torch

#: The kernel wants K divisible by 32*8 and N by 32 before it will take swizzled
#: scales; below that it wants them plain. Both are supported, but only the
#: swizzled path is fast, and K3's routed experts (3584 -> 3072) satisfy it.
SWIZZLE_K_MULTIPLE = 32 * 8
SWIZZLE_N_MULTIPLE = 32


def available() -> bool:
    try:
        import aiter.ops.triton.moe.moe_op_gemm_a8w4  # noqa: F401
        import aiter.ops.triton.moe.quant_moe  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _swizzle(scales: torch.Tensor, n: int, k: int) -> Tuple[torch.Tensor, Optional[str]]:
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import swizzle_scales

    if n % SWIZZLE_N_MULTIPLE == 0 and k % SWIZZLE_K_MULTIPLE == 0:
        return swizzle_scales(scales), "CDNA4_SCALE"
    return scales, None


def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
    """A `[N, K]` linear weight -> the kernel's `[1, K, N]` MXFP4 form."""
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    n, k = weight.shape
    packed, scales = downcast_to_mxfp(weight.t().contiguous().unsqueeze(0).bfloat16(),
                                      torch.uint8, axis=1)
    swizzled, mode = _swizzle(scales, n, k)
    return packed, swizzled, mode


def _single_expert_routing(tokens: int, device):
    from aiter.ops.triton.moe.moe_routing.routing import routing

    return routing(torch.zeros(tokens, 1, device=device, dtype=torch.bfloat16), 1)


class _A8W4Matmul(torch.autograd.Function):
    """Forward through the kernel; backward against the dequantised weight."""

    @staticmethod
    def forward(ctx, x, master, packed, scales, mode, dequantized):
        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
        from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp

        flat = x.reshape(-1, x.shape[-1])
        xq, xs = downcast_to_mxfp(flat.bfloat16(), torch.float8_e4m3fn, axis=1)
        rdata, gather, _ = _single_expert_routing(flat.shape[0], x.device)
        out = moe_gemm_a8w4(
            xq, packed, xs, scales, None, None, None, rdata,
            gather_indx=gather, swizzle_mx_scale=mode,
        )
        # wgrad is taken against the activations that actually multiplied the
        # weight -- the quantised ones -- not the originals. Using the originals
        # is the easy mistake: it still trains, and the gradient is subtly wrong.
        ctx.save_for_backward(upcast_from_mxfp(xq, xs, torch.bfloat16, axis=1), dequantized)
        ctx.shape = x.shape
        return out.reshape(*x.shape[:-1], out.shape[-1]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        quantized_x, dequantized = ctx.saved_tensors
        g = grad_out.reshape(-1, grad_out.shape[-1]).float()
        grad_x = (g @ dequantized.float()).reshape(ctx.shape).to(grad_out.dtype)
        grad_w = (g.t() @ quantized_x.float()).to(grad_out.dtype)
        return grad_x, grad_w, None, None, None, None


def a8w4_linear(x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
    """`F.linear(x, master)` with an MXFP8 x MXFP4 forward and an STE backward.

    `master` is the fp32 (or bf16) weight in `[N, K]` linear layout. It is
    quantised here rather than read from a cache, so this is the honest but slow
    call; the expert module holds the cache.
    """
    from .k3_qat import fake_quantize_mxfp4

    packed, scales, mode = quantize_weight(master)
    dequantized = fake_quantize_mxfp4(master.detach().float()).t().contiguous()
    return _A8W4Matmul.apply(x, master, packed, scales, mode, dequantized.t().contiguous())


def assert_swizzle_matters(x: torch.Tensor, master: torch.Tensor) -> Tuple[float, float]:
    """Return (rel-L2 swizzled, rel-L2 unswizzled) against the dequantised matmul.

    The second number is the reason this helper exists: an unswizzled call does
    not raise, it just returns the wrong answer.
    """
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp

    n, k = master.shape
    packed, raw = downcast_to_mxfp(
        master.t().contiguous().unsqueeze(0).bfloat16(), torch.uint8, axis=1
    )
    swizzled, mode = _swizzle(raw, n, k)
    flat = x.reshape(-1, k)
    xq, xs = downcast_to_mxfp(flat.bfloat16(), torch.float8_e4m3fn, axis=1)
    rdata, gather, _ = _single_expert_routing(flat.shape[0], x.device)

    reference = (
        upcast_from_mxfp(xq, xs, torch.bfloat16, axis=1).float()
        @ upcast_from_mxfp(packed, raw, torch.bfloat16, axis=1).float()[0]
    )
    scale = reference.norm()

    def error(scales, swizzle_mode):
        out = moe_gemm_a8w4(xq, packed, xs, scales, None, None, None, rdata,
                            gather_indx=gather, swizzle_mx_scale=swizzle_mode)
        return ((out.float().reshape(reference.shape) - reference).norm() / scale).item()

    return error(swizzled, mode), error(raw, mode)
