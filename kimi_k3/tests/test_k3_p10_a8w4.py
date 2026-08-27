"""P10 / gates G39 and G40 -- the a8w4 forward and the STE backward.

The plan's kernel path (`aiter/ops/opus/moe_stage{1,2}_a8w4`) is not in the pinned
AITER checkout -- P0 recorded that as finding D1 and deferred it to a checkout
bump. It turns out no bump is needed: `aiter.ops.triton.moe.moe_op_gemm_a8w4` is
Triton, is present, and runs on this gfx950 box.
"""

import pytest
import torch

from kimi_k3.moe import k3_a8w4
from kimi_k3.moe.k3_qat import fake_quantize_mxfp4, ste_mxfp4, ste_mxfp8

pytestmark = pytest.mark.skipif(not k3_a8w4.available(), reason="needs a GPU with AITER")

#: The kernel accumulates in bf16, so this is accumulation noise, not format loss.
A8W4_VS_DEQUANTIZED = 5e-3

SHAPES = [(256, 256, 32), (512, 1024, 64), pytest.param((3584, 3072, 128), marks=pytest.mark.slow)]


def operands(k, n, m, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda") * 0.02
    return x, w


@pytest.mark.parametrize("shape", SHAPES)
def test_the_kernel_matches_the_dequantised_matmul(shape):
    """G39, against the dequantisation of the *same* quantised operands.

    That is the comparison that tests the kernel. Comparing against a separately
    quantised reference would instead measure two quantisers agreeing, which is a
    different question and a looser number.
    """
    x, w = operands(*shape)
    swizzled, _ = k3_a8w4.assert_swizzle_matters(x, w)
    assert swizzled < A8W4_VS_DEQUANTIZED, swizzled


@pytest.mark.parametrize("shape", SHAPES[:2])
def test_unswizzled_scales_are_silently_wrong(shape):
    """The failure mode this is guarding: no error, just the wrong answer.

    The kernel reads each weight scale from the wrong position and returns a
    plausible tensor that is off by half its own magnitude.
    """
    x, w = operands(*shape)
    swizzled, unswizzled = k3_a8w4.assert_swizzle_matters(x, w)
    assert unswizzled > 0.3, unswizzled
    assert unswizzled > 100 * swizzled


def test_the_backward_is_a_straight_through_estimator():
    """G40: given the same incoming gradient, the STE backward is *exact*.

    `gradcheck` is invalid here by construction -- an STE backward is
    deliberately not the derivative of its forward (rule R4.4) -- so the
    comparison is against an explicit fake-quant reference, and it is exact
    rather than approximate.
    """
    x, w = operands(512, 256, 32)
    x = x.float().requires_grad_(True)
    master = w.clone().requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_w = w.clone().requires_grad_(True)

    torch.manual_seed(9)
    seed_grad = torch.randn(32, 256, device="cuda")

    k3_a8w4.a8w4_linear(x, master).backward(seed_grad)
    torch.nn.functional.linear(
        ste_mxfp8(reference_x), ste_mxfp4(reference_w)
    ).backward(seed_grad)

    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(master.grad, reference_w.grad, rtol=0, atol=0)


def test_the_gradient_reaches_the_master_not_the_packed_copy():
    """An STE that silently zeroes the weight gradient still trains -- badly."""
    x, w = operands(256, 128, 16)
    master = w.clone().requires_grad_(True)
    k3_a8w4.a8w4_linear(x.float(), master).sum().backward()
    assert master.grad is not None
    assert float(master.grad.abs().max()) > 0


def test_the_plan_s_kernel_path_is_still_absent():
    """Records why the Triton op is used, so the substitution stays visible."""
    with pytest.raises(ImportError):
        import aiter.ops.opus.moe_stage1_a8w4  # noqa: F401


def test_aiter_has_two_mxfp4_quantisers_and_we_match_the_serving_one():
    """`downcast_to_mxfp` and `per_1x32_f4_quant` round differently.

    Both are valid MXFP4 and their errors are within 0.02 % of each other, but
    only one is byte-identical to ours -- `per_1x32_f4_quant`, which is the
    `QuantType.per_1x32` path the release serves through. Asserting this stops
    "cross-checked against AITER" from drifting to the other entry point.
    """
    import aiter
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    from kimi_k3.moe.k3_qat import quantize_mxfp4

    torch.manual_seed(0)
    w = (torch.randn(256, 512, device="cuda") * 0.02).bfloat16()
    packed, scales = quantize_mxfp4(w.float())
    assert torch.equal(aiter.per_1x32_f4_quant(w)[0].view(torch.uint8), packed)

    other, _ = downcast_to_mxfp(w.t().contiguous().unsqueeze(0), torch.uint8, axis=1)
    assert not torch.equal(other[0].t().contiguous(), packed)
