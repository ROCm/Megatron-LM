import os
import unittest.mock

import pytest
import torch

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl


def _te_swiglu_available():
    try:
        import transformer_engine_torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.skipif(not _te_swiglu_available(), reason="transformer_engine_torch not installed")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(16, 64), (4, 8, 64)])
def test_te_swiglu_forward_backward(input_dtype, shape):
    """Verify NVTE_SWIGLU=1 (TE kernel) matches the default fused path for SwiGLUFunction."""
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-5, atol=1.0e-5)
    else:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)

    x = torch.randn(shape, dtype=input_dtype, device="cuda")
    grad_out_shape = list(shape)
    grad_out_shape[-1] //= 2
    grad_out = torch.randn(grad_out_shape, dtype=input_dtype, device="cuda")

    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = bias_swiglu_impl(x_ref, None)
    y_ref.backward(grad_out)

    import megatron.core.fusions.fused_bias_swiglu as _mod
    with unittest.mock.patch.object(_mod, "_use_te_swiglu", True):
        x_te = x.clone().detach().requires_grad_(True)
        y_te = bias_swiglu_impl(x_te, None)
        y_te.backward(grad_out)

    assert y_te.shape == y_ref.shape
    assert torch.allclose(y_te, y_ref, **tols), (
        f"Forward mismatch: max diff = {(y_te - y_ref).abs().max().item()}"
    )
    assert x_te.grad.shape == x_ref.grad.shape
    assert torch.allclose(x_te.grad, x_ref.grad, **tols), (
        f"Backward mismatch: max diff = {(x_te.grad - x_ref.grad).abs().max().item()}"
    )
