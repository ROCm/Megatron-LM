"""P0 -- torch.compile must keep working on the Triton we pinned.

We upgraded Triton (3.6.0 -> 3.7.1) to unblock fla's KDA backward, and Triton is
not a package torch is indifferent to: inductor generates Triton code against its
internal APIs. Megatron also puts `torch.compile` on the *default* training path
-- `megatron/core/jit.py` sets `jit_fuser = torch.compile` for torch >= 2.2, and
that decorator is applied to the bias-swiglu, bias-dropout, activation, norm,
MoE-router and token-dispatcher fusions.

So this is not a hypothetical: if the pinned Triton and torch disagree, ordinary
K3 training breaks, not just a P11 experiment. These tests are cheap and run in
CI stage 1.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_inductor_can_import_its_triton_shim():
    from torch.utils._triton import has_triton

    assert has_triton()
    import torch._inductor.runtime.triton_compat  # noqa: F401


def test_compiled_forward_and_backward_match_eager():
    def fn(x, w):
        return torch.nn.functional.gelu(x @ w).sum(-1)

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, 128, device="cuda", requires_grad=True)

    ref = fn(x, w)
    ref.sum().backward()
    ref_gx, ref_gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None

    out = torch.compile(fn)(x, w)
    out.sum().backward()

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad, ref_gx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w.grad, ref_gw, rtol=1e-4, atol=1e-4)


def test_megatron_jit_fuser_is_torch_compile_and_still_runs():
    """The default training path, not an opt-in one."""
    from megatron.core.jit import jit_fuser
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    assert jit_fuser is torch.compile, (
        "core stopped routing its fusions through torch.compile; re-check whether "
        "the Triton pin still matters for the default path"
    )
    inp = torch.randn(32, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    out = bias_swiglu_impl(inp, bias)
    out.sum().backward()
    assert out.shape == (32, 2, 128)
    assert torch.isfinite(inp.grad).all()
