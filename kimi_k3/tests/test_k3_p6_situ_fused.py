"""G61 -- the fused SiTU kernel against the eager oracle."""

import pytest
import torch

from kimi_k3.moe.situ import SITU_BETA, SITU_LINEAR_BETA, situ_glu
from kimi_k3.moe.situ_triton import HAVE_TRITON, fused_situ_glu

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="triton unavailable")


@pytest.mark.parametrize("dtype,bound", [(torch.float32, 1e-5), (torch.bfloat16, 1e-3)])
@pytest.mark.parametrize("shape", [(7, 64), (8072, 3072), (129, 6144)])
def test_forward_and_backward_match(dtype, bound, shape):
    """Both directions, in the tanh-limited regime.

    Inputs are scaled x3 on purpose: SiTU is defined by its limiting at beta=4 and
    linear_beta=25, and near zero it degenerates towards a plain product, where a
    wrong kernel would still look right (the same trap G53 documents).
    """
    n, d = shape
    g = torch.Generator("cuda").manual_seed(0)
    x = (torch.randn(n, 2 * d, device="cuda", dtype=dtype, generator=g) * 3).requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    situ_glu(x).sum().backward()
    fused_situ_glu(y).sum().backward()
    fwd = ((fused_situ_glu(y.detach()).float() - situ_glu(x.detach()).float()).norm()
           / situ_glu(x.detach()).float().norm()).item()
    grad = ((y.grad.float() - x.grad.float()).norm() / x.grad.float().norm()).item()
    assert fwd < bound, f"forward rel-L2 {fwd:.3e}"
    assert grad < bound, f"grad rel-L2 {grad:.3e}"


def test_no_linear_beta_branch():
    """`linear_beta=None` leaves the up branch untouched; both paths must agree."""
    g = torch.Generator("cuda").manual_seed(1)
    x = torch.randn(64, 256, device="cuda", dtype=torch.float32, generator=g)
    a = situ_glu(x, beta=SITU_BETA, linear_beta=None)
    b = fused_situ_glu(x, beta=SITU_BETA, linear_beta=None)
    assert ((b - a).norm() / a.norm()).item() < 1e-5


def test_saturates_like_the_reference():
    """Huge inputs: tanh must saturate rather than overflow.

    `tl.math.tanh` is absent in this Triton build so the kernel uses
    2*sigmoid(2x)-1. That identity is only safe because sigmoid saturates; this
    asserts it does rather than trusting the algebra.
    """
    x = torch.full((4, 8), 1e4, device="cuda", dtype=torch.float32)
    out = fused_situ_glu(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, situ_glu(x), rtol=1e-5, atol=1e-5)


def test_the_module_uses_the_kernel_by_default():
    from kimi_k3.moe.situ import SituGLU

    assert SituGLU(None).fused is True
