"""G62 -- the fused KDA kernels against their eager oracles."""

import pytest
import torch

from kimi_k3.attention.kda import causal_short_conv, gated_rms_norm
from kimi_k3.attention.kda_triton import (
    HAVE_TRITON, fused_causal_short_conv, fused_gated_rms_norm,
)

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="triton unavailable")
BOUNDS = {torch.float32: 1e-5, torch.bfloat16: 5e-3}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B", [1, 2, 3])
def test_short_conv_matches(dtype, B):
    """Several batch sizes on purpose.

    The first version of the backward indexed its `dw` partials by time-tile only,
    so every batch wrote the same slot and the last one won. `dx` and the forward
    were both correct; only `dw` was wrong, and only for B > 1.
    """
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(B, 129, 512, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    w = torch.randn(512, 1, 4, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    x2, w2 = (t.detach().clone().requires_grad_(True) for t in (x, w))
    causal_short_conv(x, w).sum().backward()
    fused_causal_short_conv(x2, w2).sum().backward()
    rel = lambda a, b: ((b.float() - a.float()).norm() / a.float().norm().clamp_min(1e-12)).item()
    bound = BOUNDS[dtype]
    assert rel(x.grad, x2.grad) < bound, f"dx {rel(x.grad, x2.grad):.3e}"
    assert rel(w.grad, w2.grad) < bound, f"dw {rel(w.grad, w2.grad):.3e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_rms_norm_matches(dtype):
    g = torch.Generator("cuda").manual_seed(0)
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    x, w, gate = mk(2, 129, 8, 128), mk(128), mk(2, 129, 8, 128)
    x2, w2, g2 = (t.detach().clone().requires_grad_(True) for t in (x, w, gate))
    gated_rms_norm(x, w, gate, 1e-5).sum().backward()
    fused_gated_rms_norm(x2, w2, g2, 1e-5).sum().backward()
    rel = lambda a, b: ((b.float() - a.float()).norm() / a.float().norm().clamp_min(1e-12)).item()
    bound = BOUNDS[dtype]
    for name, a, b in (("dx", x.grad, x2.grad), ("dw", w.grad, w2.grad), ("dgate", gate.grad, g2.grad)):
        assert rel(a, b) < bound, f"{name} {rel(a, b):.3e}"


def test_causality_is_preserved():
    """A late token must not influence an earlier output.

    The kernel replaces `F.pad` with masked loads, so an off-by-one in the tap
    offset would leak the future and still look numerically plausible.
    """
    g = torch.Generator("cuda").manual_seed(3)
    x = torch.randn(1, 32, 64, device="cuda", dtype=torch.float32, generator=g)
    w = torch.randn(64, 1, 4, device="cuda", dtype=torch.float32, generator=g)
    a = fused_causal_short_conv(x, w)
    x2 = x.clone()
    x2[:, 20:] += 100.0
    b = fused_causal_short_conv(x2, w)
    torch.testing.assert_close(a[:, :20], b[:, :20], rtol=1e-5, atol=1e-5)
