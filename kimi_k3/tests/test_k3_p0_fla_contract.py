"""P0 / gate G1 -- the released `chunk_kda` call must actually work.

This check is **functional, not signature-based**, and that is deliberate. An
earlier version of this project inspected `inspect.signature(chunk_kda)`,
concluded that `A_log` / `dt_bias` / `transpose_state_layout` were being silently
swallowed by `**kwargs`, and was wrong: fla reads them from kwargs on purpose. A
name-based check would have failed on a working library
(develop/notes/2026-08-27-fla-signature-check.md §1, rule R3.6).

So: call it the way the release does, and assert behaviour.
"""

import pytest
import torch

fla = pytest.importorskip("fla.ops.kda", reason="fla is a pinned optional backend")
chunk_kda = fla.chunk_kda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: Exactly the kwargs `KimiDeltaAttention.forward` passes in the release.
RELEASED_KWARGS = dict(
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
    transpose_state_layout=True,
)


def _inputs(T=128, H=4, K=64, requires_grad=True):
    torch.manual_seed(0)
    mk = lambda *s, d=torch.bfloat16: torch.randn(
        *s, device="cuda", dtype=d, requires_grad=requires_grad
    )
    return dict(
        q=mk(1, T, H, K),
        k=mk(1, T, H, K),
        v=mk(1, T, H, K),
        g=mk(1, T, H, K),
        beta=mk(1, T, H, d=torch.float32),
        A_log=torch.rand(H, device="cuda", dtype=torch.float32)
        .log_()
        .requires_grad_(requires_grad),
        dt_bias=mk(H * K, d=torch.float32),
    )


def test_released_call_runs_forward():
    args = _inputs()
    out, state = chunk_kda(**args, **RELEASED_KWARGS)
    assert out.shape == args["q"].shape
    assert torch.isfinite(out.float()).all()
    assert state is not None


def test_a_log_is_not_ignored():
    """The claim that fla drops `A_log` was wrong; this is what disproves it."""
    args = _inputs(requires_grad=False)
    with torch.no_grad():
        a, _ = chunk_kda(**args, **RELEASED_KWARGS)
        b, _ = chunk_kda(**{**args, "A_log": args["A_log"] * 0 + 3.0}, **RELEASED_KWARGS)
    assert not torch.equal(a, b)
    assert (a.float() - b.float()).abs().max() > 1e-3


def test_transpose_state_layout_is_honoured():
    """It is an accepted alias for `state_v_first`, not an ignored kwarg."""
    args = _inputs(requires_grad=False)
    with torch.no_grad():
        _, s_true = chunk_kda(**args, **RELEASED_KWARGS)
        _, s_false = chunk_kda(**args, **{**RELEASED_KWARGS, "transpose_state_layout": False})
    assert s_true.shape != s_false.shape or not torch.equal(s_true, s_false)


def test_missing_a_log_is_rejected_not_ignored():
    args = _inputs(requires_grad=False)
    args.pop("A_log")
    with pytest.raises(Exception):
        chunk_kda(**args, **{**RELEASED_KWARGS, "lower_bound": None})


@pytest.mark.parametrize("shape", [(128, 4, 64), (2048, 8, 128)])
def test_backward_compiles_and_produces_finite_gradients(shape):
    """The G1 blocker until triton 3.7.1: `chunk_intra` failed to compile on gfx950.

    Keep the loss a `sum`, not a `mean`: averaging over millions of elements puts
    the gradients near 1e-6 and makes a healthy result look like zeros.
    """
    T, H, K = shape
    args = _inputs(T, H, K)
    out, _ = chunk_kda(**args, **RELEASED_KWARGS)
    out.float().pow(2).sum().backward()

    for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"):
        grad = args[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{name} gradient is all zero"
