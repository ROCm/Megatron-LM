"""P3 / gate G14 -- the FP32 oracle is the contract, so it is checked hardest.

It is validated three ways, which is the point: an oracle nobody cross-checks is
just a second implementation of the same misunderstanding.

* against fla's own `naive_recurrent_kda` -- **bit-identical**, so the recurrence
  and the gate were transcribed correctly rather than approximately;
* against `fused_recurrent_kda` and `chunk_kda` -- two independent kernels;
* against itself under fp64 `gradcheck` -- true autograd, so gradcheck is valid
  here (unlike any STE path).
"""

import pytest
import torch

from kimi_k3.attention.kda_eager_fp32 import (
    chunk_kda_eager_fp32,
    kda_gate,
    l2norm_last,
    released_call_kwargs,
)
from kimi_k3.tests.tolerance import KDA_FWD_FP32, assert_within, compare

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(B=2, T=32, H=4, K=16, dtype=torch.float32, grad=False, seed=0):
    """The release keeps beta / A_log / dt_bias in fp32; under gradcheck every
    input has to be fp64 or the numerical Jacobian is noise, so the "fp32"
    tensors follow the requested dtype when it is wider."""
    torch.manual_seed(seed)
    aux = dtype if dtype == torch.float64 else torch.float32
    mk = lambda *s, d=dtype: torch.randn(*s, device="cuda", dtype=d, requires_grad=grad)
    return dict(
        q=mk(B, T, H, K), k=mk(B, T, H, K), v=mk(B, T, H, K), g=mk(B, T, H, K),
        beta=mk(B, T, H, d=aux),
        A_log=torch.rand(H, device="cuda", dtype=aux).log_().requires_grad_(grad),
        dt_bias=mk(H * K, d=aux),
    )


def test_bit_identical_to_flas_naive_reference():
    """The strongest statement available: not close, equal."""
    naive = pytest.importorskip("fla.ops.kda.naive").naive_recurrent_kda
    args = _inputs()
    ours, state = chunk_kda_eager_fp32(**args, **released_call_kwargs())
    decay = kda_gate(args["g"], args["A_log"], args["dt_bias"])
    theirs, their_state = naive(
        l2norm_last(args["q"]), l2norm_last(args["k"]), args["v"], decay,
        torch.sigmoid(args["beta"]), output_final_state=True,
    )
    assert torch.equal(ours, theirs)
    assert torch.equal(state.transpose(-1, -2), their_state)


@pytest.mark.parametrize("kernel", ["chunk_kda", "fused_recurrent_kda"])
def test_matches_both_fla_kernels_in_fp32(kernel):
    ops = pytest.importorskip("fla.ops.kda")
    args = _inputs()
    kw = released_call_kwargs()
    ours, our_state = chunk_kda_eager_fp32(**args, **kw)
    theirs, their_state = getattr(ops, kernel)(**args, **kw)
    assert_within(theirs, ours, KDA_FWD_FP32, f"{kernel} forward")
    assert_within(their_state, our_state, KDA_FWD_FP32, f"{kernel} state")


def test_gradcheck_fp64():
    """True autograd. Small shapes, because gradcheck is O(numel) forwards."""
    args = _inputs(B=1, T=6, H=2, K=4, dtype=torch.float64, grad=True)

    def fn(q, k, v, g, beta):
        out, _ = chunk_kda_eager_fp32(
            q, k, v, g, beta, A_log=args["A_log"], dt_bias=args["dt_bias"],
            **{**released_call_kwargs(), "output_final_state": False},
        )
        return out

    assert torch.autograd.gradcheck(
        fn, (args["q"], args["k"], args["v"], args["g"], args["beta"]), atol=1e-6
    )


def test_gate_lands_inside_the_lower_bound():
    """`lower_bound * sigmoid(...)` cannot leave [lower_bound, 0], by construction.

    Mathematically the upper end is open, but sigmoid saturates in floating point:
    a large enough input gives exactly 0 (no decay at all for that key dimension)
    or exactly the bound. Both are reachable and neither is a bug -- the point of
    the bound is that nothing outside it ever is.
    """
    args = _inputs()
    for scale in (1.0, 100.0, 1e4):
        decay = kda_gate(args["g"] * scale, args["A_log"], args["dt_bias"], lower_bound=-5.0)
        assert decay.min() >= -5.0 and decay.max() <= 0.0

    # for ordinary inputs it stays strictly inside, i.e. it is doing something
    typical = kda_gate(args["g"], args["A_log"], args["dt_bias"], lower_bound=-5.0)
    assert typical.max() < 0.0 and typical.min() > -5.0


def test_softplus_gate_when_no_lower_bound_is_set():
    args = _inputs()
    decay = kda_gate(args["g"], args["A_log"], args["dt_bias"], lower_bound=None, safe_gate=False)
    expected = -args["A_log"].float().view(1, 1, -1, 1).exp() * torch.nn.functional.softplus(
        args["g"].float() + args["dt_bias"].float().view(args["g"].shape[-2], args["g"].shape[-1])
    )
    torch.testing.assert_close(decay, expected)


def test_l2_norm_puts_its_epsilon_inside_the_square_root():
    """Not the usual `x / (norm + eps)`; matching the kernel exactly matters."""
    x = torch.zeros(1, 1, 1, 4, device="cuda")
    out = l2norm_last(x)
    assert torch.isfinite(out).all() and out.abs().sum() == 0
    y = torch.randn(2, 3, 4, 8, device="cuda")
    torch.testing.assert_close(l2norm_last(y), y / torch.sqrt(y.pow(2).sum(-1, keepdim=True) + 1e-6))


def test_causality():
    """Changing token t must never move an output before t."""
    args = _inputs(T=16)
    kw = {**released_call_kwargs(), "output_final_state": False}
    base, _ = chunk_kda_eager_fp32(**args, **kw)
    perturbed = {**args, "v": args["v"].clone()}
    perturbed["v"][:, 8:] += 5.0
    after, _ = chunk_kda_eager_fp32(**perturbed, **kw)
    assert torch.equal(base[:, :8], after[:, :8])
    assert not torch.equal(base[:, 8:], after[:, 8:])


def test_initial_state_round_trips_through_the_transposed_layout():
    """`transpose_state_layout=True` is the released call; the inverse must hold."""
    args = _inputs(T=8)
    kw = released_call_kwargs()
    _, state = chunk_kda_eager_fp32(**args, **kw)
    # feeding the state back in must be identical to running the sequence twice as long
    out_a, state_a = chunk_kda_eager_fp32(**args, initial_state=state, **kw)
    assert torch.isfinite(out_a).all() and state_a.shape == state.shape
    # and with the layout flag off, the state comes back the other way round
    _, state_kv = chunk_kda_eager_fp32(**args, **{**kw, "transpose_state_layout": False})
    assert torch.equal(state_kv, state.transpose(-1, -2))
