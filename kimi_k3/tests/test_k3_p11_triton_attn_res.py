"""G58 -- the fused Triton AttnRes kernel against the eager oracle.

Unlike the chunked mixer (G43), this one **cannot** be bit-identical: it reduces
over `H` in tiles rather than in torch's order. So the forward is gated on a
measured tolerance and the backward on exact equality -- the backward recomputes
through the oracle, so anything other than 0.0 there means the wrapper is wrong.
"""

import pytest
import torch

from kimi_k3.block.attn_res import attn_res_mix
from kimi_k3.block.attn_res_triton import HAVE_TRITON, fused_attn_res_mix

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="triton unavailable")


def operands(T, K, H, dtype=torch.float32, seed=0):
    g = torch.Generator("cuda").manual_seed(seed)
    return (
        torch.randn(T, H, device="cuda", dtype=dtype, generator=g),
        torch.randn(T, K, H, device="cuda", dtype=dtype, generator=g),
        torch.randn(H, device="cuda", dtype=dtype, generator=g),
        torch.randn(1, H, device="cuda", dtype=dtype, generator=g),
    )


@pytest.mark.parametrize("T,K,H", [(64, 2, 512), (129, 4, 7168), (512, 8, 7168)])
def test_forward_matches_the_oracle(T, K, H):
    """Tolerance, not equality -- and tight enough to catch a real error.

    Measured fp32 rel-L2 is ~1e-06; 1e-4 leaves three orders of headroom while
    still failing anything that has the arithmetic wrong. T=129 is deliberately
    not a multiple of any block size.
    """
    p, s, nw, pw = operands(T, K, H)
    want = attn_res_mix(p, s, nw, pw, 1e-6)
    got = fused_attn_res_mix(p, s, nw, pw, 1e-6)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, f"rel-L2 {rel:.3e}"


@pytest.mark.parametrize("dtype,bound", [(torch.float32, 1e-4), (torch.bfloat16, 2e-3)])
def test_backward_matches_the_eager_gradient(dtype, bound):
    """Tolerance, not equality: the backward is its own kernel now (G59).

    It was an eager recompute at first, which made gradients bit-identical but
    re-materialised the [T, K+1, H] temporaries the forward avoids -- the memory
    win was forward-only. With a real backward kernel the reduction order over H
    differs, so equality is gone and a bound takes its place.

    Measured fp32 ~9e-06 and bf16 ~3e-04; the bounds leave an order of headroom.
    The norm/proj entries are the sensitive ones -- they are cross-token
    reductions and an earlier bf16 GEMV version of that reduction sat at 3.05e-03,
    which this bound would have caught.
    """
    ref, fused = {}, {}
    for store, fn in ((ref, attn_res_mix), (fused, fused_attn_res_mix)):
        p, s, nw, pw = operands(129, 4, 7168, dtype=dtype, seed=1)
        for t in (p, s, nw, pw):
            t.requires_grad_(True)
        fn(p, s, nw, pw, 1e-6).sum().backward()
        store.update(prefix=p.grad, slots=s.grad, norm=nw.grad, proj=pw.grad)
    for name in ref:
        a, b = ref[name].float(), fused[name].float()
        rel = ((b - a).norm() / a.norm().clamp_min(1e-12)).item()
        assert rel < bound, f"{name} gradient rel-L2 {rel:.3e} ({dtype})"


def test_backward_allocates_no_big_temporary():
    """The point of the backward kernel: no [T, K+1, H] fp32 stack.

    The eager recompute it replaced was correct but peaked at the eager figure,
    so the forward's memory win never survived a training step.
    """
    T, K, H = 2048, 8, 7168
    peaks = {}
    for label, fn in (("eager", attn_res_mix), ("fused", fused_attn_res_mix)):
        p, s, nw, pw = operands(T, K, H, dtype=torch.bfloat16, seed=2)
        for t in (p, s, nw, pw):
            t.requires_grad_(True)
        fn(p, s, nw, pw, 1e-6).sum().backward()      # warm allocator
        for t in (p, s, nw, pw):
            t.grad = None
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn(p, s, nw, pw, 1e-6).sum().backward()
        torch.cuda.synchronize()
        peaks[label] = (torch.cuda.max_memory_allocated() - base) / 2**30
        del p, s, nw, pw
        torch.cuda.empty_cache()
    assert peaks["fused"] < peaks["eager"] / 8, peaks


def test_it_is_differentiable_at_all():
    """A bare Triton kernel returns a tensor with no grad_fn.

    Calling the kernel directly inside a training step would silently train
    nothing through this path while the loss still fell, because every other path
    still trains. That is the whole reason the autograd wrapper exists.
    """
    from kimi_k3.block.attn_res_triton import attn_res_mix_triton

    p, s, nw, pw = operands(32, 2, 512)
    p.requires_grad_(True)
    assert attn_res_mix_triton(p, s, nw, pw, 1e-6).grad_fn is None, (
        "the raw kernel became differentiable; this test's premise is stale"
    )
    assert fused_attn_res_mix(p, s, nw, pw, 1e-6).grad_fn is not None


def test_no_slots_is_a_no_op():
    p, s, nw, pw = operands(8, 0, 512)
    assert torch.equal(fused_attn_res_mix(p, s, nw, pw, 1e-6), p)


def test_the_flag_reaches_every_mixer(single_rank_world):
    """Same two-construction-site hazard that G56 exposed for the chunked flag."""
    from kimi_k3.model.build import build_k3_model

    model = build_k3_model("tiny", k3_attn_res_triton=True)
    mixers = [m for m in model.modules() if hasattr(m, "triton")]
    assert len(mixers) == 2 * 4 + 1, len(mixers)
    assert all(m.triton for m in mixers), "the flag did not reach every site"
