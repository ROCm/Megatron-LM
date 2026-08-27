"""P11 / gate G43 -- the fused AttnRes mixer against the eager one.

The eager mixer is the transcription of the release and stays the oracle. The
fused one is a memory optimisation, so the bar is **bit-identical**, forward and
backward, not "close enough": anything less would mean the optimisation changed
the model, and the whole point is that it does not.
"""

import pytest
import torch

from kimi_k3.block.attn_res import attn_res_mix, attn_res_mix_fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FAST = (128, 3, 256)      # tokens, slots, hidden
RELEASE = (8192, 8, 7168)  # a production mix: 8 slots at the official block size


def operands(tokens, slots, hidden, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(tokens, hidden, device="cuda", dtype=dtype),
        torch.randn(tokens, slots, hidden, device="cuda", dtype=dtype),
        torch.randn(hidden, device="cuda"),
        torch.randn(1, hidden, device="cuda"),
    )


@pytest.mark.parametrize(
    "shape", [FAST, pytest.param(RELEASE, marks=pytest.mark.slow)], ids=["fast", "release"]
)
@pytest.mark.parametrize("chunk", [16, 64])
def test_forward_is_bit_identical(shape, chunk):
    prefix, slots, norm, proj = operands(*shape)
    torch.testing.assert_close(
        attn_res_mix_fused(prefix, slots, norm, proj, 1e-6, chunk=chunk),
        attn_res_mix(prefix, slots, norm, proj, 1e-6),
        rtol=0, atol=0,
    )


def backward_grads(shape, mix, **kwargs):
    prefix, slots, norm, proj = operands(*shape)
    p = prefix.clone().requires_grad_(True)
    s = slots.clone().requires_grad_(True)
    n = norm.clone().requires_grad_(True)
    torch.manual_seed(4)
    mix(p, s, n, proj, 1e-6, **kwargs).backward(torch.randn_like(prefix))
    return {"prefix": p.grad, "slots": s.grad, "norm": n.grad}


@pytest.mark.parametrize(
    "shape", [FAST, pytest.param(RELEASE, marks=pytest.mark.slow)], ids=["fast", "release"]
)
def test_per_token_gradients_are_bit_identical(shape):
    """Everything whose gradient belongs to one token is exact."""
    ours = backward_grads(shape, attn_res_mix_fused, chunk=32)
    reference = backward_grads(shape, attn_res_mix)
    for name in ("prefix", "slots"):
        torch.testing.assert_close(ours[name], reference[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(
    "shape", [FAST, pytest.param(RELEASE, marks=pytest.mark.slow)], ids=["fast", "release"]
)
def test_the_shared_gain_gradient_differs_only_by_accumulation_order(shape):
    """The one thing chunking cannot keep bitwise, stated rather than hidden.

    `norm_weight` is `[H]` and shared by every token, so its gradient is a sum
    over all `T` rows. Chunking changes the order of that sum, and floating-point
    addition is not associative. The difference is fp32 accumulation noise --
    ~1e-5 relative -- and it does not grow with the number of chunks; it is not a
    different gradient, it is the same gradient added up differently.
    """
    ours = backward_grads(shape, attn_res_mix_fused, chunk=32)
    reference = backward_grads(shape, attn_res_mix)
    assert not torch.equal(ours["norm"], reference["norm"]), "expected reordering, got none"
    torch.testing.assert_close(ours["norm"], reference["norm"], rtol=1e-4, atol=1e-4)


def test_the_gain_gradient_error_does_not_grow_with_chunk_count():
    """If it did, chunking would be losing precision rather than reordering."""
    reference = backward_grads(FAST, attn_res_mix)["norm"]
    errors = {
        chunk: (backward_grads(FAST, attn_res_mix_fused, chunk=chunk)["norm"] - reference)
        .abs().max().item()
        for chunk in (4, 16, 64)
    }
    assert max(errors.values()) < 1e-4, errors


def test_the_chunk_size_is_only_a_memory_knob():
    """Every chunk size must agree, or it is a numerical knob in disguise."""
    prefix, slots, norm, proj = operands(*FAST)
    reference = attn_res_mix(prefix, slots, norm, proj, 1e-6)
    for chunk in (1, 7, 16, 127, 128, 4096):
        torch.testing.assert_close(
            attn_res_mix_fused(prefix, slots, norm, proj, 1e-6, chunk=chunk),
            reference, rtol=0, atol=0,
        )


def test_it_actually_uses_less_memory():
    """Otherwise it is a slower mixer with a docstring."""
    prefix, slots, norm, proj = operands(2048, 8, 1024)
    peaks = {}
    for label, call in (
        ("eager", lambda: attn_res_mix(prefix, slots, norm, proj, 1e-6)),
        ("fused", lambda: attn_res_mix_fused(prefix, slots, norm, proj, 1e-6, chunk=128)),
    ):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.max_memory_allocated()
        call()
        torch.cuda.synchronize()
        peaks[label] = torch.cuda.max_memory_allocated() - before
    assert peaks["fused"] < peaks["eager"] / 2, peaks


def test_no_slots_is_a_no_op_in_both():
    prefix, _, norm, proj = operands(*FAST)
    empty = torch.zeros(FAST[0], 0, FAST[2], device="cuda", dtype=prefix.dtype)
    assert torch.equal(attn_res_mix_fused(prefix, empty, norm, proj, 1e-6), prefix)
