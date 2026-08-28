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
def test_per_token_gradients_are_bit_identical_at_the_default_chunk(shape):
    """Everything whose gradient belongs to one token is exact -- above the threshold."""
    from kimi_k3.block.attn_res import ATTN_RES_CHUNK

    ours = backward_grads(shape, attn_res_mix_fused, chunk=ATTN_RES_CHUNK)
    reference = backward_grads(shape, attn_res_mix)
    for name in ("prefix", "slots"):
        torch.testing.assert_close(ours[name], reference[name], rtol=0, atol=0, msg=name)


@pytest.mark.slow
def test_a_small_chunk_stops_being_bitwise_at_release_width():
    """The limit of the guarantee, pinned rather than left to be discovered.

    A small chunk makes the batched matmul a different shape, rocBLAS picks a
    different kernel, and the per-token gradients change in the last bits. The
    claim is therefore scoped to chunks at or above `ATTN_RES_BITWISE_MIN_CHUNK`
    rather than made for every chunk size.

    The difference is characterised by *how many* elements move and by how far
    relative to the tensor's own scale -- about one element in ten thousand, and
    4e-04 of the gradient's magnitude. Two tighter formulations were tried and
    both were wrong: an elementwise `assert_close` at 2e-3 failed on a max-abs of
    0.25 that is a single ulp of a large gradient, and a per-element relative
    bound blew up on near-zero gradients around 1e-25 where the ratio is all
    divisor.
    """
    from kimi_k3.block.attn_res import ATTN_RES_BITWISE_MIN_CHUNK

    reference = backward_grads(RELEASE, attn_res_mix)
    small = backward_grads(RELEASE, attn_res_mix_fused, chunk=32)

    differing = small["prefix"] != reference["prefix"]
    fraction = differing.float().mean().item()
    assert 0 < fraction < 1e-3, fraction  # measured 1.05e-04 of 58.7M elements

    # Scale-relative, not per-element: most of the elements that move are
    # near-zero gradients (1e-25 and smaller), where a per-element ratio is
    # dominated by the divisor and says nothing. Against the tensor's own
    # magnitude the whole difference is 4e-04.
    a, b = small["prefix"].float(), reference["prefix"].float()
    assert (a - b).abs().max().item() / b.abs().max().item() < 1e-3

    big = backward_grads(RELEASE, attn_res_mix_fused, chunk=ATTN_RES_BITWISE_MIN_CHUNK)
    torch.testing.assert_close(big["prefix"], reference["prefix"], rtol=0, atol=0)
    torch.testing.assert_close(big["slots"], reference["slots"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape", [FAST, pytest.param(RELEASE, marks=pytest.mark.slow)], ids=["fast", "release"]
)
def test_the_shared_gain_gradient_differs_only_by_accumulation_order(shape):
    """The one thing chunking cannot keep bitwise, stated rather than hidden.

    `norm_weight` is `[H]` and shared by every token, so its gradient is a sum
    over all `T` rows. Chunking changes the order of that sum, and floating-point
    addition is not associative.

    The bound is relative to the gradient's own magnitude, not absolute. At
    release width that magnitude is ~5e+03, so an absolute tolerance tuned at the
    fast tier is meaningless there -- which is exactly how the first version of
    this test passed at 128 tokens and failed at 8192.
    """
    ours = backward_grads(shape, attn_res_mix_fused, chunk=32)
    reference = backward_grads(shape, attn_res_mix)
    assert not torch.equal(ours["norm"], reference["norm"]), "expected reordering, got none"

    scale = reference["norm"].abs().max().item()
    relative = (ours["norm"] - reference["norm"]).abs().max().item() / scale
    assert relative < 1e-5, relative


@pytest.mark.slow
def test_the_gain_gradient_error_grows_sublinearly_in_the_chunk_count():
    """Reordering a sum, not losing precision -- and the difference is measurable.

    Precision loss would grow at least linearly in the number of partial sums.
    Random-walk reordering grows like its square root. Measured at release width:

        1 chunk    -> 0.0 (bitwise; the whole-tensor path)
        4 chunks   -> 1.92e-07 of the gradient's magnitude
        64 chunks  -> 4.80e-07
        256 chunks -> 7.68e-07

    64x the chunks for 4x the error. An earlier version of this test asserted the
    error "does not grow with the chunk count", which is simply false -- it grows,
    slowly, and that is the property worth pinning.
    """
    T = RELEASE[0]
    reference = backward_grads(RELEASE, attn_res_mix)["norm"]
    scale = reference.abs().max().item()

    def error(chunk):
        got = backward_grads(RELEASE, attn_res_mix_fused, chunk=chunk)["norm"]
        return (got - reference).abs().max().item() / scale

    assert error(T) == 0.0, "one chunk is the whole tensor and must be bitwise"

    few, many = error(T // 4), error(T // 256)
    assert 0 < few <= many, (few, many)
    assert many < 1e-6, many
    # 64x the chunks must cost far less than 64x the error
    assert many < 8 * few, (few, many)


def test_the_forward_is_bitwise_at_every_chunk_size():
    """The forward carries the guarantee the backward only carries above a threshold."""
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


def test_the_flag_selects_the_path_and_the_model_agrees(single_rank_world):
    """G43 end to end: the whole model with `--k3-attn-res-fused` matches without it.

    The mixer is used at three sites per model (two per layer plus the output),
    so a flag that reaches only some of them would still pass a unit test.
    """
    from kimi_k3.model.build import build_k3_model

    torch.manual_seed(0)
    eager = build_k3_model("tiny")
    fused = build_k3_model("tiny", k3_attn_res_fused=True, k3_attn_res_chunk=4)
    fused.load_state_dict(eager.state_dict(), strict=False)

    mixers = [m for m in fused.modules() if hasattr(m, "fused")]
    assert len(mixers) == 2 * 4 + 1, len(mixers)
    assert all(m.fused for m in mixers), "the flag did not reach every site"

    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    with torch.no_grad():
        a = eager(input_ids=tokens, position_ids=None, attention_mask=None)
        b = fused(input_ids=tokens, position_ids=None, attention_mask=None)
    torch.testing.assert_close(b, a, rtol=0, atol=0)
