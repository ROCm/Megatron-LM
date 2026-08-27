"""P0 / gate G8 -- MXFP4 quantiser and the one-expert QAT prototype.

TE 2.12's MXFP4 path is a stub at our pin (`quantize_impl` and
dequantize-from-packed both raise `NotImplementedError`), so there is no
third-party round-trip to check against: `moe/k3_qat.py` *is* the reference, and
these tests pin it against the OCP-MX definition and against the released
checkpoint's shapes.

**No `gradcheck` appears here, by construction.** An STE backward is deliberately
not the derivative of its forward; the check is against an explicit fake-quant
reference module (rule R4.4).
"""

import pytest
import torch

from kimi_k3.moe.k3_qat import (
    FP4_E2M1_VALUES,
    MX_GROUP,
    compute_e8m0_scale,
    dequantize_e8m0,
    dequantize_mxfp4,
    fake_quantize_mxfp4,
    fake_quantize_mxfp8,
    pack_nibbles,
    quantize_mxfp4,
    quantize_to_fp4_codes,
    ste_mxfp4,
    unpack_nibbles,
)
from kimi_k3.moe.k3_qat_experts import KimiK3FakeQuantExpertReference, KimiK3QATExpert
from kimi_k3.moe.situ import situ_glu

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture()
def x():
    torch.manual_seed(0)
    return torch.randn(64, 256, dtype=torch.float32)


# --- quantiser ---------------------------------------------------------------


def test_packing_shapes_match_the_released_checkpoint():
    """w1 is U8 [3072, 1792] with U8 [3072, 112] scales for 3584 inputs."""
    w = torch.randn(8, 3584)
    packed, scale = quantize_mxfp4(w)
    assert packed.shape == (8, 3584 // 2) and packed.dtype == torch.uint8
    assert scale.shape == (8, 3584 // MX_GROUP) and scale.dtype == torch.uint8


def test_nibble_packing_is_lossless(x):
    codes = quantize_to_fp4_codes(x, compute_e8m0_scale(x))
    assert torch.equal(unpack_nibbles(pack_nibbles(codes)), codes)


def test_every_value_lands_on_the_e2m1_grid(x):
    packed, scale = quantize_mxfp4(x)
    deq = dequantize_mxfp4(packed, scale)
    grouped = deq.unflatten(-1, (-1, MX_GROUP)) / dequantize_e8m0(scale).unsqueeze(-1)
    levels = torch.tensor(FP4_E2M1_VALUES)
    on_grid = torch.isclose(grouped.abs().unsqueeze(-1), levels.view(1, 1, 1, -1), atol=1e-6)
    assert on_grid.any(-1).all()


def test_exactly_representable_values_survive_unchanged():
    """A group whose peak is a power of two round-trips exactly."""
    levels = torch.tensor([v for v in FP4_E2M1_VALUES if v <= 4.0])  # seven of the eight
    row = torch.cat([levels, -levels] * 2 + [torch.zeros(MX_GROUP - 4 * levels.numel())])
    row = row * 2.0**-3
    assert row.numel() == MX_GROUP
    packed, scale = quantize_mxfp4(row.unsqueeze(0))
    assert torch.equal(dequantize_mxfp4(packed, scale).squeeze(0), row)


def test_a_group_peaking_at_6_does_not_round_trip_and_that_is_the_format():
    """The one case exact round-trip fails, stated rather than discovered later.

    The scale targets the group peak at 4.0 and rounds the exponent to nearest,
    with 1.5 as the midpoint. A peak of `6 * 2**k` puts `amax / 4` at exactly
    `1.5 * 2**(k-1)` -- the tie -- and the released rule rounds it **up**. The
    grid then doubles, and the smallest level in the group falls off it.

    This is a property of the released format, not of this implementation:
    `develop/results/mxfp4_scale_rule.md` measures the rule off real weights.
    """
    levels = torch.tensor(FP4_E2M1_VALUES)
    row = torch.cat([levels, -levels] * 2) * 2.0**-3
    packed, scale = quantize_mxfp4(row.unsqueeze(0))
    back = dequantize_mxfp4(packed, scale).squeeze(0)
    assert not torch.equal(back, row)
    assert back.abs().amax() == row.abs().amax(), "the peak itself is still exact"
    assert back[1] == 0.0, "the 0.5 level is what falls off the doubled grid"


def test_zero_group_is_handled():
    z = torch.zeros(1, MX_GROUP)
    packed, scale = quantize_mxfp4(z)
    assert torch.equal(dequantize_mxfp4(packed, scale), z)


def test_error_ordering_is_sane(x):
    """MXFP8 must beat MXFP4; both must beat doing nothing."""
    rel = lambda y: ((y - x).norm() / x.norm()).item()
    e4, e8 = rel(fake_quantize_mxfp4(x)), rel(fake_quantize_mxfp8(x))
    assert e8 < e4 < 0.25, (e8, e4)


def test_scaling_never_clamps(x):
    """The reason the released rule targets 4.0 instead of 6.0.

    Rounding `amax / 4` to the nearest power of two leaves the scaled peak in
    `[4/sqrt(2), 4*sqrt(2)] = [2.83, 5.66]`, always below E2M1's 6.0. The OCP
    formula `2**(floor(log2(amax)) - 2)` instead leaves it in `[4, 8)`, so any
    group whose peak has a mantissa above 1.5 gets clamped -- up to 25 % error on
    the group's largest element.
    """
    scale = compute_e8m0_scale(x)
    scaled = x.unflatten(-1, (-1, MX_GROUP)) / dequantize_e8m0(scale).unsqueeze(-1)
    assert scaled.abs().amax() <= FP4_E2M1_VALUES[-1]
    assert scaled.abs().amax() >= 2.8


def test_ste_forward_is_quantised_and_backward_is_identity(x):
    xq = x.clone().requires_grad_(True)
    out = ste_mxfp4(xq)
    assert torch.equal(out.detach(), fake_quantize_mxfp4(x))
    out.backward(torch.ones_like(out))
    assert torch.equal(xq.grad, torch.ones_like(x))


# --- SiTU --------------------------------------------------------------------


def test_situ_matches_the_released_formula():
    torch.manual_seed(0)
    gate_up = torch.randn(16, 64)
    d = 32
    gate, up = gate_up[:, :d].float(), gate_up[:, d:].float()
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(up / 25.0)
    )
    torch.testing.assert_close(situ_glu(gate_up), expected.to(gate_up.dtype))


def test_situ_soft_caps_both_branches():
    """Why the release needs no Hadamard for QAT: outliers are tanh-limited."""
    big = torch.full((1, 2), 1e4)
    out = situ_glu(torch.cat([big, big], dim=-1))
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 4.0 * 25.0 * 1.001


# --- one-expert QAT prototype ------------------------------------------------


@pytest.fixture()
def expert():
    torch.manual_seed(0)
    return KimiK3QATExpert(hidden=128, inter=256).cuda()


@CUDA
def test_cache_is_in_sync_after_construction(expert):
    assert all(v == 0.0 for v in expert.cache_matches_masters().values())


@CUDA
def test_ste_grads_match_the_fake_quant_reference_exactly(expert):
    """G8's core assertion. Exact, not approximate: same math, two spellings."""
    ref = KimiK3FakeQuantExpertReference(expert).cuda()
    torch.manual_seed(1)
    x = torch.randn(32, 128, device="cuda")

    out_a = expert(x)
    out_b = ref(x)
    assert torch.equal(out_a, out_b), (out_a - out_b).abs().max().item()

    out_a.sum().backward()
    out_b.sum().backward()
    for name in KimiK3QATExpert.WEIGHTS:
        ga = getattr(expert, name).grad
        gb = getattr(ref, name).grad
        assert torch.equal(ga, gb), f"{name}: {(ga - gb).abs().max().item()}"
        assert ga.abs().sum() > 0


@CUDA
def test_optimizer_step_then_refresh_keeps_the_cache_honest(expert):
    opt = torch.optim.SGD(expert.parameters(), lr=1e-2)
    torch.manual_seed(2)
    x = torch.randn(32, 128, device="cuda")

    before = expert.w1.detach().clone()
    expert(x).sum().backward()
    opt.step()
    assert not torch.equal(expert.w1.detach(), before), "the master did not move"

    # stale until refreshed, exact afterwards
    assert max(expert.cache_matches_masters().values()) > 0
    expert.refresh_packed_cache()
    assert all(v == 0.0 for v in expert.cache_matches_masters().values())


@CUDA
def test_checkpoint_round_trip_carries_masters_and_packed_state(expert):
    opt = torch.optim.SGD(expert.parameters(), lr=1e-2)
    torch.manual_seed(3)
    x = torch.randn(32, 128, device="cuda")
    expert(x).sum().backward()
    opt.step()
    expert.refresh_packed_cache()

    state = {k: v.clone() for k, v in expert.state_dict().items()}
    fresh = KimiK3QATExpert(hidden=128, inter=256).cuda()
    fresh.load_state_dict(state)

    for k, v in expert.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v), k
    assert all(v == 0.0 for v in fresh.cache_matches_masters().values())
    torch.testing.assert_close(fresh(x), expert(x), rtol=0, atol=0)


@CUDA
def test_activation_quantisation_changes_the_result(expert):
    """Guards against the activation path being silently skipped."""
    torch.manual_seed(4)
    x = torch.randn(32, 128, device="cuda")
    assert not torch.equal(expert(x, quantize_activations=True),
                           expert(x, quantize_activations=False))
