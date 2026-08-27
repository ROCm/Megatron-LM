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
    """A group built from E2M1 levels times a power of two must round-trip exactly."""
    levels = torch.tensor(FP4_E2M1_VALUES)
    row = torch.cat([levels, -levels, levels, -levels]) * 2.0**-3
    packed, scale = quantize_mxfp4(row.unsqueeze(0))
    assert torch.equal(dequantize_mxfp4(packed, scale).squeeze(0), row)


def test_zero_group_is_handled():
    z = torch.zeros(1, MX_GROUP)
    packed, scale = quantize_mxfp4(z)
    assert torch.equal(dequantize_mxfp4(packed, scale), z)


def test_error_ordering_is_sane(x):
    """MXFP8 must beat MXFP4; both must beat doing nothing."""
    rel = lambda y: ((y - x).norm() / x.norm()).item()
    e4, e8 = rel(fake_quantize_mxfp4(x)), rel(fake_quantize_mxfp8(x))
    assert e8 < e4 < 0.25, (e8, e4)


def test_scale_keeps_groups_in_range(x):
    """After scaling, no element should be clamped away from its group's peak."""
    scale = compute_e8m0_scale(x)
    scaled = x.unflatten(-1, (-1, MX_GROUP)) / dequantize_e8m0(scale).unsqueeze(-1)
    # OCP's rule allows the peak to exceed 6.0 by up to 2x before clamping
    assert scaled.abs().amax() <= 2 * FP4_E2M1_VALUES[-1]


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
