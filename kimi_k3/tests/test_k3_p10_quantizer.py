"""P10 / gate G38 -- the quantisers, cross-checked against something that is not us.

The plan's cross-check target was TE's `MXFP4Quantizer`. It does not exist at the
pin: TE ships `MXFP8Quantizer` and `NVFP4Quantizer`, and NVFP4 is a different
format (block 16, E4M3 scales, plus a global scale) that cannot stand in. AITER
is the better target anyway -- it is the library whose kernels the release's
serving path actually runs -- and it agrees **byte for byte**.
"""

import pytest
import torch

from kimi_k3.moe.k3_qat import (
    MX_GROUP,
    compute_e8m0_scale,
    f32_to_e8m0,
    fake_quantize_mxfp8,
    quantize_mxfp4,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

aiter = pytest.importorskip("aiter")


@pytest.mark.parametrize("scale", [3.0, 0.02, 100.0])
def test_mxfp4_is_byte_identical_to_aiter(scale):
    """G38: same packed nibbles and same e8m0 scales, not merely the same values."""
    torch.manual_seed(0)
    x = (torch.randn(64, 256, device="cuda") * scale).bfloat16()
    a_packed, a_scale = aiter.per_1x32_f4_quant(x)
    packed, scales = quantize_mxfp4(x.float())
    assert torch.equal(a_scale.view(torch.uint8), scales)
    assert torch.equal(a_packed.view(torch.uint8), packed)


def test_mxfp8_is_bit_identical_to_aiter_with_e8m0_scales():
    """The MX form. AITER's *default* fp32-scale variant is a different format."""
    from aiter import dtypes

    torch.manual_seed(1)
    x = (torch.randn(8, 64, device="cuda") * 3).bfloat16()
    q, s = aiter.per_1x32_f8_scale_f8_quant(x, scale_type=dtypes.fp8_e8m0)
    theirs = q.float().view(8, 2, MX_GROUP) * s.view(torch.uint8).float().sub(127).exp2().view(8, 2, 1)
    assert torch.equal(theirs.reshape(8, 64), fake_quantize_mxfp8(x.float()))


def test_aiters_default_fp8_scale_is_not_the_mx_format():
    """Guards against 'cross-checked against AITER' meaning the wrong AITER call."""
    torch.manual_seed(2)
    x = (torch.randn(8, 64, device="cuda") * 3).bfloat16()
    _, s = aiter.per_1x32_f8_scale_f8_quant(x)
    assert s.dtype == torch.float32
    powers = torch.log2(s.float())
    assert not bool((powers == powers.round()).all()), "expected non-power-of-two scales"


def test_te_has_no_mxfp4_quantizer_at_the_pin():
    """The plan named one; recording its absence keeps the substitution honest."""
    import transformer_engine.pytorch.tensor as te_tensor

    assert not hasattr(te_tensor, "MXFP4Quantizer")
    assert hasattr(te_tensor, "MXFP8Quantizer")


def test_the_scale_rule_is_round_to_nearest_of_amax_over_four():
    """The rule measured off the released weights, pinned against hand cases.

    Under the OCP formula `2**(floor(log2(amax)) - 2)` the peak of 3.99 would get
    exponent -1 and clamp to 3.0. It gets 0 instead.
    """
    cases = {6.0: 1, 7.0: 1, 8.0: 1, 5.0: 0, 4.0: 0, 12.0: 2, 0.75: -2, 1.0: -2, 1.5: -1, 3.99: 0}
    for amax, expected in cases.items():
        group = torch.zeros(1, MX_GROUP, device="cuda")
        group[0, 0] = amax
        assert int(compute_e8m0_scale(group)[0, 0]) - 127 == expected, amax


def test_the_tie_rounds_up_like_the_hardware():
    """A mantissa of exactly 1.5 is the midpoint, and it rounds away from zero."""
    assert int(f32_to_e8m0(torch.tensor([1.5]))) - 127 == 1
    assert int(f32_to_e8m0(torch.tensor([1.4999]))) - 127 == 0
    assert int(f32_to_e8m0(torch.tensor([1.0]))) - 127 == 0


def test_a_zero_group_scales_to_one():
    scale = compute_e8m0_scale(torch.zeros(1, MX_GROUP, device="cuda"))
    assert int(scale[0, 0]) == 127
