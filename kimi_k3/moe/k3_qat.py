"""MXFP4 / MXFP8 quantisation for Kimi K3 QAT.

The release ships routed experts as `compressed-tensors` `mxfp4-pack-quantized`:
group 32 along the reduction axis, uint8 (e8m0) scales, symmetric, two 4-bit
values per byte. Verified against the checkpoint headers -- `w1` is
`U8 [3072, 1792]` (logical `[3072, 3584]`) with `U8 [3072, 112]` scales
(3584 / 32 = 112).

TransformerEngine ships no NumPy reference at our pin (review finding D2), so
this module *is* the reference: an explicit OCP-MX implementation whose round-trip
is cross-checked against TE's `MXFP4Quantizer` in the tests.

Nothing here is a kernel. AITER's a8w4 fused-MoE path (`aiter/ops/opus/
moe_stage{1,2}_a8w4.py`, tuned for gfx950 at expert=896 / topk=16 /
model_dim=3584 / `ActivationType.Situv2`) is the fast forward and lands in P10;
this module owns the numerics contract it has to match.
"""

from typing import Tuple

import torch

# E2M1: the eight representable magnitudes, and the max normal.
FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX = 6.0
#: E2M1's largest normal is 6.0 = 1.5 x 2^2, so the shared scale targets exponent 2.
FP4_MAX_EXP = 2
E8M0_BIAS = 127
MX_GROUP = 32


def _e2m1_levels(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(FP4_E2M1_VALUES, device=device, dtype=dtype)


def compute_e8m0_scale(x: torch.Tensor, group_size: int = MX_GROUP) -> torch.Tensor:
    """Shared exponent per group, as an e8m0 byte. ``x`` is ``[..., N]``.

    OCP MX: take the group's max magnitude, keep its exponent, and shift so the
    largest element lands at E2M1's top exponent. Returns ``uint8`` of shape
    ``[..., N / group_size]``.
    """
    assert x.shape[-1] % group_size == 0, (x.shape, group_size)
    grouped = x.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(amax.clamp(min=torch.finfo(torch.float32).tiny)))
    exp = exp - FP4_MAX_EXP
    # amax == 0 -> a scale of 1.0 keeps the group at zero and stays in range.
    exp = torch.where(amax == 0, torch.zeros_like(exp), exp)
    return (exp + E8M0_BIAS).clamp(0, 255).to(torch.uint8)


def dequantize_e8m0(scale_u8: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale_u8.float() - E8M0_BIAS)


def quantize_to_fp4_codes(
    x: torch.Tensor, scale_u8: torch.Tensor, group_size: int = MX_GROUP
) -> torch.Tensor:
    """Round each element to the nearest E2M1 level. Returns codes in ``[0, 15]``.

    Bit layout matches `compressed-tensors`: sign in bit 3, magnitude index in
    bits 0-2.
    """
    grouped = x.float().unflatten(-1, (-1, group_size))
    scaled = grouped / dequantize_e8m0(scale_u8).unsqueeze(-1)
    sign = (scaled < 0).to(torch.uint8)
    mag = scaled.abs().clamp(max=FP4_MAX)
    levels = _e2m1_levels(x.device)
    # round-to-nearest, ties to even level index (RNE on the level grid)
    idx = torch.bucketize(mag, levels)
    lo = (idx - 1).clamp(0, len(FP4_E2M1_VALUES) - 1)
    hi = idx.clamp(0, len(FP4_E2M1_VALUES) - 1)
    lo_v, hi_v = levels[lo], levels[hi]
    take_hi = (hi_v - mag) < (mag - lo_v)
    tie = (hi_v - mag) == (mag - lo_v)
    take_hi = torch.where(tie, (lo % 2 == 1), take_hi)
    code_mag = torch.where(take_hi, hi, lo).to(torch.uint8)
    return ((sign << 3) | code_mag).flatten(-2)


def dequantize_fp4_codes(
    codes: torch.Tensor, scale_u8: torch.Tensor, group_size: int = MX_GROUP
) -> torch.Tensor:
    levels = _e2m1_levels(codes.device)
    mag = levels[(codes & 0x7).long()]
    signed = torch.where((codes & 0x8) != 0, -mag, mag)
    grouped = signed.unflatten(-1, (-1, group_size))
    return (grouped * dequantize_e8m0(scale_u8).unsqueeze(-1)).flatten(-2)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """``[..., N]`` codes -> ``[..., N/2]`` bytes, even index in the low nibble."""
    assert codes.shape[-1] % 2 == 0
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def quantize_mxfp4(x: torch.Tensor, group_size: int = MX_GROUP) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x`` -> (packed uint8 ``[..., N/2]``, e8m0 scales ``[..., N/group]``)."""
    scale = compute_e8m0_scale(x, group_size)
    codes = quantize_to_fp4_codes(x, scale, group_size)
    return pack_nibbles(codes), scale


def dequantize_mxfp4(
    packed: torch.Tensor, scale_u8: torch.Tensor, group_size: int = MX_GROUP
) -> torch.Tensor:
    return dequantize_fp4_codes(unpack_nibbles(packed), scale_u8, group_size)


def fake_quantize_mxfp4(x: torch.Tensor, group_size: int = MX_GROUP) -> torch.Tensor:
    """Quantise and immediately dequantise -- the QAT forward's weight path."""
    packed, scale = quantize_mxfp4(x, group_size)
    return dequantize_mxfp4(packed, scale, group_size).to(x.dtype)


def fake_quantize_mxfp8(
    x: torch.Tensor, group_size: int = MX_GROUP, dtype=torch.float8_e4m3fn
) -> torch.Tensor:
    """MXFP8 activation quantisation, matching AITER's ``QuantType.per_1x32``."""
    grouped = x.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    fmax = torch.finfo(dtype).max
    # ceil, not floor: the scale must be large enough that amax/scale <= fmax,
    # otherwise every group's peak element clamps and the error explodes.
    exp = torch.ceil(torch.log2((amax / fmax).clamp(min=torch.finfo(torch.float32).tiny)))
    exp = torch.where(amax == 0, torch.zeros_like(exp), exp)
    scale = torch.exp2(exp)
    q = (grouped / scale).clamp(-fmax, fmax).to(dtype).float()
    return (q * scale).flatten(-2).to(x.dtype)


class STEFakeQuantFP8(torch.autograd.Function):
    """STE for the activation path.

    Without this the cast into `float8_e4m3fn` produces no usable gradient, so
    everything upstream of an activation quantisation point silently receives a
    zero gradient -- in an expert, that is the whole gate/up half. Caught by
    `test_ste_grads_match_the_fake_quant_reference_exactly`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        return fake_quantize_mxfp8(x, group_size)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out, None


def ste_mxfp8(x: torch.Tensor, group_size: int = MX_GROUP) -> torch.Tensor:
    return STEFakeQuantFP8.apply(x, group_size)


class STEFakeQuant(torch.autograd.Function):
    """Straight-through estimator: quantised forward, identity backward.

    An STE backward is deliberately **not** the derivative of its forward, so
    `torch.autograd.gradcheck` is invalid here by construction (rule R4.4). The
    gate compares against an explicit fake-quant reference module instead.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        return fake_quantize_mxfp4(x, group_size)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out, None


def ste_mxfp4(x: torch.Tensor, group_size: int = MX_GROUP) -> torch.Tensor:
    return STEFakeQuant.apply(x, group_size)
