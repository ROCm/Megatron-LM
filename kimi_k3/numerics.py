"""Dtype promotion for the FP32 oracles.

The released code writes `.float()` in its numerically sensitive spots, which is
an *upcast* for the bf16 it runs on. Transcribing that literally makes an oracle
that is bit-identical in production and silently wrong under test: `.float()`
**downcasts** an fp64 input, so `torch.autograd.gradcheck` compares a float64
numerical Jacobian against a float32 analytical one and fails for a reason that
has nothing to do with the code being checked.

Promoting instead keeps the released behaviour exactly (bf16 and fp32 both
promote to fp32) while letting an fp64 test input stay fp64. Both oracles use
this; it has already been the cause of two failures.
"""

import torch


def hi_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32, or the input dtype when that is already wider."""
    return torch.promote_types(dtype, torch.float32)


def to_hi(x: torch.Tensor) -> torch.Tensor:
    return x.to(hi_dtype(x.dtype))
