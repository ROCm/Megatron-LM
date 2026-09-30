# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import unpermute
from megatron.core.utils import is_te_min_version


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_te_min_version("2.1.0"), reason="TE fused unpermute requires TE >= 2.1.0"
)
def test_fused_unpermute_empty_restores_shape():
    """Cold EP rank: fused unpermute must yield zeros(restore_shape), not [0, H].

    TE's fused unpermute short-circuits on an empty input. Megatron's wrapper
    must still restore the dispatch-buffer shape so MORI combine is well-defined.
    """
    hidden = 16
    max_recv = 32
    permuted = torch.zeros((0, hidden), device="cuda", dtype=torch.bfloat16)
    sorted_indices = torch.zeros((0,), device="cuda", dtype=torch.int32)
    restore_shape = torch.Size([max_recv, hidden])

    out = unpermute(
        permuted, sorted_indices, restore_shape=restore_shape, fused=True
    )

    assert out.shape == restore_shape, (
        f"fused unpermute on empty input returned {tuple(out.shape)}, "
        f"expected {tuple(restore_shape)}"
    )
    assert out.dtype == permuted.dtype
    assert out.device == permuted.device
    assert torch.count_nonzero(out) == 0
