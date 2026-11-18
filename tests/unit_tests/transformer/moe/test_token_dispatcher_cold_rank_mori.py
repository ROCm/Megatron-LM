# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MORI cold-rank token-dispatcher test.

Kept out of ``test_token_dispatcher_mori.py`` so it runs in its own fresh ``torchrun``
process. Forcing every token onto the first ``topk`` experts leaves the other EP ranks
with zero tokens, so a regression here can desynchronize the ranks and leave the MORI
op and process groups in a state the other MORI tests cannot recover from. The MORI op
and shmem lifecycle is owned by ``conftest.py``.
"""
import pytest
import torch

from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    is_mori_available,
)


@pytest.mark.skipif(not is_mori_available(), reason="MORI is not available")
class TestMoriColdRank:
    """MORI flex dispatcher with fused permute on a rank that receives 0 tokens."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not is_te_min_version("2.1.0"), reason="TE fused permute is required for permute_fusion"
    )
    @pytest.mark.internal
    @pytest.mark.timeout(600)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    def test_cold_rank_fused_permute_sequential_mlp(self, tp_size, ep_size):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="mori",
            moe_permute_fusion=True,
            moe_grouped_gemm=False,
            moe_mori_max_tokens_per_rank=4096,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_cold_rank_test()
