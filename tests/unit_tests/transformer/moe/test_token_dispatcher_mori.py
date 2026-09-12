# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dedicated MORI token-dispatcher tests.

Extracted from ``test_token_dispatcher.py`` so MORI runs in its own fresh ``torchrun``
process, never parametrized alongside the DeepEP/HybridEP cases. MORI's
``EpDispatchCombineHandle`` needs a node-spanning expert communicator and its symmetric
memory is process-scoped (init once, finalize once at session end via the shared
``conftest.py``); interleaving it with the other flex backends in one process risks
corrupting the shared process-group / shmem state.

MORI only runs when the expert-parallel group spans the whole node
(``require_node_spanning_mori_ep``); the non-spanning ``(tp, ep)`` parametrizations skip,
exactly as they did in the original mixed tests.
"""
import pytest
import torch

from megatron.core import config
from megatron.core.transformer.moe.fused_a2a import reset_mori_op
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    is_mori_available,
    permute_fusion_params,
    require_node_spanning_mori_ep,
)


@pytest.mark.skipif(not is_mori_available(), reason="MORI is not available")
class TestMoriFlexDispatcher:
    """MORI variants of the flex token-dispatcher tests, isolated from other backends."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        # Drop the per-test op; keep shmem alive. Finalize is owned solely by the
        # session-scoped conftest fixture (MORI shmem is process-scoped and cannot be
        # finalized then reinitialized).
        reset_mori_op()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        require_node_spanning_mori_ep(ep_size)
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend="mori",
            moe_mori_max_tokens_per_rank=4096,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()
        # reset experimental flag to False
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_forward_backward(self, tp_size, ep_size, permute_fusion):
        require_node_spanning_mori_ep(ep_size)
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend="mori",
            moe_mori_max_tokens_per_rank=4096,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_capacity_test()
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", [True])
    def test_router_padding_for_fp8_forward_backward(self, tp_size, ep_size, permute_fusion):
        require_node_spanning_mori_ep(ep_size)
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend="mori",
            moe_mori_max_tokens_per_rank=4096,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False


@pytest.mark.skipif(not is_mori_available(), reason="MORI is not available")
class TestMoriSharedOp:
    """MORI-only tests for process-wide EpDispatchCombineOp reuse."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_mori_op()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("num_layers,num_iters", [(3, 4), (2, 8)])
    def test_multi_layer_multi_iter_forward_backward(
        self, tp_size, ep_size, num_layers, num_iters
    ):
        require_node_spanning_mori_ep(ep_size)
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="mori",
            moe_mori_max_tokens_per_rank=4096,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_multi_layer_multi_iter_test(
            num_layers=num_layers, num_iters=num_iters
        )
