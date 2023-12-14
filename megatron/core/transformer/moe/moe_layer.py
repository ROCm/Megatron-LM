# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.base_moe_layer import ZeroDropSinkhornRouter
from megatron.core.transformer.moe.grouped_mlp import GroupedMLP
from megatron.core.transformer.moe.switch_mlp import SwitchMLP
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    def __init__(self, config: TransformerConfig):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        self.router = self.initialize_router()
        self.experts = self.initialize_experts()

    def initialize_experts(self):
        pass

    def initialize_router(self):
        pass

    def forward(self, hidden_states):
        # process MoE
        gatings, indices = self.router(hidden_states)
        (
            dispatched_input,
            tokens_per_expert,
            probs,
            indices,
            global_local_map,
        ) = self.router.token_dispatcher.dispatch(hidden_states, gatings, indices)
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.router.token_dispatcher.restore(
            expert_output, probs, indices, global_local_map, mlp_bias
        )

        if mlp_bias is None:
            mlp_bias = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # output = output.reshape(hidden_states.shape)
        return output, mlp_bias


class GroupedGemmMoELayer(BaseMoELayer):
    def __init__(self, config: TransformerConfig):
        super(GroupedGemmMoELayer, self).__init__(config=config)

    def initialize_experts(self):
        experts = GroupedMLP(self.num_local_experts, self.config)
        return experts

    def initialize_router(self):
        router = ZeroDropSinkhornRouter(
            self.num_local_experts, self.local_expert_indices, self.config
        )
        return router


class SwitchMLPLayer(BaseMoELayer):
    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        self.submodules = submodules
        super(SwitchMLPLayer, self).__init__(config=config)

    def initialize_experts(self):
        experts = SwitchMLP(self.num_local_experts, self.config, self.submodules)
        return experts

    def initialize_router(self):
        router = ZeroDropSinkhornRouter(
            self.num_local_experts, self.local_expert_indices, self.config
        )
        return router
