# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE layer whose dispatch + experts + combine run as Primus-Turbo MegaMoE (fused_mega_moe, bf16).

Routing (router, aux losses, expert bias) and the shared expert stay in Megatron. The routed part
(dispatch, GEMM1, SwiGLU, GEMM2, combine, weighted top-k reduce) is Primus-Turbo's
``fused_mega_moe(group, x, topk_idx, topk_weights, w1, w2)`` and its fused backward
(grads for x, topk_weights, w1, w2).

Primus-Turbo (primus_turbo) must be importable; it needs a GPU at import, so the import is
deferred to layer construction.
"""

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer

_OP = None


def _fused_mega_moe():
    global _OP
    if _OP is None:
        from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

        _OP = fused_mega_moe
    return _OP


class MegaMoEExperts(MegatronModule):
    """Local experts in MegaMoE layout: weight1 [E_loc, 2I, H] = [gate; up] halves (Megatron's fc1
    layout), weight2 [E_loc, H, I]. Initialised from the SequentialMLP of the same spec, so the
    initial weights equal the SequentialMLP ones."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None, name=None):
        super().__init__(config=config)
        assert config.gated_linear_unit and not config.add_bias_linear
        seq = SequentialMLP(
            num_local_experts, config, submodules, pg_collection=pg_collection, name=name
        )
        with torch.no_grad():
            w1 = torch.stack([e.linear_fc1.weight.detach() for e in seq.local_experts])
            w2 = torch.stack([e.linear_fc2.weight.detach() for e in seq.local_experts])
        del seq
        self.num_local_experts = num_local_experts
        self.weight1 = torch.nn.Parameter(w1.contiguous())
        self.weight2 = torch.nn.Parameter(w2.contiguous())
        expert_parallel = config.expert_model_parallel_size > 1
        for p in (self.weight1, self.weight2):
            setattr(p, "allreduce", not expert_parallel)

    def backward_dw(self):
        pass


class MegaMoELayer(MoELayer):
    """MoELayer with the routed part (dispatch, experts, combine) replaced by fused_mega_moe."""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
    ):
        super().__init__(config, submodules, layer_number, pg_collection, is_mtp_layer, name)
        assert config.tensor_model_parallel_size == 1 and config.expert_tensor_parallel_size == 1
        assert config.moe_expert_capacity_factor is None, "MegaMoE is dropless"
        assert config.moe_latent_size is None and not config.moe_shared_expert_overlap
        assert config.gated_linear_unit and not config.add_bias_linear, "MegaMoE is SwiGLU, no bias"
        _fused_mega_moe()

    def preprocess(self, hidden_states, probs, routing_map):
        self._in_shape = hidden_states.shape
        return hidden_states.reshape(-1, hidden_states.shape[-1]), (probs, routing_map)

    def dispatch(self, hidden_states, probs):
        return hidden_states, probs

    def routed_experts_compute(self, hidden_states, probs):
        probs, routing_map = probs
        k = self.config.moe_router_topk
        idx = routing_map.to(torch.float32).topk(k, dim=1, sorted=False).indices
        w = probs.gather(1, idx).to(torch.float32)
        out = _fused_mega_moe()(
            self.ep_group,
            hidden_states.contiguous(),
            idx,
            w,
            self.experts.weight1,
            self.experts.weight2,
        )
        return out, None

    def combine(self, output):
        return output

    def postprocess(self, output, shared_expert_output):
        output = output.view(self._in_shape)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output


def megamoe_spec(moe_spec):
    """Turn an MoELayer spec whose experts are SequentialMLP into the MegaMoE layer spec."""
    from megatron.core.transformer.moe.kosmos_moe import replace_moe_experts

    return replace_moe_experts(moe_spec, MegaMoELayer, MegaMoEExperts, "--moe-use-megamoe")
