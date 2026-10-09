# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE layer whose dispatch + experts + combine run as Primus-Turbo MegaMoE (fused_mega_moe, bf16; with
``--fp8-recipe mxfp8`` the staged MXFP8 op, fused_mega_moe_fp8_stage1 / stage2).

Routing (router, aux losses, expert bias) and the shared expert stay in Megatron. The routed part
(dispatch, GEMM1, SwiGLU, GEMM2, combine, weighted top-k reduce) is Primus-Turbo's
``fused_mega_moe(group, x, topk_idx, topk_weights, w1, w2)`` and its fused backward
(grads for x, topk_weights, w1, w2).

MXFP8: the symmetric pool holds MEGAMOE_POOL_MULT (default Primus's 2) times the mean rows per rank; a routing that
sends more rows to one rank overflows it and the combine times out, so skewed runs size it to their measured peak (as
KOSMOS sizes its arena). Larger pools also cost memory, and pool rows * hidden must stay in int32. w1 / w2 stay bf16 parameters; the op keeps their MXFP8 quantization in a cache that
``advance_weight_generation()`` invalidates, called once per training iteration (the weights move only at
the optimizer step), as Primus's MegaMoEFP8Experts does.

Primus-Turbo (primus_turbo) must be importable; it needs a GPU at import, so the import is
deferred to layer construction.
"""

import os

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer

_OP = None
_FP8 = None
_FP8_ITER = None


def _fused_mega_moe():
    global _OP
    if _OP is None:
        from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

        _OP = fused_mega_moe
    return _OP


def _mxfp8(config):
    return bool(config.fp8) and getattr(config, "fp8_recipe", None) == "mxfp8"


def _fp8_symm_buffer(group, num_experts, tokens, topk, hidden, intermediate):
    """Allocates (or keeps) the MXFP8 symmetric buffer; the op reuses any buffer at least its size."""
    from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe

    pool_mult = int(os.environ.get("MEGAMOE_POOL_MULT", "2"))
    get_symm_buffer_for_mega_moe(group, num_experts=num_experts, num_max_tokens_per_rank=tokens, num_topk=topk,
                                 hidden=hidden, intermediate_hidden=intermediate, pool_mult=pool_mult,
                                 use_mxfp8=True)


def _fused_mega_moe_fp8():
    global _FP8
    if _FP8 is None:
        from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation
        from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
            fused_mega_moe_fp8_stage1,
            fused_mega_moe_fp8_stage2,
        )

        _FP8 = (fused_mega_moe_fp8_stage1, fused_mega_moe_fp8_stage2, advance_weight_generation)
    return _FP8


def _fp8_weight_generation():
    """Invalidates the MXFP8 weight cache once per training iteration."""
    global _FP8_ITER
    from megatron.training import get_args

    it = getattr(get_args(), "curr_iteration", None)
    if it != _FP8_ITER:
        _FP8_ITER = it
        _fused_mega_moe_fp8()[2]()


class MegaMoEExperts(MegatronModule):
    """Local experts in MegaMoE layout: weight1 [E_loc, 2I, H] = [gate; up] halves (Megatron's fc1
    layout), weight2 [E_loc, H, I]. Initialised from the SequentialMLP of the same spec, so the
    initial weights equal the SequentialMLP ones."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None):
        super().__init__(config=config)
        assert config.gated_linear_unit and not config.add_bias_linear
        seq = SequentialMLP(num_local_experts, config, submodules, pg_collection=pg_collection)
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
        self, config, submodules=None, layer_number=None, pg_collection=None, is_mtp_layer=False
    ):
        super().__init__(config, submodules, layer_number, pg_collection, is_mtp_layer)
        assert config.tensor_model_parallel_size == 1 and config.expert_tensor_parallel_size == 1
        assert config.moe_expert_capacity_factor is None, "MegaMoE is dropless"
        assert config.moe_latent_size is None and not config.moe_shared_expert_overlap
        assert config.gated_linear_unit and not config.add_bias_linear, "MegaMoE is SwiGLU, no bias"
        self.mxfp8 = _mxfp8(config)
        if self.mxfp8:
            _fused_mega_moe_fp8()
        else:
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
        x = hidden_states.contiguous()
        if self.mxfp8:
            stage1, stage2, _ = _fused_mega_moe_fp8()
            w1 = self.experts.weight1
            _fp8_symm_buffer(self.ep_group, w1.shape[0] * self.ep_group.size(), x.shape[0], k, x.shape[1],
                             w1.shape[1] // 2)
            if self.training:
                _fp8_weight_generation()
            l1, dw, handle, state = stage1(x, idx, w, self.experts.weight1, self.ep_group)
            return stage2(l1, dw, handle, state, idx, w, self.experts.weight2, self.ep_group), None
        out = _fused_mega_moe()(
            self.ep_group,
            x,
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
