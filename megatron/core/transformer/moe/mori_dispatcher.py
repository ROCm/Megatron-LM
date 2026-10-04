"""MORI-EP token dispatcher for Megatron-LM MoE (intra-node, bf16).

Drop-in for MoEFlexTokenDispatcher: same dispatch_preprocess / token_dispatch / dispatch_postprocess /
combine_preprocess / token_combine / combine_postprocess contract, with MORI's IntraNode
dispatch/combine kernels (github.com/ROCm/mori) in place of DeepEP. Local permute/unpermute and
tokens_per_expert are Megatron's own (inherited from _DeepepManager).

Plug-in (after the model is built, before the first forward):
    from megatron.core.transformer.moe.mori_dispatcher import install_mori_dispatcher
    install_mori_dispatcher(model)      # swaps every MoELayer.token_dispatcher

Environment (MORI built and importable as `mori`):
    MORI_MAX_TOKENS_PER_RANK  capacity of the symmetric buffers (tokens per rank per call), default 8192
    MORI_SHMEM_HEAP_SIZE      symmetric heap, default 24G (must be set before the first dispatch)
    MORI_EP_LAUNCH_CONFIG_MODE AUTO (default here) = MORI's shipped gfx950 tuning tables

Limits: tp_size == 1 (EP group == tp_ep group); dropless only (no capacity factor / -1 indices);
one MORI op is shared by all layers with a per-call routing handle, so dispatch/combine of one
layer must not interleave with another layer's (true for the default schedule; not for the
EP-overlap / combined-1F1B schedules).
"""

import os
from typing import List, Optional

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
    _DeepepManager,
)
from megatron.core.transformer.transformer_config import TransformerConfig

_STATE = {"group": None, "ops": {}}


def _max_tokens_per_rank() -> int:
    return int(os.environ.get("MORI_MAX_TOKENS_PER_RANK", "8192"))


def _init_shmem(group: dist.ProcessGroup) -> None:
    if _STATE["group"] is not None:
        assert _STATE["group"] is group, "MORI shmem is bound to one EP group per process"
        return
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "24G")
    os.environ.setdefault("MORI_EP_LAUNCH_CONFIG_MODE", "AUTO")
    import mori

    rank = dist.get_rank(group)
    obj = [mori.shmem.shmem_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=dist.get_global_rank(group, 0), group=group)
    mori.shmem.shmem_init_attr(
        mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, dist.get_world_size(group), obj[0]
    )
    _STATE["group"] = group


def _get_op(group, hidden: int, num_local_experts: int, topk: int):
    key = (hidden, num_local_experts, topk)
    op = _STATE["ops"].get(key)
    if op is None:
        import mori

        _init_shmem(group)
        cfg = mori.ops.EpDispatchCombineConfig(
            data_type=torch.bfloat16,
            rank=dist.get_rank(group),
            world_size=dist.get_world_size(group),
            hidden_dim=hidden,
            scale_dim=0,
            scale_type_size=1,
            max_token_type_size=4,
            max_num_inp_token_per_rank=_max_tokens_per_rank(),
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=topk,
            use_external_inp_buf=True,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            gpu_per_node=dist.get_world_size(group),
        )
        op = mori.ops.EpDispatchCombineOp(cfg)
        _STATE["ops"][key] = op
    return op


class _MoriDispatch(torch.autograd.Function):
    """fwd: MORI dispatch (new routing handle). bwd: MORI combine of grad x and grad probs."""

    @staticmethod
    def forward(ctx, x, indices, probs, op, ctx_box):
        num_tokens = x.size(0)
        if num_tokens > op.config.max_num_inp_token_per_rank:
            raise RuntimeError(
                f"MORI: {num_tokens} tokens > MORI_MAX_TOKENS_PER_RANK="
                f"{op.config.max_num_inp_token_per_rank}"
            )
        out, out_w, _, out_idx, total, routing = op.dispatch(
            x, probs, None, indices, return_routing=True
        )
        n = int(total[0].item())
        recv_x = out[:n].clone()
        recv_idx = out_idx[:n].clone()
        recv_w = out_w[:n].clone()
        ctx_box["routing"] = routing
        ctx_box["indices"] = indices
        ctx_box["num_recv"] = n
        ctx_box["src_key"] = routing.disp_tok_id_to_src_tok_id_local[:n].long()
        ctx.op = op
        ctx.box = ctx_box
        ctx.num_tokens = num_tokens
        ctx.mark_non_differentiable(recv_idx)
        return recv_x, recv_idx, recv_w

    @staticmethod
    def backward(ctx, grad_x, grad_idx, grad_w):
        box = ctx.box
        w = grad_w.float().contiguous() if grad_w is not None else None
        out, out_w = ctx.op.combine(
            grad_x.contiguous(), w, box["indices"], routing=box["routing"], call_reset=True
        )
        grad_in = out[: ctx.num_tokens].clone()
        grad_probs = out_w[: ctx.num_tokens].clone() if w is not None else None
        return grad_in, None, grad_probs, None, None


class _MoriCombine(torch.autograd.Function):
    """fwd: MORI combine (sum over destination ranks). bwd: MORI dispatch of grad, reordered."""

    @staticmethod
    def forward(ctx, y, op, ctx_box):
        out, _ = op.combine(
            y.contiguous(), None, ctx_box["indices"], routing=ctx_box["routing"], call_reset=True
        )
        num_tokens = ctx_box["indices"].size(0)
        ctx.op = op
        ctx.box = ctx_box
        return out[:num_tokens].clone()

    @staticmethod
    def backward(ctx, grad_out):
        # IntraNode ignores routing replay (the kernel skips the payload copy in replay mode), so
        # dispatch afresh and reorder the received rows to the forward layout by source-token key.
        box, op = ctx.box, ctx.op
        n = box["num_recv"]
        out, _, _, _, total, r2 = op.dispatch(
            grad_out.contiguous(), None, None, box["indices"], return_routing=True
        )
        key2 = r2.disp_tok_id_to_src_tok_id_local[:n].long()
        inv = torch.empty(
            op.config.world_size * op.max_num_tokens_to_send(), dtype=torch.long, device=out.device
        )
        inv[key2] = torch.arange(n, device=out.device)
        return out[inv[box["src_key"]]], None, None


class _MoriManager(_DeepepManager):
    """_DeepepManager with MORI in place of deep_ep's fused_dispatch/fused_combine."""

    def __init__(self, group, num_local_experts, router_topk, num_experts, config):
        self.group = group
        self.num_local_experts = num_local_experts
        self.config = config
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion
        self.token_indices = None
        self.token_probs = None
        self.handle = None
        self.rank = dist.get_rank(group)
        assert self.capacity_factor is None, "MORI dispatcher is dropless only"

    def dispatch(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        op = _get_op(self.group, hidden_states.size(-1), self.num_local_experts, self.router_topk)
        self.handle = {}
        indices = self.token_indices.to(torch.int32).contiguous()
        probs = self.token_probs.float().contiguous()
        recv_x, recv_idx, recv_w = _MoriDispatch.apply(
            hidden_states.contiguous(), indices, probs, op, self.handle
        )
        base = self.rank * self.num_local_experts
        local = (recv_idx >= base) & (recv_idx < base + self.num_local_experts)
        self.dispatched_indices = torch.where(local, recv_idx - base, -1).long()
        self.dispatched_probs = recv_w * local
        self.tokens_per_expert = torch.bincount(
            self.dispatched_indices[local], minlength=self.num_local_experts
        ).cpu()
        return recv_x

    def combine(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        op = _get_op(self.group, hidden_states.size(-1), self.num_local_experts, self.router_topk)
        out = _MoriCombine.apply(hidden_states, op, self.handle)
        self.handle = None
        return out


class MoEMoriTokenDispatcher(MoEFlexTokenDispatcher):
    """Flex token dispatcher whose communication manager is MORI-EP (IntraNode kernels)."""

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        MoETokenDispatcher.__init__(self, config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size == 1, "MORI dispatcher: expert TP must be 1"
        assert self.ep_size > 1, "MORI dispatcher requires EP > 1"
        self._comm_manager = _MoriManager(
            group=self.tp_ep_group,
            num_local_experts=num_local_experts,
            router_topk=config.moe_router_topk,
            num_experts=config.num_moe_experts,
            config=config,
        )
        self.cudagraph_attrs = []


def install_mori_dispatcher(model) -> int:
    """Replace the token dispatcher of every MoELayer under `model`. Returns the count."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    n = 0
    for m in model.modules() if hasattr(model, "modules") else []:
        if isinstance(m, MoELayer):
            old = m.token_dispatcher
            pgc = ProcessGroupCollection()
            pgc.ep, pgc.expt_tp, pgc.tp_ep = old.ep_group, old.tp_group, old.tp_ep_group
            m.token_dispatcher = MoEMoriTokenDispatcher(
                m.num_local_experts, m.local_expert_indices, m.config, pgc
            )
            n += 1
    return n
