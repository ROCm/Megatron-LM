# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE layer whose dispatch + experts + combine run as KOSMOS single-launch kernels (EP=8, bf16).

The routed part (dispatch, GEMM1 + SwiGLU, GEMM2, combine, weighted sum) is one forward launch,
which takes the routing in device memory and also stores the activations the backward needs, and
its backward is one launch, which returns the bf16 input gradient and the fp32 probs gradient.

Routing: for the softmax (top-k on the logits) and sigmoid (expert bias, group-limited) score
functions with aux_loss or seq_aux_loss, the fused router runs inside the same two launches, and
Megatron's router module only holds the router weight and the expert bias. Other router configs,
and calls with a padding mask, keep Megatron's router. The shared expert stays in Megatron.

Every MoE layer has its own context (its arena and tables); all layers of one shape share one
workspace (the per-call buffers), so any recompute pattern works. kosmos_gate.py keeps Megatron
DDP communication out of the KOSMOS kernels.

Environment:
  KOSMOS_PYTHON             directory of the kosmos module (KOSMOS `make python`: build/python);
                            unset: the module is found on PYTHONPATH
  KOSMOS_WGCAP              launch width (workgroups), default 256
  KOSMOS_CAPACITY_FACTOR    arena rows reserved per layer = factor * T * k, default 1.5
  KOSMOS_ROUTER             1 (default): the fused router where supported; 0: Megatron's router
  KOSMOS_DDP_GATE           1 (default): the DDP gate (kosmos_gate.py); 0: off

This file also holds the PyTorch+RCCL baseline (--moe-use-torch-experts): SequentialMLP experts
on Megatron's local (torch matmul) linears.
"""

import dataclasses
import os
import sys
from functools import partial

import torch
import torch.distributed as dist

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import kosmos_gate
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec

ROUTER = bool(int(os.environ.get("KOSMOS_ROUTER", "1")))

# ---------------------------------------------------------------------------------------------
# KOSMOS binding: every call into it goes through the helpers below.

_KOSMOS = None


def _kosmos():
    """The framework-free kosmos module (pybind11 over libkosmos_moe.a)."""
    global _KOSMOS
    if _KOSMOS is None:
        d = os.environ.get("KOSMOS_PYTHON")
        if d and d not in sys.path:
            sys.path.insert(0, d)
        import kosmos

        _KOSMOS = kosmos
    return _KOSMOS


def _ptr(t):
    """Device address of a contiguous tensor; None -> 0 (null)."""
    if t is None:
        return 0
    assert t.is_contiguous(), "KOSMOS takes contiguous tensors"
    return t.data_ptr()


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _host_allgather(ep_group):
    """All-gather of host bytes over the EP group, for the library's collective setup steps
    (first forward of each layer): send (bytes) -> the ranks' sends concatenated in rank order."""

    def allgather(send):
        src = torch.frombuffer(bytearray(send), dtype=torch.uint8).cuda()
        dev = torch.empty(src.numel() * ep_group.size(), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(dev, src, group=ep_group)
        return dev.cpu().numpy().tobytes()

    return allgather


def _new_workspace(config, ep_group, num_local_experts, T, router):
    kosmos = _kosmos()
    k = config.moe_router_topk
    capacity_rows = int(float(os.environ.get("KOSMOS_CAPACITY_FACTOR", "1.5")) * T * k)
    rcfg = None
    if router is not None:
        # router: (sigmoid, groups, group_topk, aux_groups, aux_coeff, scale, force)
        score = kosmos.ROUTER_SIGMOID if router[0] else kosmos.ROUTER_SOFTMAX
        rcfg = kosmos.MoeRouterConfig(score, *router[1:])
    return kosmos.MoeWorkspace(
        config.hidden_size,
        config.moe_ffn_hidden_size,
        num_local_experts,
        k,
        T,
        ep_group.rank(),
        _host_allgather(ep_group),
        wgcap=int(os.environ.get("KOSMOS_WGCAP", "256")),
        capacity_rows=capacity_rows,
        router=rcfg,
    )


def _new_context(ws):
    return _kosmos().MoeContext(ws)


def _new_g(K, x, need_g):
    """The forward's stored activations for the backward (K.g_bytes bytes), or None."""
    return torch.empty((K.g_bytes + 1) // 2, dtype=x.dtype, device=x.device) if need_g else None


def _forward(K, x, w1, w2, idx, w, need_g):
    """-> (out, G or None)."""
    out = torch.empty_like(x)
    G = _new_g(K, x, need_g)
    K.forward(*map(_ptr, (x, w1, w2, idx, w, out, G)), _stream())
    return out, G


def _backward(K, dout, G, w1, w2):
    """-> (dX, dprobs, dW1, dW2)."""
    dX = torch.empty_like(dout)
    dprobs = torch.empty(
        (dout.shape[0], _kosmos().EP * w1.shape[0]), dtype=torch.float32, device=dout.device
    )
    dW1, dW2 = torch.empty_like(w1), torch.empty_like(w2)
    K.backward(*map(_ptr, (dout, G, w1, w2, dX, dprobs, dW1, dW2)), _stream())
    return dX, dprobs, dW1, dW2


def _forward_router(K, x, wr, bias, w1, w2, seed, need_g):
    """-> (out, G or None, aux loss, tokens per expert)."""
    out = torch.empty_like(x)
    G = _new_g(K, x, need_g)
    aux = torch.empty(1, dtype=torch.float32, device=x.device)
    tpe = torch.empty(wr.shape[0], dtype=torch.int32, device=x.device)
    K.forward_router(*map(_ptr, (x, wr, bias, w1, w2, out, G, aux, tpe, seed)), _stream())
    return out, G, aux, tpe


def _backward_router(K, dout, G, x, w1, w2, wr, aux_scale):
    """-> (dX, dW1, dW2, dWr)."""
    dX = torch.empty_like(dout)
    dW1, dW2, dWr = torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(wr)
    ins = map(_ptr, (dout, G, x, w1, w2, wr))
    K.backward_router(*ins, aux_scale, *map(_ptr, (dX, dW1, dW2, dWr)), _stream())
    return dX, dW1, dW2, dWr


# ---------------------------------------------------------------------------------------------

_WS = {}


def _workspace(config, ep_group, num_local_experts, T, router=None):
    """The workspace shared by every MoE layer of one shape (and fused-router config)."""
    key = (
        config.hidden_size,
        config.moe_ffn_hidden_size,
        num_local_experts,
        config.moe_router_topk,
        T,
        tuple(dist.get_process_group_ranks(ep_group)),
        router,
    )
    if key not in _WS:
        _WS[key] = _new_workspace(config, ep_group, num_local_experts, T, router)
    return _WS[key]


def w1_kosmos_perm(I, device=None):
    """Row permutation: KOSMOS W1 row r = Megatron [gate; up] row perm[r] (128-row interleave)."""
    r = torch.arange(2 * I, device=device)
    return ((r % 256) // 128) * I + (r // 256) * 128 + r % 128


def w1_to_kosmos(w1):
    """[E, 2I, H] contiguous halves [gate; up] -> KOSMOS interleave."""
    return w1[:, w1_kosmos_perm(w1.shape[1] // 2, w1.device), :].contiguous()


class KosmosContext:
    """One MoE layer's KOSMOS context (its arena and tables), created at first use on the shared
    workspace; router: the fused router's config (KosmosMoELayer._router_cfg) or None."""

    def __init__(self, config, ep_group, num_local_experts):
        self.config = config
        self.ep_group = ep_group
        self.E_loc = num_local_experts
        self.ctx = None
        self.T = None

    def get(self, T, router=None):
        if self.ctx is None:
            self.ctx = _new_context(_workspace(self.config, self.ep_group, self.E_loc, T, router))
            self.T = T
        assert T == self.T, f"KOSMOS context built for T={self.T}, got T={T}"
        return self.ctx


class KosmosMoEFunction(torch.autograd.Function):
    """out = sum_j w_j * Expert_{idx_j}(x) over the EP group; grads for x, probs, W1, W2."""

    @staticmethod
    def forward(ctx, hidden, probs, routing_map, w1, w2, kctx, need_g):
        T, _ = hidden.shape
        k = kctx.config.moe_router_topk
        assert (
            probs.dtype == torch.float32
        ), "KOSMOS writes the probs gradient in fp32 (--moe-router-dtype fp32)"
        idx = routing_map.to(torch.float32).topk(k, dim=1, sorted=False).indices
        w = probs.gather(1, idx).to(torch.float32)
        K = kctx.get(T)
        idx32 = idx.to(torch.int32).contiguous()
        kosmos_gate.before_kernel()
        out, G = _forward(K, hidden.contiguous(), w1, w2, idx32, w.contiguous(), need_g)
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        ctx.save_for_backward(w1, w2)
        ctx.G = G
        ctx.kctx = kctx
        return out

    @staticmethod
    def backward(ctx, dout):
        w1, w2 = ctx.saved_tensors
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        dX, dprobs, dW1, dW2 = _backward(K, dout, ctx.G, w1, w2)
        kosmos_gate.after_kernel(backward=True)
        ctx.G = None
        return dX, dprobs, None, dW1, dW2, None, None


_AUX_SCALE = None


def _aux_scale():
    """MoEAuxLossAutoScaler.main_loss_backward_scale, read once on the first backward."""
    global _AUX_SCALE
    if _AUX_SCALE is None:
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        s = MoEAuxLossAutoScaler.main_loss_backward_scale
        _AUX_SCALE = 1.0 if s is None else float(s)
    return float(_AUX_SCALE)


class KosmosRouterMoEFunction(torch.autograd.Function):
    """Router + MoE in the two KOSMOS launches: out, aux-loss value and the dispatch counts per
    expert from hidden; grads for hidden (MoE + router), the router weight, W1 and W2. The aux-loss
    gradient is applied inside the backward launch."""

    @staticmethod
    def forward(ctx, hidden, wr, bias, w1, w2, kctx, need_g, seed, router_cfg):
        T, _ = hidden.shape
        K = kctx.get(T, router_cfg)
        kosmos_gate.before_kernel()
        out, G, aux, tpe = _forward_router(
            K, hidden.contiguous(), wr.contiguous(), bias, w1, w2, seed, need_g
        )
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        ctx.save_for_backward(hidden, wr, w1, w2)
        ctx.G = G
        ctx.kctx = kctx
        ctx.mark_non_differentiable(aux, tpe)
        return out, aux, tpe

    @staticmethod
    def backward(ctx, dout, daux, dtpe):
        hidden, wr, w1, w2 = ctx.saved_tensors
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        dX, dW1, dW2, dWr = _backward_router(K, dout, ctx.G, hidden, w1, w2, wr, _aux_scale())
        kosmos_gate.after_kernel(backward=True)
        ctx.G = None
        return dX, dWr, None, dW1, dW2, None, None, None, None


class KosmosExperts(MegatronModule):
    """Local experts stored in KOSMOS layout: weight1 [E_loc, 2I, H] (128-row gate|up interleave),
    weight2 [E_loc, H, I]. Initialised by building the SequentialMLP of the same spec and
    converting its weights, so a KOSMOS model starts from the same weights as SequentialMLP."""

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
        self.weight1 = torch.nn.Parameter(w1_to_kosmos(w1))
        self.weight2 = torch.nn.Parameter(w2.contiguous())
        expert_parallel = config.expert_model_parallel_size > 1
        for p in (self.weight1, self.weight2):
            setattr(p, "allreduce", not expert_parallel)

    def backward_dw(self):
        pass


class KosmosMoELayer(MoELayer):
    """MoELayer with the routed part (dispatch, experts, combine) replaced by KOSMOS."""

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
        kosmos_gate.enable()
        assert config.tensor_model_parallel_size == 1 and config.expert_tensor_parallel_size == 1
        assert self.ep_group.size() == 8, "KOSMOS MoE is EP=8"
        assert config.moe_router_topk in (6, 8, 10)
        assert config.moe_latent_size is None and not config.moe_shared_expert_overlap
        self.kctx = KosmosContext(config, self.ep_group, self.num_local_experts)
        self._fused_router = ROUTER and self._router_supported()

    def _router_supported(self):
        c, r = self.config, self.router
        return (
            c.moe_router_score_function in ("sigmoid", "softmax")
            and not (c.moe_router_score_function == "softmax" and c.moe_router_pre_softmax)
            and not r.get_aux_loss_coeff("global_aux_loss")
            and not c.moe_z_loss_coeff
            and not c.moe_input_jitter_eps
            and not (r.get_aux_loss_coeff("seq_aux_loss") and r.get_aux_loss_coeff("aux_loss"))
            and c.moe_router_force_biased is None
            and not c.moe_enable_routing_replay
        )

    def route(self, hidden_states, padding_mask=None):
        # (None, None): the fused router routes inside the KOSMOS launches (routed_experts_compute).
        if not self._fused_router or padding_mask is not None:
            return super().route(hidden_states, padding_mask)
        return None, None

    def _router_cfg(self, bsz):
        """(sigmoid, groups, group_topk, aux_groups, aux_coeff, scale, force) of Megatron's router
        config."""
        c, r = self.config, self.router
        seq, aux = r.get_aux_loss_coeff("seq_aux_loss"), r.get_aux_loss_coeff("aux_loss")
        return (
            int(c.moe_router_score_function == "sigmoid"),
            c.moe_router_num_groups or 0,
            c.moe_router_group_topk or 0,
            bsz if seq else (1 if aux else 0),
            seq or aux,
            c.moe_router_topk_scaling_factor or 1.0,
            int(bool(c.moe_router_force_load_balancing)),
        )

    def _routed_with_router(self, hidden_states):
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_expert_parallel_rng_tracker_name,
        )

        r = self.router
        r._maintain_float32_expert_bias()
        # The forced draw's seed comes from the expert-parallel RNG stream (as RandomSTE's
        # normal_), so a recomputed forward reproduces it.
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=hidden_states.device)
        cfg = self._router_cfg(self._in_shape[1])
        out, aux, tpe = KosmosRouterMoEFunction.apply(
            hidden_states,
            r.weight,
            r.expert_bias if r.enable_expert_bias else None,
            self.experts.weight1,
            self.experts.weight2,
            self.kctx,
            torch.is_grad_enabled(),
            seed,
            cfg,
        )
        if r.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                r.local_tokens_per_expert += tpe.to(r.local_tokens_per_expert.dtype)
        if self.training and torch.is_grad_enabled() and cfg[3]:
            name = (
                "seq_load_balancing_loss"
                if r.get_aux_loss_coeff("seq_aux_loss")
                else "load_balancing_loss"
            )
            out = r.attach_and_log_load_balancing_loss(
                out, cfg[4], aux[0], name, r.tp_cp_group, valid_token_count=hidden_states.shape[0]
            )
        return out

    def preprocess(self, hidden_states, probs, routing_map):
        self._in_shape = hidden_states.shape
        return hidden_states.reshape(-1, hidden_states.shape[-1]), (probs, routing_map)

    def dispatch(self, hidden_states, probs):
        return hidden_states, probs

    def routed_experts_compute(self, hidden_states, probs):
        probs, routing_map = probs
        if probs is None:
            return self._routed_with_router(hidden_states), None
        if self._fused_router:
            # Megatron's routing on a fused-router layer: the context still goes on the router
            # workspace.
            self.kctx.get(hidden_states.shape[0], self._router_cfg(self._in_shape[1]))
        out = KosmosMoEFunction.apply(
            hidden_states,
            probs,
            routing_map,
            self.experts.weight1,
            self.experts.weight2,
            self.kctx,
            torch.is_grad_enabled(),
        )
        return out, None

    def combine(self, output):
        return output

    def postprocess(self, output, shared_expert_output):
        output = output.view(self._in_shape)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output


def _builder_parts(builder):
    """(class, submodules, other kwargs) of a module builder: functools.partial (the MoE specs
    since core_r0.18) or ModuleSpec (older spec helpers)."""
    if isinstance(builder, ModuleSpec):
        return builder.module, builder.submodules, dict(builder.params)
    assert isinstance(builder, partial) and not builder.args, f"unsupported builder {builder!r}"
    kwargs = dict(builder.keywords)
    submodules = kwargs.pop("submodules", None)
    return builder.func, submodules, kwargs


def replace_moe_experts(moe_spec, layer_cls, experts_cls, flag, experts_submodules=None):
    """MoELayer builder whose experts are SequentialMLP -> layer_cls builder with experts_cls
    experts (built from experts_submodules, default the SequentialMLP's MLPSubmodules)."""
    _, sub, kwargs = _builder_parts(moe_spec)
    e_cls, e_sub, e_kwargs = _builder_parts(sub.experts)
    assert (
        e_cls is SequentialMLP
    ), f"{flag} needs the SequentialMLP expert spec (do not pass --moe-grouped-gemm)"
    experts = partial(
        experts_cls,
        submodules=e_sub if experts_submodules is None else experts_submodules,
        **e_kwargs,
    )
    return partial(layer_cls, submodules=dataclasses.replace(sub, experts=experts), **kwargs)


def kosmos_moe_spec(moe_spec):
    """Turn an MoELayer spec whose experts are SequentialMLP into the KOSMOS layer spec."""
    return replace_moe_experts(moe_spec, KosmosMoELayer, KosmosExperts, "--moe-use-kosmos")


class TorchSwiGLU(torch.nn.Module):
    """silu(gate) * up on [.., 2I] with contiguous halves [gate; up] (torch ops; used when
    --use-te-activation-func is set, so the torch-experts baseline stays TE-free)."""

    def __init__(self, config=None):
        super().__init__()

    def forward(self, x):
        g, u = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(g) * u


def torch_experts_spec(moe_spec):
    """PyTorch+RCCL baseline: SequentialMLP experts on Megatron's local (torch matmul) linears."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLPSubmodules

    layer_cls = _builder_parts(moe_spec)[0]
    local = MLPSubmodules(
        linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear, activation_func=TorchSwiGLU
    )
    return replace_moe_experts(
        moe_spec, layer_cls, SequentialMLP, "--moe-use-torch-experts", experts_submodules=local
    )
