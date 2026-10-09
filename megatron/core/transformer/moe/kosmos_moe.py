# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE layer whose dispatch + experts + combine run as KOSMOS single-launch kernels (EP=8, bf16).

The routed part (dispatch, GEMM1 + SwiGLU, GEMM2, combine, weighted sum) is one forward launch,
which takes the routing in device memory and also stores the activations the backward needs, and
its backward is one launch, which returns the bf16 input gradient and the fp32 probs gradient.

Routing: for the softmax (top-k on the logits) and sigmoid (expert bias, group-limited) score
functions with aux_loss or seq_aux_loss, the fused router runs inside the same two launches, and
Megatron's router module only holds the router weight and the expert bias. Other router configs,
and calls with a padding mask, keep Megatron's router.

Shared expert (KOSMOS_SHARED=1): when the layer's shared expert is one ungated SwiGLU MLP of the
experts' intermediate size, it runs inside the same two launches (fc1 + SwiGLU + fc2 alongside the
routed experts, its output added at the combine; its dgrad/wgrad in the backward launch, its dX
added to the input gradient). Megatron's SharedExpertMLP keeps the parameters; its forward is
skipped. With gradient-accumulation fusion the shared weight gradients are added into main_grad
(fp32 or bf16) by the kernel (grad_added_to_main_grad, dummy grads, as TE does); otherwise they
are returned as bf16 gradients. Unsupported configs, and a token count that is not a multiple of
512, keep Megatron's shared expert.

MXFP8 (KOSMOS_MXFP8=1, --fp8-recipe mxfp8): on the layers Megatron runs in FP8, the four expert
GEMMs run in MXFP8 (TE recipe, no requantization): the forward quantizes X at the source and
dispatches it in MXFP8, the backward dispatches dY in bf16 and quantizes it at the experts. The
weights are quantized both ways once per step: with one microbatch per step, the row-wise copy in
the forward (kept from a checkpointed first pass for its recompute) and the column-wise copy in
the backward, each freed after its use; with more, both on the first microbatch and cached for
the step (as TE's weight cache);
the layer's inputs, outputs and gradients stay bf16 (probs gradient fp32). MXFP8 layers use
Megatron's router and shared expert, and need a hidden size that is a multiple of 1024.

Every MoE layer has its own context (its arena and tables); all layers of one shape share one
workspace (the per-call buffers), so any recompute pattern works. kosmos_gate.py keeps Megatron
DDP communication out of the KOSMOS kernels.

Capacity: a call that routes more rows to one rank than the arena holds (capacity_rows, per rank,
experts padded to 256 rows) gets the empty layout and undefined outputs, and sets a status bit on
the device that stays set until read. At exit every layer's status is read once (one device sync)
and each rank prints "[KOSMOS] plan status rank R: ..." (OVERFLOW if any call exceeded capacity).

Environment:
  KOSMOS_PYTHON             directory of the kosmos module (KOSMOS `make python`: build/python);
                            unset: the module is found on PYTHONPATH
  KOSMOS_WGCAP              launch width (workgroups), default 256
  KOSMOS_CAPACITY_FACTOR    arena rows reserved per layer and rank = factor * T * k, default 1.5
  KOSMOS_ROUTER             1 (default): the fused router where supported; 0: Megatron's router
  KOSMOS_DDP_GATE           1 (default): the DDP gate (kosmos_gate.py); 0: off
  KOSMOS_SHARED             1 (default): the shared expert in the KOSMOS launches where supported;
                            0: in Megatron
  KOSMOS_MXFP8              1 (default): MXFP8 experts on the FP8 layers of an mxfp8-recipe run;
                            0: bf16 experts

This file also holds the PyTorch+RCCL baseline (--moe-use-torch-experts): SequentialMLP experts
on Megatron's local (torch matmul) linears.
"""

import atexit
import dataclasses
import os
import sys

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_first_last_bf16_layer
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import kosmos_gate
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import record_routing_stats
from megatron.core.transformer.spec_utils import ModuleSpec

ROUTER = bool(int(os.environ.get("KOSMOS_ROUTER", "1")))
SHARED = bool(int(os.environ.get("KOSMOS_SHARED", "1")))
MXFP8 = bool(int(os.environ.get("KOSMOS_MXFP8", "1")))
MXFP8_H_MULTIPLE = 1024
SHARED_T_MULTIPLE = 512

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


def _new_workspace(config, ep_group, num_local_experts, T, router, shared, mxfp8):
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
        shared=shared,
        mxfp8=mxfp8,
    )


def _new_context(ws):
    return _kosmos().MoeContext(ws)


def _new_g(K, x, need_g):
    """The forward's stored activations for the backward (K.g_bytes bytes), or None."""
    return torch.empty((K.g_bytes + 1) // 2, dtype=x.dtype, device=x.device) if need_g else None


def _topk(probs, routing_map, k):
    """Megatron's routing -> (topk_idx int32, topk_w fp32), contiguous."""
    assert (
        probs.dtype == torch.float32
    ), "KOSMOS writes the probs gradient in fp32 (--moe-router-dtype fp32)"
    idx = routing_map.to(torch.float32).topk(k, dim=1, sorted=False).indices
    w = probs.gather(1, idx).to(torch.float32)
    return idx.to(torch.int32).contiguous(), w.contiguous()


def _quantize_mxfp8(w, col):
    """bf16 [E, R, C] -> (e4m3 [E, R, C], e8m0 scales [E, R, C / 32] or column-wise
    [E, R / 32, C]), both as uint8."""
    E, R, C = w.shape
    q = torch.empty(w.shape, dtype=torch.uint8, device=w.device)
    s = torch.empty((E, R // 32, C) if col else (E, R, C // 32), dtype=torch.uint8, device=w.device)
    _kosmos().quantize_mxfp8(_ptr(w), E, R, C, col, _ptr(q), _ptr(s), _stream())
    return q, s


def _forward_mxfp8(K, x, mxw, idx, w):
    """-> (out, G); mxw: KosmosExperts.mxfp8_row(). The MXFP8 forward always stores G."""
    out = torch.empty_like(x)
    G = _new_g(K, x, True)
    K.forward_mxfp8_train(*map(_ptr, (x, *mxw[:4], idx, w, out, G)), _stream())
    return out, G


def _backward_mxfp8(K, dout, G, x, w1, w2, mxc):
    """-> (dX, dprobs, dW1, dW2); mxc: KosmosExperts.mxfp8_col()."""
    dX = torch.empty_like(dout)
    dprobs = torch.empty(
        (dout.shape[0], _kosmos().EP * w1.shape[0]), dtype=torch.float32, device=dout.device
    )
    dW1, dW2 = torch.empty_like(w1), torch.empty_like(w2)
    K.backward_mxfp8(*map(_ptr, (dout, G, x, *mxc, dX, dprobs, dW1, dW2)), _stream())
    return dX, dprobs, dW1, dW2


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


# Shared expert: ws1 [2I, H] ([gate; up], Megatron's linear_fc1.weight), ws2 [H, I]; Gs [T, 2I] its
# stored pre-activations.


def _new_gs(x, ws1, need_g):
    return torch.empty((x.shape[0], ws1.shape[0]), dtype=x.dtype, device=x.device) if need_g else None


def _forward_shared(K, x, w1, w2, ws1, ws2, idx, w, need_g):
    """-> (out, G or None, Gs or None)."""
    out = torch.empty_like(x)
    G, Gs = _new_g(K, x, need_g), _new_gs(x, ws1, need_g)
    K.forward_shared(*map(_ptr, (x, w1, w2, ws1, ws2, idx, w, out, G, Gs)), _stream())
    return out, G, Gs


def _forward_router_shared(K, x, wr, bias, w1, w2, ws1, ws2, seed, need_g):
    """-> (out, G or None, Gs or None, aux loss, tokens per expert)."""
    out = torch.empty_like(x)
    G, Gs = _new_g(K, x, need_g), _new_gs(x, ws1, need_g)
    aux = torch.empty(1, dtype=torch.float32, device=x.device)
    tpe = torch.empty(wr.shape[0], dtype=torch.int32, device=x.device)
    K.forward_router_shared(
        *map(_ptr, (x, wr, bias, w1, w2, ws1, ws2, out, G, Gs, aux, tpe, seed)), _stream()
    )
    return out, G, Gs, aux, tpe


_ACC_MODE = {torch.float32: 1, torch.bfloat16: 2}


def _main_grad_mode(config, p):
    """The library's accumulate mode for p.main_grad (1 fp32, 2 bf16) if the kernel may add p's
    gradient into it (gradient-accumulation fusion), else 0."""
    mg = getattr(p, "main_grad", None)
    if (
        config.gradient_accumulation_fusion
        and mg is not None
        and mg.is_contiguous()
        and mg.shape == p.shape
        and hasattr(p, "grad_added_to_main_grad")
    ):
        return _ACC_MODE.get(mg.dtype, 0)
    return 0


def _shared_dw(config, ws1, ws2):
    """-> (dWs1, dWs2, acc): the main_grads (acc 1 fp32 / 2 bf16, both the same) or new bf16
    buffers (acc 0)."""
    a1, a2 = _main_grad_mode(config, ws1), _main_grad_mode(config, ws2)
    if a1 and a1 == a2:
        return ws1.main_grad, ws2.main_grad, a1
    return torch.empty_like(ws1), torch.empty_like(ws2), 0


def _dummy_wgrad(p):
    """The gradient returned for a weight whose gradient the kernel added into main_grad: a dummy, so
    the DDP hook runs on the backward thread (Megatron's LinearWithGradAccumulation does the same)."""
    zero = getattr(p, "zero_out_wgrad", False)
    try:
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        return get_dummy_wgrad(list(p.main_grad.shape), p.dtype, zero=zero)
    except ImportError:
        f = torch.zeros if zero else torch.empty
        return f(p.main_grad.shape, dtype=p.dtype, device=p.device, requires_grad=False)


def _shared_grads(ws1, ws2, dWs1, dWs2, acc):
    if not acc:
        return dWs1, dWs2
    ws1.grad_added_to_main_grad = True
    ws2.grad_added_to_main_grad = True
    return _dummy_wgrad(ws1), _dummy_wgrad(ws2)


def _backward_shared(K, config, dout, G, Gs, x, w1, w2, ws1, ws2):
    """-> (dX, dprobs, dW1, dW2, dWs1 grad, dWs2 grad)."""
    dX = torch.empty_like(dout)
    dprobs = torch.empty(
        (dout.shape[0], _kosmos().EP * w1.shape[0]), dtype=torch.float32, device=dout.device
    )
    dW1, dW2 = torch.empty_like(w1), torch.empty_like(w2)
    dWs1, dWs2, acc = _shared_dw(config, ws1, ws2)
    ins = map(_ptr, (dout, G, Gs, x, w1, w2, ws1, ws2, dX, dprobs, dW1, dW2, dWs1, dWs2))
    K.backward_shared(*ins, acc, _stream())
    return (dX, dprobs, dW1, dW2) + _shared_grads(ws1, ws2, dWs1, dWs2, acc)


def _backward_router_shared(K, config, dout, G, Gs, x, w1, w2, ws1, ws2, wr, aux_scale):
    """-> (dX, dW1, dW2, dWr, dWs1 grad, dWs2 grad)."""
    dX = torch.empty_like(dout)
    dW1, dW2, dWr = torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(wr)
    dWs1, dWs2, acc = _shared_dw(config, ws1, ws2)
    ins = map(_ptr, (dout, G, Gs, x, w1, w2, ws1, ws2, wr))
    outs = map(_ptr, (dX, dW1, dW2, dWr, dWs1, dWs2))
    K.backward_router_shared(*ins, aux_scale, *outs, acc, _stream())
    return (dX, dW1, dW2, dWr) + _shared_grads(ws1, ws2, dWs1, dWs2, acc)


# ---------------------------------------------------------------------------------------------

_WS = {}


def _workspace(config, ep_group, num_local_experts, T, router=None, shared=False, mxfp8=False):
    """The workspace shared by every MoE layer of one shape (and fused-router, shared-expert and
    MXFP8 config)."""
    key = (
        config.hidden_size,
        config.moe_ffn_hidden_size,
        num_local_experts,
        config.moe_router_topk,
        T,
        tuple(dist.get_process_group_ranks(ep_group)),
        router,
        shared,
        mxfp8,
    )
    if key not in _WS:
        _WS[key] = _new_workspace(config, ep_group, num_local_experts, T, router, shared, mxfp8)
    return _WS[key]


def w1_kosmos_perm(I, device=None):
    """Row permutation: KOSMOS W1 row r = Megatron [gate; up] row perm[r] (128-row interleave)."""
    r = torch.arange(2 * I, device=device)
    return ((r % 256) // 128) * I + (r // 256) * 128 + r % 128


def w1_to_kosmos(w1):
    """[E, 2I, H] contiguous halves [gate; up] -> KOSMOS interleave."""
    return w1[:, w1_kosmos_perm(w1.shape[1] // 2, w1.device), :].contiguous()


_CONTEXTS = []


def _report_status():
    """At exit: each layer's device status (the OR over all its calls), one line per rank."""
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    flagged, n = [], 0
    try:
        for kc in _CONTEXTS:
            st = kc.ctx.plan_status()
            n += 1
            if st:
                flagged.append(f"layer {kc.layer_number} status {st}")
        rows = max(kc.ctx.info()["arena_rows"] for kc in _CONTEXTS)
    except Exception as e:  # pylint: disable=broad-except
        print(f"[KOSMOS] plan status rank {rank}: QUERY FAILED: {e}", flush=True)
        return
    cf = os.environ.get("KOSMOS_CAPACITY_FACTOR", "1.5")
    if flagged:
        print(
            f"[KOSMOS] plan status rank {rank}: OVERFLOW or bad routing in {len(flagged)} of {n} "
            f"layers ({', '.join(flagged)}; 4 = over capacity, 1 = bad expert id, 2 = duplicate); "
            f"arena {rows} rows (KOSMOS_CAPACITY_FACTOR {cf}): those calls' outputs are undefined",
            flush=True,
        )
    else:
        print(
            f"[KOSMOS] plan status rank {rank}: ok, {n} layers, arena {rows} rows "
            f"(KOSMOS_CAPACITY_FACTOR {cf})",
            flush=True,
        )


class KosmosContext:
    """One MoE layer's KOSMOS context (its arena and tables), created at first use on the shared
    workspace; router: the fused router's config (KosmosMoELayer._router_cfg) or None."""

    def __init__(self, config, ep_group, num_local_experts, shared=False, mxfp8=False):
        self.config = config
        self.ep_group = ep_group
        self.E_loc = num_local_experts
        self.shared = shared
        self.mxfp8 = mxfp8
        self.layer_number = None
        self.ctx = None
        self.T = None

    def get(self, T, router=None):
        if self.ctx is None:
            ws = _workspace(
                self.config, self.ep_group, self.E_loc, T, router, self.shared, self.mxfp8
            )
            self.ctx = _new_context(ws)
            self.T = T
            if not _CONTEXTS:
                atexit.register(_report_status)
            _CONTEXTS.append(self)
        assert T == self.T, f"KOSMOS context built for T={self.T}, got T={T}"
        return self.ctx


class KosmosMoEFunction(torch.autograd.Function):
    """out = sum_j w_j * Expert_{idx_j}(x) over the EP group; grads for x, probs, W1, W2."""

    @staticmethod
    def forward(ctx, hidden, probs, routing_map, w1, w2, kctx, need_g, ws1=None, ws2=None):
        idx32, w = _topk(probs, routing_map, kctx.config.moe_router_topk)
        K = kctx.get(hidden.shape[0])
        ctx.shared = ws1 is not None
        hidden = hidden.contiguous()
        kosmos_gate.before_kernel()
        if ctx.shared:
            out, G, Gs = _forward_shared(K, hidden, w1, w2, ws1, ws2, idx32, w, need_g)
        else:
            out, G = _forward(K, hidden, w1, w2, idx32, w, need_g)
            Gs = None
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        if ctx.shared:
            ctx.save_for_backward(w1, w2, hidden, ws1, ws2)
        else:
            ctx.save_for_backward(w1, w2)
        ctx.G, ctx.Gs = G, Gs
        ctx.kctx = kctx
        return out

    @staticmethod
    def backward(ctx, dout):
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        if ctx.shared:
            w1, w2, hidden, ws1, ws2 = ctx.saved_tensors
            dX, dprobs, dW1, dW2, dWs1, dWs2 = _backward_shared(
                K, ctx.kctx.config, dout, ctx.G, ctx.Gs, hidden, w1, w2, ws1, ws2
            )
        else:
            w1, w2 = ctx.saved_tensors
            dX, dprobs, dW1, dW2 = _backward(K, dout, ctx.G, w1, w2)
            dWs1 = dWs2 = None
        kosmos_gate.after_kernel(backward=True)
        ctx.G = ctx.Gs = None
        return dX, dprobs, None, dW1, dW2, None, None, dWs1, dWs2


class KosmosMXFP8MoEFunction(torch.autograd.Function):
    """KosmosMoEFunction with the expert GEMMs in MXFP8 (experts: the KosmosExperts of w1, w2)."""

    @staticmethod
    def forward(ctx, hidden, probs, routing_map, w1, w2, kctx, need_g, experts):
        idx32, w = _topk(probs, routing_map, kctx.config.moe_router_topk)
        K = kctx.get(hidden.shape[0])
        hidden = hidden.contiguous()
        # Quantize after the gate: it waits for the param all-gather of the step's updated weights.
        kosmos_gate.before_kernel()
        out, G = _forward_mxfp8(K, hidden, experts.mxfp8_row(keep=not need_g), idx32, w)
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        ctx.save_for_backward(w1, w2, hidden)
        ctx.G = G if need_g else None
        ctx.experts, ctx.kctx = experts, kctx
        return out

    @staticmethod
    def backward(ctx, dout):
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        w1, w2, hidden = ctx.saved_tensors
        kosmos_gate.before_kernel()
        dX, dprobs, dW1, dW2 = _backward_mxfp8(
            ctx.kctx.ctx, dout.contiguous(), ctx.G, hidden, w1, w2, ctx.experts.mxfp8_col()
        )
        kosmos_gate.after_kernel(backward=True)
        ctx.G = ctx.experts = None
        return dX, dprobs, None, dW1, dW2, None, None, None


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
    def forward(ctx, hidden, wr, bias, w1, w2, kctx, need_g, seed, router_cfg, ws1=None, ws2=None):
        T, _ = hidden.shape
        K = kctx.get(T, router_cfg)
        ctx.shared = ws1 is not None
        hidden = hidden.contiguous()
        kosmos_gate.before_kernel()
        if ctx.shared:
            out, G, Gs, aux, tpe = _forward_router_shared(
                K, hidden, wr.contiguous(), bias, w1, w2, ws1, ws2, seed, need_g
            )
        else:
            out, G, aux, tpe = _forward_router(K, hidden, wr.contiguous(), bias, w1, w2, seed, need_g)
            Gs = None
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        if ctx.shared:
            ctx.save_for_backward(hidden, wr, w1, w2, ws1, ws2)
        else:
            ctx.save_for_backward(hidden, wr, w1, w2)
        ctx.G, ctx.Gs = G, Gs
        ctx.kctx = kctx
        ctx.mark_non_differentiable(aux, tpe)
        return out, aux, tpe

    @staticmethod
    def backward(ctx, dout, daux, dtpe):
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        if ctx.shared:
            hidden, wr, w1, w2, ws1, ws2 = ctx.saved_tensors
            dX, dW1, dW2, dWr, dWs1, dWs2 = _backward_router_shared(
                K, ctx.kctx.config, dout, ctx.G, ctx.Gs, hidden, w1, w2, ws1, ws2, wr, _aux_scale()
            )
        else:
            hidden, wr, w1, w2 = ctx.saved_tensors
            dX, dW1, dW2, dWr = _backward_router(K, dout, ctx.G, hidden, w1, w2, wr, _aux_scale())
            dWs1 = dWs2 = None
        kosmos_gate.after_kernel(backward=True)
        ctx.G = ctx.Gs = None
        return dX, dWr, None, dW1, dW2, None, None, None, None, dWs1, dWs2


class KosmosExperts(MegatronModule):
    """Local experts stored in KOSMOS layout: weight1 [E_loc, 2I, H] (128-row gate|up interleave),
    weight2 [E_loc, H, I]. Initialised by building the SequentialMLP of the same spec and
    converting its weights, so a KOSMOS model starts from the same weights as SequentialMLP."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None):
        super().__init__(config=config)
        assert config.gated_linear_unit and not config.add_bias_linear
        seq = SequentialMLP(num_local_experts, config, submodules, pg_collection=pg_collection)
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
        # Set by Megatron's set_is_first_microbatch at each step (FP8 runs): refresh mxfp8_weights.
        self.is_first_microbatch = True
        self._mxw = None

    def mxfp8_weights(self):
        """(W1, W1_scale, W2, W2_scale, W1c, W1c_scale, W2c, W2c_scale): the weights row-wise
        (forward) and column-wise (dgrad), quantized on the first microbatch of each step and cached
        for the step (more than one microbatch per step)."""
        if self._mxw is None or self.is_first_microbatch:
            w1, w2 = self.weight1.detach(), self.weight2.detach()
            self._mxw = (
                *_quantize_mxfp8(w1, False),
                *_quantize_mxfp8(w2, False),
                *_quantize_mxfp8(w1, True),
                *_quantize_mxfp8(w2, True),
            )
        self.is_first_microbatch = False
        return self._mxw

    def mxfp8_row(self, keep):
        """(W1, W1_scale, W2, W2_scale). One microbatch per step: quantized once per step and held
        only while used; keep (a forward with grad disabled, the checkpointed first pass) holds it
        for the recompute's forward, which takes it. Otherwise from mxfp8_weights."""
        if get_num_microbatches() > 1:
            return self.mxfp8_weights()[:4]
        if self.is_first_microbatch:
            self._mxw = None
            self.is_first_microbatch = False
        row = self._mxw
        if row is None:
            w1, w2 = self.weight1.detach(), self.weight2.detach()
            row = (*_quantize_mxfp8(w1, False), *_quantize_mxfp8(w2, False))
        self._mxw = row if keep else None
        return row

    def mxfp8_col(self):
        """(W1c, W1c_scale, W2c, W2c_scale): one microbatch per step, quantized for the step's one
        backward and freed after it; otherwise from mxfp8_weights."""
        if get_num_microbatches() > 1:
            return self.mxfp8_weights()[4:]
        w1, w2 = self.weight1.detach(), self.weight2.detach()
        return (*_quantize_mxfp8(w1, True), *_quantize_mxfp8(w2, True))

    def backward_dw(self):
        pass


class KosmosMoELayer(MoELayer):
    """MoELayer with the routed part (dispatch, experts, combine) replaced by KOSMOS."""

    def __init__(
        self, config, submodules=None, layer_number=None, pg_collection=None, is_mtp_layer=False
    ):
        super().__init__(config, submodules, layer_number, pg_collection, is_mtp_layer)
        kosmos_gate.enable()
        assert config.tensor_model_parallel_size == 1 and config.expert_tensor_parallel_size == 1
        assert self.ep_group.size() == 8, "KOSMOS MoE is EP=8"
        assert config.moe_router_topk in (6, 8, 10)
        assert config.moe_latent_size is None and not config.moe_shared_expert_overlap
        self._sh_active = False
        self.kctx = KosmosContext(config, self.ep_group, self.num_local_experts)
        self._configure(layer_number)

    def set_layer_number(self, layer_number):
        super().set_layer_number(layer_number)
        self._configure(layer_number)

    def _configure(self, layer_number):
        """MXFP8, fused shared expert and fused router, before the first forward: the layer number
        (set by TransformerLayer after construction) decides Megatron's bf16 first/last layers."""
        assert self.kctx.ctx is None, "KOSMOS layer reconfigured after its first forward"
        self._mxfp8 = self._mxfp8_supported(layer_number)
        self._fused_shared = SHARED and not self._mxfp8 and self._shared_supported()
        if self._fused_shared and layer_number is not None:
            print(f"[KOSMOS] layer {layer_number}: fused shared expert", flush=True)
        self._fused_router = ROUTER and not self._mxfp8 and self._router_supported()
        self.kctx.shared, self.kctx.mxfp8 = self._fused_shared, self._mxfp8
        self.kctx.layer_number = layer_number

    def _mxfp8_supported(self, layer_number):
        """MXFP8 experts: an mxfp8-recipe run, a layer Megatron runs in FP8, H a multiple of 1024
        and a library with the MXFP8 entry points."""
        c = self.config
        if not MXFP8 or c.fp8 is None or c.fp8_recipe != "mxfp8":
            return False
        if layer_number is not None and is_first_last_bf16_layer(c, layer_number - 1):
            return False
        why = None
        if c.hidden_size % MXFP8_H_MULTIPLE:
            why = f"hidden size not a multiple of {MXFP8_H_MULTIPLE}"
        elif not hasattr(_kosmos(), "quantize_mxfp8"):
            why = "kosmos module without the MXFP8 entry points"
        if why is not None:
            if layer_number is not None:
                print(f"[KOSMOS] layer {layer_number}: bf16 experts: {why}", flush=True)
            return False
        if layer_number is not None and dist.get_rank() == 0:
            print(f"[KOSMOS] layer {layer_number}: MXFP8 experts", flush=True)
        return True

    def _shared_supported(self):
        """The shared expert fits the KOSMOS launches: one ungated SwiGLU MLP of the experts' intermediate
        size, bf16 linears without bias or fused norm, and a library with the shared entry points."""
        c, sh = self.config, getattr(self, "shared_experts", None)
        if not getattr(self, "use_shared_expert", False) or sh is None:
            return False
        fc1, fc2 = getattr(sh, "linear_fc1", None), getattr(sh, "linear_fc2", None)
        w1, w2 = getattr(fc1, "weight", None), getattr(fc2, "weight", None)
        H, I = c.hidden_size, c.moe_ffn_hidden_size
        why = None
        if c.moe_shared_expert_gate or getattr(sh, "use_shared_expert_gate", False):
            why = "shared expert gate"
        elif not c.gated_linear_unit or c.activation_func is not torch.nn.functional.silu:
            why = "not SwiGLU"
        elif c.add_bias_linear or any(getattr(m, "bias", None) is not None and m.bias.numel() for m in (fc1, fc2)):
            why = "bias"
        elif w1 is None or w2 is None or tuple(w1.shape) != (2 * I, H) or tuple(w2.shape) != (H, I):
            why = "shared intermediate size != moe_ffn_hidden_size"
        elif w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            why = "not bf16"
        elif hasattr(fc1, "layer_norm_weight") or c.fp8 or getattr(c, "fp4", None):
            why = "fused norm or low precision"
        elif not hasattr(_kosmos().MoeContext, "forward_router_shared"):
            why = "kosmos module without the shared entry points"
        if why is not None:
            if self.layer_number in (None, 1):
                print(f"[KOSMOS] shared expert stays in Megatron: {why}", flush=True)
            return False
        return True

    def shared_experts_compute(self, hidden_states):
        # Fused: the shared expert runs in the KOSMOS launches (routed_experts_compute); no output here.
        T = hidden_states.numel() // hidden_states.shape[-1]
        self._sh_active = self._fused_shared and T % SHARED_T_MULTIPLE == 0
        if self._sh_active:
            return None
        return super().shared_experts_compute(hidden_states)

    def _shared_weights(self):
        if not self._sh_active:
            return None, None
        sh = self.shared_experts
        return sh.linear_fc1.weight, sh.linear_fc2.weight

    def _router_supported(self):
        c, r = self.config, self.router
        return (
            c.moe_router_score_function in ("sigmoid", "softmax")
            and not (c.moe_router_score_function == "softmax" and c.moe_router_pre_softmax)
            and not r.get_aux_loss_coeff("global_aux_loss")
            and not c.moe_z_loss_coeff
            and not c.moe_input_jitter_eps
            and not (r.get_aux_loss_coeff("seq_aux_loss") and r.get_aux_loss_coeff("aux_loss"))
            and getattr(c, "moe_router_force_biased", None) is None
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
            *self._shared_weights(),
        )
        record_routing_stats(r.layer_number, tpe)
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
        if self._mxfp8:
            out = KosmosMXFP8MoEFunction.apply(
                hidden_states,
                probs,
                routing_map,
                self.experts.weight1,
                self.experts.weight2,
                self.kctx,
                torch.is_grad_enabled(),
                self.experts,
            )
            return out, None
        out = KosmosMoEFunction.apply(
            hidden_states,
            probs,
            routing_map,
            self.experts.weight1,
            self.experts.weight2,
            self.kctx,
            torch.is_grad_enabled(),
            *self._shared_weights(),
        )
        return out, None

    def combine(self, output):
        return output

    def postprocess(self, output, shared_expert_output):
        output = output.view(self._in_shape)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output


def _builder_parts(spec):
    """(class, submodules, params) of a ModuleSpec, or of a bare module class (build_module
    accepts both)."""
    if isinstance(spec, ModuleSpec):
        return spec.module, spec.submodules, dict(spec.params)
    assert isinstance(spec, type), f"unsupported module spec {spec!r}"
    return spec, None, {}


def replace_moe_experts(moe_spec, layer_cls, experts_cls, flag, experts_submodules=None):
    """MoELayer ModuleSpec whose experts are SequentialMLP -> layer_cls ModuleSpec with
    experts_cls experts (built from experts_submodules, default the SequentialMLP's
    MLPSubmodules). layer_cls takes MoELayer's constructor arguments (TransformerLayer passes
    pg_collection and is_mtp_layer); experts_cls takes SequentialMLP's."""
    _, sub, params = _builder_parts(moe_spec)
    e_cls, e_sub, e_params = _builder_parts(sub.experts)
    assert (
        e_cls is SequentialMLP
    ), f"{flag} needs the SequentialMLP expert spec (do not pass --moe-grouped-gemm)"
    experts = ModuleSpec(
        module=experts_cls,
        params=e_params,
        submodules=e_sub if experts_submodules is None else experts_submodules,
    )
    return ModuleSpec(
        module=layer_cls,
        params=params,
        submodules=dataclasses.replace(sub, experts=experts),
        metainfo=dict(getattr(moe_spec, "metainfo", {}) or {}),
    )


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
