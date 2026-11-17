# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE layer whose dispatch + experts + combine run as KOSMOS single-launch kernels (EP=8, bf16).

The routed part (dispatch, GEMM1 + SwiGLU, GEMM2, combine, weighted sum) is one forward launch,
which takes the routing in device memory and keeps what its backward needs in the workspace's
saved-state pool, and its backward is one launch, which returns the bf16 input gradient and the
fp32 probs gradient. The backward may read the forward's input again, so it is kept, unchanged,
until then.

Experts: bias-free SwiGLU, or DeepSeek-V4's clamped SwiGLU (activation_func_clamp_value=10:
silu(min(g, 10)) * clamp(u, -10, 10), routed and shared experts; top-k 6, 8 or 10). Top-k 1, 2, 4
(SwiGLU) run the library's small-k kernels; with top-1, moe_apply_probs_on_input (Llama 4) scales
each token by its weight at the dispatch and combines unweighted (Megatron's router; the backward
also reads the forward's input, kept in the autograd context).

Routing: for the softmax (top-k on the logits, or with moe_router_pre_softmax a softmax over all
experts, then top-k), sigmoid and sqrtsoftplus (expert bias, group-limited; one group, or
group_topk = groups, is plain top-k) score functions with aux_loss or seq_aux_loss, the fused router
runs inside the same two launches, and Megatron's router module only holds the router weight and the
expert bias. The aux-loss gradient is scaled by MoEAuxLossAutoScaler's scale, which Megatron rewrites
every microbatch: the backward launch reads it on the device (no host sync). Other router configs (and a
kosmos module without that score function), per-token loss, MTP layers, context parallelism, an
expert capacity factor, a token count that is not a multiple of 256, and calls with a padding mask,
keep Megatron's router; its routing is passed to the kernels as top-k ids and weights, so any router
Megatron implements works.

Shared expert (KOSMOS_SHARED=1): when the layer's shared expert is one ungated SwiGLU MLP of the
experts' intermediate size (with or without moe_shared_expert_gate's sigmoid gate), it runs inside
the same two launches (fc1 + SwiGLU + fc2 alongside the routed experts, its output added at the
combine; its dgrad/wgrad in the backward launch, its dX added to the input gradient). The gate
s_t = sigmoid(x_t w_sg) is computed here in torch and scales the shared output at the combine; the
kernel returns ds_t = <dout_t, shared_t> and autograd takes it through the sigmoid into w_sg and x.
Megatron's SharedExpertMLP keeps the parameters; its forward is skipped. With gradient-accumulation
fusion the shared weight gradients are added into main_grad (fp32 or bf16) by the kernel
(grad_added_to_main_grad, dummy grads, as TE does); otherwise they are returned as bf16 gradients.
Unsupported configs (a gate with top-k < 6, FP8 recipes other than mxfp8, FP8 primary weights),
and a token count that is not a multiple of 512, keep Megatron's shared expert.

MXFP8 (KOSMOS_MXFP8=1, --fp8-recipe mxfp8): on the layers Megatron runs in FP8, the four expert
GEMMs run in MXFP8 (TE recipe, no requantization): the forward quantizes X at the source and
dispatches it in MXFP8, the backward dispatches dY in bf16 and quantizes it at the experts. The
weights are quantized both ways once per step: with one microbatch per step, the row-wise copy in
the forward (kept from a checkpointed first pass for its recompute) and the column-wise copy in
the backward, each freed after its use; with more, both on the first microbatch and cached for
the step (as TE's weight cache);
the layer's inputs, outputs and gradients stay bf16 (probs gradient fp32). MXFP8 layers run what
bf16 layers run, through the library's MXFP8 training calls: top-k 1, 2, 4, probs on input (the
library quantizes bf16(w x)), the fused router (bf16 inside the launch), and the fused shared
expert with its gate, whose fc1 and fc2 run in MXFP8 too (weights quantized with the experts').
They need a hidden size that is a multiple of 1024 and a token count that is a multiple of 256;
otherwise the layer runs bf16 experts ("[KOSMOS] layer N: bf16 experts: ...").

Each layer's decisions (precision, router, shared expert, and every feature left to Megatron with
its reason) are printed once, at rank 0, at the layer's first forward.

Every MoE layer has its own context (its tables); all layers of one shape share one workspace (the
per-call buffers and the saved-state pool), so any recompute pattern works. kosmos_gate.py
keeps Megatron DDP communication out of the KOSMOS kernels.

Capacity: capacity_rows per rank (experts padded to 256 rows) sizes the per-call buffers. A call
that routes more rows to one rank runs in rounds inside its two launches. Pool: a bf16 forward
keeps its whole block (Xp, G, A) while the pool has room, else Xp only (its backward then
recomputes GEMM1); an MXFP8 forward keeps its whole block (G, the column-wise A, and with I < 1024
the column-wise Xp) or nothing; a call with no room for its Xp (MXFP8: its block), or over
capacity, runs in rounds inside its launches and keeps nothing (its backward re-sends X and
recomputes GEMM1).
Capacity (automatic unless KOSMOS_CAPACITY_FACTOR fixes it): each workspace (all its
layers' calls share its buffers) starts at 1.5 T k + 256 E_loc rows. At a queried step boundary
(below) the rows any call since the previous query needed (a running max each forward keeps on the
device, read with the status words) are reduced over layers and ranks to the peak; the capacity
grows to peak (1 + KOSMOS_CAPACITY_HEADROOM) when the peak exceeds capacity / (1 + headroom), and
shrinks to it after 5 queries in a row with the peak below capacity / (1 + 2 headroom) while a
storing pool holds less than all its calls (kosmos MoeWorkspace.set_capacity, collective over the
EP group). A growth is limited to the memory the pools leave (as below), the ranks' least, and an
allocation failure keeps the old capacity. With CUDA graphs the capacity adapts at the first two
boundaries only.
Step boundary: the optimizer step post-hook flags it, and the next MoE forward with no stored
block alive acts on it (rank-identical). The first two boundaries, those within 5 of a capacity or
pool change and those at Megatron's log interval are queried: one device sync, the capacity
resizes, then the pool resizes (their reservations depend on the capacity), and one line on rank
0: "[KOSMOS] iteration(s) N[-M]: rounds in R/L layers (C over capacity)" (the layers with a call
in rounds since the previous query, and of those the ones over capacity; the others the pool
forced into rounds, which never grows the capacity) followed by each capacity (old -> new rows,
reason, peak) and pool (old -> new GiB) change. The others only restart the step's memory
statistics (no host sync).
Pool size (automatic unless KOSMOS_POOL_GB or KOSMOS_POOL_LAYERS fixes it): each workspace
starts at one call's block (the first step's storing calls run in rounds, a one-time warm-up). At
the top of the first MoE forward after each of the first two optimizer steps, with no stored block
alive, the pools are re-allocated (kosmos MoeWorkspace.pool_resize, collective over the EP group)
to what the rank can spare: free memory (after the old pools and PyTorch's cache are released) less
what PyTorch will still reserve (the step's peak reserved plus its growth in allocated, less what
it holds now) less KOSMOS_POOL_HEADROOM_GB, split over the workspaces whose calls store in
proportion to, and at most, what keeps all their stored calls; the pools are re-allocated so again
after a capacity change. CUDA graphs capture the pool's and the capacity's addresses: with TE
graphs --cuda-graph-warmup-steps is raised to 3 so that capture follows both resizes; other graphs
with fewer warmup steps get fixed sizes (capacity factor 4.1, a pool of every local layer's block
and one more), with a warning.
At exit every layer's status is read once (one device sync) and each rank prints
"[KOSMOS] plan status rank R: ..." (OVERFLOW: bad routing, or a KOSMOS error).

Environment:
  KOSMOS_PYTHON             directory of the kosmos module (KOSMOS `make python`: build/python);
                            unset: the module is found on PYTHONPATH
  KOSMOS_WGCAP              launch width (workgroups), default 256
  KOSMOS_CAPACITY_FACTOR    a fixed capacity, per-call rows per rank = factor * T * k + 256 E_loc
                            (each local expert's rows padded to 256; a larger call runs in rounds
                            inside its launches, correct but slower; the pool's blocks scale with
                            it); unset, empty or "auto": automatic (above)
  KOSMOS_CAPACITY_HEADROOM  automatic capacity: the rows kept above the peak, default 0.125
  KOSMOS_POOL_GB            a fixed saved-state pool of this size (no resize); a smaller pool
                            keeps (bf16) Xp only for some layers (their backward recomputes GEMM1),
                            and a call with no room for its Xp (MXFP8: its block) runs in rounds
  KOSMOS_POOL_LAYERS        a fixed pool for this many whole blocks (no resize)
  KOSMOS_POOL_HEADROOM_GB   automatic pool size: memory left free, default 8
  KOSMOS_ROUTER             1 (default): the fused router where supported; 0: Megatron's router
  KOSMOS_DDP_GATE           1 (default): the DDP gate (kosmos_gate.py); 0: off
  KOSMOS_SHARED             1 (default): the shared expert in the KOSMOS launches where supported;
                            0: in Megatron
  KOSMOS_MXFP8              1 (default): MXFP8 experts on the FP8 layers of an mxfp8-recipe run;
                            0: bf16 experts
"""

import atexit
import dataclasses
import os
import sys
from functools import partial

import torch
import torch.distributed as dist
from torch.optim.optimizer import register_optimizer_step_post_hook

from megatron.core.fp8_utils import is_first_last_bf16_layer, is_float8tensor
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
MXFP8_T_MULTIPLE = 256
ROUTER_T_MULTIPLE = 256
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
    """Device address of a contiguous tensor; None -> 0 (null); an int (kosmos.MOE_SAVE) as is."""
    if t is None:
        return 0
    if isinstance(t, int):
        return t
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


_CF_START = 1.5
_CF_FIXED = 4.1
_SHRINK_AFTER = 5
_ROWS = 256


def _cf():
    """KOSMOS_CAPACITY_FACTOR, or None (unset, empty or "auto": automatic capacity)."""
    v = os.environ.get("KOSMOS_CAPACITY_FACTOR", "auto")
    return None if v in ("", "auto") else float(v)


def _cap_headroom():
    return float(os.environ.get("KOSMOS_CAPACITY_HEADROOM", "0.125"))


def _pool_fixed(config, capacity_rows, mxfp8):
    """The fixed pool size of KOSMOS_POOL_GB, or of KOSMOS_POOL_LAYERS whole blocks of capacity_rows
    rows (KOSMOS moe_pool_bytes; KOSMOS takes the ranks' least); None: automatic."""
    if os.environ.get("KOSMOS_POOL_GB") is not None:
        return int(float(os.environ["KOSMOS_POOL_GB"]) * 2**30)
    n = os.environ.get("KOSMOS_POOL_LAYERS")
    if n is None:
        return None
    return _pool_bytes(config, capacity_rows, int(n), mxfp8)


def _pool_bytes(config, capacity_rows, n, mxfp8):
    """Pool bytes for n whole blocks of capacity_rows rows and one more call (moe_pool_bytes; MXFP8:
    G, the column-wise A and with I < 1024 the column-wise Xp, with scales)."""
    H, I = config.hidden_size, config.moe_ffn_hidden_size
    if mxfp8:
        row = I * 165 // 32 + (H * 33 // 32 if I < 1024 else 0)
        return (n + 1) * ((capacity_rows + 512) * row + 2 * H)
    return (n + 1) * (capacity_rows + 512) * (2 * H + 6 * I)


def _local_layers(config):
    """The transformer layers this pipeline stage holds (an upper bound on its MoE layers)."""
    pp = getattr(config, "pipeline_model_parallel_size", 1) or 1
    return -(-config.num_layers // pp)


class _Pool:
    """A training workspace's saved-state pool: its sizes so far (bytes per rank), its contexts,
    and its stored blocks still awaiting their backward (now and at most); with an automatic
    capacity (auto_cap) also the capacities so far (rows), the boundaries in a row below the shrink
    threshold and the rows whose allocation failed."""

    def __init__(self, ws, name, fixed, group, auto_cap, rows):
        self.ws, self.name, self.fixed, self.group, self.auto_cap = ws, name, fixed, group, auto_cap
        self.sizes = []
        self.nctx = self.live = self.peak = 0
        self.caps = [-(-rows // _ROWS) * _ROWS]
        self.low = 0
        self.cap_fail = None

    def stored(self):
        self.live += 1
        self.peak = max(self.peak, self.live)

    def released(self):
        self.live -= 1

    def size(self):
        return self.ws.pool_info()["bytes"]

    def keep_all(self):
        """Bytes that keep every stored call in full: the peak's blocks, the next call's block and the
        Xp KOSMOS reserves (bf16) for each context without a stored block (kosmos_moe_pool_info)."""
        i = self.ws.pool_info()
        return (self.peak + 1) * i["head_bytes"] + max(0, self.nctx - self.peak) * i["reserve_bytes"]


def _new_workspace(config, ep_group, num_local_experts, T, router, shared, mxfp8):
    """-> (workspace, its _Pool)."""
    kosmos = _kosmos()
    k = config.moe_router_topk
    graphs_fixed = _graphs_fixed(config)
    cf = _cf()
    if cf is None and graphs_fixed:
        cf = _CF_FIXED
    auto_cap = cf is None
    # Each local expert's rows are padded to 256 in the arena: up to 255 more rows per expert.
    capacity_rows = int((cf if cf is not None else _CF_START) * T * k) + num_local_experts * 256
    fixed = _pool_fixed(config, capacity_rows, mxfp8)
    if fixed is None and graphs_fixed:
        fixed = _pool_bytes(config, capacity_rows, _local_layers(config), mxfp8)
    _auto_setup(fixed is None, auto_cap)
    rcfg = None
    if router is not None:
        # router: (score, groups, group_topk, aux_groups, aux_coeff, scale, force)
        rcfg = kosmos.MoeRouterConfig(*router)
    act = _experts_act(config)
    extra = {} if act is None else {"act": act}
    if config.moe_apply_probs_on_input:
        extra["probs_in"] = True
    ws = kosmos.MoeWorkspace(
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
        pool_bytes=fixed or 0,
        **extra,
    )
    name = f"T={T} H={config.hidden_size} I={config.moe_ffn_hidden_size}" + (" router" if router else "")
    name += " shared" if shared else ""
    name += " mxfp8" if mxfp8 else ""
    return ws, _Pool(ws, name, fixed is not None, ep_group, auto_cap, capacity_rows)


def _new_context(ws):
    return _kosmos().MoeContext(ws)


def _new_g(need_g):
    """kosmos.MOE_SAVE (the forward saves into the workspace's pool), or None."""
    return _kosmos().MOE_SAVE if need_g else None


def _topk(probs, routing_map, k):
    """Megatron's routing -> (topk_idx int32, topk_w fp32), contiguous."""
    assert (
        probs.dtype == torch.float32
    ), "KOSMOS writes the probs gradient in fp32 (--moe-router-dtype fp32)"
    # A stable sort, not topk: topk's order among the tied ones may change run to run, and with it
    # the fold order.
    idx = torch.argsort(routing_map.to(torch.int8), dim=1, descending=True, stable=True)[:, :k]
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


# MXFP8 (KosmosExperts.mxfp8_row / mxfp8_col): mxw = (W1, W1_scale, W2, W2_scale) row-wise, mxc the
# same column-wise, each followed by (Ws1, Ws1_scale, Ws2, Ws2_scale) with the fused shared expert;
# None: bf16. The MXFP8 forwards always write Gs, and the MXFP8 backwards read the forward's X.


def _new_dprobs(dout, w1):
    return torch.empty(
        (dout.shape[0], _kosmos().EP * w1.shape[0]), dtype=torch.float32, device=dout.device
    )


def _forward(K, x, w1, w2, idx, w, need_g, mxw=None):
    """-> (out, G or None)."""
    out = torch.empty_like(x)
    G = _new_g(need_g)
    if mxw is None:
        K.forward(*map(_ptr, (x, w1, w2, idx, w, out, G)), _stream())
    else:
        K.forward_mxfp8_train(*map(_ptr, (x, *mxw[:4], idx, w, out, G)), _stream())
    return out, G


def _backward(K, dout, G, w1, w2, x=None, mxc=None):
    """-> (dX, dprobs, dW1, dW2)."""
    dX, dprobs = torch.empty_like(dout), _new_dprobs(dout, w1)
    dW1, dW2 = torch.empty_like(w1), torch.empty_like(w2)
    if mxc is None:
        K.backward(*map(_ptr, (dout, G, w1, w2, dX, dprobs, dW1, dW2)), _stream())
    else:
        K.backward_mxfp8(*map(_ptr, (dout, G, x, *mxc[:4], dX, dprobs, dW1, dW2)), _stream())
    return dX, dprobs, dW1, dW2


def _forward_router(K, x, wr, bias, w1, w2, seed, need_g, mxw=None):
    """-> (out, G or None, aux loss, tokens per expert)."""
    out = torch.empty_like(x)
    G = _new_g(need_g)
    aux = torch.empty(1, dtype=torch.float32, device=x.device)
    tpe = torch.empty(wr.shape[0], dtype=torch.int32, device=x.device)
    outs = (out, G, aux, tpe, seed)
    if mxw is None:
        K.forward_router(*map(_ptr, (x, wr, bias, w1, w2, *outs)), _stream())
    else:
        K.forward_router_mxfp8_train(*map(_ptr, (x, wr, bias, *mxw[:4], *outs)), _stream())
    return out, G, aux, tpe


def _backward_router(K, dout, G, x, w1, w2, wr, aux_scale, mxc=None):
    """-> (dX, dW1, dW2, dWr)."""
    dX = torch.empty_like(dout)
    dW1, dW2, dWr = torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(wr)
    outs = map(_ptr, (dX, dW1, dW2, dWr))
    if mxc is None:
        ins = map(_ptr, (dout, G, x, w1, w2, wr))
        K.backward_router(*ins, _ptr(aux_scale), *outs, _stream())
    else:
        ins = map(_ptr, (dout, G, x, *mxc[:4], wr))
        K.backward_router_mxfp8(*ins, _ptr(aux_scale), *outs, _stream())
    return dX, dW1, dW2, dWr


# Shared expert: ws1 [2I, H] ([gate; up], Megatron's linear_fc1.weight), ws2 [H, I]; Gs [T, 2I] its
# stored pre-activations.


def _new_gs(x, ws1, need_g):
    return torch.empty((x.shape[0], ws1.shape[0]), dtype=x.dtype, device=x.device) if need_g else None


def _sgate_kw(sgate):
    """The gate argument of the shared calls: sgate [T] fp32 (None: ungated, no argument)."""
    return {} if sgate is None else {"sgate": _ptr(sgate)}


def _dsgate(sgate):
    """-> (dsgate [T] fp32 or None, its keyword arguments)."""
    if sgate is None:
        return None, {}
    ds = torch.empty_like(sgate)
    return ds, {"sgate": _ptr(sgate), "dsgate": _ptr(ds)}


def _forward_shared(K, x, w1, w2, ws1, ws2, idx, w, need_g, sgate, mxw=None):
    """-> (out, G or None, Gs or None)."""
    out = torch.empty_like(x)
    G, Gs = _new_g(need_g), _new_gs(x, ws1, need_g or mxw is not None)
    if mxw is None:
        ins = map(_ptr, (x, w1, w2, ws1, ws2, idx, w, out, G, Gs))
        K.forward_shared(*ins, _stream(), **_sgate_kw(sgate))
    else:
        assert len(mxw) == 8, "MXFP8 shared expert without its quantized weights"
        ins = map(_ptr, (x, *mxw, idx, w, out, G, Gs))
        K.forward_shared_mxfp8_train(*ins, _stream(), **_sgate_kw(sgate))
    return out, G, Gs


def _forward_router_shared(K, x, wr, bias, w1, w2, ws1, ws2, seed, need_g, sgate, mxw=None):
    """-> (out, G or None, Gs or None, aux loss, tokens per expert)."""
    out = torch.empty_like(x)
    G, Gs = _new_g(need_g), _new_gs(x, ws1, need_g or mxw is not None)
    aux = torch.empty(1, dtype=torch.float32, device=x.device)
    tpe = torch.empty(wr.shape[0], dtype=torch.int32, device=x.device)
    outs = (out, G, Gs, aux, tpe, seed)
    if mxw is None:
        ins = map(_ptr, (x, wr, bias, w1, w2, ws1, ws2, *outs))
        K.forward_router_shared(*ins, _stream(), **_sgate_kw(sgate))
    else:
        assert len(mxw) == 8, "MXFP8 shared expert without its quantized weights"
        ins = map(_ptr, (x, wr, bias, *mxw, *outs))
        K.forward_router_shared_mxfp8_train(*ins, _stream(), **_sgate_kw(sgate))
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


def _backward_shared(K, config, dout, G, Gs, x, w1, w2, ws1, ws2, sgate, mxc=None):
    """-> (dX, dprobs, dW1, dW2, dWs1 grad, dWs2 grad, dsgate or None)."""
    dX, dprobs = torch.empty_like(dout), _new_dprobs(dout, w1)
    dW1, dW2 = torch.empty_like(w1), torch.empty_like(w2)
    dWs1, dWs2, acc = _shared_dw(config, ws1, ws2)
    dsgate, kw = _dsgate(sgate)
    outs = (dX, dprobs, dW1, dW2, dWs1, dWs2)
    if mxc is None:
        ins = map(_ptr, (dout, G, Gs, x, w1, w2, ws1, ws2, *outs))
        K.backward_shared(*ins, acc, _stream(), **kw)
    else:
        assert len(mxc) == 8, "MXFP8 shared expert without its quantized weights"
        ins = map(_ptr, (dout, G, Gs, x, *mxc, *outs))
        K.backward_shared_mxfp8(*ins, acc, _stream(), **kw)
    return (dX, dprobs, dW1, dW2) + _shared_grads(ws1, ws2, dWs1, dWs2, acc) + (dsgate,)


def _backward_router_shared(
    K, config, dout, G, Gs, x, w1, w2, ws1, ws2, wr, aux_scale, sgate, mxc=None
):
    """-> (dX, dW1, dW2, dWr, dWs1 grad, dWs2 grad, dsgate or None)."""
    dX = torch.empty_like(dout)
    dW1, dW2, dWr = torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(wr)
    dWs1, dWs2, acc = _shared_dw(config, ws1, ws2)
    dsgate, kw = _dsgate(sgate)
    outs = map(_ptr, (dX, dW1, dW2, dWr, dWs1, dWs2))
    if mxc is None:
        ins = map(_ptr, (dout, G, Gs, x, w1, w2, ws1, ws2, wr))
        K.backward_router_shared(*ins, _ptr(aux_scale), *outs, acc, _stream(), **kw)
    else:
        assert len(mxc) == 8, "MXFP8 shared expert without its quantized weights"
        ins = map(_ptr, (dout, G, Gs, x, *mxc, wr))
        K.backward_router_shared_mxfp8(*ins, _ptr(aux_scale), *outs, acc, _stream(), **kw)
    return (dX, dW1, dW2, dWr) + _shared_grads(ws1, ws2, dWs1, dWs2, acc) + (dsgate,)


# ---------------------------------------------------------------------------------------------

_WS = {}


def _workspace(config, ep_group, num_local_experts, T, router=None, shared=False, mxfp8=False):
    """(The workspace shared by every MoE layer of one shape (and fused-router, shared-expert and
    MXFP8 config), its _Pool)."""
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
        config.activation_func_clamp_value,
    )
    if key not in _WS:
        _WS[key] = _new_workspace(config, ep_group, num_local_experts, T, router, shared, mxfp8)
    return _WS[key]


# Step boundaries: the optimizer step post-hook only flags one; the next MoE forward with no stored
# block alive acts on it (_step_boundary).
_AUTO = {
    "on": False,
    "step": False,
    "resizes": 0,
    "refusals": 0,
    "alloc0": None,
    "boundaries": 0,
    "graphs": False,
    "iter": 0,
    "stable": 0,
    "queried": 0,
}
_MAX_RESIZES = 2
_MAX_REFUSALS = 5


def _rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def _iteration():
    """Megatron's number of the iteration whose optimizer step just ran (as its log prints it)."""
    try:
        from megatron.training.global_vars import get_args

        return get_args().curr_iteration + 1
    except Exception:  # pylint: disable=broad-except
        return _AUTO["iter"] + 1


def _on_optimizer_step(optimizer, args, kwargs):
    _AUTO["step"] = True
    _AUTO["iter"] = _iteration()


def _graphs_fixed(config):
    """CUDA graphs keep the addresses captured, so the pool and capacity resizes must come before
    the capture (the first _MAX_RESIZES boundaries): with TE graphs the warmup steps are raised to
    _MAX_RESIZES + 1 (captured after them); otherwise (local or full-iteration graphs, fixed when
    the model was built) True: fixed sizes. Decided once, the same on every rank."""
    if "graphs_fixed" in _AUTO:
        return _AUTO["graphs_fixed"]
    impl = getattr(config, "cuda_graph_impl", "none")
    graphs = impl != "none" or getattr(config, "enable_cuda_graph", False)
    fixed = False
    if graphs and config.cuda_graph_warmup_steps < _MAX_RESIZES + 1:
        args = None
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
        except Exception:  # pylint: disable=broad-except
            pass
        scope = str(getattr(config, "cuda_graph_scope", "") or "")
        if impl == "transformer_engine" and args is not None and "full_iteration" not in scope:
            old = args.cuda_graph_warmup_steps
            args.cuda_graph_warmup_steps = config.cuda_graph_warmup_steps = _MAX_RESIZES + 1
            if _rank0():
                print(
                    f"[KOSMOS] CUDA graphs: --cuda-graph-warmup-steps raised {old} -> "
                    f"{_MAX_RESIZES + 1} (capture after the pool and capacity resizes)",
                    flush=True,
                )
        else:
            fixed = True
            if _rank0():
                print(
                    f"[KOSMOS] WARNING: CUDA graphs ({impl}) with --cuda-graph-warmup-steps "
                    f"{config.cuda_graph_warmup_steps} < {_MAX_RESIZES + 1} cannot be raised here: "
                    f"fixed sizes (KOSMOS_CAPACITY_FACTOR {_CF_FIXED}, pool of every local layer's "
                    "block and one more) unless set explicitly",
                    flush=True,
                )
    _AUTO["graphs"] = graphs
    _AUTO["graphs_fixed"] = fixed
    return fixed


def _auto_setup(pool, capacity):
    """Every workspace: register the step hook once (the boundary's status line)."""
    ws_type = _kosmos().MoeWorkspace
    assert hasattr(ws_type, "pool_resize") and hasattr(ws_type, "set_capacity"), (
        "KOSMOS module without MoeWorkspace.pool_resize / set_capacity: rebuild it (make python)"
    )
    if not _AUTO["on"]:
        register_optimizer_step_post_hook(_on_optimizer_step)
        _AUTO["on"] = True
    if pool and not _AUTO.get("pool_said") and _rank0():
        _AUTO["pool_said"] = True
        print(
            "[KOSMOS] saved-state pool: automatic (one call's block until the first optimizer step)",
            flush=True,
        )
    if capacity and not _AUTO.get("cap_said") and _rank0():
        _AUTO["cap_said"] = True
        print(
            f"[KOSMOS] capacity: automatic from {_CF_START} T k rows (headroom {_cap_headroom()})",
            flush=True,
        )


def _auto_pools():
    return [p for _, p in _WS.values() if not p.fixed]


def _iteration_start():
    torch.cuda.reset_peak_memory_stats()
    _AUTO["alloc0"] = torch.cuda.memory_allocated()


def _pool_refused(why):
    """A resize refused on every rank (nothing changed for that pool): retried at the next step
    boundary; an error after _MAX_REFUSALS while the pools are still at one block."""
    _AUTO["refusals"] += 1
    if _rank0():
        print(f"[KOSMOS] pool resize refused ({why}), refusal {_AUTO['refusals']}", flush=True)
    if _AUTO["resizes"] == 0 and _AUTO["refusals"] >= _MAX_REFUSALS:
        raise RuntimeError(
            f"KOSMOS: the saved-state pool resize was refused {_MAX_REFUSALS} times; the pools are "
            "still one call's block (set KOSMOS_POOL_GB to fix the size)"
        )


def _resize(pool, nbytes):
    """True if done; False if refused (a stored block alive or a stream capture)."""
    if torch.cuda.is_current_stream_capturing():
        _pool_refused("inside a CUDA graph capture")
        return False
    r = pool.ws.pool_resize(int(nbytes))
    if r:
        _pool_refused("a stored block awaits its backward" if r == 1 else "inside a stream capture")
    return r == 0


def _tag(p):
    return f" ({p.name})" if len(_WS) > 1 else ""


def _capacity_target(p, q, pressure):
    """(rows, reason) for an automatic capacity at a boundary, or None: grow to peak (1 + h) when
    the peak exceeds capacity / (1 + h); shrink to it after _SHRINK_AFTER boundaries in a row with
    the peak below capacity / (1 + 2 h), and only under memory pressure (a storing pool holds less
    than all its calls). Inputs and state are the same on every rank."""
    peak, cap, h = q["peak"], q["rows"], _cap_headroom()
    if not peak:
        return None
    want = min(-(-int(peak * (1 + h)) // _ROWS) * _ROWS, q["max_rows"])
    if peak * (1 + h) > cap and want > cap:
        p.low = 0
        if p.cap_fail is not None and want >= p.cap_fail:
            return None
        return want, "overflow" if peak > cap else "grow"
    if peak * (1 + 2 * h) < cap:
        p.low += 1
        if p.low >= _SHRINK_AFTER and want < cap and pressure:
            p.low = 0
            return want, "shrink"
        return None
    p.low = 0
    return None


def _agree_min(group, v):
    t = torch.tensor([v], dtype=torch.int64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return int(t.item())


def _set_capacity(p, cur, rows, reason, peak, mem, events):
    """One workspace's capacity cur -> rows; a growth limited to what the memory leaves (mem: the
    step's peak reserved and growth in allocated, the headroom), the ranks' least."""
    if rows > cur:
        free, _ = torch.cuda.mem_get_info()
        room = free - max(0, mem[0] + mem[1] - torch.cuda.memory_reserved()) - mem[2]
        need = p.ws.capacity_bytes(rows) - p.ws.capacity_bytes(cur)
        if need > room:
            rows = cur + max(0, room) // max(1, need // (rows - cur)) // _ROWS * _ROWS
            reason += ", memory-limited"
        rows = _agree_min(p.group, rows)
    if rows == cur:
        events.append(f"capacity{_tag(p)} {cur} rows kept ({reason}, peak {peak})")
        return
    r = p.ws.set_capacity(int(rows))
    if r == 0:
        p.caps.append(rows)
        p.cap_fail = None if rows < cur else p.cap_fail
        events.append(f"capacity{_tag(p)} {cur} -> {rows} rows ({reason}, peak {peak})")
    elif r == 3:
        p.cap_fail = rows
        events.append(
            f"capacity{_tag(p)} {cur} -> {rows} rows FAILED, out of memory ({reason}, peak {peak})"
        )
    else:
        events.append(f"capacity{_tag(p)} {cur} -> {rows} rows refused ({reason}, peak {peak})")


def _resizes(changes, events):
    """The boundary's capacity changes (p, cur, rows, reason, peak), then the automatic pools'
    resize (at the first _MAX_RESIZES boundaries and after a capacity change): to what the rank can
    spare, split over the storing workspaces in proportion to, and at most, what keeps all their
    calls. The step boundary has checked what refuses a resize (a stored block, a capture), so a
    refusal here is a bug: the pools shrunk so far are restored and it raises."""
    auto = _auto_pools()
    first = _AUTO["resizes"] < _MAX_RESIZES
    pools_due = bool(auto) and (first or bool(changes))
    if not pools_due and not changes:
        return
    headroom = int(float(os.environ.get("KOSMOS_POOL_HEADROOM_GB", "8")) * 2**30)
    grown = torch.cuda.memory_allocated() - _AUTO["alloc0"]
    mem = (torch.cuda.max_memory_reserved(), grown, headroom)
    storing = [p for p in auto if p.peak] if pools_due else []
    old = [p.size() for p in storing]
    for p, o in zip(storing, old):
        if not p.sizes:
            p.sizes.append(o)
    done = []
    for p in storing:
        if not _resize(p, 0):
            _restore(done, storing, old)
        done.append(p)
    torch.cuda.empty_cache()
    for c in changes:
        _set_capacity(*c, mem, events)
    if storing:
        free, _ = torch.cuda.mem_get_info()
        need = [p.keep_all() for p in storing]
        torch_more = max(0, mem[0] + mem[1] - torch.cuda.memory_reserved())
        head = sum(p.size() for p in storing)
        budget = max(0, free + head - torch_more - headroom)
        total = sum(need) or 1
        for i, (p, n, o) in enumerate(zip(storing, need, old)):
            if not _resize(p, min(n, budget * n // total)):
                _restore(storing[i:], storing, old)
            p.sizes.append(p.size())
            events.append(
                f"pool{_tag(p)} {o / 2**30:.2f} -> {p.sizes[-1] / 2**30:.2f} GiB (keep-all "
                f"{n / 2**30:.2f}, budget {budget / 2**30:.2f})"
            )
    if pools_due and first:
        _AUTO["resizes"] += 1


def _restore(shrunk, storing, old):
    """A refused pool resize: the pools in shrunk back to their old sizes, then an error."""
    for p in shrunk:
        p.ws.pool_resize(int(old[storing.index(p)]))
    raise RuntimeError(
        "KOSMOS: a pool resize was refused at a step boundary that had checked no stored block is "
        "alive and no capture is running (pools restored to their old sizes)"
    )


def _log_interval():
    try:
        from megatron.training.global_vars import get_args

        return max(1, int(get_args().log_interval))
    except Exception:  # pylint: disable=broad-except
        return 1


def _step_boundary():
    """At the top of an MoE forward after an optimizer step, with no stored block alive (rank-
    identical): the step's memory statistics restart; at the first _MAX_RESIZES boundaries, within
    _SHRINK_AFTER boundaries of a change, and at Megatron's log interval (else no host sync), one
    device sync and one query per workspace (status words and peak rows since the previous query),
    the capacity and pool resizes due, and one line on rank 0."""
    if not _AUTO["on"]:
        return
    if _AUTO["alloc0"] is None:
        _iteration_start()
    if not _AUTO["step"] or torch.cuda.is_current_stream_capturing():
        return
    if any(p.live for _, p in _WS.values()):
        return
    _AUTO["step"] = False
    _AUTO["boundaries"] += 1
    it = _AUTO["iter"]
    busy = _AUTO["boundaries"] <= _MAX_RESIZES or _AUTO["stable"] < _SHRINK_AFTER
    if not busy and it % _log_interval():
        _AUTO["stable"] += 1
        _iteration_start()
        return
    try:
        torch.cuda.synchronize()
        frozen = _AUTO["graphs"] and _AUTO["boundaries"] > _MAX_RESIZES
        pools = [p for _, p in _WS.values() if p.peak]
        pressure = any(p.size() < p.keep_all() for p in pools)
        nl = nr = nc = 0
        changes, events = [], []
        for ws, p in _WS.values():
            q = ws.capacity_query()
            nl, nr = nl + q["nctx"], nr + q["rounds"]
            nc += q["cap_rounds"]
            auto = p.auto_cap and not frozen
            d = _capacity_target(p, q, pressure) if auto else None
            if d is not None:
                changes.append((p, q["rows"], d[0], d[1], q["peak"]))
        n0 = sum(len(p.caps) + len(p.sizes) for _, p in _WS.values())
        _resizes(changes, events)
        n1 = sum(len(p.caps) + len(p.sizes) for _, p in _WS.values())
        _AUTO["stable"] = 0 if n1 != n0 else _AUTO["stable"] + 1
        first = _AUTO["queried"] + 1
        span = f"iteration {it}" if first >= it else f"iterations {first}-{it}"
        _AUTO["queried"] = it
        if _rank0():
            more = "".join(f"; {e}" for e in events)
            line = f"[KOSMOS] {span}: rounds in {nr}/{nl} layers ({nc} over capacity){more}"
            print(line, flush=True)
    finally:
        _iteration_start()


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
    flagged, n, notes = [], 0, [0, 0, 0]
    try:
        for kc in _CONTEXTS:
            st = kc.ctx.plan_status()
            n += 1
            if st & 71:
                flagged.append(f"layer {kc.layer_number} status {st & 71}")
            for i in range(3):
                notes[i] += bool(st & (8 << i))
        rows = max(kc.ctx.info()["arena_rows"] for kc in _CONTEXTS)
    except Exception as e:  # pylint: disable=broad-except
        print(f"[KOSMOS] plan status rank {rank}: QUERY FAILED: {e}", flush=True)
        return
    cf = _cf()
    pools = [p for _, p in _WS.values()]
    autos = "; ".join(
        f"{p.name}: " + " -> ".join(str(r) for r in p.caps) for p in pools if p.auto_cap
    )
    if cf is not None:
        cf = f"KOSMOS_CAPACITY_FACTOR {cf}"
    else:
        cf = f"capacity auto, {autos or 'none'} rows"
    for p in pools:
        if not p.sizes:
            p.sizes.append(p.size())
    pool = "; ".join(
        f"{p.name}: " + " -> ".join(f"{b / 2**30:.2f}" for b in p.sizes) + " GiB" for p in pools if p.sizes
    )
    pool = f", pool ({pool})" if pool else ""
    if any(notes):
        pool += (
            f"; layers with calls in rounds {notes[0]}, keeping Xp only {notes[1]}, "
            f"keeping nothing {notes[2]}"
        )
    if flagged:
        print(
            f"[KOSMOS] plan status rank {rank}: OVERFLOW or bad routing in {len(flagged)} of {n} "
            f"layers ({', '.join(flagged)}; 4 = over capacity (KOSMOS error), 1 = bad expert id, "
            f"2 = duplicate, 64 = a single pass kept nothing (KOSMOS error)); arena {rows} rows "
            f"({cf}){pool}: those calls' "
            f"outputs are undefined",
            flush=True,
        )
    else:
        print(
            f"[KOSMOS] plan status rank {rank}: ok, {n} layers, arena {rows} rows "
            f"({cf}){pool}",
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
        self.pool = None
        self.T = None

    def get(self, T, router=None):
        if self.ctx is None:
            ws, self.pool = _workspace(
                self.config, self.ep_group, self.E_loc, T, router, self.shared, self.mxfp8
            )
            self.ctx = _new_context(ws)
            self.pool.nctx += 1
            self.T = T
            if not _CONTEXTS:
                atexit.register(_report_status)
            _CONTEXTS.append(self)
        assert T == self.T, f"KOSMOS context built for T={self.T}, got T={T}"
        return self.ctx


class KosmosMoEFunction(torch.autograd.Function):
    """out = sum_j w_j * Expert_{idx_j}(x) over the EP group; grads for x, probs, W1, W2 (and the
    fused shared expert's Ws1, Ws2, sgate). experts: the KosmosExperts of w1, w2 for MXFP8 GEMMs,
    None for bf16."""

    @staticmethod
    def forward(
        ctx,
        hidden,
        probs,
        routing_map,
        w1,
        w2,
        kctx,
        need_g,
        ws1=None,
        ws2=None,
        sgate=None,
        experts=None,
    ):
        idx32, w = _topk(probs, routing_map, kctx.config.moe_router_topk)
        K = kctx.get(hidden.shape[0])
        _step_boundary()
        ctx.shared = ws1 is not None
        hidden = hidden.contiguous()
        kosmos_gate.before_kernel()
        # MXFP8: quantize after the gate: it waits for the param all-gather of the step's updated
        # weights.
        mxw = None if experts is None else experts.mxfp8_row(keep=not need_g)
        if ctx.shared:
            out, G, Gs = _forward_shared(K, hidden, w1, w2, ws1, ws2, idx32, w, need_g, sgate, mxw)
        else:
            out, G = _forward(K, hidden, w1, w2, idx32, w, need_g, mxw)
            Gs = None
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        if need_g:
            kctx.pool.stored()
        # The backward may read the forward's X (MXFP8, probs on input, the bf16 pool, a call over
        # capacity): keep it alive and unchanged.
        if ctx.shared:
            ctx.save_for_backward(w1, w2, hidden, ws1, ws2, sgate)
        else:
            ctx.save_for_backward(w1, w2, hidden)
        ctx.G, ctx.Gs = (G, Gs) if need_g else (None, None)
        ctx.kctx, ctx.experts = kctx, experts
        return out

    @staticmethod
    def backward(ctx, dout):
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        mxc = None if ctx.experts is None else ctx.experts.mxfp8_col()
        if ctx.shared:
            w1, w2, hidden, ws1, ws2, sgate = ctx.saved_tensors
            dX, dprobs, dW1, dW2, dWs1, dWs2, dsgate = _backward_shared(
                K, ctx.kctx.config, dout, ctx.G, ctx.Gs, hidden, w1, w2, ws1, ws2, sgate, mxc
            )
        else:
            w1, w2 = ctx.saved_tensors[:2]
            hidden = ctx.saved_tensors[2] if mxc is not None else None
            dX, dprobs, dW1, dW2 = _backward(K, dout, ctx.G, w1, w2, hidden, mxc)
            dWs1 = dWs2 = dsgate = None
        kosmos_gate.after_kernel(backward=True)
        ctx.kctx.pool.released()
        ctx.G = ctx.Gs = ctx.experts = None
        return dX, dprobs, None, dW1, dW2, None, None, dWs1, dWs2, dsgate, None


_AUX_SCALE_ONE = None


def _aux_scale():
    """MoEAuxLossAutoScaler.main_loss_backward_scale, the device fp32 scale the backward launch
    reads when it runs (Megatron updates it in place); a device 1 if unset."""
    global _AUX_SCALE_ONE
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    s = MoEAuxLossAutoScaler.main_loss_backward_scale
    if s is None:
        if _AUX_SCALE_ONE is None:
            _AUX_SCALE_ONE = torch.ones(1, dtype=torch.float32, device="cuda")
        return _AUX_SCALE_ONE
    assert (
        s.is_cuda and s.dtype == torch.float32 and s.numel() == 1
    ), "KOSMOS reads a device fp32 scale"
    return s.contiguous()


class KosmosRouterMoEFunction(torch.autograd.Function):
    """Router + MoE in the two KOSMOS launches: out, aux-loss value and the dispatch counts per
    expert from hidden; grads for hidden (MoE + router), the router weight, W1 and W2 (and the fused
    shared expert's). The aux-loss gradient is applied inside the backward launch. experts: as
    KosmosMoEFunction's (the router itself stays bf16)."""

    @staticmethod
    def forward(
        ctx,
        hidden,
        wr,
        bias,
        w1,
        w2,
        kctx,
        need_g,
        seed,
        router_cfg,
        ws1=None,
        ws2=None,
        sgate=None,
        experts=None,
    ):
        T, _ = hidden.shape
        K = kctx.get(T, router_cfg)
        _step_boundary()
        ctx.shared = ws1 is not None
        hidden = hidden.contiguous()
        kosmos_gate.before_kernel()
        mxw = None if experts is None else experts.mxfp8_row(keep=not need_g)
        if ctx.shared:
            out, G, Gs, aux, tpe = _forward_router_shared(
                K, hidden, wr.contiguous(), bias, w1, w2, ws1, ws2, seed, need_g, sgate, mxw
            )
        else:
            out, G, aux, tpe = _forward_router(
                K, hidden, wr.contiguous(), bias, w1, w2, seed, need_g, mxw
            )
            Gs = None
        kosmos_gate.after_kernel(backward=False, need_backward=need_g)
        if need_g:
            kctx.pool.stored()
        if ctx.shared:
            ctx.save_for_backward(hidden, wr, w1, w2, ws1, ws2, sgate)
        else:
            ctx.save_for_backward(hidden, wr, w1, w2)
        ctx.G, ctx.Gs = (G, Gs) if need_g else (None, None)
        ctx.kctx, ctx.experts = kctx, experts
        ctx.mark_non_differentiable(aux, tpe)
        return out, aux, tpe

    @staticmethod
    def backward(ctx, dout, daux, dtpe):
        assert ctx.G is not None, "KOSMOS backward without G (forward ran with grad disabled)"
        K = ctx.kctx.ctx
        dout = dout.contiguous()
        kosmos_gate.before_kernel()
        mxc = None if ctx.experts is None else ctx.experts.mxfp8_col()
        if ctx.shared:
            hidden, wr, w1, w2, ws1, ws2, sgate = ctx.saved_tensors
            dX, dW1, dW2, dWr, dWs1, dWs2, dsgate = _backward_router_shared(
                K,
                ctx.kctx.config,
                dout,
                ctx.G,
                ctx.Gs,
                hidden,
                w1,
                w2,
                ws1,
                ws2,
                wr,
                _aux_scale(),
                sgate,
                mxc,
            )
        else:
            hidden, wr, w1, w2 = ctx.saved_tensors
            dX, dW1, dW2, dWr = _backward_router(
                K, dout, ctx.G, hidden, w1, w2, wr, _aux_scale(), mxc
            )
            dWs1 = dWs2 = dsgate = None
        kosmos_gate.after_kernel(backward=True)
        ctx.kctx.pool.released()
        ctx.G = ctx.Gs = ctx.experts = None
        return dX, dWr, None, dW1, dW2, None, None, None, None, dWs1, dWs2, dsgate, None


# activation_func_clamp_value -> the kosmos module's clamped-SwiGLU activation (top-k 6, 8, 10).
_CLAMP_ACTS = {10.0: "ACT_SWIGLU_CLAMP10"}


def _experts_act(config):
    """The kosmos activation of the experts' MLP config (None: plain SwiGLU on a kosmos module
    without activations); raise on a config the KOSMOS kernels (bias-free SwiGLU or clamped SwiGLU,
    probs applied at the combine) would otherwise run as something else."""
    c = config
    clamp = c.activation_func_clamp_value
    why = None
    if not c.gated_linear_unit:
        why = "an ungated expert MLP (gated_linear_unit=False)"
    elif c.activation_func is not torch.nn.functional.silu:
        why = f"activation_func={getattr(c.activation_func, '__name__', c.activation_func)} (SwiGLU only)"
    elif c.glu_linear_offset:
        why = f"glu_linear_offset={c.glu_linear_offset}"
    elif clamp is not None and float(clamp) not in _CLAMP_ACTS:
        why = f"activation_func_clamp_value={clamp} (10 only)"
    elif clamp is not None and c.moe_router_topk not in (6, 8, 10):
        why = f"activation_func_clamp_value with moe_router_topk={c.moe_router_topk} (6/8/10 only)"
    elif clamp is not None and (c.bias_activation_fusion or c.use_te_activation_func):
        # Megatron's fused / TE SwiGLU drop the clamp: its shared expert would run unclamped.
        why = "activation_func_clamp_value with fused SwiGLU (pass --no-bias-swiglu-fusion)"
    elif clamp is not None and not hasattr(_kosmos(), _CLAMP_ACTS[float(clamp)]):
        why = "activation_func_clamp_value (kosmos module without the clamped SwiGLU)"
    elif c.add_bias_linear:
        why = "expert bias (add_bias_linear)"
    elif c.moe_apply_probs_on_input and (
        c.moe_router_topk != 1 or "probs_in" not in (_kosmos().MoeWorkspace.__init__.__doc__ or "")
    ):
        why = "moe_apply_probs_on_input (top-1 only, and a kosmos module with probs_in)"
    if why is not None:
        raise ValueError(f"KOSMOS MoE (--moe-use-kosmos) does not support {why}")
    if clamp is not None:
        return getattr(_kosmos(), _CLAMP_ACTS[float(clamp)])
    return getattr(_kosmos(), "ACT_SWIGLU", None)


class KosmosExperts(MegatronModule):
    """Local experts stored in KOSMOS layout: weight1 [E_loc, 2I, H] (128-row gate|up interleave),
    weight2 [E_loc, H, I]. Initialised by building the SequentialMLP of the same spec and
    converting its weights, so a KOSMOS model starts from the same weights as SequentialMLP."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None, name=None):
        super().__init__(config=config)
        _experts_act(config)
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
        # Set by Megatron's set_is_first_microbatch at each step (FP8 runs): refresh mxfp8_weights.
        self.is_first_microbatch = True
        self._mxw = None
        # (linear_fc1.weight, linear_fc2.weight) of the layer's fused shared expert on an MXFP8
        # layer (set by KosmosMoELayer): quantized with the experts, as one more expert each.
        self.shared_weights = None

    def _quantize(self, col):
        """(W1, W1_scale, W2, W2_scale[, Ws1, Ws1_scale, Ws2, Ws2_scale]) row-wise or column-wise."""
        q = ()
        for w in (self.weight1, self.weight2, *(self.shared_weights or ())):
            w = w.detach()
            q += _quantize_mxfp8(w if w.dim() == 3 else w.unsqueeze(0), col)
        return q

    def mxfp8_weights(self):
        """(row-wise, column-wise) weights (_quantize), quantized on the first microbatch of each
        step and cached for the step (more than one microbatch per step)."""
        if self._mxw is None or self.is_first_microbatch:
            self._mxw = (self._quantize(False), self._quantize(True))
        self.is_first_microbatch = False
        return self._mxw

    def mxfp8_row(self, keep):
        """The row-wise weights (forward). One microbatch per step: quantized once per step and held
        only while used; keep (a forward with grad disabled, the checkpointed first pass) holds them
        for the recompute's forward, which takes them. Otherwise from mxfp8_weights."""
        if get_num_microbatches() > 1:
            return self.mxfp8_weights()[0]
        if self.is_first_microbatch:
            self._mxw = None
            self.is_first_microbatch = False
        row = self._mxw
        if row is None:
            row = self._quantize(False)
        self._mxw = row if keep else None
        return row

    def mxfp8_col(self):
        """The column-wise weights (dgrad): one microbatch per step, quantized for the step's one
        backward and freed after it; otherwise from mxfp8_weights."""
        if get_num_microbatches() > 1:
            return self.mxfp8_weights()[1]
        return self._quantize(True)

    def backward_dw(self):
        pass


# Megatron's moe_router_score_function -> the kosmos module's (pre-softmax: _router_score).
_ROUTER_SCORES = {
    "softmax": "ROUTER_SOFTMAX",
    "sigmoid": "ROUTER_SIGMOID",
    "sqrtsoftplus": "ROUTER_SQRTSOFTPLUS",
}

# The MXFP8 training calls an MXFP8 layer may use.
_MXFP8_CALLS = (
    "forward_mxfp8_train",
    "backward_mxfp8",
    "forward_router_mxfp8_train",
    "backward_router_mxfp8",
    "forward_shared_mxfp8_train",
    "backward_shared_mxfp8",
    "forward_router_shared_mxfp8_train",
    "backward_router_shared_mxfp8",
)


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
        assert config.moe_router_topk in (1, 2, 4, 6, 8, 10)
        assert config.moe_latent_size is None and not config.moe_shared_expert_overlap
        self.kctx = KosmosContext(config, self.ep_group, self.num_local_experts)
        self._resolved = False
        self._configure(layer_number)

    def set_layer_number(self, layer_number):
        super().set_layer_number(layer_number)
        self._configure(layer_number)

    def _configure(self, layer_number):
        """MXFP8, fused shared expert and fused router from the config, before the first forward:
        the layer number (set by TransformerLayer after construction) decides Megatron's bf16
        first/last layers. _notes: why features stay in Megatron, printed at the first forward."""
        assert self.kctx.ctx is None, "KOSMOS layer reconfigured after its first forward"
        self._notes = []
        self._mxfp8 = self._mxfp8_supported(layer_number)
        self._fused_shared = SHARED and self._shared_supported()
        self._fused_router = ROUTER and self._router_supported()
        self.kctx.layer_number = layer_number
        self._apply_decisions()

    def _apply_decisions(self):
        """The decisions to the context and, for an MXFP8 fused shared expert, to the experts."""
        self.kctx.shared, self.kctx.mxfp8 = self._fused_shared, self._mxfp8
        sh = self.shared_experts if self._fused_shared and self._mxfp8 else None
        self.experts.shared_weights = (
            None if sh is None else (sh.linear_fc1.weight, sh.linear_fc2.weight)
        )

    def _resolve(self, hidden_states):
        """At the first forward (the context is built for one token count T, the same on every
        rank): MXFP8 training and the fused router need T % 256, the fused router's seq_aux_loss at
        most 4 sequences, the fused shared expert T % 512; then the layer's decisions are printed
        once, at rank 0."""
        if self._resolved:
            return
        self._resolved = True
        bsz = hidden_states.shape[1]
        T = hidden_states.numel() // hidden_states.shape[-1]
        if self._mxfp8 and T % MXFP8_T_MULTIPLE:
            self._mxfp8 = False
            self._notes.append(
                f"bf16 experts: {T} tokens per call, not a multiple of {MXFP8_T_MULTIPLE}"
            )
        why = None
        if self._fused_router and T % ROUTER_T_MULTIPLE:
            why = f"{T} tokens per call, not a multiple of {ROUTER_T_MULTIPLE}"
        elif self._fused_router and self.router.get_aux_loss_coeff("seq_aux_loss") and bsz > 4:
            why = f"seq_aux_loss over {bsz} sequences (at most 4)"
        if why is not None:
            self._fused_router = False
            self._notes.append(f"Megatron's router: {why}")
        if self._fused_shared and T % SHARED_T_MULTIPLE:
            self._fused_shared = False
            self._notes.append(
                f"shared expert stays in Megatron: {T} tokens per call, not a multiple of "
                f"{SHARED_T_MULTIPLE}"
            )
        self._apply_decisions()
        if dist.get_rank() != 0:
            return
        c, n = self.config, self.layer_number
        for why in self._notes:
            print(f"[KOSMOS] layer {n}: {why}", flush=True)
        what = ["MXFP8 experts" if self._mxfp8 else "KOSMOS experts (bf16)"]
        what.append(f"top-k {c.moe_router_topk}")
        if c.moe_apply_probs_on_input:
            what.append("probs on input")
        what.append("fused router" if self._fused_router else "Megatron's router")
        if getattr(self, "use_shared_expert", False):
            gated = self._fused_shared and self.shared_experts.use_shared_expert_gate
            what.append(
                ("fused shared expert" + (" with gate" if gated else ""))
                if self._fused_shared
                else "Megatron's shared expert"
            )
        print(f"[KOSMOS] layer {n}: {', '.join(what)}", flush=True)

    def _mxfp8_supported(self, layer_number):
        """MXFP8 experts: an mxfp8-recipe run, a layer Megatron runs in FP8, H a multiple of 1024
        and a library with the MXFP8 training entry points."""
        c = self.config
        if c.fp8 is None or c.fp8_recipe != "mxfp8":
            return False
        if layer_number is not None and is_first_last_bf16_layer(c, layer_number - 1):
            return False
        why = None
        if not MXFP8:
            why = "KOSMOS_MXFP8=0"
        elif c.hidden_size % MXFP8_H_MULTIPLE:
            why = f"hidden size not a multiple of {MXFP8_H_MULTIPLE}"
        elif not hasattr(_kosmos(), "quantize_mxfp8") or not all(
            hasattr(_kosmos().MoeContext, f) for f in _MXFP8_CALLS
        ):
            why = "kosmos module without the MXFP8 training entry points"
        if why is not None:
            self._notes.append(f"bf16 experts: {why}")
            return False
        return True

    def _shared_supported(self):
        """The shared expert fits the KOSMOS launches: one SwiGLU MLP of the experts' intermediate
        size, bf16 linears without bias or fused norm (bf16 run, or an mxfp8 run: in the layer's
        precision), and a library with the shared entry points."""
        c, sh = self.config, getattr(self, "shared_experts", None)
        if not getattr(self, "use_shared_expert", False) or sh is None:
            return False
        fc1, fc2 = getattr(sh, "linear_fc1", None), getattr(sh, "linear_fc2", None)
        w1, w2 = getattr(fc1, "weight", None), getattr(fc2, "weight", None)
        H, I = c.hidden_size, c.moe_ffn_hidden_size
        gated = c.moe_shared_expert_gate or getattr(sh, "use_shared_expert_gate", False)
        why = None
        if gated and "sgate" not in (_kosmos().MoeContext.forward_shared.__doc__ or ""):
            why = "shared expert gate (kosmos module without sgate)"
        elif gated and c.moe_router_topk < 6:
            why = "shared expert gate with top-k < 6 (the library's gate needs top-k 6, 8 or 10)"
        elif not c.gated_linear_unit or c.activation_func is not torch.nn.functional.silu:
            why = "not SwiGLU"
        elif c.add_bias_linear or any(getattr(m, "bias", None) is not None and m.bias.numel() for m in (fc1, fc2)):
            why = "bias"
        elif w1 is None or w2 is None or tuple(w1.shape) != (2 * I, H) or tuple(w2.shape) != (H, I):
            why = "shared intermediate size != moe_ffn_hidden_size"
        elif is_float8tensor(w1) or is_float8tensor(w2):
            why = "FP8 primary weights (--fp8-param-gather)"
        elif w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            why = "not bf16"
        elif hasattr(fc1, "layer_norm_weight"):
            why = "fused norm"
        elif c.fp8 and c.fp8_recipe != "mxfp8":
            why = f"FP8 recipe {c.fp8_recipe} (KOSMOS runs it in bf16 or MXFP8)"
        elif getattr(c, "fp4", None):
            why = "FP4"
        elif not hasattr(_kosmos().MoeContext, "forward_router_shared"):
            why = "kosmos module without the shared entry points"
        if why is not None:
            self._notes.append(f"shared expert stays in Megatron: {why}")
            return False
        return True

    def shared_experts_compute(self, hidden_states):
        # Fused: the shared expert runs in the KOSMOS launches (routed_experts_compute); no output here.
        self._resolve(hidden_states)
        if self._fused_shared:
            return None
        return super().shared_experts_compute(hidden_states)

    def _shared_args(self, hidden_states):
        """(Ws1, Ws2, sgate) of the fused shared expert, or Nones. sgate [T] fp32: the gate's
        sigmoid(x w_sg) (SharedExpertMLP.forward's), in autograd; None: ungated."""
        if not self._fused_shared:
            return None, None, None
        sh = self.shared_experts
        sgate = None
        if sh.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, sh.gate_weight)
            sgate = torch.sigmoid(logits.float()).view(-1)
        return sh.linear_fc1.weight, sh.linear_fc2.weight, sgate

    def _mx_experts(self):
        """The experts argument of the KOSMOS functions: MXFP8 GEMMs or (None) bf16."""
        return self.experts if self._mxfp8 else None

    def _router_score(self):
        """The kosmos score function of Megatron's router config, or None (no fused router)."""
        c = self.config
        name = _ROUTER_SCORES.get(c.moe_router_score_function)
        if c.moe_router_score_function == "softmax" and c.moe_router_pre_softmax:
            name = "ROUTER_SOFTMAX_PRE"
        return getattr(_kosmos(), name, None) if name else None

    def _router_groups(self):
        """(groups, group_topk) of the group-limited routing; (0, 0) when every group is selected
        (one group, or group_topk = groups: plain top-k)."""
        c = self.config
        groups, gtopk = c.moe_router_num_groups or 0, c.moe_router_group_topk or 0
        if groups <= 1 or gtopk == groups:
            return 0, 0
        return groups, gtopk

    def _router_supported(self):
        """The fused router computes Megatron's routing, aux loss and their gradients for this
        config (the per-call limits are _resolve's); else Megatron's router, with the reason noted."""
        c, r = self.config, self.router
        k, E = c.moe_router_topk, c.num_moe_experts
        groups, gtopk = self._router_groups()
        why = None
        if self._router_score() is None:
            why = f"score function {c.moe_router_score_function} (not in the kosmos module)"
        elif c.moe_apply_probs_on_input:
            why = "moe_apply_probs_on_input"
        elif r.get_aux_loss_coeff("global_aux_loss"):
            why = "global_aux_loss"
        elif c.moe_z_loss_coeff:
            why = "moe_z_loss_coeff"
        elif c.moe_input_jitter_eps:
            why = "moe_input_jitter_eps"
        elif r.get_aux_loss_coeff("seq_aux_loss") and r.get_aux_loss_coeff("aux_loss"):
            why = "seq_aux_loss with aux_loss"
        elif c.moe_router_force_biased is not None:
            why = "moe_router_force_biased"
        elif c.moe_enable_routing_replay:
            why = "moe_enable_routing_replay"
        elif c.calculate_per_token_loss:
            why = "calculate_per_token_loss (the in-kernel aux-loss gradient does not model it)"
        elif self.is_mtp_layer:
            why = "MTP layer (the in-kernel aux-loss gradient does not model it)"
        elif c.context_parallel_size > 1:
            why = "context parallelism (the in-kernel aux-loss gradient does not model it)"
        elif c.moe_expert_capacity_factor is not None:
            why = "moe_expert_capacity_factor (token dropping)"
        elif E > 512:
            why = f"{E} experts (at most 512)"
        elif groups and (
            groups > 64
            or E % groups
            or gtopk < 1
            or gtopk > groups
            or k % gtopk
            or k // gtopk > 4
            or k // gtopk > E // groups
        ):
            why = f"group-limited routing (groups {groups}, group_topk {gtopk}, top-k {k})"
        if why is not None:
            self._notes.append(f"Megatron's router: {why}")
            return False
        return True

    def route(self, hidden_states, padding_mask=None):
        # (None, None): the fused router routes inside the KOSMOS launches (routed_experts_compute).
        self._resolve(hidden_states)
        if not self._fused_router or padding_mask is not None:
            return super().route(hidden_states, padding_mask)
        return None, None

    def _router_cfg(self, bsz):
        """(score, groups, group_topk, aux_groups, aux_coeff, scale, force) of Megatron's router
        config."""
        c, r = self.config, self.router
        seq, aux = r.get_aux_loss_coeff("seq_aux_loss"), r.get_aux_loss_coeff("aux_loss")
        return (
            self._router_score(),
            *self._router_groups(),
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
            *self._shared_args(hidden_states),
            self._mx_experts(),
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
        out = KosmosMoEFunction.apply(
            hidden_states,
            probs,
            routing_map,
            self.experts.weight1,
            self.experts.weight2,
            self.kctx,
            torch.is_grad_enabled(),
            *self._shared_args(hidden_states),
            self._mx_experts(),
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


def replace_moe_experts(moe_spec, layer_cls, experts_cls, flag):
    """MoELayer builder whose experts are SequentialMLP -> layer_cls builder with experts_cls
    experts (built from the SequentialMLP's MLPSubmodules)."""
    _, sub, kwargs = _builder_parts(moe_spec)
    e_cls, e_sub, e_kwargs = _builder_parts(sub.experts)
    assert (
        e_cls is SequentialMLP
    ), f"{flag} needs the SequentialMLP expert spec (do not pass --moe-grouped-gemm)"
    experts = partial(
        experts_cls,
        submodules=e_sub,
        **e_kwargs,
    )
    return partial(layer_cls, submodules=dataclasses.replace(sub, experts=experts), **kwargs)


def kosmos_moe_spec(moe_spec):
    """Turn an MoELayer spec whose experts are SequentialMLP into the KOSMOS layer spec."""
    return replace_moe_experts(moe_spec, KosmosMoELayer, KosmosExperts, "--moe-use-kosmos")
