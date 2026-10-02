# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""KOSMOS kernel windows, and the gate that keeps Megatron DDP communication out of them.

A KOSMOS MoE kernel is a collective (EP dispatch/combine) that occupies every CU. Nothing DDP
enqueues may share the GPU with it: no grad reduce-scatter, no param all-gather, no 1/dp scaling,
and no AccumulateGrad add on the DDP hook stream. The adapter (kosmos_moe.py) brackets each KOSMOS
launch with before_kernel() / after_kernel(); DDP (megatron/core/distributed) calls the functions
below. Every call is a no-op until a KOSMOS layer calls enable(), so other MoE backends are
unaffected. KOSMOS_DDP_GATE=0 leaves the gate off.

Streams: S0 is the stream KOSMOS is enqueued on (the training compute stream). Side streams are the
other streams DDP enqueues on: the AccumulateGrad stream, on which the DDP hook adds, checks, scales
and launches its collectives. Collectives run on NCCL streams that wait on the stream they were
launched from.

  before_kernel  S0 waits on every side stream and on every DDP collective already started (grad
                 reduce-scatter or param all-gather handle not yet finished): nothing enqueued
                 earlier overlaps the kernel.
  after_kernel   records END on S0 after the kernel. A side stream waits on END before its next
                 DDP work (side_stream()), and deferred grad syncs are released behind END: nothing
                 enqueued later overlaps the kernel either.
  defer          a bucket whose grads become ready while a KOSMOS backward kernel is still to come
                 is not synced then: its scaling + collective are released right after the next
                 KOSMOS backward kernel is enqueued, so they overlap the attention/dense backward
                 that follows that kernel. Once no KOSMOS backward is to come, buckets sync at once
                 (still behind END). flush() (finish_grad_sync, start_grad_sync, start_param_sync)
                 releases whatever is left.

Release points depend only on the autograd order, which is the same on every DP rank, so every rank
issues the same collectives in the same order relative to its KOSMOS kernels (as without the gate).
"""

import os
import weakref

import torch

_enabled = False
_ddps = []  # weakrefs to the DistributedDataParallel instances (bucket groups hold the handles)
_side = {}  # side stream handle -> [stream, number of the KOSMOS kernel it last waited on]
_end = None  # event recorded on S0 after the last KOSMOS kernel
_s0 = None  # S0 handle
_kernels = 0  # KOSMOS kernels enqueued
_pending_bwd = 0  # KOSMOS forwards run with grad whose backward kernel is not enqueued yet
_deferred = []  # (side stream, release callable), in ready order


def enable():
    """Called by each KOSMOS layer at construction; KOSMOS_DDP_GATE=0 leaves the gate off."""
    global _enabled
    _enabled = os.environ.get("KOSMOS_DDP_GATE", "1") != "0"


def register_ddp(ddp):
    """DistributedDataParallel.__init__: remember the instance so before_kernel can fence its
    collectives."""
    if not _enabled:
        return
    cfg = ddp.ddp_config
    assert (
        cfg.num_distributed_optimizer_instances == 1
    ), "KOSMOS DDP gate: one distributed-optimizer instance"
    assert (
        not cfg.reduce_scatter_with_fp32_accumulation
    ), "KOSMOS DDP gate: plain reduce-scatter only"
    _ddps.append(weakref.ref(ddp))


def _bucket_groups():
    for r in _ddps:
        ddp = r()
        if ddp is not None:
            yield from ddp.bucket_groups + ddp.expert_parallel_bucket_groups


def side_stream():
    """DDP backward hook entry: order the current stream, if it is not S0, after the last KOSMOS
    kernel."""
    if not _enabled:
        return
    s = torch.cuda.current_stream()
    if s.cuda_stream == _s0:
        return
    e = _side.setdefault(s.cuda_stream, [s, 0])
    if e[1] != _kernels:
        s.wait_event(_end)
        e[1] = _kernels


def before_kernel():
    """Adapter, right before enqueuing a KOSMOS kernel on the current stream (S0)."""
    if not _enabled:
        return
    s = torch.cuda.current_stream()
    for side, _ in _side.values():
        if side.cuda_stream != s.cuda_stream:
            s.wait_stream(side)
    for g in _bucket_groups():
        # Work.wait() makes the current stream wait on the collective's end; Megatron's own wait()
        # later is unaffected.
        if g.grad_reduce_handle is not None:
            g.grad_reduce_handle.wait()
        if g.param_gather_handle is not None:
            g.param_gather_handle.wait()


def after_kernel(backward, need_backward=False):
    """Adapter, right after enqueuing a KOSMOS kernel: mark its end, release deferred grad syncs
    after a backward kernel."""
    global _end, _s0, _kernels, _pending_bwd
    if not _enabled:
        return
    s = torch.cuda.current_stream()
    _end = torch.cuda.Event()
    _end.record(s)
    _s0 = s.cuda_stream
    _kernels += 1
    if backward:
        _pending_bwd = max(_pending_bwd - 1, 0)
        _release()
    elif need_backward:
        _pending_bwd += 1


def defer_grad_sync():
    """register_grad_ready: True if the bucket's sync must wait for the next KOSMOS backward
    kernel."""
    if not _enabled:
        return False
    return _pending_bwd > 0


def defer(release):
    """Queue a bucket's sync, to run on the current (side) stream after the next KOSMOS backward
    kernel."""
    _deferred.append((torch.cuda.current_stream(), release))


def flush(end_of_backward=False):
    """Release every deferred sync now (in ready order); end_of_backward also resets the backward
    count."""
    global _pending_bwd
    if not _enabled:
        return
    _release()
    if end_of_backward:
        _pending_bwd = 0


def _release():
    items = list(_deferred)
    _deferred.clear()
    for s, release in items:
        with torch.cuda.stream(s):
            if _end is not None and s.cuda_stream != _s0:
                s.wait_event(_end)
                e = _side.setdefault(s.cuda_stream, [s, 0])
                e[1] = _kernels
            release()
