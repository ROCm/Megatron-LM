# Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mori SDMA backend for the Megatron FSDP parameter all-gather.

When the user opts in (via ``--enable-mori-sdma-ag`` /
``DistributedDataParallelConfig.enable_mori_sdma_ag``), the Megatron FSDP
all-gather pipeline routes ``all_gather_into_tensor`` through the mori
``mori.ccl.AllGatherIntoTensor`` SDMA dispatcher.

The gather is driven entirely on the caller's CUDA stream (the FSDP all-gather
stream). Completion is exposed as a stream event via ``_SdmaWork``.
Any failure (mori missing, non-AMD/ROCm runtime, shmem init error, oversized
call, or a group other than the one bound at init time) yields ``None`` and the
caller falls back to the underlying RCCL/NCCL
``torch.distributed.all_gather_into_tensor``.

Megatron FSDP all-gathers parameters over the data-parallel (FSDP) shard group. 
The backend therefore binds (lazily, on thefirst call) 
to the process group passed by the all-gather pipeline and only
accepts subsequent calls on that same group.

Environment overrides:

* ``MEGATRON_FSDP_SDMA_ALLGATHER_MAX_NUMEL=N`` overrides the transit buffer
  size in elements (default 64M = 256 MiB per-rank input, ~2 GiB output on 8
  ranks).
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from megatron.core.utils import log_single_rank

# Module-level lazy state. Populated by ``init`` on the first all-gather call.
_handle = None
_dtype_map = None
_max_numel = 0
_bound_group = None
_init_attempted = False
_call_failed_warned = False

# Direct (non-blocking) path state. When the output bucket can be registered with
# mori's symmetric-memory layer, the SDMA kernel writes straight into it and we can
# launch without the host-blocking ``finish_sync``, recording a CUDA event instead so
# the gather actually overlaps prefetch-window compute. ``_direct_unavailable`` latches
# off the fast path (and logs once) if registration or the launch hook is missing.
_registered_output_ptrs = set()
_direct_unavailable = False

# Dedicated stream for the SDMA all-gather kernels, created lazily. Mirrors how
# ProcessGroupNCCL runs collectives on its own internal stream: by NOT running on the
# FSDP all-gather stream we avoid the per-module ``all_gather_stream.wait_stream(
# current_stream())`` barrier that otherwise chains every SDMA gather behind the
# previous module's compute (measured: ~0% gather/compute overlap). The local weight
# shard is the gather's only real input and is read-only during fwd/bwd (it is written
# only by the optimizer step), so the dedicated stream needs to synchronize with the
# compute stream just *once per iteration* -- see ``notify_weights_updated`` -- after
# which gathers run freely and overlap compute.
_sdma_stream = None
_needs_compute_sync = True


def notify_weights_updated() -> None:
    """Signal that the local weight shards may have changed (e.g. optimizer step).

    The next SDMA gather will re-synchronize its dedicated stream with the compute
    stream exactly once, so it observes the updated weights; subsequent gathers in the
    same iteration then run without per-gather compute dependencies and overlap
    compute. Safe to call unconditionally (no-op cost when SDMA is unused).
    """
    global _needs_compute_sync
    _needs_compute_sync = True

# Name under which the FSDP data-parallel group is registered for mori's shmem
# layer. mori looks the process group up by this label during shmem init.
_PG_NAME = "megatron_fsdp_sdma"

# Default transit buffer size, in elements (matches the DeepSpeed default).
_DEFAULT_MAX_NUMEL = 64 * 1024 * 1024


class _SdmaWork:
    """Duck-type compatible with ``torch.distributed.Work``.

    On the direct path the one-shot SDMA kernel (put + cross-rank wait + direct
    write into the registered output bucket) is issued on the all-gather stream at
    gather time and completion is captured by a CUDA event -- no host sync -- so the
    gather runs concurrently with prefetch-window compute, the same way an async
    RCCL collective overlaps. :meth:`wait` issues a stream-level ``wait_event`` and
    never blocks the CPU, mirroring RCCL ``Work.wait()`` semantics so the FSDP
    prefetch pipeline keeps queueing ahead.

    (On the synchronous fallback path the event is recorded after mori's blocking
    ``finish_sync`` has already completed the gather, so it is trivially signaled.)
    """

    def __init__(self, event):
        self._event = event

    def wait(self):
        """Make the current stream wait on SDMA completion (no CPU block)."""
        torch.cuda.current_stream().wait_event(self._event)

    def is_completed(self) -> bool:
        """Return True if the gather has completed."""
        return self._event.query()


def _resolve_max_numel(default: int) -> int:
    raw = os.environ.get("MEGATRON_FSDP_SDMA_ALLGATHER_MAX_NUMEL")
    if raw is None:
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        return default


def _register_group(group) -> None:
    """Register the given process group under ``_PG_NAME`` in PyTorch's C++
    GroupRegistry so mori's shmem layer can look it up by name."""
    if group is None:
        group = torch.distributed.group.WORLD
    assert group is not None, "torch.distributed must be initialized before SDMA allgather"
    torch._C._distributed_c10d._register_process_group(_PG_NAME, group)


def _build_dtype_map():
    """torch.dtype -> mori.ccl.DataType (NCCL-style enum)."""
    from mori.ccl import DataType

    return {
        torch.uint8: DataType.Uint8,
        torch.int8: DataType.Int8,
        torch.int16: DataType.Int16,
        torch.int32: DataType.Int32,
        torch.int64: DataType.Int64,
        torch.float16: DataType.Float16,
        torch.bfloat16: DataType.BFloat16,
        torch.float32: DataType.Float32,
        torch.float64: DataType.Float64,
    }


def _create_sdma_handle(my_pe: int, npes: int, input_bytes: int):
    """Construct the mori ``AllGatherIntoTensor`` SDMA all-gather handle."""
    from mori.ccl import AllGatherIntoTensor

    return AllGatherIntoTensor(
        my_pe=my_pe,
        npes=npes,
        input_buffer_size=input_bytes,
        output_buffer_size=input_bytes * npes,
        copy_output_to_user=True,
    )


def init(group, max_numel: int = _DEFAULT_MAX_NUMEL) -> None:
    """Best-effort, idempotent SDMA handle construction.

    Builds one mori SDMA all-gather handle sized for the largest expected
    per-rank shard, bound to ``group``. All subsequent allgather calls on
    ``group`` reuse this handle. Safe to call unconditionally: any failure
    leaves ``_handle`` unset and logs a single rank-0 info line, so callers
    transparently fall back to RCCL/NCCL.
    """
    global _handle, _dtype_map, _max_numel, _bound_group, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    max_numel = _resolve_max_numel(max_numel)
    # mori's SymmMemManager only allocates the uncached transit buffers required
    # by the SDMA kernel when MORI_ENABLE_SDMA is set; setdefault so users who
    # already exported it (or want to override) win.
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    try:
        _register_group(group)
        import mori.shmem as shmem

        shmem.shmem_torch_process_group_init(_PG_NAME)
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        # Per-rank input transit buffer must hold the largest shard we'll ever
        # see; output buffer = npes * input. 4 B/element is the SDMA kernel's
        # uint32 lane width.
        input_bytes = max_numel * 4
        _handle = _create_sdma_handle(my_pe, npes, input_bytes)
        _dtype_map = _build_dtype_map()
        _max_numel = max_numel
        _bound_group = group if group is not None else torch.distributed.group.WORLD
        log_single_rank(
            logger,
            logging.INFO,
            f"Megatron FSDP SDMA allgather enabled via mori.ccl.AllGatherIntoTensor "
            f"(max_numel={max_numel})",
        )
    except Exception as e:  # noqa: BLE001 - best-effort, always fall back to RCCL
        _handle = None
        _dtype_map = None
        _max_numel = 0
        _bound_group = None
        log_single_rank(
            logger,
            logging.INFO,
            f"Megatron FSDP SDMA allgather unavailable ({type(e).__name__}: {e}); "
            f"using RCCL/NCCL allgather",
        )


def is_enabled() -> bool:
    """Return True if the SDMA handle was successfully constructed."""
    return _handle is not None


def supports(input_tensor: torch.Tensor, group=None) -> bool:
    """Cheap pre-check used before routing an all-gather through mori.

    SDMA is only safe when:
    - the backend is initialised (``_handle`` set),
    - the call is on the process group bound at init time (mori's shmem layer
      was bound to that group),
    - the per-rank shard fits inside the pre-allocated transit buffer,
    - the dtype is supported by mori's public dispatcher.
    """
    if _handle is None:
        return False
    call_group = group if group is not None else torch.distributed.group.WORLD
    if call_group is not _bound_group:
        return False
    if input_tensor.numel() > _max_numel:
        return False
    if _dtype_map is None or input_tensor.dtype not in _dtype_map:
        return False
    return True


def _try_register_output(output_tensor: torch.Tensor) -> bool:
    """Register the output bucket so the SDMA kernel can write into it directly.

    Returns True when the direct (non-blocking) path is usable for this buffer.
    Registration is done once per distinct output buffer pointer and cached; FSDP's
    double buffer reuses a small fixed set of bucket buffers, so this is cheap. Any
    failure (registration unsupported on this build/runtime, or the launch hook
    missing) latches ``_direct_unavailable`` so we stop trying and fall back to the
    blocking synchronous path.
    """
    global _direct_unavailable
    if _direct_unavailable:
        return False
    if not hasattr(_handle, "launch_no_sync") or not hasattr(_handle, "register_output_buffer"):
        _direct_unavailable = True
        log_single_rank(
            logger,
            logging.INFO,
            "Megatron FSDP SDMA direct launch hook unavailable; using synchronous "
            "SDMA path (host-blocking, no compute overlap)",
        )
        return False
    ptr = output_tensor.data_ptr()
    if ptr in _registered_output_ptrs:
        return True
    try:
        size = output_tensor.numel() * output_tensor.element_size()
        _handle.register_output_buffer(ptr, size)
        _registered_output_ptrs.add(ptr)
        return True
    except Exception as e:  # noqa: BLE001 - fall back to the blocking path on any failure
        _direct_unavailable = True
        log_single_rank(
            logger,
            logging.INFO,
            f"Megatron FSDP SDMA output registration unavailable "
            f"({type(e).__name__}: {e}); using synchronous SDMA path "
            f"(host-blocking, no compute overlap)",
        )
        return False


def allgather_into_tensor(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    group=None,
    max_numel: int = _DEFAULT_MAX_NUMEL,
    serialize_after_stream: Optional["torch.cuda.Stream"] = None,
) -> Optional[_SdmaWork]:
    """Issue one all_gather_into_tensor through the SDMA handle.

    Prefers the *direct, non-blocking* path: the output bucket is registered with
    mori's symmetric-memory layer so the one-shot SDMA kernel writes straight into
    it, the kernel is enqueued on the caller's stream via ``launch_no_sync`` (no
    ``hipStreamSynchronize``), and a CUDA event captures completion. The CPU thread
    returns immediately, so the gather overlaps prefetch-window compute and
    ``_SdmaWork.wait()`` defers to a device-side ``wait_event``.

    If the output cannot be registered (or the launch hook is missing), falls back
    to mori's blocking ``__call__`` (``finish_sync`` -> ``hipStreamSynchronize``),
    which is correct but serializes against compute.

    Lazily initialises (bound to ``group``) on the first call. Returns an
    ``_SdmaWork`` (Work-compatible) on success, or ``None`` when SDMA is not
    applicable (uninitialised, a group other than the bound group, dtype not
    supported, shard larger than the transit buffer) or the call fails -- the
    caller then falls back to ``torch.distributed.all_gather_into_tensor``.
    """
    global _call_failed_warned, _sdma_stream, _needs_compute_sync
    if not _init_attempted:
        init(group, max_numel)
    if not supports(input_tensor, group):
        return None
    try:
        compute_stream = torch.cuda.current_stream()

        # Run the gather on a dedicated stream decoupled from the FSDP all-gather
        # stream's per-module compute barrier, mirroring NCCL's internal stream.
        if _sdma_stream is None:
            _sdma_stream = torch.cuda.Stream()
        # Once per iteration, honor the optimizer's weight write (and any other
        # producer of the local shard) by syncing the dedicated stream with compute.
        # Afterwards the shard is read-only, so further gathers need no compute
        # dependency and can overlap compute.
        if _needs_compute_sync:
            _sdma_stream.wait_stream(compute_stream)
            _needs_compute_sync = False
        # Serialize the gather behind in-flight reduce-scatter (NCCL serializes its
        # all-gather and reduce-scatter on one communicator; our SDMA stream is
        # independent and would otherwise co-run with the NCCL reduce-scatter and
        # steal HBM bandwidth, slowing it and pushing it onto the critical path). The
        # passed stream is ordered after the RS kernel (synchronous coalescing), so a
        # one-way wait keeps the two collectives from overlapping while still letting
        # both hide under compute.
        if serialize_after_stream is not None:
            _sdma_stream.wait_stream(serialize_after_stream)
        sdma_stream = _sdma_stream

        dtype = _dtype_map[input_tensor.dtype]
        issued = False
        if _try_register_output(output_tensor):
            # Direct path: enqueue only; defer completion to the CUDA event below.
            issued = _handle.launch_no_sync(
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                input_tensor.numel(),
                dtype,
                sdma_stream.cuda_stream,
            )
        if not issued:
            # Blocking fallback (host-syncs inside finish_sync; no compute overlap).
            ok = _handle(
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                input_tensor.numel(),
                dtype,
                sdma_stream.cuda_stream,
            )
            if not ok:
                return None

        # Keep the input shard and output bucket alive for the async stream and let
        # the caching allocator order any buffer reuse against this stream (the same
        # cross-stream safety mechanism NCCL relies on).
        input_tensor.record_stream(sdma_stream)
        output_tensor.record_stream(sdma_stream)

        event = torch.cuda.Event()
        event.record(sdma_stream)
        return _SdmaWork(event)
    except Exception as e:  # noqa: BLE001 - best-effort, always fall back to RCCL
        if not _call_failed_warned:
            log_single_rank(
                logger,
                logging.WARNING,
                f"Megatron FSDP SDMA allgather failed ({e}); falling back to "
                f"torch.distributed.all_gather_into_tensor",
            )
            _call_failed_warned = True
        return None
