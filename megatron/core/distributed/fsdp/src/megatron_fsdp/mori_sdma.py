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
``mori.ccl.AllgatherSdma`` SDMA dispatcher.

The gather is driven on a dedicated CUDA stream. Completion is exposed as a
stream event via ``_SdmaWork``.
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
_max_numel = 0
_bound_group = None
_init_attempted = False
_call_failed_warned = False

# Output-registration state. Registering the output bucket with mori's symmetric-memory
# layer lets the SDMA kernel write straight into it.
_registered_output_ptrs = set()
_direct_unavailable = False

def _direct_write_enabled(double_buffer: bool) -> bool:
    """Whether to register + direct-write the output bucket for this gather.

    Follows the FSDP double-buffer setting (safe lockstep registration).
    """
    return bool(double_buffer)

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

    The SDMA put + cross-rank wait kernels are issued on a dedicated stream and
    completion is captured by a CUDA event, so the gather runs concurrently with
    prefetch-window compute, the same way an async RCCL collective overlaps.
    :meth:`wait` issues a stream-level ``wait_event`` and never blocks the CPU,
    mirroring RCCL ``Work.wait()`` semantics so the FSDP prefetch pipeline keeps
    queueing ahead.
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


def _create_sdma_handle(my_pe: int, npes: int, input_bytes: int):
    """Construct the mori ``AllgatherSdma`` SDMA all-gather handle.

    ``AllgatherSdma`` operates on raw bytes (it gathers the input tensor as a
    ``uint32`` byte stream), so no dtype enum is needed: the dispatcher accepts the
    user tensors directly along with the element count.
    """
    from mori.ccl import AllgatherSdma

    return AllgatherSdma(
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
    global _handle, _max_numel, _bound_group, _init_attempted
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
        _max_numel = max_numel
        _bound_group = group if group is not None else torch.distributed.group.WORLD
        log_single_rank(
            logger,
            logging.INFO,
            f"Megatron FSDP SDMA allgather enabled via mori.ccl.AllgatherSdma "
            f"(max_numel={max_numel})",
        )
    except Exception as e:  # noqa: BLE001 - best-effort, always fall back to RCCL
        _handle = None
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
    - the per-rank shard fits inside the pre-allocated transit buffer.

    No dtype check is needed: ``AllgatherSdma`` gathers the tensor as a raw byte
    stream, so any dtype is supported.
    """
    if _handle is None:
        return False
    call_group = group if group is not None else torch.distributed.group.WORLD
    if call_group is not _bound_group:
        return False
    if input_tensor.numel() > _max_numel:
        return False
    return True


def _maybe_register_output(output_tensor: torch.Tensor, double_buffer: bool) -> None:
    """Best-effort: register the output bucket with mori's symmetric-memory layer so
    the SDMA kernel can write straight into it instead of copying through the transit
    buffer.

    This is purely an optimization -- the gather still works via the transit-buffer
    copy path (``copy_output_to_user=True``) when registration is unavailable.
    Registration is done once per distinct output buffer pointer and cached; FSDP's
    double buffer reuses a small fixed set of bucket buffers, so this is cheap. Any
    failure latches ``_direct_unavailable`` so we stop trying.
    """
    global _direct_unavailable
    # The per-buffer registration is a cross-rank collective that deadlocks unless every
    # rank registers the same buffers in lockstep, which only holds with the FSDP double
    # buffer on (see ``_direct_write_enabled``). The transit-buffer copy path is correct
    # without it.
    if not _direct_write_enabled(double_buffer):
        return
    if _direct_unavailable:
        return
    ptr = output_tensor.data_ptr()
    if ptr in _registered_output_ptrs:
        return
    try:
        if not _handle.is_output_registered(output_tensor):
            _handle.register_output_buffer(output_tensor)
        _registered_output_ptrs.add(ptr)
    except Exception as e:  # noqa: BLE001 - registration is optional; keep gathering
        _direct_unavailable = True
        log_single_rank(
            logger,
            logging.INFO,
            f"Megatron FSDP SDMA output registration unavailable "
            f"({type(e).__name__}: {e}); using the transit-buffer copy path",
        )


def allgather_into_tensor(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    group=None,
    max_numel: int = _DEFAULT_MAX_NUMEL,
    serialize_after_stream: Optional["torch.cuda.Stream"] = None,
    double_buffer: bool = False,
) -> Optional[_SdmaWork]:
    """Issue one all_gather_into_tensor through the mori ``AllgatherSdma`` handle.

    The gather is driven on a dedicated SDMA stream via the async put/wait kernels
    (``start_async`` + ``wait_async``), both enqueued on that stream so they overlap
    prefetch-window compute on the main stream; a CUDA event captures completion and
    ``_SdmaWork.wait()`` defers to a device-side ``wait_event``. The output bucket is
    registered with mori's symmetric-memory layer when possible so the kernel can
    write into it directly.

    Lazily initialises (bound to ``group``) on the first call. Returns an
    ``_SdmaWork`` (Work-compatible) on success, or ``None`` when SDMA is not
    applicable (uninitialised, a group other than the bound group, or shard larger
    than the transit buffer) or the call fails -- the caller then falls back to
    ``torch.distributed.all_gather_into_tensor``.
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

        # Best-effort direct-write registration of the output bucket (optional). Only
        # safe with lockstep buffers (FSDP double buffer on); see _direct_write_enabled.
        _maybe_register_output(output_tensor, double_buffer)

        count = input_tensor.numel()
        # Enqueue the async put + cross-rank wait kernels on the dedicated SDMA
        # stream. They run in order on this stream and overlap compute on the main
        # stream; the recorded event signals completion to _SdmaWork.wait().
        if not _handle.start_async(
            input_tensor, output_tensor, count, sdma_stream.cuda_stream
        ):
            return None
        _handle.wait_async(sdma_stream.cuda_stream)

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
