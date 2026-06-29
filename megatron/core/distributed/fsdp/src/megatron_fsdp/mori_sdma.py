# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
MORI SDMA all-gather backend for Megatron-FSDP.

This module provides an optional, self-contained backend that performs the parameter
all-gather using MORI's SDMA copy engines instead of ``torch.distributed.all_gather_into_tensor``.
Offloading the all-gather to the GPU DMA engines frees compute units (CUs/SMs) for overlapping
compute.

Key properties / constraints (see MORI source):
  - SDMA one-shot all-gather is *intra-node only*: the kernels write directly into peer GPUs'
    symmetric memory via peer pointers (XGMI/PCIe P2P). It cannot span nodes.
  - Output is produced in copy mode (``copy_output_to_user=True``), so it works with arbitrary
    (PyTorch-cached) bucket tensors and is byte-exact.
  - Completion is fully asynchronous: ``start_async`` + ``wait_async(blocking=False)`` only enqueue
    GPU work on the all-gather stream; the host never blocks. A ``torch.cuda.Event`` recorded on
    that stream bridges the cross-stream dependency to the consuming compute stream, matching the
    device-side semantics of ``torch.distributed`` ``Work.wait()``.

Whenever SDMA cannot apply (multi-node all-gather group, oversized bucket vs. the preallocated
transit buffer, or an unsupported dtype), ``all_gather_into_tensor`` returns ``None`` to signal the
caller to fall back to ``torch.distributed.all_gather_into_tensor``, and a one-time warning is
emitted.
"""

import logging
import os
import socket
from typing import Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# MORI's SHMEM context is a process-global singleton; it can only be initialized once per process.
_SHMEM_INITIALIZED = False
# A stable name used to register the all-gather process group for the SHMEM bootstrap.
_SHMEM_PG_NAME = "megatron_fsdp_mori_sdma_ag"

# torch dtypes supported by MORI's typed AllgatherSdma handles.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.uint32)


def is_mori_available() -> bool:
    """Return True if the ``mori`` package (with the SDMA all-gather API) is importable."""
    try:
        import mori  # noqa: F401
        from mori.ccl import AllgatherSdma  # noqa: F401
        import mori.shmem  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


class _EventWork:
    """
    Minimal stand-in for a ``torch.distributed`` ``Work`` handle, backed by a CUDA event.

    The Megatron-FSDP all-gather pipeline stores work handles in ``param_gather_event_map`` and
    later calls ``.wait()`` on them when the bucket is about to be consumed. ``Work.wait()`` for an
    async collective enqueues a *device-side* wait on the current (consuming) stream rather than
    blocking the host. ``torch.cuda.Event.wait()`` has identical semantics, so this wrapper is a
    drop-in replacement.
    """

    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Make the current (or given) stream wait on the gather-completion event (device-side)."""
        if stream is None:
            self.event.wait()
        else:
            self.event.wait(stream)


class MoriSdmaAllGather:
    """
    MORI SDMA backend for the parameter all-gather.

    A single instance is created per ``AllGatherPipeline`` and shared across all buckets. MORI
    ``AllgatherSdma`` handles are created lazily per dtype and sized to the largest per-rank shard
    seen so far (re-created if a larger shard appears). All collectives run over ``ag_process_group``.
    """

    def __init__(self, ag_process_group: dist.ProcessGroup):
        self.group = ag_process_group
        self.active = False
        self.my_pe = 0
        self.npes = 1
        # dtype -> (handle, capacity_in_bytes_per_rank)
        self._handles: Dict[torch.dtype, "object"] = {}
        self._handle_capacity: Dict[torch.dtype, int] = {}
        # Track which fallback reasons have already been warned about (warn once each).
        self._warned: set = set()

        if not is_mori_available():
            self._warn_once(
                "unavailable",
                "MORI SDMA all-gather requested but the `mori` package is not importable; "
                "falling back to torch.distributed.all_gather_into_tensor.",
            )
            return

        if not self._is_single_node():
            self._warn_once(
                "multinode",
                "MORI SDMA all-gather is intra-node only, but the all-gather process group spans "
                "multiple nodes; falling back to torch.distributed.all_gather_into_tensor.",
            )
            return

        # SDMA must be explicitly enabled for MORI's CCL handles before construction.
        os.environ["MORI_ENABLE_SDMA"] = "1"

        try:
            self._init_shmem()
            self.active = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._warn_once(
                "init_failed",
                f"Failed to initialize MORI SHMEM for SDMA all-gather ({exc}); "
                "falling back to torch.distributed.all_gather_into_tensor.",
            )
            self.active = False

    def _is_single_node(self) -> bool:
        """Return True iff every rank in the all-gather group is on the same host."""
        world_size = dist.get_world_size(self.group)
        if world_size <= 1:
            return True
        hostname = socket.gethostname()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, hostname, group=self.group)
        return len(set(gathered)) == 1

    def _init_shmem(self) -> None:
        """Bootstrap MORI's (process-global) SHMEM context over the all-gather group, once."""
        global _SHMEM_INITIALIZED
        import mori.shmem as shmem

        if not _SHMEM_INITIALIZED:
            # Register the all-gather process group so MORI can bootstrap SHMEM from it.
            torch._C._distributed_c10d._register_process_group(_SHMEM_PG_NAME, self.group)
            shmem.shmem_torch_process_group_init(_SHMEM_PG_NAME)
            _SHMEM_INITIALIZED = True

        self.my_pe = shmem.shmem_mype()
        self.npes = shmem.shmem_npes()

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    def _get_handle(self, dtype: torch.dtype, shard_numel: int):
        """Get (or lazily (re)create) an AllgatherSdma handle for ``dtype`` sized to ``shard_numel``."""
        from mori.ccl import AllgatherSdma

        element_size = torch.tensor([], dtype=dtype).element_size()
        shard_bytes = shard_numel * element_size

        capacity = self._handle_capacity.get(dtype, 0)
        if dtype not in self._handles or shard_bytes > capacity:
            # (Re)create the handle sized to the largest shard seen so far for this dtype.
            handle = AllgatherSdma(
                my_pe=self.my_pe,
                npes=self.npes,
                input_buffer_size=shard_bytes,
                output_buffer_size=shard_bytes * self.npes,
                copy_output_to_user=True,
                dtype=dtype,
            )
            self._handles[dtype] = handle
            self._handle_capacity[dtype] = shard_bytes
        return self._handles[dtype]

    def presize(self, full_bucket_numel_by_dtype: Dict[torch.dtype, int]) -> None:
        """Pre-build the per-dtype handles sized to the largest bucket up front.

        ``_get_handle`` otherwise grows (re-creates) the handle the first time a larger shard
        appears, which tears down the (multi-hundred-MB) transit buffers and rebuilds bigger ones.
        Sizing each handle to the maximum bucket here makes the handle stable for the whole run.

        ``full_bucket_numel_by_dtype`` maps dtype -> largest *unsharded* bucket element count. That
        value is identical on every rank (it is the full bucket size, not the per-rank shard), so
        the underlying symmetric allocations are built collectively with matching sizes. We process
        dtypes in a stable (sorted) order for the same reason.
        """
        if not self.active:
            return
        for dtype in sorted(full_bucket_numel_by_dtype, key=str):
            if dtype not in _SUPPORTED_DTYPES:
                continue
            full_numel = full_bucket_numel_by_dtype[dtype]
            if not full_numel or full_numel <= 0:
                continue
            # Upper bound on the per-rank shard (ceil-divide); >= every rank's actual shard.
            shard_numel = (full_numel + self.npes - 1) // self.npes
            try:
                self._get_handle(dtype, shard_numel)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._warn_once(
                    "presize_failed",
                    f"MORI SDMA handle pre-sizing failed for dtype {dtype} ({exc}); "
                    "handles will be built lazily on first use instead.",
                )

    @torch.no_grad()
    def all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> Optional[_EventWork]:
        """
        Issue an asynchronous SDMA all-gather of ``input_tensor`` (the local shard) into
        ``output_tensor`` (the full unsharded bucket) on ``stream``.

        Returns an ``_EventWork`` whose ``.wait()`` makes the consuming stream wait on completion,
        or ``None`` to signal the caller to fall back to ``torch.distributed.all_gather_into_tensor``
        (when SDMA is inactive, the dtype is unsupported, or the shard is too large to fit the
        transit buffer at a reasonable size).
        """
        if not self.active:
            return None

        dtype = input_tensor.dtype
        if dtype not in _SUPPORTED_DTYPES:
            self._warn_once(
                f"dtype_{dtype}",
                f"MORI SDMA all-gather does not support dtype {dtype}; "
                "falling back to torch.distributed.all_gather_into_tensor for these buckets.",
            )
            return None

        # The all-gather must produce npes copies of the per-rank shard.
        shard_numel = input_tensor.numel()
        if output_tensor.numel() != shard_numel * self.npes:
            self._warn_once(
                "shape_mismatch",
                "MORI SDMA all-gather expected output.numel == input.numel * npes; "
                "falling back to torch.distributed.all_gather_into_tensor.",
            )
            return None

        try:
            handle = self._get_handle(dtype, shard_numel)
            # Enqueue PUT (input -> peers' transit buffers) on the all-gather stream.
            handle.start_async(input_tensor, output_tensor, shard_numel, stream=stream)
            # Enqueue the wait kernel and the transit->user copy on the same stream;
            # blocking=False means the host does NOT synchronize - only GPU work is enqueued.
            handle.wait_async(stream=stream, blocking=False)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._warn_once(
                "launch_failed",
                f"MORI SDMA all-gather launch failed ({exc}); "
                "falling back to torch.distributed.all_gather_into_tensor.",
            )
            return None

        # Record completion of the full gather sequence (put -> wait -> copy) on the stream.
        event = torch.cuda.Event()
        event.record(stream)
        return _EventWork(event)
