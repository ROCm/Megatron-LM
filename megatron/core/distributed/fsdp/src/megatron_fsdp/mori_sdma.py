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

# Safety cap on the number of distinct output buffers we register per dtype. With
# ``--fsdp-double-buffer`` the all-gather outputs come from a small fixed pool (a handful of
# buffers), so this is reached only during warmup. If it is exceeded (e.g. a dynamic allocator is
# in use and output addresses change every iteration) we stop registering new buffers and fall
# back rather than registering unboundedly. All of these buffers are registered on the *single*
# per-dtype handle (see ``MoriSdmaAllGather``); MORI supports many registered output buffers per
# handle, but multiple concurrent handles each owning their own buffer corrupt each other's gathers.
_MAX_REGISTERED_OUTPUTS = 64

# Minimal transit-buffer size (bytes) used in no-copy (registered-output) mode. The SDMA async PUT
# kernel reads the input shard directly and writes into the peer's *registered* output buffer, so
# neither the input nor the output transit buffer is used in this mode. Sizing them to a few bytes
# (instead of ``shard_bytes`` / ``shard_bytes * npes``) reclaims the otherwise-wasted symmetric VRAM.
_MIN_TRANSIT_BYTES = 4


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

    Two construction modes:

    - **eager** (``event`` given): the full gather (PUT + WAIT [+ copy]) was already enqueued at
      issue time; ``wait()`` just makes the consuming stream wait on the recorded event. Used by the
      copy-output path.
    - **deferred** (``owner`` given): only the PUT was enqueued at issue time. The SDMA WAIT kernel
      is enqueued lazily by ``owner._flush_pending(...)`` -- either when the next gather is issued on
      the (single, shared) handle, or here the first time ``wait()`` is called, whichever comes
      first. Enqueuing the WAIT lazily keeps the long-running WAIT kernel off the all-gather stream
      during the prefetch window, where it otherwise pins the stream and contends for XGMI
      bandwidth. The flush records this work's completion ``event``; ``wait()`` then makes the
      consuming stream wait on it. ``_flush_pending`` is idempotent via the ``_flushed`` flag, so it
      is safe whether the WAIT was already enqueued at the next issue or not.
    """

    def __init__(
        self,
        event: Optional[torch.cuda.Event] = None,
        owner: "Optional[MoriSdmaAllGather]" = None,
    ):
        self.event = event
        self._owner = owner
        self._flushed = event is not None

    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Make the current (or given) stream wait on the gather-completion event (device-side)."""
        if not self._flushed and self._owner is not None:
            # Deferred path: enqueue the WAIT kernel now (records ``self.event`` on the AG stream).
            self._owner._flush_pending(self)
        if self.event is None:
            return
        if stream is None:
            self.event.wait()
        else:
            self.event.wait(stream)


class MoriSdmaAllGather:
    """
    MORI SDMA backend for the parameter all-gather.

    A single instance is created per ``AllGatherPipeline`` and shared across all buckets. There is
    exactly **one** MORI ``AllgatherSdma`` handle per dtype (in both copy and no-copy modes), sized
    to the largest per-rank shard for the dtype (pre-sized via ``presize`` so it does not grow
    mid-run). In no-copy (registered-output) mode every double-buffer output buffer is registered on
    that single per-dtype handle: MORI supports many registered output buffers per handle, but
    multiple *concurrent* handles corrupt each other's gathers (verified by
    ``scripts/mori_nocopy_allgather_check.py``: two buffers on one handle pass, two handles fail).

    Because a MORI handle holds only one outstanding async op, no-copy gathers are kept to one in
    flight at a time: when a new gather is issued, the previous gather's WAIT is flushed first
    (enqueued on the all-gather stream, recording its completion event). The WAIT therefore still
    lives in the ``_EventWork`` event -- it fires lazily either at the next issue or at
    ``Work.wait()``, whichever comes first -- but is never concurrent across handles. All
    collectives run over ``ag_process_group``.
    """

    def __init__(self, ag_process_group: dist.ProcessGroup, register_output: bool = False):
        self.group = ag_process_group
        self.active = False
        self.my_pe = 0
        self.npes = 1
        # One handle per dtype (both modes). ``_handle_capacity`` maps dtype -> the per-rank shard
        # byte size the handle was built for (used to detect when a larger shard forces a re-create).
        self._handles: Dict[torch.dtype, "object"] = {}
        self._handle_capacity: Dict[torch.dtype, int] = {}
        # dtype -> largest per-rank shard element count seen / pre-sized. Handles are built to this
        # maximum so they never need to grow (which would drop output-buffer registrations).
        self._max_shard_numel: Dict[torch.dtype, int] = {}
        # Track which fallback reasons have already been warned about (warn once each).
        self._warned: set = set()
        # The single in-flight no-copy gather whose WAIT has not yet been enqueued, plus the handle
        # and stream needed to flush it. At most one exists at a time (see class docstring); it is
        # flushed before the next gather is issued or at its own ``Work.wait()``.
        self._pending_work: Optional[_EventWork] = None
        self._pending_handle: "object" = None
        self._pending_stream: Optional[torch.cuda.Stream] = None

        # "No-copy" mode: register the all-gather output buffers with MORI so SDMA writes the
        # gathered result directly into them, eliminating the transit->user DtoD copy that the
        # default copy_output_to_user=True path performs. This requires the output buffers to have
        # stable addresses, which only the fixed-pool double buffer (FixedPoolAllocator) provides,
        # so it is enabled exclusively when ``fsdp_double_buffer`` is set and never otherwise.
        # Registration is a collective over the all-gather group, so it must happen in the same
        # order on every rank; the FSDP bucket-gather order is deterministic and identical across
        # ranks, so registering lazily on first sight of each output buffer stays in sync. With the
        # fixed pool this registers only ``2 x buckets_per_fsdp_unit`` buffers per dtype (just 2 in
        # the common single-bucket-per-unit case), all during warmup.
        self.register_output = register_output
        # dtype -> set of already-registered output data_ptrs (capped by _MAX_REGISTERED_OUTPUTS).
        self._registered_ptrs: Dict[torch.dtype, set] = {}
        # dtype -> {output_ptr: registered output tensor}. Kept so every buffer registered on the
        # per-dtype handle can be deregistered before that handle is torn down / re-created (avoids
        # orphaned IPC mappings).
        self._registered_tensors: Dict[torch.dtype, Dict[int, "object"]] = {}

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

    def _make_handle(self, dtype: torch.dtype, shard_numel: int):
        """Construct a fresh ``AllgatherSdma`` handle for ``dtype`` sized to ``shard_numel``.

        In no-copy (registered-output) mode the gathered result is written straight into the
        registered user output buffer and the transit buffers are unused, so they are sized to a
        few bytes. In copy mode the output transit buffer receives the gather and is copied to the
        user buffer, so it must be full-sized.
        """
        from mori.ccl import AllgatherSdma

        element_size = torch.tensor([], dtype=dtype).element_size()
        shard_bytes = shard_numel * element_size
        if self.register_output:
            input_buffer_size = _MIN_TRANSIT_BYTES
            output_buffer_size = _MIN_TRANSIT_BYTES
        else:
            input_buffer_size = shard_bytes
            output_buffer_size = shard_bytes * self.npes
        return AllgatherSdma(
            my_pe=self.my_pe,
            npes=self.npes,
            input_buffer_size=input_buffer_size,
            output_buffer_size=output_buffer_size,
            copy_output_to_user=not self.register_output,
            dtype=dtype,
        )

    def _get_handle(self, dtype: torch.dtype, shard_numel: int):
        """Get (or lazily (re)create) the single AllgatherSdma handle for ``dtype``.

        The same handle is shared by every output buffer of this dtype (no-copy mode registers them
        all on it). It is built to the per-dtype maximum shard (``_max_shard_numel``, populated by
        ``presize``) so it does not need to grow mid-run -- a re-create would tear down the transit
        buffers and drop output-buffer registrations.
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        # Build to the largest shard seen / pre-sized for this dtype so the handle is stable.
        target_shard = max(shard_numel, self._max_shard_numel.get(dtype, 0))
        target_bytes = target_shard * element_size

        capacity = self._handle_capacity.get(dtype, 0)
        if dtype not in self._handles or target_bytes > capacity:
            # Deregister-before-realloc: if we are replacing an existing handle that owns registered
            # buffers, deregister them first so their IPC mappings are not orphaned.
            if dtype in self._handles:
                self._teardown_handle(dtype)
            handle = self._make_handle(dtype, target_shard)
            self._handles[dtype] = handle
            self._handle_capacity[dtype] = target_bytes
        return self._handles[dtype]

    def _teardown_handle(self, dtype: torch.dtype) -> None:
        """Deregister every output buffer owned by ``dtype``'s handle and drop its registrations."""
        handle = self._handles.get(dtype)
        tensors = self._registered_tensors.pop(dtype, {})
        if handle is not None:
            for tensor in tensors.values():
                try:
                    handle.deregister_output_buffer(tensor)
                except Exception as exc:  # pragma: no cover - environment dependent
                    self._warn_once(
                        "dereg_failed",
                        f"MORI SDMA deregister_output_buffer failed during handle re-create ({exc}).",
                    )
        self._registered_ptrs.pop(dtype, None)

    def _flush_pending(self, work: "Optional[_EventWork]" = None) -> None:
        """Enqueue the pending no-copy gather's WAIT kernel and record its completion event.

        Called before issuing the next gather (so the single per-dtype handle has no outstanding op)
        and from ``_EventWork.wait()`` (whichever happens first). Idempotent: if ``work`` is given
        and it is not the current pending work, it was already flushed and this is a no-op.
        """
        pending = self._pending_work
        if pending is None:
            return
        if work is not None and work is not pending:
            return
        # blocking=False only enqueues the WAIT kernel on the AG stream (no host sync).
        self._pending_handle.wait_async(stream=self._pending_stream, blocking=True)
        pending.event = torch.cuda.Event()
        pending.event.record(self._pending_stream)
        pending._flushed = True
        self._pending_work = None
        self._pending_handle = None
        self._pending_stream = None

    def presize(self, full_bucket_numel_by_dtype: Dict[torch.dtype, int]) -> None:
        """Record the largest per-dtype shard (and, in copy mode, pre-build the handle) up front.

        ``_get_handle`` otherwise grows (re-creates) the handle the first time a larger shard
        appears, which tears down the transit buffers, rebuilds them, and drops any output-buffer
        registrations - forcing a re-registration collective. Recording the maximum shard here (and
        sizing the per-dtype handle to it) makes the handle stable for the whole run. The handle is
        also constructed now to preallocate its transit buffers; in no-copy mode the output buffers
        are still registered lazily on first gather (their addresses are not known until then).

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
            # Record the max shard so the per-dtype handle is always built to this size and never
            # has to grow mid-run (which would drop output-buffer registrations).
            self._max_shard_numel[dtype] = max(self._max_shard_numel.get(dtype, 0), shard_numel)
            try:
                self._get_handle(dtype, shard_numel)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._warn_once(
                    "presize_failed",
                    f"MORI SDMA handle pre-sizing failed for dtype {dtype} ({exc}); "
                    "the handle will be built lazily on first use instead.",
                )

    def _ensure_output_registered(self, handle, dtype: torch.dtype, output_tensor) -> bool:
        """Register ``output_tensor`` with ``handle`` for direct (no-copy) SDMA writes.

        Returns True if the output is registered (now or previously), False if it could not be
        registered (cap exceeded or registration error) so the caller can fall back. Registration
        is collective across the all-gather group and is performed at most once per distinct buffer.
        """
        regset = self._registered_ptrs.setdefault(dtype, set())
        ptr = output_tensor.data_ptr()
        if ptr in regset:
            return True
        if len(regset) >= _MAX_REGISTERED_OUTPUTS:
            self._warn_once(
                "reg_cap",
                f"MORI SDMA exceeded {_MAX_REGISTERED_OUTPUTS} registered output buffers for "
                f"dtype {dtype} (output addresses are not stable - is --fsdp-double-buffer set?); "
                "falling back to torch.distributed.all_gather_into_tensor for new buffers.",
            )
            return False
        try:
            handle.register_output_buffer(output_tensor)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._warn_once(
                "reg_failed",
                f"MORI SDMA register_output_buffer failed ({exc}); "
                "falling back to torch.distributed.all_gather_into_tensor.",
            )
            return False
        regset.add(ptr)
        # Keep a reference to the registered tensor so the buffer can be deregistered from this
        # handle if it is ever torn down / re-created (see ``_teardown_handle``).
        self._registered_tensors.setdefault(dtype, {})[ptr] = output_tensor
        # One-time (per buffer) rank-0 log so the fixed pool registering exactly once during warmup
        # is observable, and a regression that starts churning addresses is visible as a growing count.
        if self.my_pe == 0:
            nbytes = output_tensor.numel() * output_tensor.element_size()
            logger.info(
                "MORI SDMA registered output buffer #%d for dtype %s (%.1f MiB) for no-copy gather.",
                len(regset),
                dtype,
                nbytes / (1024 * 1024),
            )
        return True

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

        # Flush any in-flight no-copy gather first: the single per-dtype handle holds one
        # outstanding op, and multiple concurrent handles corrupt each other, so only one SDMA
        # gather may be live at a time. This enqueues the previous gather's WAIT on the AG stream.
        self._flush_pending()

        try:
            if self.register_output:
                # No-copy mode: the single per-dtype handle, with this output buffer registered so
                # SDMA writes directly into it (skipping the transit->user copy).
                handle = self._get_handle(dtype, shard_numel)
                # If it cannot be registered, fall back: with copy_output_to_user=False an
                # unregistered output would not receive the result.
                if not self._ensure_output_registered(handle, dtype, output_tensor):
                    return None
                # Enqueue only the PUT (input -> peers' registered output buffers) now. The WAIT
                # kernel is deferred -- flushed either at the next issue (above) or at
                # ``_EventWork.wait()`` -- so it does not pin the all-gather stream during prefetch.
                handle.start_async(input_tensor, output_tensor, shard_numel, stream=stream)
                work = _EventWork(owner=self)
                self._pending_work = work
                self._pending_handle = handle
                self._pending_stream = stream
                return work

            # Copy mode: the per-dtype handle. Enqueue PUT then the WAIT kernel + transit->user copy
            # back-to-back; blocking=False only enqueues GPU work (no host sync).
            handle = self._get_handle(dtype, shard_numel)
            handle.start_async(input_tensor, output_tensor, shard_numel, stream=stream)
            handle.wait_async(stream=stream, blocking=True)
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
        return _EventWork(event=event)
