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
all-gather pipeline routes ``all_gather_into_tensor`` through
``mori.ccl.AllGatherIntoTensor`` (intra-node SDMA copy on ROCm).
Any failure (mori missing, non-AMD/ROCm runtime, shmem init error, oversized
call, unsupported dtype, or a group other than the one bound at init time)
yields ``None`` and the caller falls back to the underlying RCCL/NCCL
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

try:
    # Default to Megatron-LM FW.
    from megatron.core.utils import log_single_rank
except ImportError:
    # Megatron-LM is not installed, use Megatron-FSDP as a standalone module.
    def log_single_rank(
        logger_: logging.Logger, level: int, msg: str, *args, rank: int = 0, **kwargs
    ):
        """Fallback log_single_rank when Megatron Core is not available."""
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == rank:
                logger_.log(level, msg, *args, **kwargs)
        else:
            logger_.log(level, msg, *args, **kwargs)

# Module-level lazy state. Populated by ``init`` on the first all-gather call.
_handle = None
_dtype_map = None
_max_numel = 0
_bound_group = None
_init_attempted = False
_call_failed_warned = False

# Name under which the FSDP data-parallel group is registered for mori's shmem
# layer. mori looks the process group up by this label during shmem init.
_PG_NAME = "megatron_fsdp_sdma"

# Default transit buffer size, in elements (matches the DeepSpeed default).
_DEFAULT_MAX_NUMEL = 64 * 1024 * 1024


class _SdmaWork:
    """Duck-type compatible with ``torch.distributed.Work``.

    ``wait()`` issues a stream-level dependency only and does NOT block the
    CPU, mirroring RCCL ``Work.wait()`` semantics. The FSDP prefetch pipeline
    relies on the CPU staying free so the next bucket can be queued ahead of
    time while bucket N is still in flight.
    """

    def __init__(self, event):
        self._event = event

    def wait(self):
        """Make the current stream wait on the SDMA completion event."""
        torch.cuda.current_stream().wait_event(self._event)

    def is_completed(self) -> bool:
        """Return True if the SDMA copy has finished."""
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


def init(group, max_numel: int = _DEFAULT_MAX_NUMEL) -> None:
    """Best-effort, idempotent SDMA handle construction.

    Builds one ``mori.ccl.AllGatherIntoTensor`` (NCCL/RCCL-style C++
    dispatcher) sized for the largest expected per-rank shard, bound to
    ``group``. All subsequent allgather calls on ``group`` reuse this handle.
    Safe to call unconditionally: any failure leaves ``_handle`` unset and logs
    a single rank-0 info line, so callers transparently fall back to RCCL/NCCL.
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
        from mori.ccl import AllGatherIntoTensor

        shmem.shmem_torch_process_group_init(_PG_NAME)
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        # Per-rank input transit buffer must hold the largest shard we'll ever
        # see; output buffer = npes * input. 4 B/element is the SDMA kernel's
        # uint32 lane width.
        input_bytes = max_numel * 4
        _handle = AllGatherIntoTensor(
            my_pe=my_pe,
            npes=npes,
            input_buffer_size=input_bytes,
            output_buffer_size=input_bytes * npes,
            copy_output_to_user=True,
        )
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
    - the input dtype is in ``_dtype_map``.
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


def allgather_into_tensor(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    group=None,
    max_numel: int = _DEFAULT_MAX_NUMEL,
) -> Optional[_SdmaWork]:
    """Run one all_gather_into_tensor through the SDMA handle.

    Lazily initialises (bound to ``group``) on the first call. Returns an
    ``_SdmaWork`` (Work-compatible) on success. Returns ``None`` when SDMA is
    not applicable for this call (uninitialised, group other than the bound
    group, dtype not supported, shard larger than the transit buffer) or the
    call fails for any reason -- the caller falls back to
    ``torch.distributed.all_gather_into_tensor``.
    """
    global _call_failed_warned
    if not _init_attempted:
        init(group, max_numel)
    if not supports(input_tensor, group):
        return None
    try:
        stream = torch.cuda.current_stream()
        dtype = _dtype_map[input_tensor.dtype]
        ok = _handle(
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            input_tensor.numel(),
            dtype,
            stream.cuda_stream,
        )
        if not ok:
            return None
        event = torch.cuda.Event()
        event.record(stream)
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
