# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# ---------------------------------------------------------------------------
# MORI early bootstrap (must run before any megatron / TransformerEngine
# import below). Mirrors `pretrain_gpt.py:21-84`.
#
# THIS IS A WORKAROUND, NOT A ROOT-CAUSE FIX. The proper fix lives
# upstream in MORI (or in how MORI drives HIP relative to a primary
# context that already has loaded modules + allocations). Until that
# lands, we sidestep the bug here by ordering imports.
#
# What we know empirically:
#   * Symptom: glibc `free(): invalid size` SIGABRT, with no Python
#     exception — the log just shows `Fatal Python error: Aborted`.
#   * Crash site: `mori_cpp.load_shmem_module(hsaco)` called from
#     `mori/shmem/api.py:39 _ensure_shmem_module`, which is in turn
#     called from `shmem_torch_process_group_init`.
#   * Trigger: TransformerEngine has already been imported before that
#     point (which happens transitively via any `from megatron.core ...`
#     below this block, since the MoE / attention modules pull in TE).
#   * Workaround: init MORI shmem first, while HIP's module table and
#     allocator are still untouched by TE. No crash.
#
# What we DO NOT know (would need ASAN/Valgrind on the failing process,
# or a MORI source review, to nail down):
#   * Which MORI-side buffer is being overrun.
#   * Whether the bug is in MORI's hsaco-blob handling, its hipModuleLoad
#     wrapper, the `__device__ globalGpuStates` symbol-patching path, or
#     glibc cleanup along an internal MORI error edge.
#
# The function is gated on:
#   1. Being launched under `torchrun` (RANK / WORLD_SIZE / LOCAL_RANK set),
#      so plain `pytest` invocations stay no-op.
#   2. The `mori` package being importable.
#   3. The argv mentioning "mori" or "test_token_dispatcher" — cheap heuristic
#      that avoids burning ~32 GiB of symmetric heap on test runs that
#      don't exercise the MORI backend.
# ---------------------------------------------------------------------------
def _mori_early_bootstrap_for_pytest():
    import os as _os
    import sys as _sys

    if not all(k in _os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return
    _argv_str = " ".join(_sys.argv).lower()
    if "mori" not in _argv_str and "test_token_dispatcher" not in _argv_str:
        return

    try:
        import torch as _torch
        import torch.distributed as _dist
    except ImportError:
        return
    try:
        import mori.shmem as _mori_shmem
    except ImportError:
        return

    # Match the production env (`examples/qwen3/train_qwen3.sh:432-433`):
    # `vmm_heap` switches MORI's symmetric-heap allocator from
    # `hipExtMallocWithFlags` to the VMM path (`hipMemAddressReserve` +
    # `hipMemCreate` + `hipMemMap`). VMM is also what HSA uses internally
    # for primary-context-owned allocations, so the heap shares a single
    # virtual-address namespace with TE/torch's allocations and avoids the
    # double-mmap collision that the old `hipExtMallocWithFlags` path was
    # prone to. `setdefault` lets the user override on the command line.
    # _os.environ.setdefault("MORI_SHMEM_MODE", "vmm_heap")
    _os.environ.setdefault("MORI_SHMEM_LOG_LEVEL", "INFO")

    rank = int(_os.environ["RANK"])
    local_rank = int(_os.environ["LOCAL_RANK"])
    world_size = int(_os.environ["WORLD_SIZE"])

    _torch.cuda.set_device(local_rank)
    device = _torch.device(f"cuda:{local_rank}")

    if not _dist.is_initialized():
        _dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

    mori_group = _dist.new_group(list(range(world_size)), backend="gloo")
    _torch._C._distributed_c10d._register_process_group("mori", mori_group)

    try:
        _mori_shmem.shmem_torch_process_group_init("mori")
    except Exception:
        try:
            _mori_shmem.shmem_finalize()
        except Exception:
            pass
        _dist.destroy_process_group()
        raise

    # Signal to test code (test_token_dispatcher.py::_ensure_mori_shmem) that
    # it should skip its own bootstrap and treat shmem as already-initialized.
    _os.environ["MEGATRON_MORI_SHMEM_BOOTSTRAPPED"] = "1"

    if rank == 0:
        print(
            f"[MORI BOOTSTRAP] shmem init OK"
            f" (world_size={world_size}"
            f", MORI_SHMEM_MODE={_os.environ.get('MORI_SHMEM_MODE')}"
            f", MORI_SHMEM_HEAP_SIZE={_os.environ.get('MORI_SHMEM_HEAP_SIZE')})",
            flush=True,
        )


_mori_early_bootstrap_for_pytest()
del _mori_early_bootstrap_for_pytest

import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core import config
from megatron.core.utils import is_te_min_version
from tests.test_utils.python_scripts.download_unit_tests_dataset import download_and_extract_asset
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --experimental: Enable the Mcore experimental flag (DEFAULT: False)
    """
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


@pytest.fixture(autouse=True)
def experimental(request):
    """Simple fixture setting the experimental flag [CPU | GPU]"""
    config.ENABLE_EXPERIMENTAL = request.config.getoption("--experimental") is True


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def set_env():
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    # Ensure directory exists for all ranks
    os.makedirs(tmp_dir, exist_ok=True)
    
    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    else:
        yield tmp_dir
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available at /opt/data by downloading if necessary."""
    data_path = Path("/opt/data")

    # Check if data directory exists and has content
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Test data not found at /opt/data. Downloading...")

        try:
            # Download assets to /opt/data
            download_and_extract_asset(assets_dir=str(data_path))

            print("Test data downloaded successfully.")

        except ImportError as e:
            print(f"Failed to import download function: {e}")
            # Don't fail the tests, just warn
        except Exception as e:
            print(f"Failed to download test data: {e}")
            # Don't fail the tests, just warn
    else:
        print("Test data already available at /opt/data")


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)
