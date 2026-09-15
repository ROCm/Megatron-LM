# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified for portability across upstream and ROCm CI environments.

import logging
import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core.transformer.moe.fused_a2a import (
    HAVE_MORI,
    finalize_mori_shmem,
    reset_mori_op,
)
import megatron.core.parallel_state as parallel_state
from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _destroy_tracked_process_groups():
    """Destroy non-default process groups left after model-parallel init."""
    if not torch.distributed.is_initialized():
        return

    group_list = parallel_state._global_process_group_list
    if group_list is None:
        return

    default_pg = torch.distributed.group.WORLD
    pg_map = torch.distributed.distributed_c10d._world.pg_map
    for group in reversed(group_list):
        if group is None or group == default_pg:
            continue
        if pg_map.get(group, None) is None:
            continue
        try:
            torch.distributed.destroy_process_group(group)
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to destroy %s: %s", group, e)

    parallel_state._global_process_group_list = None


def _destroy_model_parallel_with_subgroups():
    """Teardown model parallel state and destroy tracked NCCL subgroups."""
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    if not Utils.inited:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.distributed.barrier()
    _destroy_tracked_process_groups()
    parallel_state.destroy_model_parallel()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    Utils.inited = False


@pytest.fixture(autouse=True)
def moe_model_parallel_teardown(monkeypatch):
    """Ensure MoE unit tests destroy tracked subgroups between topology changes."""
    monkeypatch.setattr(
        Utils,
        "destroy_model_parallel",
        staticmethod(_destroy_model_parallel_with_subgroups),
    )
    yield
    if HAVE_MORI:
        # MORI shmem is process-scoped and cannot be finalized then initialized
        # again safely. Only reset the per-test dispatch/combine operator here.
        reset_mori_op()
    # Safety net: a test that errors before its own teardown would otherwise
    # leak NCCL/RCCL subgroups into the next test.
    if Utils.inited:
        _destroy_model_parallel_with_subgroups()


@pytest.fixture(scope="session", autouse=True)
def mori_session_teardown(cleanup):
    """Finalize MORI once, before the parent session fixture destroys the default group."""
    yield
    if HAVE_MORI:
        finalize_mori_shmem()


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="function", autouse=True)
def set_env():
    """Configure TE env vars for MoE unit tests.

    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` enables TE's cuDSL fused grouped MLP path.
    The kernel additionally requires SM100 (Blackwell), so on H100/A100 CI this
    is a no-op; setting it here means the kernel is picked up automatically when
    Blackwell hardware joins the unit-test matrix.
    """
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_CUTEDSL_FUSED_GROUPED_MLP'] = '1'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir
