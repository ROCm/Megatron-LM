# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MORI lifecycle fixtures shared by the MORI unit-test conftests.

Follows the harness used by MORI's own test suite; see
``docs/developer/mori_test_lifecycle.md`` for the rationale behind each fixture.
Import these names into a directory's ``conftest.py`` to activate them there.
"""

import faulthandler
import os
import sys

import pytest
import torch

from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.fused_a2a import HAVE_MORI, finalize_mori_shmem, reset_mori_op

# Read by MORI when shmem is initialized and when each op is built, so they must be
# in place before the first MORI dispatch of the session. The heap size is MORI's
# static-heap default; pinning it keeps a stale value inherited from the shell or
# runner from shrinking the heap. MANUAL is MORI's own default launch mode and the
# only one its test suite exercises.
MORI_TEST_ENV = {"MORI_SHMEM_HEAP_SIZE": "4G", "MORI_EP_LAUNCH_CONFIG_MODE": "MANUAL"}

MORI_TEST_TIMEOUT_ENV = "MORI_TEST_TIMEOUT"
DEFAULT_MORI_TEST_TIMEOUT_S = 600.0


def drain_and_reset_mori_op():
    """Quiesce every rank's MORI work, then release the cached op on all ranks.

    The op's symmetric buffers are freed by its C++ destructor when the reference is
    dropped, and the heap hands out addresses deterministically, so every rank must
    release at the same point. The barrier makes that explicit instead of relying on
    the next op constructor's barrier.
    """
    if not HAVE_MORI or fused_a2a._mori_op is None:
        return
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    reset_mori_op()


@pytest.fixture(scope="session", autouse=True)
def mori_session_env():
    """Pin MORI's environment for the whole session and restore it afterwards."""
    saved = {name: os.environ.get(name) for name in MORI_TEST_ENV}
    os.environ.update(MORI_TEST_ENV)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def mori_hang_watchdog(request):
    """Fail a hung MORI test fast, with a stack dump from every rank.

    MORI kernels spin-wait on peers with no in-kernel watchdog, so a protocol hang
    never raises; without a bound it only ends at the runner's per-file timeout.
    Only armed in dedicated ``*_mori`` modules. Override the budget in seconds with
    ``MORI_TEST_TIMEOUT``.
    """
    if not HAVE_MORI or not request.module.__name__.endswith("_mori"):
        yield
        return
    budget = float(os.environ.get(MORI_TEST_TIMEOUT_ENV) or DEFAULT_MORI_TEST_TIMEOUT_S)
    faulthandler.dump_traceback_later(budget, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session", autouse=True)
def mori_session_teardown(cleanup):
    """Finalize MORI shmem once, before ``cleanup`` destroys the default process group."""
    yield
    if HAVE_MORI:
        drain_and_reset_mori_op()
        finalize_mori_shmem()
