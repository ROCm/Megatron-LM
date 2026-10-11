# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Session-scoped MORI cleanup for the a2a-overlap unit tests.

The MORI overlap tests finalize symmetric memory in their per-class
``teardown_class`` only. That teardown is skipped whenever a test process aborts
(NCCL/RCCL watchdog SIGABRT) or is killed on its per-file timeout, which leaves
MORI symmetric memory and the registered ``mori_ep`` process-group handle
un-finalized. The next torchrun/retry then re-enters that wedged MORI state and
hangs at ``shmem_torch_process_group_init`` -- reproducing as a persistent
per-file TIMEOUT in CI while every isolated run passes.

Mirror ``tests/unit_tests/transformer/moe/conftest.py``: finalize MORI once at
session teardown, ordered (via the ``cleanup`` dependency) to run before the
parent session fixture destroys the default process group.
"""

import pytest

from megatron.core.transformer.moe.fused_a2a import HAVE_MORI, finalize_mori_shmem, reset_mori_op


@pytest.fixture(autouse=True)
def mori_op_reset():
    """Drop the per-test MORI dispatch/combine op between cases.

    Every a2a-overlap test also resets in its own ``teardown_method``; this is a
    centralized safety net for a test whose ``setup_method`` raises (so its
    ``teardown_method`` never runs) and for any future file that omits the reset.
    Function-scoped, so it fires only after the whole test -- never between the
    intra-test data variants that deliberately share one op (PR #150). Never
    finalizes shmem: that is process-scoped and must not be reinitialized.
    """
    yield
    if HAVE_MORI:
        reset_mori_op()


@pytest.fixture(scope="session", autouse=True)
def mori_session_teardown(cleanup):
    """Finalize MORI once, before the parent session fixture destroys the default group."""
    yield
    if HAVE_MORI:
        finalize_mori_shmem()
