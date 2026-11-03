# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MORI lifecycle for the a2a-overlap unit tests.

See ``docs/developer/mori_test_lifecycle.md``: shmem is initialized once per process
and finalized once at session end, the cached op is released after every test, and
dedicated ``*_mori`` modules run under a hang watchdog.
"""

import pytest

from tests.unit_tests.mori_fixtures import (  # noqa: F401
    drain_and_reset_mori_op,
    mori_hang_watchdog,
    mori_session_env,
    mori_session_teardown,
)


@pytest.fixture(autouse=True)
def mori_op_reset():
    """Release the per-test MORI op on every rank once the whole test has finished.

    Function-scoped, so the intra-test data variants that deliberately share one op
    keep sharing it.
    """
    yield
    drain_and_reset_mori_op()
