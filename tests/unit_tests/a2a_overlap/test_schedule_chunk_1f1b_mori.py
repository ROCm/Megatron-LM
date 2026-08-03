# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_chunk_1f1b import (
    TestA2AOverlap as _ChunkScheduleTests,
)
from tests.unit_tests.a2a_overlap.utils import (
    is_mori_available,
    reinitialize_model_parallel_for_mori,
)
from tests.unit_tests.test_utilities import Utils


class TestA2AOverlapMori:
    """Run process-scoped MORI coverage in a fresh torchrun invocation."""

    def teardown_method(self, method):
        # Keep process-scoped MORI symmetric allocations alive while TP changes.
        Utils.destroy_model_parallel()

    @classmethod
    def teardown_class(cls):
        from megatron.core.transformer.moe.fused_a2a import finalize_mori_shmem

        finalize_mori_shmem()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.skipif(not is_mori_available(), reason="MORI is not available")
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_1f1b_schedule_model_chunk_mori(self, tp_size):
        ep_size = reinitialize_model_parallel_for_mori(tp_size)
        for use_padding_mask in (False, True):
            for layers in ([2, 1], [1, 1]):
                _ChunkScheduleTests.run_1f1b_schedule_model_chunk_mori(
                    layers, use_padding_mask, tp_size, ep_size
                )
