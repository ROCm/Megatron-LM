# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Added for portability across upstream and ROCm CI environments.

"""Shared path resolution for unit tests across CI environments."""

import os
from pathlib import Path

_UNIT_TESTS_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Megatron-LM repository root.

    Upstream CI checks out to /opt/megatron-lm; ROCm CI uses /workspace/Megatron-LM.
    Override with MEGATRON_LM_ROOT when needed.
    """
    if root := os.environ.get("MEGATRON_LM_ROOT"):
        return Path(root)
    return _UNIT_TESTS_DIR.parents[1]


def unit_test_data_dir() -> Path:
    """Root directory for unit test datasets and tokenizers.

    Upstream and ROCm CI typically mount or populate /opt/data. The session-scoped
    ensure_test_data fixture downloads assets here when missing. Override with
    MEGATRON_TEST_DATA_DIR.
    """
    if data_dir := os.environ.get("MEGATRON_TEST_DATA_DIR"):
        return Path(data_dir)
    return Path("/opt/data")


def unit_test_data_path(*parts: str) -> Path:
    """Return a path under unit_test_data_dir()."""
    return unit_test_data_dir().joinpath(*parts)
