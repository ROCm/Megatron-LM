# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified for portability across upstream and ROCm CI environments.

import os

# Pin Mamba/SSM Triton autotune to a deterministic config for the whole unit-test
# session. The SSM kernels evaluate their autotune config list at @triton.autotune
# decoration (import) time, so MAMBA_DETERMINISTIC must be set before any test
# module imports megatron.core.ssm.*. Setting it in a per-file module (e.g.
# test_dynamic_engine.py) is too late in a full-suite run, because an earlier test
# may import the SSM kernels first and freeze the non-deterministic config list --
# which lets timing-based autotuning flip a greedy token across runs of the hybrid
# inference tests. Setting it here (before the megatron imports below) guarantees
# determinism is engaged regardless of collection order. Use setdefault so an
# explicit override from the environment still wins.
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core import config
from megatron.core.utils import is_te_min_version
from tests.test_utils.python_scripts.download_unit_tests_dataset import download_and_extract_asset
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.paths import unit_test_data_dir
from tests.unit_tests.test_utilities import Utils


def _insert_rank_suffix(path: str, rank: str) -> str:
    """Insert a ``.rank<N>`` suffix before the file extension."""
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext}"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Give each distributed rank its own report file.

    Under ``torchrun`` every rank runs pytest and, by default, writes to the
    same ``--junitxml``/``--csv`` path. The ranks race and the last writer wins,
    so a failure on a non-zero rank can be silently overwritten by a passing
    rank and the CI reporter shows green. Suffixing the path with the rank keeps
    each rank's result; ``run_unit_tests.sh`` then merges them so a test is
    reported as failed if it failed on ANY rank. This runs ``tryfirst`` so the
    paths are rewritten before the junit/csv plugins capture them.
    """
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    if rank is None:
        return
    xmlpath = getattr(config.option, "xmlpath", None)
    if xmlpath:
        config.option.xmlpath = _insert_rank_suffix(xmlpath, rank)
    csvpath = getattr(config.option, "csvpath", None)
    if csvpath:
        config.option.csvpath = _insert_rank_suffix(csvpath, rank)


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
        try:
            torch.distributed.barrier()
        except Exception:
            return
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
    """Ensure unit test assets are available, downloading them if necessary."""
    data_path = unit_test_data_dir()

    # Check if data directory exists and has content
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"Test data not found at {data_path}. Downloading...")

        try:
            download_and_extract_asset(assets_dir=data_path)

            print("Test data downloaded successfully.")

        except ImportError as e:
            print(f"Failed to import download function: {e}")
            # Don't fail the tests, just warn
        except Exception as e:
            print(f"Failed to download test data: {e}")
            # Don't fail the tests, just warn
    else:
        print(f"Test data already available at {data_path}")


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
