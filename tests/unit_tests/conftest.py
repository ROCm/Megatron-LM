import os
import sys
from pathlib import Path
import subprocess

import pytest
import torch
import torch.distributed

from megatron.core import config
from megatron.core.utils import is_te_min_version
from tests.test_utils.python_scripts.download_unit_tests_dataset import (
    get_oldest_release_and_assets,
)
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

    else:
        yield tmp_dir


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available at /opt/data by downloading if necessary."""
    data_path = Path("/opt/data")

    # Check if data directory exists and has content
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Test data not found at /opt/data. Downloading...")

        try:
            # Download assets to /opt/data
            get_oldest_release_and_assets(assets_dir=str(data_path))

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


def pytest_runtest_teardown(item, nextitem):
    """Run rocm-smi and resource monitoring after each test."""
    # Only run on rank 0 to avoid duplicate output
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        import subprocess
        
        print("\n" + "=" * 80)
        print(f"Resource Monitor after test: {item.nodeid}")
        print("=" * 80)
        
        # System Memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    if 'MemTotal:' in line:
                        meminfo['total'] = int(line.split()[1]) // 1024  # MB
                    elif 'MemAvailable:' in line:
                        meminfo['available'] = int(line.split()[1]) // 1024  # MB
                    elif 'MemFree:' in line:
                        meminfo['free'] = int(line.split()[1]) // 1024  # MB
                if meminfo:
                    print(f"System Memory: {meminfo.get('available', 'N/A')} MB available / "
                          f"{meminfo.get('total', 'N/A')} MB total")
        except Exception as e:
            print(f"Could not read memory info: {e}")
        
        # Process count
        try:
            import os
            process_count = len([p for p in os.listdir('/proc') if p.isdigit()])
            python_count = subprocess.run(
                ["pgrep", "-c", "python"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"Processes: {process_count} total, "
                  f"{python_count.stdout.strip() if python_count.returncode == 0 else 'N/A'} python")
        except Exception as e:
            print(f"Could not count processes: {e}")
        
        # File descriptors
        try:
            fd_count = 0
            for pid_dir in os.listdir('/proc'):
                if not pid_dir.isdigit():
                    continue
                try:
                    fd_count += len(os.listdir(f'/proc/{pid_dir}/fd'))
                except:
                    pass
            print(f"File Descriptors: {fd_count} open")
        except Exception as e:
            print(f"Could not count file descriptors: {e}")
        
        # Shared memory usage
        try:
            if os.path.exists('/dev/shm'):
                result = subprocess.run(
                    ["du", "-sh", "/dev/shm"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    shm_size = result.stdout.split()[0]
                    print(f"/dev/shm usage: {shm_size}")
                
                # Count NCCL and HSA files
                nccl_count = len([f for f in os.listdir('/dev/shm') if f.startswith('nccl-')])
                hsa_count = len([f for f in os.listdir('/dev/shm') if f.startswith('hsa')])
                torch_count = len([f for f in os.listdir('/dev/shm') if f.startswith('torch_')])
                if nccl_count or hsa_count or torch_count:
                    print(f"  NCCL files: {nccl_count}, HSA files: {hsa_count}, Torch files: {torch_count}")
        except Exception as e:
            print(f"Could not check /dev/shm: {e}")
        
        print("-" * 80)
        
        # ROCm-SMI
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"rocm-smi returned error code {result.returncode}")
                print(result.stderr)
        except FileNotFoundError:
            # rocm-smi not available (probably on CUDA system)
            pass
        except subprocess.TimeoutExpired:
            print("rocm-smi command timed out")
        except Exception as e:
            print(f"Failed to run rocm-smi: {e}")
        finally:
            print("=" * 80 + "\n")