"""Shared fixtures for the Kimi K3 test suite.

Default tier is the ``tiny`` preset (rule R4.2). Production widths are marked
``pytest.mark.slow`` and run only in the nightly 8-GPU job.
"""

import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run release-tier tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: release-tier shapes; opt in with --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="release tier; pass --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def single_rank_world():
    """A 1-rank world with model-parallel state and the TP rng tracker seeded.

    Megatron's parallel layers refuse to initialise without ``model-parallel-rng``,
    and container images commonly pin NVTE_* backend selectors that core's
    attention-backend check rejects, so both are handled here.
    """
    from megatron.core import parallel_state, tensor_parallel

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, world_size=1, rank=0)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
    yield


@pytest.fixture()
def tiny_spec():
    """Transformer-layer spec for skeleton construction tests.

    The TE spec, not the local one: at the pinned SHA ``MLASelfAttention`` always
    passes ``k_channels`` / ``v_channels`` to its ``core_attention`` submodule
    (multi_latent_attention.py:219-226), and core's own ``DotProductAttention``
    accepts neither, so the local MLA path cannot be constructed at all. See
    develop/plan-0/00-review-findings.md finding A13.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from kimi_k3.config.presets import preset

    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=preset("tiny")["config"]["num_moe_experts"],
        moe_grouped_gemm=False,
        multi_latent_attention=True,
    )


@pytest.fixture()
def tiny_config():
    """The ``tiny`` preset as a K3 transformer config."""
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    return config_from_preset(preset("tiny")["config"])
