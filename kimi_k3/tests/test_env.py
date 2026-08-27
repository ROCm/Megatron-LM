"""P1 / gate G10 -- the pinned dependencies are importable and at the right versions.

Cheap, and it fails with a pointer to `kimi_k3/PINS.md` rather than an ImportError
three layers deep in a training run.
"""

import pytest
import torch


def test_torch_and_triton_versions():
    """triton 3.7.1+ is required: 3.6.0 cannot compile fla's KDA backward on gfx950."""
    import triton

    major, minor = (int(p) for p in triton.__version__.split(".")[:2])
    assert (major, minor) >= (3, 7), (
        f"triton {triton.__version__} is too old; 3.6.0 fails to compile "
        "chunk_kda_bwd_intra on gfx950. See PINS.md and "
        "develop/notes/2026-08-27-triton-unblocks-fla-backward.md"
    )
    assert torch.__version__.startswith("2."), torch.__version__


def test_transformer_engine_imports():
    import transformer_engine.pytorch as tep

    assert hasattr(tep, "Linear")


def test_fla_kda_entry_points_import():
    ops = pytest.importorskip("fla.ops.kda", reason="fla is a pinned optional backend")
    assert callable(ops.chunk_kda)
    assert callable(ops.fused_recurrent_kda)


def test_aiter_k3_kernels():
    """AITER's K3 a8w4 path. Expected to be absent until the checkout is bumped."""
    pytest.importorskip("aiter", reason="AITER is an optional backend")
    opus = pytest.importorskip(
        "aiter.ops.opus.moe_stage1_a8w4",
        reason="AITER checkout predates the K3 a8w4 kernels; bump to the PINS.md SHA "
        "and rebuild (P10 owns this)",
    )
    assert hasattr(opus, "opus_moe_stage1_a8w4_fwd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_is_the_expected_architecture():
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    assert torch.cuda.device_count() >= 1
    # Not an assertion on gfx950 specifically -- tests must run on dev boxes too.
    print(f"\ndevice: {name} ({arch}), count={torch.cuda.device_count()}")
