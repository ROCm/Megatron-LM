"""P9 — two things about Muon that look available and are not (A19, A20).

Both are tripwires. They assert the *current* behaviour, so if core or
`emerging_optimizers` fixes either one, the test fails and the finding comes out
of the register instead of quietly going stale.
"""

import inspect

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def build_dist_muon(**optimizer_kwargs):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.muon import get_megatron_muon_optimizer

    from kimi_k3.model.build import build_k3_model

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                                      use_distributed_optimizer=False),
        model,
    )
    config = OptimizerConfig(
        optimizer="dist_muon", lr=1e-4, bf16=False, params_dtype=torch.float32,
        use_distributed_optimizer=False, weight_decay=0.1, clip_grad=1.0, **optimizer_kwargs,
    )
    optimizer = get_megatron_muon_optimizer(config, [ddp], layer_wise_distributed_optimizer=True)
    inner = set()
    for sub in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner.add(type(getattr(sub, "optimizer", sub)).__name__)
    return inner


def test_cpu_offload_leaves_the_muon_group_on_the_gpu(single_rank_world):
    """A19: the flag builds cleanly and reaches only the scalar optimizer."""
    baseline = build_dist_muon()
    offloaded = build_dist_muon(optimizer_cpu_offload=True)

    assert "TensorParallelMuon" in baseline and "TensorParallelMuon" in offloaded, (
        "the Muon group should exist either way"
    )
    assert "HybridDeviceOptimizer" not in baseline
    assert "HybridDeviceOptimizer" in offloaded, "the scalar group should have been swapped"
    assert "TensorParallelMuon" in offloaded, (
        "if Muon ever gains an offload path this fails and finding A19 can be retired"
    )


def test_offload_still_enforces_its_precondition_via_the_scalar_group(single_rank_world):
    """A19, corrected: the guard *does* fire, through the group that reaches it.

    An earlier draft of the finding claimed `assert config.decoupled_weight_decay`
    never runs because Muon bypasses that code path. Only half of that is true:
    the scalar group *does* go through it, so the guard fires normally. The reason
    the first experiment saw no error is duller -- `decoupled_weight_decay`
    defaults on.
    """
    with pytest.raises(AssertionError, match="decoupled_weight_decay"):
        build_dist_muon(optimizer_cpu_offload=True, decoupled_weight_decay=False)


def test_only_a_rounding_error_of_k3_could_offload_at_all():
    """A19: the ceiling is structural -- P9's policy sends the matrices to Muon."""
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.tools.mem_budget import breakdown

    cfg = config_from_preset(preset("93L")["config"])
    total = breakdown(cfg, 163840).total
    embedding_and_output = 2 * 163840 * cfg.hidden_size
    assert embedding_and_output / total < 0.002, "the offloadable share should be negligible"


def test_transformer_engine_here_has_no_newton_schulz(single_rank_world):
    """A20(b): our pinned build predates the kernel, whatever the fork has at HEAD."""
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex

    assert transformer_engine.__version__.startswith("2.12.0.dev0+40434cf6")
    assert not hasattr(te, "newton_schulz")
    assert not hasattr(tex, "newton_schulz")
    assert not hasattr(tex, "cusolvermp_ctx_create"), (
        "TE gained the cuSOLVERMp entry point; re-check whether ROCm builds it "
        "(develop/notes/2026-08-30-te-bump.md) before retiring A20"
    )


def test_the_newton_schulz_triton_kernel_is_unreachable_from_the_tp_path():
    """A20: `use_syrk` exists, defaults off, and `newton_schulz_tp` cannot forward it."""
    from emerging_optimizers.orthogonalized_optimizers import muon_utils

    plain = inspect.signature(muon_utils.newton_schulz).parameters
    assert "use_syrk" in plain, "the Triton fast path is gone; re-check A20"
    assert plain["use_syrk"].default is False

    tp = inspect.signature(muon_utils.newton_schulz_tp).parameters
    assert "use_syrk" not in tp, "newton_schulz_tp gained use_syrk; A20 can be retired"
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in tp.values()), (
        "newton_schulz_tp gained **kwargs, so the fast path may now be reachable"
    )


def test_megatron_never_asks_for_the_fast_path():
    """A20: and core does not pass it on the non-TP fallback either."""
    from megatron.core.optimizer import muon

    assert "use_syrk" not in inspect.getsource(muon)


def test_te_optimizers_expose_no_orthogonalisation():
    """A20: TE's optimizer surface is Adam/SGD; the NS work is a separate module.

    Stated narrowly on purpose. The ROCm fork *does* carry a Newton-Schulz at
    HEAD (`a073ad5`, via cuSOLVERMp) -- it is simply excluded from ROCm builds
    (`cuda_only_cpp_sources`) and absent from our pin. "TE has no Newton-Schulz"
    would be wrong; "TE's optimizers do not orthogonalise" is what holds.
    """
    import transformer_engine.pytorch.optimizers as te_optimizers

    names = " ".join(dir(te_optimizers)).lower()
    assert "newton" not in names and "schulz" not in names and "orthogonal" not in names
    assert {"FusedAdam", "FusedSGD"} <= set(dir(te_optimizers))
