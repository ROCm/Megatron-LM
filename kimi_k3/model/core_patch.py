"""Scoped namespace patching, plus the pin contracts that guard it.

``GPTModel.__init__`` constructs ``TransformerBlock`` directly
(megatron/core/models/gpt/gpt_model.py:209), resolving the name from its own
module scope (imported at :34). Rebinding that attribute for the duration of
construction gives us our block with **no transient core block allocated** and
**no diff under megatron/**, which is what the core-isolation rule requires
(kimi_k3/develop/rules/rule.md R2.2).

This is the only file in the tree allowed to patch a core namespace.
"""

import contextlib
import inspect


@contextlib.contextmanager
def k3_block_class(block_cls):
    """Rebind the symbol GPTModel resolves when it builds its decoder."""
    import megatron.core.models.gpt.gpt_model as gm

    original = gm.TransformerBlock
    gm.TransformerBlock = block_cls
    try:
        yield
    finally:
        gm.TransformerBlock = original


_ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS = None


def install_router_bias_dispatch() -> None:
    """Let a router own its own expert-bias update.

    `finalize_model_grads._update_router_expert_bias` collects every module with
    an `expert_bias` attribute and overwrites it with the free function
    `get_updated_expert_bias` -- core's fixed `sign()` step. It never consults
    the module. So K3's `QuantileBalancingRouter.update_expert_bias` was dead
    code the moment training ran through `megatron.training.pretrain`: core
    recomputed the bias from scratch every step and copied its own value in,
    silently discarding quantile balancing.

    This rebinds that module-scope symbol (resolved at
    `finalize_model_grads.py:481`) so routers that implement
    `update_expert_bias` handle themselves, and every other module still goes
    through core's original path untouched. Idempotent.
    """
    global _ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS
    # NB: the package rebinds `finalize_model_grads` to the *function* of that
    # name, so both `from ... import` and `import ... as` hand back the function.
    # importlib returns the actual module object.
    import importlib

    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    from megatron.core.utils import get_attr_wrapped_model

    if _ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS is not None:
        return
    _ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS = fmg._update_router_expert_bias

    def _dispatching_update(model, config):
        handled, remaining = [], []
        for model_chunk in model:
            for module in get_attr_wrapped_model(model_chunk, 'modules')():
                if not (hasattr(module, "expert_bias") and module.training):
                    continue
                (handled if callable(getattr(module, "update_expert_bias", None))
                 else remaining).append(module)
        for module in handled:
            module.update_expert_bias()
        if remaining:
            # Core's own path, for any router that does not own its update.
            _ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS(model, config)

    fmg._update_router_expert_bias = _dispatching_update


def _contract_router_bias_dispatch():
    """Core must still route the bias update through a module-scope symbol."""
    import importlib

    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    assert hasattr(fmg, "_update_router_expert_bias"), (
        "finalize_model_grads no longer defines _update_router_expert_bias; "
        "the K3 quantile-balancing dispatch would silently do nothing."
    )
    source = inspect.getsource(fmg.finalize_model_grads)
    assert "_update_router_expert_bias(model, config)" in source, (
        "finalize_model_grads no longer calls _update_router_expert_bias at "
        "module scope; re-check install_router_bias_dispatch()."
    )
    original = _ORIGINAL_UPDATE_ROUTER_EXPERT_BIAS or fmg._update_router_expert_bias
    assert "expert_bias" in inspect.getsource(original), (
        "core's expert-bias update no longer reads module.expert_bias."
    )


_ORIGINAL_EXTRA_TE_KWARGS = None


def install_grouped_linear_single_param() -> None:
    """Ask TE to store grouped expert weights as ONE parameter, not 112.

    TE's `GroupedLinear` keeps `weight0 .. weight{n-1}` as separate parameters, so
    DDP registers a backward hook per expert. The trace shows what that costs:
    **672 launches and 22.4 ms** of `main_grad (fp32) += grad (bf16)`, 336 per
    matrix over 112 experts x 3 MoE layers, each a small add where one contiguous
    add would do.

    TE supports `single_grouped_weight`, but **core never passes it**, and the
    `NVTE_GROUPED_LINEAR_SINGLE_PARAM` env var only *gates* an explicitly
    requested True -- setting the env var alone does nothing. So the argument has
    to be injected where core builds its TE kwargs.

    EXPERIMENTAL upstream, and it changes the parameter layout, which the QAT
    wiring reads directly (`_get_weight_tensors` does `getattr(self, f"weight{i}")`).
    Off unless asked for.
    """
    global _ORIGINAL_EXTRA_TE_KWARGS
    import functools
    import os

    import transformer_engine.pytorch as te

    if _ORIGINAL_EXTRA_TE_KWARGS is not None:
        return
    # Patch TE's GroupedLinear.__init__, not core's `_get_extra_te_kwargs`: that
    # helper is shared by *every* TE module, so injecting the argument there makes
    # `RMSNorm.__init__() got an unexpected keyword argument` -- only GroupedLinear
    # takes it.
    _ORIGINAL_EXTRA_TE_KWARGS = te.GroupedLinear.__init__

    @functools.wraps(_ORIGINAL_EXTRA_TE_KWARGS)
    def _init(self, *args, **kwargs):
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        kwargs.setdefault("single_grouped_weight", True)
        return _ORIGINAL_EXTRA_TE_KWARGS(self, *args, **kwargs)

    te.GroupedLinear.__init__ = _init


def _contract_grouped_linear_single_param():
    """TE must still accept the argument, and core must still build its kwargs here."""
    import inspect

    from megatron.core.extensions import transformer_engine as te_ext

    assert hasattr(te_ext, "_get_extra_te_kwargs"), (
        "core no longer builds TE kwargs through _get_extra_te_kwargs; the "
        "single-grouped-weight injection would silently do nothing."
    )
    try:
        import transformer_engine.pytorch as te

        sig = inspect.signature(te.GroupedLinear.__init__)
        assert "single_grouped_weight" in sig.parameters, (
            "TE's GroupedLinear no longer takes single_grouped_weight."
        )
    except ImportError:  # pragma: no cover
        pass


_ORIGINAL_NEWTON_SCHULZ_TP = None


def install_muon_syrk() -> None:
    """Pass `use_syrk=True` into Newton-Schulz.

    `A = X @ X.mT` is symmetric, so SYRK computes one triangle instead of a full
    GEMM -- roughly half the FLOPs on the `mm` half of Newton-Schulz, which the
    trace puts at ~420 ms of a 1,836 ms iteration.

    The kernel exists (`emerging_optimizers/triton_kernels/syrk.py`) and its gate
    is already open: `muon_utils.py:209` reaches the SYRK branch only when
    `float32_matmul_precision == "medium"`, and `OrthogonalizedOptimizer.step`
    enters exactly that context for the duration of the step
    (`emerging_optimizers/utils/__init__.py:29`). Measured inside a real step:
    precision is `medium` and the NS step function already receives **bfloat16**.

    The single reason it never runs is that Megatron builds `ns_kwargs` from
    `steps`, `tp_group`, `partition_dim`, `tp_mode` and `coefficient_type`
    (`muon.py:106`) and never includes `use_syrk`, so it falls to the library
    default `False`. This injects it at the boundary rather than editing core.

    Requires Triton >= 3.4 for the tensor-descriptor API (`HAS_TRITON_340`); we
    are on 3.7.1. The kernel asserts 2-D input, so it is skipped for the batched
    (per-head) path.
    """
    global _ORIGINAL_NEWTON_SCHULZ_TP
    import functools

    from emerging_optimizers.orthogonalized_optimizers import muon_utils

    if _ORIGINAL_NEWTON_SCHULZ_TP is not None:
        return
    # Wrap `newton_schulz`, not `newton_schulz_tp`: only the former takes
    # `use_syrk` (the TP wrapper has an explicit signature without it and raises
    # TypeError), and it delegates down to `newton_schulz(...)` by module-level
    # name, so patching here catches both the TP and non-TP paths.
    _ORIGINAL_NEWTON_SCHULZ_TP = muon_utils.newton_schulz

    @functools.wraps(_ORIGINAL_NEWTON_SCHULZ_TP)
    def _with_syrk(x, *args, **kwargs):
        # The kernel asserts 2-D; the batched per-head path must stay on the
        # general GEMM ("use_syrk does not support N-d input").
        if x.ndim == 2:
            kwargs.setdefault("use_syrk", True)
        return _ORIGINAL_NEWTON_SCHULZ_TP(x, *args, **kwargs)

    muon_utils.newton_schulz = _with_syrk


_MUON_BATCH_INSTALLED = False


def _quack_batched_tsyrk():
    """Resolve quack's batched symmetric GEMM, or explain why it is unavailable.

    From `wenchenvincent/quack-flydsl` @ `wip/amd-flydsl-port`. Needs FlyDSL 0.2.x
    on the path: 0.1.1.dev409 has no `flydsl.expr.math`, 0.3.2 renamed
    `expr.vector` -> `expr.Vector`. The global install is pinned by `amd-aiter`
    (MoRI imports it), so 0.2.4 lives in a separate tree reached by PYTHONPATH.
    """
    try:
        from quack.amd.gemm_gfx950_nt_pingpong import batched_tsyrk_ex, can_use_batched_tsyrk
    except ImportError as exc:  # pragma: no cover - depends on PYTHONPATH
        raise RuntimeError(
            "quack.amd.gemm_gfx950_nt_pingpong is not importable. Clone "
            "github.com/wenchenvincent/quack-flydsl @ wip/amd-flydsl-port and put it plus "
            "FlyDSL 0.2.4 on PYTHONPATH (do NOT upgrade the global flydsl: amd-aiter/MoRI "
            f"pin it). Underlying error: {exc}"
        ) from exc
    return batched_tsyrk_ex, can_use_batched_tsyrk


def install_muon_batched_ns(batch_size: int = 16, use_quack_syrk: bool = False) -> None:
    """Batch same-shaped Muon parameters through one Newton-Schulz call.

    K3's Muon population is dominated by expert weights: per rank 336 of
    (6144, 3584) and 336 of (3584, 3072) at 3 MoE layers x 112 local experts
    (`kimi_k3.tools.muon_shapes`). `OrthogonalizedOptimizer.step` orthogonalizes
    one parameter at a time, so those are 672 separate 5-step iterations --
    ~10k kernel launches for three GEMMs per step.

    `newton_schulz` already dispatches 3-D input to `batched_newton_schulz_step`
    (`muon_utils.py:201`), and `scaled_orthogonalize_fn` reads its scale from
    `size(-2)/size(-1)`, so a stacked (B, M, N) tensor needs no change there.
    The only requirement is `partition_dim is None`, which `--muon-tp-mode
    blockwise` (our default) already guarantees for every parameter -- the TP
    branches of `newton_schulz_tp` all-gather along `partition_dim` and would
    mis-handle the extra leading dim. TP-partitioned and QKV-split parameters
    therefore stay on the serial path.

    With `use_quack_syrk`, the Gram matmuls inside the batched step go to quack's
    triangular kernel. Both K3 Gram shapes satisfy its N%256/K%128 constraints.
    Measured against `torch.baddbmm` at B=16: 1.44x for the (3584, 3584) Gram and
    1.20x for (3072, 3072); at B=1 it *loses* (0.84x / 0.63x) because a single
    expert's Gram cannot fill the GPU, so the halved triangular tile count just
    leaves CUs idle. Batching is what makes the kernel win.
    """
    global _MUON_BATCH_INSTALLED
    import collections

    import torch
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import muon_utils

    from megatron.core.optimizer.muon import TensorParallelMuon

    if _MUON_BATCH_INSTALLED:
        return
    _MUON_BATCH_INSTALLED = True

    if use_quack_syrk:
        batched_tsyrk_ex, can_use_batched_tsyrk = _quack_batched_tsyrk()
        _baddbmm_step = muon_utils.batched_newton_schulz_step

        def _syrk_step(X, a, b, c, tp_group=None):
            # A = X @ X.mT and B = A @ A are both symmetric; the third matmul
            # (B @ X) is not. tp_group would need an all-reduce between the two
            # syrk calls -- unreachable here (partition_dim is None) but kept
            # honest by falling back.
            if tp_group is not None or X.dtype != torch.bfloat16 or not can_use_batched_tsyrk(X):
                return _baddbmm_step(X, a, b, c, tp_group=tp_group)
            A = batched_tsyrk_ex(X)
            Bm = batched_tsyrk_ex(A, c=A, alpha=c, beta=b)
            return torch.baddbmm(X, Bm, X, alpha=1.0, beta=a)

        muon_utils.batched_newton_schulz_step = _syrk_step

    def _ns_context(self, p):
        """(tp_group, partition_dim) exactly as TensorParallelMuon.orthogonalize derives them."""
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        return tp_group, partition_dim

    def _momentum_into(self, p, group, out):
        """The per-parameter prologue of the original step, writing the update into `out`."""
        state = self.state[p]
        self._apply_weight_decay_inplace(p, p.grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
        if self.nesterov:
            torch.lerp(p.grad, state["momentum_buffer"], group["momentum"], out=out)
        else:
            out.copy_(state["momentum_buffer"])

    @torch.no_grad()
    def _batched_step(self, closure=None):
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            buckets = collections.defaultdict(list)
            serial = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                tp_group, partition_dim = _ns_context(self, p)
                batchable = (
                    p.ndim == 2
                    and partition_dim is None
                    and not (self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p))
                    # The batched path calls scaled_orthogonalize_fn directly, so it
                    # bypasses any `orthogonalize` override. `PerHeadMuon` is one
                    # (`optim/per_head_muon.py:189`) and keys on this attribute; batching
                    # a tagged parameter would silently drop the head split.
                    and getattr(p, "k3_head_split", None) is None
                )
                if batchable:
                    buckets[(tuple(p.shape), p.dtype, id(tp_group))].append((p, tp_group))
                else:
                    serial.append(p)

            # Serial path: byte-for-byte the original loop.
            for p in serial:
                grad = torch.empty_like(p.grad)
                _momentum_into(self, p, group, grad)
                with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    orth_grad = self.orthogonalize(p, grad, **group_kwargs)
                self.pre_weight_update_fn_inplace(p, orth_grad)
                p.add_(orth_grad, alpha=-group["lr"])
                self.post_weight_update_fn_inplace(p)

            for (shape, dtype, _), members in buckets.items():
                for start in range(0, len(members), batch_size):
                    chunk = members[start : start + batch_size]
                    tp_group = chunk[0][1]
                    buf = torch.empty((len(chunk), *shape), dtype=dtype, device=chunk[0][0].device)
                    for i, (p, _) in enumerate(chunk):
                        _momentum_into(self, p, group, buf[i])
                    with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        orth = self.scaled_orthogonalize_fn(buf, tp_group, None)
                    for i, (p, _) in enumerate(chunk):
                        self.pre_weight_update_fn_inplace(p, orth[i])
                        p.add_(orth[i], alpha=-group["lr"])
                        self.post_weight_update_fn_inplace(p)
        return None

    TensorParallelMuon.step = _batched_step


def _contract_muon_batched_ns():
    """Everything the batching rewrite of `step` reaches into must still exist."""
    import inspect

    from emerging_optimizers.orthogonalized_optimizers import muon_utils
    from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
        OrthogonalizedOptimizer,
    )

    src = inspect.getsource(OrthogonalizedOptimizer.step)
    for needle in (
        "_apply_weight_decay_inplace",
        'state["momentum_buffer"].lerp_',
        "pre_weight_update_fn_inplace",
        "fp32_matmul_precision",
    ):
        assert needle in src, (
            f"OrthogonalizedOptimizer.step no longer contains {needle!r}; the batched "
            "step in install_muon_batched_ns is a copy of that loop and has drifted."
        )
    assert "batched_newton_schulz_step" in inspect.getsource(muon_utils.newton_schulz), (
        "newton_schulz no longer dispatches 3-D input to batched_newton_schulz_step."
    )
    ns_src = inspect.getsource(muon_utils.newton_schulz_tp)
    from kimi_k3.optim.per_head_muon import PerHeadMuon

    assert "k3_head_split" in inspect.getsource(PerHeadMuon.orthogonalize), (
        "PerHeadMuon no longer keys on p.k3_head_split; the batched step excludes "
        "parameters by that attribute and would now silently skip the head split."
    )
    assert "if partition_dim is None:" in ns_src, (
        "newton_schulz_tp no longer short-circuits on partition_dim=None; batching "
        "same-shaped params may now hit a TP all-gather that cannot take 3-D input."
    )


def _contract_muon_syrk():
    """The gate SYRK sits behind, and the kernel itself, must both still exist."""
    import inspect

    from emerging_optimizers.orthogonalized_optimizers import muon_utils

    assert "use_syrk" in inspect.signature(muon_utils.newton_schulz).parameters, (
        "newton_schulz no longer takes use_syrk; the SYRK injection would be ignored."
    )
    src = inspect.getsource(muon_utils.newton_schulz)
    assert 'get_float32_matmul_precision() == "medium"' in src, (
        "the SYRK branch is no longer gated on medium precision; re-check whether "
        "OrthogonalizedOptimizer.step still opens that context."
    )
    import megatron.core.optimizer.muon as mm

    assert hasattr(mm, "newton_schulz_tp"), (
        "muon.py no longer resolves newton_schulz_tp at module scope."
    )


def _contract_gpt_model_block():
    """GPTModel must still resolve TransformerBlock at module scope and build it."""
    import megatron.core.models.gpt.gpt_model as gm
    from megatron.core.transformer.transformer_block import TransformerBlock

    assert getattr(gm, "TransformerBlock", None) is TransformerBlock, (
        "gpt_model no longer resolves TransformerBlock at module scope; the K3 "
        "block injection in k3_block_class() would silently do nothing."
    )
    assert "self.decoder = TransformerBlock(" in inspect.getsource(gm.GPTModel.__init__), (
        "GPTModel.__init__ no longer constructs TransformerBlock directly; "
        "re-check the block-injection design."
    )


def _contract_adjust_tensor_shapes_fn():
    """1F1B must accept the hook; the other two schedules must still reject it."""
    from megatron.core.pipeline_parallel import schedules

    sig = inspect.signature(schedules.forward_backward_pipelining_without_interleaving)
    assert "adjust_tensor_shapes_fn" in sig.parameters, (
        "the 1F1B schedule no longer accepts adjust_tensor_shapes_fn; the AttnRes "
        "payload transport has no hook to bind."
    )
    for fn in (
        schedules.forward_backward_no_pipelining,
        schedules.forward_backward_pipelining_with_interleaving,
    ):
        assert "adjust_tensor_shapes_fn is None" in inspect.getsource(fn), (
            f"{fn.__name__} no longer asserts the hook is None; re-check whether "
            "the K3 schedule binding must stay conditional on PP > 1."
        )


def _contract_backward_step_single_tensor():
    """Only output_tensor[0] is back-propped -- why the payload is one tensor."""
    from megatron.core.pipeline_parallel import schedules

    assert "output_tensor[0], output_tensor_grad[0]" in inspect.getsource(schedules.backward_step), (
        "backward_step no longer back-props only output_tensor[0]; the "
        "single-packed-tensor payload may no longer be required -- re-read "
        "develop/notes/2026-08-26-attn-res-pp-transport.md before changing it."
    )


def _contract_moe_router_and_postprocess():
    """The router injection point, and the missing latent norm K3MoELayer adds."""
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules

    assert "router" in MoESubmodules.__dataclass_fields__, (
        "MoESubmodules.router is gone; the QuantileBalancingRouter injection path "
        "no longer exists."
    )
    post = inspect.getsource(MoELayer.postprocess)
    assert "fc2_latent_proj" in post and "routed_expert_norm" not in post, (
        "core MoELayer.postprocess changed; K3MoELayer overrides it to insert the "
        "latent RMSNorm before the up-projection."
    )


def _contract_qk_clip():
    from megatron.core.optimizer import qk_clip

    src = inspect.getsource(qk_clip.clip_qk)
    assert "decoder.layers" in src and "clip_qk" in src, (
        "core clip_qk no longer walks decoder.layers looking for a clip_qk "
        "attribute; the K3 MLA hook will not be reached."
    )


def _contract_muon():
    """dist_muon shards; every muon variant still rejects the distributed optimizer."""
    from megatron.core.optimizer import muon
    from megatron.training import arguments

    assert "LayerWiseDistributedOptimizer" in inspect.getsource(muon.get_megatron_muon_optimizer), (
        "dist_muon no longer routes to LayerWiseDistributedOptimizer; the measured "
        "6 + 8/DP memory model in develop/plan-0/06-capacity-and-parallelism.md "
        "no longer applies."
    )
    validate = inspect.getsource(arguments.validate_args)
    assert "Muon optimizer does not support distributed optimizer" in validate, (
        "the muon / --use-distributed-optimizer rejection changed; re-check whether "
        "dist_muon is still the only sharded Muon path."
    )


def _contract_config_substitution():
    from megatron.training import arguments

    assert "config_class = MLATransformerConfig" in inspect.getsource(
        arguments.core_transformer_config_from_args
    ), (
        "core_transformer_config_from_args no longer substitutes "
        "MLATransformerConfig; k3_config_from_args may be able to delegate again."
    )


def _contract_mla_core_attention_kwargs():
    """MLA passes k_channels/v_channels, which core's DotProductAttention rejects.

    Review finding A13: this is why construction gates need the TE spec and a GPU,
    and why K3GatedMLA's core_attention submodule cannot be core's local one.
    """
    from megatron.core.transformer import dot_product_attention as dpa
    from megatron.core.transformer import multi_latent_attention as mla

    # The kwargs are built in MultiLatentAttention.__init__, which MLASelfAttention
    # inherits -- not in the subclass.
    assert "k_channels" in inspect.getsource(mla.MultiLatentAttention.__init__), (
        "MLA no longer passes k_channels to core_attention; the local MLA spec may "
        "work again (finding A13)."
    )
    params = inspect.signature(dpa.DotProductAttention.__init__).parameters
    assert "k_channels" not in params and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), "core DotProductAttention now accepts k_channels; finding A13 may be stale."


#: Every core mechanism K3 rides. An IFU breaks one of these first, and silently.
PIN_CONTRACTS = (
    ("gpt_model.TransformerBlock", _contract_gpt_model_block),
    ("schedules.adjust_tensor_shapes_fn", _contract_adjust_tensor_shapes_fn),
    ("schedules.backward_step single-tensor backward", _contract_backward_step_single_tensor),
    ("MoESubmodules.router + MoELayer.postprocess", _contract_moe_router_and_postprocess),
    ("optimizer.qk_clip.clip_qk", _contract_qk_clip),
    ("muon -> LayerWiseDistributedOptimizer + dist-opt rejection", _contract_muon),
    ("core_transformer_config_from_args substitution", _contract_config_substitution),
    ("MLA passes k_channels to core_attention", _contract_mla_core_attention_kwargs),
    ("router expert-bias update dispatches to the module", _contract_router_bias_dispatch),
    ("TE GroupedLinear accepts single_grouped_weight", _contract_grouped_linear_single_param),
    ("Muon Newton-Schulz still exposes use_syrk", _contract_muon_syrk),
    ("Muon step loop the batched rewrite copies", _contract_muon_batched_ns),
)


def assert_pin_contracts() -> list:
    """Run every contract. Returns the names checked (rule R4.5, R10.3)."""
    for name, fn in PIN_CONTRACTS:
        fn()
    return [name for name, _ in PIN_CONTRACTS]
