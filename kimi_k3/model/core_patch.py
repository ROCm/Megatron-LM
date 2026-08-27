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


def assert_pin_contracts() -> list:
    """Assert every core mechanism K3 rides still looks the way we expect.

    Returns the list of contract names checked. Runs on CPU in CI stage 0 and is
    the first thing re-run after an IFU rebase (rule R10.3).
    """
    checked = []

    import megatron.core.models.gpt.gpt_model as gm
    from megatron.core.transformer.transformer_block import TransformerBlock

    assert getattr(gm, "TransformerBlock", None) is TransformerBlock, (
        "gpt_model no longer resolves TransformerBlock at module scope; the K3 "
        "block injection in k3_block_class() would silently do nothing."
    )
    src = inspect.getsource(gm.GPTModel.__init__)
    assert "self.decoder = TransformerBlock(" in src, (
        "GPTModel.__init__ no longer constructs TransformerBlock directly; "
        "re-check the block-injection design."
    )
    checked.append("gpt_model.TransformerBlock")

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
        body = inspect.getsource(fn)
        assert "adjust_tensor_shapes_fn is None" in body, (
            f"{fn.__name__} no longer asserts the hook is None; re-check whether "
            "the K3 schedule binding must stay conditional on PP > 1."
        )
    checked.append("schedules.adjust_tensor_shapes_fn")

    back = inspect.getsource(schedules.backward_step)
    assert "output_tensor[0], output_tensor_grad[0]" in back, (
        "backward_step no longer back-props only output_tensor[0]; the "
        "single-packed-tensor payload may no longer be required -- re-read "
        "develop/notes/2026-08-26-attn-res-pp-transport.md before changing it."
    )
    checked.append("schedules.backward_step single-tensor backward")

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
    checked.append("MoESubmodules.router + MoELayer.postprocess")

    from megatron.core.optimizer import qk_clip

    clip_src = inspect.getsource(qk_clip.clip_qk)
    assert "decoder.layers" in clip_src and "clip_qk" in clip_src, (
        "core clip_qk no longer walks decoder.layers looking for a clip_qk "
        "attribute; the K3 MLA hook will not be reached."
    )
    checked.append("optimizer.qk_clip.clip_qk")

    from megatron.core.optimizer import muon

    muon_src = inspect.getsource(muon.get_megatron_muon_optimizer)
    assert "LayerWiseDistributedOptimizer" in muon_src, (
        "dist_muon no longer routes to LayerWiseDistributedOptimizer; the measured "
        "6 + 8/DP memory model in develop/plan-0/06-capacity-and-parallelism.md "
        "no longer applies."
    )
    checked.append("muon -> LayerWiseDistributedOptimizer")

    from megatron.training import arguments

    build_src = inspect.getsource(arguments.core_transformer_config_from_args)
    assert "config_class = MLATransformerConfig" in build_src, (
        "core_transformer_config_from_args no longer substitutes "
        "MLATransformerConfig; k3_config_from_args may be able to delegate again."
    )
    checked.append("core_transformer_config_from_args substitution")

    return checked
