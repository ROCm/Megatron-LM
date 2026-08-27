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
)


def assert_pin_contracts() -> list:
    """Run every contract. Returns the names checked (rule R4.5, R10.3)."""
    for name, fn in PIN_CONTRACTS:
        fn()
    return [name for name, _ in PIN_CONTRACTS]
