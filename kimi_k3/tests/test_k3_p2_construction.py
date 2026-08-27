"""P2 / gates G12–G13 -- build from a preset, and count the official ones.

G12 promotes P0's construction prototype to the shipped path: one call from a
preset name to a model whose decoder is a K3TransformerBlock with per-layer
specs. G13 checks the official presets **analytically on meta device** -- 8L is
~215 B parameters and 93L is ~2.78 T, so they are never materialised (R4.3).
"""

import pytest
import torch

from kimi_k3.block.k3_transformer_block import K3TransformerBlock
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.model.build import build_k3_model
from kimi_k3.specs.layer_specs import k3_layer_plan
from kimi_k3.tools import mem_budget


def test_build_from_preset_gives_a_k3_block(single_rank_world):
    model = build_k3_model("tiny", device="meta")
    assert isinstance(model.decoder, K3TransformerBlock)
    assert len(model.decoder.layers) == preset("tiny")["config"]["num_layers"]


def test_layers_are_heterogeneous_as_the_plan_says(single_rank_world):
    """Layer 0 is dense; the rest are MoE. Core builds exactly the specs we pass."""
    model = build_k3_model("tiny", device="meta")
    plan = k3_layer_plan(model.config)
    for i, layer in enumerate(model.decoder.layers):
        has_experts = hasattr(layer.mlp, "experts")
        assert has_experts == (plan[i].ffn == "moe"), f"layer {i}: {type(layer.mlp).__name__}"


def test_attn_res_mixers_exist_per_layer_and_at_the_output(single_rank_world):
    """Two per layer plus one at the model output -- the released checkpoint's
    187 `*_res_proj` tensors for 93 layers (93 x 2 + 1).

    The per-layer mixes live on the layer (P5) and the model-level one on the
    block, which is what the release does too.
    """
    model = build_k3_model("tiny", device="meta")
    for layer in model.decoder.layers:
        assert hasattr(layer, "attn_res_attn") and hasattr(layer, "attn_res_mlp")
    assert model.decoder.output_attn_res is not None  # last stage only
    mixers = sum(1 for name, _ in model.named_modules() if name.endswith(("attn_res_attn", "attn_res_mlp")))
    assert mixers == 2 * len(model.decoder.layers)
    assert 2 * 93 + 1 == 187


@pytest.mark.parametrize("name,expected", [("4L", 94e9), ("8L", 215e9), ("93L", 2.78e12)])
def test_official_preset_parameter_counts(name, expected):
    """Analytic, not constructed -- these presets are never given weights."""
    p = preset(name)
    cfg = config_from_preset(p["config"])
    total = mem_budget.breakdown(cfg, p["model"]["vocab_size"]).total
    assert abs(total / expected - 1) < 0.02, f"{name}: {total / 1e9:.1f} B"


def test_official_93l_matches_the_component_table():
    cfg = config_from_preset(preset("93L")["config"])
    b = mem_budget.breakdown(cfg, 163840)
    plan = k3_layer_plan(cfg)
    assert b.kda == sum(p.attention == "kda" for p in plan) * mem_budget.kda_layer_params(cfg)
    assert b.mla == sum(p.attention == "mla" for p in plan) * mem_budget.mla_layer_params(cfg)
    assert b.moe == sum(p.ffn == "moe" for p in plan) * mem_budget.moe_layer_params(cfg)
    assert b.dense_ffn == mem_budget.dense_ffn_params(cfg)


def test_meta_device_does_not_actually_prevent_allocation(single_rank_world):
    """Recorded because it is counter-intuitive and would bite at official width.

    Megatron and TE modules place parameters on `torch.cuda.current_device()`
    explicitly, ignoring an ambient `torch.device("meta")` context. Only our own
    AttnRes mixers honour it. So "build the official preset on meta and count the
    parameters" -- which the incoming plan proposed -- would in fact try to
    allocate 2.78 T parameters.
    """
    model = build_k3_model("tiny", device="meta")
    on_meta = [p for p in model.parameters() if p.is_meta]
    assert 0 < len(on_meta) < sum(1 for _ in model.parameters()), (
        "if this now passes fully, meta construction started working and "
        "build_k3_model's official-preset guard can be revisited"
    )


@pytest.mark.parametrize("name", ["4L", "8L", "93L"])
def test_official_presets_refuse_to_be_constructed(name):
    """The guard that makes rule R4.3 enforced rather than merely written down."""
    with pytest.raises(ValueError, match="refusing to construct"):
        build_k3_model(name, device="meta")


def test_config_type_survives_the_whole_path(single_rank_world):
    from kimi_k3.config.k3_transformer_config import KimiK3TransformerConfig

    model = build_k3_model("tiny", device="meta")
    assert type(model.config) is KimiK3TransformerConfig
    assert model.config.moe_latent_size == preset("tiny")["config"]["k3_routed_expert_hidden_size"]
