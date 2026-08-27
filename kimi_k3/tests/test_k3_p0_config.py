"""P0 / gate G3 -- the K3 config must survive on the real inheritance path.

core_transformer_config_from_args replaces any caller-supplied config_class with
MLATransformerConfig when args.multi_latent_attention is set
(megatron/training/arguments.py:1230-1232). K3 therefore builds its own config
and this test proves core's builder is never on the path.
"""

import argparse

import pytest
import torch

from kimi_k3.config.k3_config_builder import config_from_preset, k3_config_from_args
from kimi_k3.config.k3_transformer_config import KimiK3TransformerConfig
from kimi_k3.config.presets import OFFICIAL_FULL_ATTN_1IDX, kda_layers_1idx, preset
from megatron.core.transformer.transformer_config import MLATransformerConfig


def _args_from_preset(name="tiny"):
    p = preset(name)
    ns = argparse.Namespace(**p["config"])
    ns.params_dtype = torch.bfloat16
    ns.num_experts = p["config"]["num_moe_experts"]
    ns.swiglu = True
    ns.no_persist_layer_norm = False
    ns.overlap_p2p_comm = False
    ns.multi_latent_attention = True
    return ns


def test_builder_returns_our_class_not_the_mla_base():
    cfg = k3_config_from_args(_args_from_preset())
    assert type(cfg) is KimiK3TransformerConfig
    assert isinstance(cfg, MLATransformerConfig)


def test_core_builder_is_never_called(monkeypatch):
    from megatron.training import arguments

    def _boom(*a, **kw):  # pragma: no cover - the point is that it never runs
        raise AssertionError("core_transformer_config_from_args must not be on the K3 path")

    monkeypatch.setattr(arguments, "core_transformer_config_from_args", _boom)
    cfg = k3_config_from_args(_args_from_preset())
    assert type(cfg) is KimiK3TransformerConfig


def test_mla_fields_are_populated():
    cfg = config_from_preset(preset("93L")["config"])
    assert (cfg.q_lora_rank, cfg.kv_lora_rank) == (1536, 512)
    assert (cfg.qk_head_dim, cfg.qk_pos_emb_head_dim, cfg.v_head_dim) == (128, 64, 128)
    assert cfg.multi_latent_attention is True


def test_latent_moe_is_mirrored_into_core_field():
    cfg = config_from_preset(preset("93L")["config"])
    assert cfg.moe_latent_size == cfg.k3_routed_expert_hidden_size == 3584


def test_lora_norm_eps_is_1e_6_not_rms_norm_eps():
    """KimiRMSNorm's class default is 1e-6 and the MLA LoRA norms take it."""
    cfg = config_from_preset(preset("93L")["config"])
    assert cfg.layernorm_epsilon == 1e-5
    assert cfg.k3_mla_lora_norm_eps == 1e-6


def test_layer_pattern_matches_the_release():
    cfg = config_from_preset(preset("93L")["config"])
    kda = [i for i in range(cfg.num_layers) if cfg.is_kda_layer(i)]
    mla = [i for i in range(cfg.num_layers) if not cfg.is_kda_layer(i)]
    assert len(kda) == 69 and len(mla) == 24
    # 1-indexed full-attention list, verbatim from config.json
    assert tuple(i + 1 for i in mla) == OFFICIAL_FULL_ATTN_1IDX
    # the tail breaks the 3:1 stride once: 1-indexed 92 and 93 are both MLA,
    # i.e. 0-indexed 91 and 92 -- and 0-indexed 90 (1-indexed 91) is still KDA.
    assert cfg.is_kda_layer(90) is True
    assert cfg.is_kda_layer(91) is False and cfg.is_kda_layer(92) is False
    assert kda_layers_1idx()[:4] == (1, 2, 3, 5)


def test_attn_res_slot_schedule():
    cfg = config_from_preset(preset("93L")["config"])
    appends = [i for i in range(cfg.num_layers) if cfg.appends_attn_res_slot(i)]
    assert appends == [0, 12, 24, 36, 48, 60, 72, 84]
    assert cfg.attn_res_slots_before(0) == 0
    assert cfg.attn_res_slots_before(1) == 1
    assert cfg.attn_res_slots_before(12) == 1  # the append at 12 has not happened yet
    assert cfg.attn_res_slots_before(13) == 2
    assert cfg.attn_res_slots_before(cfg.num_layers) == 8


def test_tiny_preset_has_the_true_kkkm_pattern():
    cfg = config_from_preset(preset("tiny")["config"])
    assert [cfg.is_kda_layer(i) for i in range(4)] == [True, True, True, False]


def test_unknown_backend_is_rejected():
    with pytest.raises(AssertionError):
        config_from_preset(preset("tiny")["config"], k3_kda_backend="flydsl")
