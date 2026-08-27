"""P2 / gate G12 -- arguments reach the config, and the builder is the only path.

`k3_config_from_args` collects by field name, so an argument whose dest does not
match a config field is silently dropped. This asserts the coverage rather than
trusting it.
"""

import argparse

import pytest
import torch

from kimi_k3.config.arguments import add_kimi_k3_args, apply_preset_defaults, k3_field_names
from kimi_k3.config.k3_config_builder import k3_config_from_args
from kimi_k3.config.k3_transformer_config import KimiK3TransformerConfig
from kimi_k3.config.presets import preset


def _parse(*argv):
    p = add_kimi_k3_args(argparse.ArgumentParser())
    ns = p.parse_args(list(argv))
    ns = apply_preset_defaults(ns, argv)
    ns.params_dtype = torch.bfloat16
    ns.swiglu = True
    ns.no_persist_layer_norm = False
    ns.overlap_p2p_comm = False
    ns.multi_latent_attention = True
    return ns


def test_every_k3_config_field_is_settable_from_the_command_line():
    parser = add_kimi_k3_args(argparse.ArgumentParser())
    dests = {a.dest for a in parser._actions}
    missing = k3_field_names() - dests
    # k3_kda_layers and k3_kda_pattern come from the preset, not a flag
    assert missing <= {"k3_kda_layers", "k3_kda_pattern"}, sorted(missing)


def test_flags_survive_into_the_config():
    ns = _parse("--k3-preset", "93L", "--k3-kda-backend", "fla",
                "--k3-attn-res-block-size", "6", "--no-k3-mla-fp32-attn-output")
    cfg = k3_config_from_args(ns)
    assert type(cfg) is KimiK3TransformerConfig
    assert cfg.k3_kda_backend == "fla"
    assert cfg.k3_attn_res_block_size == 6
    assert cfg.k3_mla_fp32_attn_output is False


def test_preset_supplies_the_released_geometry():
    cfg = k3_config_from_args(_parse("--k3-preset", "93L"))
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_moe_experts) == (93, 7168, 896)
    assert (cfg.moe_ffn_hidden_size, cfg.ffn_hidden_size) == (3072, 33792)
    assert cfg.moe_latent_size == 3584
    assert len(cfg.k3_kda_layers) == 69


def test_defaults_are_the_released_values():
    cfg = k3_config_from_args(_parse("--k3-preset", "93L"))
    assert cfg.k3_attn_res_block_size == 12
    assert cfg.k3_kda_gate_lower_bound == -5.0
    assert (cfg.k3_situ_beta, cfg.k3_situ_linear_beta) == (4.0, 25.0)
    assert cfg.k3_mla_lora_norm_eps == 1e-6
    assert cfg.k3_kda_backend == "eager", "the fast backend must not be the default (R5.3)"


def test_bad_backend_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        add_kimi_k3_args(argparse.ArgumentParser()).parse_args(["--k3-kda-backend", "flydsl"])


def test_preset_does_not_clobber_an_explicit_flag():
    """A preset fills gaps; it does not overrule the command line."""
    assert preset("tiny")["config"]["k3_attn_res_block_size"] == 2
    ns = _parse("--k3-preset", "tiny", "--k3-attn-res-block-size", "3")
    assert ns.k3_attn_res_block_size == 3
    # and an untouched field still comes from the preset
    assert ns.num_layers == 4
