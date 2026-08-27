"""P2 / gate G11 -- every layer must be the kind the release says it is.

The plan is computed from the config alone, so this runs on CPU and needs no
model. That matters: it is the cheapest possible place to catch the `+1` indexing
mistake, which is the single likeliest bug in the whole project.
"""

import pytest

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import OFFICIAL_FULL_ATTN_1IDX, preset
from kimi_k3.specs.layer_specs import DENSE, KDA, MLA, MOE, k3_layer_plan, local_layer_plan


@pytest.fixture()
def cfg93():
    return config_from_preset(preset("93L")["config"])


def test_counts_match_the_release(cfg93):
    plan = k3_layer_plan(cfg93)
    assert len(plan) == 93
    assert sum(p.attention == KDA for p in plan) == 69
    assert sum(p.attention == MLA for p in plan) == 24


def test_mla_layers_are_exactly_the_released_list(cfg93):
    plan = k3_layer_plan(cfg93)
    mla_1indexed = tuple(p.layer_idx + 1 for p in plan if p.attention == MLA)
    assert mla_1indexed == OFFICIAL_FULL_ATTN_1IDX


def test_the_tail_breaks_the_three_to_one_stride_once(cfg93):
    """1-indexed 92 and 93 are both MLA -- 0-indexed 91 and 92."""
    plan = k3_layer_plan(cfg93)
    assert plan[90].attention == KDA
    assert plan[91].attention == MLA and plan[92].attention == MLA


def test_first_layer_is_kda_with_a_dense_ffn(cfg93):
    plan = k3_layer_plan(cfg93)
    assert plan[0].attention == KDA
    assert plan[0].ffn == DENSE and plan[0].ffn_hidden_size == 33792
    assert all(p.ffn == MOE for p in plan[1:])
    assert plan[1].ffn_hidden_size == 3072


def test_attn_res_slots(cfg93):
    plan = k3_layer_plan(cfg93)
    appending = [p.layer_idx for p in plan if p.appends_attn_res_slot]
    assert appending == [0, 12, 24, 36, 48, 60, 72, 84]
    assert plan[0].slots_on_entry == 0
    assert plan[12].slots_on_entry == 1  # the append at 12 has not happened yet
    assert plan[13].slots_on_entry == 2
    assert plan[-1].slots_on_entry == 8


def test_tiny_preset_is_a_true_kkkm_block():
    cfg = config_from_preset(preset("tiny")["config"])
    plan = k3_layer_plan(cfg)
    assert [p.attention for p in plan] == [KDA, KDA, KDA, MLA]
    assert plan[0].ffn == DENSE and all(p.ffn == MOE for p in plan[1:])


def test_local_plan_is_this_stage_only(cfg93, monkeypatch):
    """A stage must be handed its own layers; TransformerBlock builds len(specs)."""
    import megatron.core.transformer.transformer_block as tb
    import kimi_k3.specs.layer_specs as ls

    monkeypatch.setattr(tb, "get_num_layers_to_build", lambda *a, **kw: 12)
    monkeypatch.setattr(ls, "get_transformer_layer_offset", lambda *a, **kw: 24)
    local = local_layer_plan(cfg93)
    assert [p.layer_idx for p in local] == list(range(24, 36))
    assert local[0].slots_on_entry == 2


def test_plan_is_pure_config(cfg93):
    """No CUDA, no distributed state, no model -- so it can run in CI stage 0."""
    a = k3_layer_plan(cfg93)
    b = k3_layer_plan(config_from_preset(preset("93L")["config"]))
    assert a == b
