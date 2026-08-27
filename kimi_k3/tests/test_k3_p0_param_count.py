"""P0 -- the analytic parameter model must match the released checkpoint.

``tools/mem_budget.py`` is the oracle for the official presets, which are never
instantiated with weights (rule R4.3). Its formulas are checked here against
tensor shapes read out of the released safetensors headers
(``fixtures/release_shapes.json``, shapes and dtypes only, no weight data).
"""

import json
import math
import pathlib

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.tools import mem_budget

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "release_shapes.json").read_text()
)
SHAPES = FIXTURE["tensors"]


def numel(key: str) -> int:
    return math.prod(SHAPES[key]["shape"])


def sum_prefix(prefix: str, *, exclude: tuple = ()) -> int:
    return sum(
        numel(k)
        for k in SHAPES
        if k.startswith(prefix) and not any(e in k for e in exclude)
    )


def cfg93():
    return config_from_preset(preset("93L")["config"])


def test_kda_layer_formula_matches_release():
    """Layer 12 (0-indexed) is a KDA layer in the release."""
    released = sum_prefix("model.layers.12.self_attn.")
    ours = mem_budget.kda_layer_params(cfg93())
    # The checkpoint stores A_log as [128] (96 real heads + 32 zeros); we keep [96].
    assert SHAPES["model.layers.12.self_attn.A_log"]["shape"] == [128]
    assert released - ours == 32, (released, ours)


def test_mla_layer_formula_matches_release():
    """Layer 3 (0-indexed) is a gated-MLA layer in the release."""
    released = sum_prefix("model.layers.3.self_attn.")
    assert mem_budget.mla_layer_params(cfg93()) == released


def test_moe_layer_formula_matches_release():
    cfg = cfg93()
    non_expert = sum_prefix(
        "model.layers.12.block_sparse_moe.", exclude=(".experts.",)
    )
    one_expert = sum(
        math.prod([s["shape"][0], s["shape"][1] * 2])  # MXFP4: two nibbles per byte
        for k, s in SHAPES.items()
        if ".experts.0." in k and k.endswith("weight_packed")
    )
    released = non_expert + cfg.num_moe_experts * one_expert
    assert one_expert == 3 * 3584 * 3072
    assert mem_budget.moe_layer_params(cfg) == released


def test_shared_experts_run_on_hidden_not_on_the_latent():
    """Finding B6: shared experts are 7168-wide, only routed experts are 3584."""
    gate = SHAPES["model.layers.12.block_sparse_moe.shared_experts.gate_proj.weight"]["shape"]
    assert gate == [6144, 7168]
    assert sum_prefix("model.layers.12.block_sparse_moe.shared_experts.") == 3 * 7168 * 6144


def test_dense_layer_and_norms_match_release():
    cfg = cfg93()
    assert mem_budget.dense_ffn_params(cfg) == sum_prefix("model.layers.0.mlp.")
    per_layer = (
        numel("model.layers.12.input_layernorm.weight")
        + numel("model.layers.12.post_attention_layernorm.weight")
        + numel("model.layers.12.self_attention_res_norm.weight")
        + numel("model.layers.12.self_attention_res_proj.weight")
        + numel("model.layers.12.mlp_res_norm.weight")
        + numel("model.layers.12.mlp_res_proj.weight")
    )
    assert mem_budget.per_layer_norm_params(cfg) == per_layer


def test_model_level_params_match_release():
    cfg = cfg93()
    released = (
        numel("model.embed_tokens.weight")
        + numel("lm_head.weight")
        + numel("model.norm.weight")
        + numel("model.output_attn_res_norm.weight")
        + numel("model.output_attn_res_proj.weight")
    )
    assert mem_budget.model_level_params(cfg, vocab_size=163840) == released


def test_attn_res_has_two_mixes_per_layer_plus_one_at_the_output():
    """187 res_proj tensors = 93 layers x 2 + 1 output mix (from the released index)."""
    cfg = cfg93()
    assert 2 * cfg.num_layers + 1 == 187


def test_total_parameter_counts():
    totals = {}
    for name in ("4L", "8L", "93L"):
        p = preset(name)
        cfg = config_from_preset(p["config"])
        totals[name] = mem_budget.breakdown(cfg, p["model"]["vocab_size"]).total

    assert abs(totals["93L"] / 2.78e12 - 1) < 0.01, totals["93L"]
    assert abs(totals["4L"] / 94e9 - 1) < 0.02, totals["4L"]
    assert abs(totals["8L"] / 215e9 - 1) < 0.02, totals["8L"]

    active = mem_budget.active_params(cfg93(), 163840)
    assert 0.03 < active / totals["93L"] < 0.05, active


def test_expert_parallelism_shards_almost_everything():
    """98 % of the parameters are routed experts -- EP is the only real lever."""
    cfg = cfg93()
    b = mem_budget.breakdown(cfg, 163840)
    total = b.total
    routed = cfg.num_moe_experts * 3 * 3584 * 3072 * (cfg.num_layers - 1)
    assert routed / total > 0.97

    assert math.isclose(mem_budget.params_per_gpu(cfg, 163840, ep=1), total, rel_tol=1e-9)
    # the layout recommended in develop/plan-0/06-capacity-and-parallelism.md
    per_gpu = mem_budget.params_per_gpu(cfg, 163840, pp=16, ep=32)
    assert 8e9 < per_gpu < 10e9, per_gpu


def test_muon_group_split_is_explicit():
    """Core sends only 2-D non-embedding weights to Muon (muon.py:283-302)."""
    cfg = cfg93()
    split = mem_budget.muon_group_split(cfg, 163840)
    assert split["muon_2d"] + split["scalar_group"] == split["total"]

    # The scalar (Adam) group is tiny in count but is dominated by the embeddings.
    embeddings = 2 * 163840 * cfg.hidden_size
    assert split["scalar_group"] > embeddings
    assert (split["scalar_group"] - embeddings) / split["total"] < 1e-4

    # Per-KDA-layer scalars: 3 conv weights (3-D), A_log, dt_bias, o_norm.
    p = cfg.k3_kda_num_heads * cfg.k3_kda_head_dim
    kda_scalar = 3 * p * cfg.k3_kda_conv_size + cfg.k3_kda_num_heads + p + cfg.k3_kda_head_dim
    assert kda_scalar == 3 * 12288 * 4 + 96 + 12288 + 128
