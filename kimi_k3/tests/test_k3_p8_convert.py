"""P8 / gates G30–G31 -- the converter accounts for every tensor.

G30 maps a rebuilt copy of the **real** released index: 497,220 tensors, and the
only acceptable outcome is that each one is either mapped or explicitly skipped.
The fixture holds the index's key patterns and counts rather than its 60 MB, and
the test reconstructs an equivalent index from them.
"""

import json
import pathlib

import pytest
import torch

from kimi_k3.tools.convert import (
    ConversionError,
    hf_to_mcore,
    mcore_to_hf,
    pad_a_log,
    trim_a_log,
)
from kimi_k3.tools.mapping import EXPECTED_SKIPPED, dry_run, infer_layer_kinds, map_key

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "release_index_patterns.json").read_text()
)


#: Tensors only a KDA layer has, and only a gated-MLA layer has. Matched as exact
#: suffixes -- a substring test gets this wrong, because "b_proj" is inside
#: "kv_b_proj" and would hand every MLA layer's keys to the KDA branch.
KDA_ONLY = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
    "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight", "self_attn.b_proj.weight",
    "self_attn.A_log", "self_attn.dt_bias", "self_attn.o_norm.weight",
)
MLA_ONLY = (
    "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
)


def rebuild_index() -> dict:
    """An index with the released structure, from the committed patterns."""
    kda = set(FIXTURE["kda_layers_0indexed"])
    mla = set(FIXTURE["mla_layers_0indexed"])
    dense = set(FIXTURE["dense_ffn_layers_0indexed"])
    experts = FIXTURE["num_experts"]

    def belongs(pattern: str, layer: int) -> bool:
        suffix = pattern.split("layers.{L}.", 1)[1]
        if suffix in KDA_ONLY:
            return layer in kda
        if suffix in MLA_ONLY:
            return layer in mla
        if suffix.startswith("mlp."):
            return layer in dense
        if suffix.startswith("block_sparse_moe."):
            return layer not in dense
        return True  # norms, attention residuals, and the two shared self_attn tensors

    keys = {}
    for pattern in FIXTURE["pattern_counts"]:
        if "{B}" in pattern:  # the vision encoder's own blocks
            for b in range(FIXTURE["vision_encoder_blocks"]):
                keys[pattern.replace("{B}", str(b))] = "shard"
            continue
        if "{L}" not in pattern:
            keys[pattern] = "shard"
            continue
        for layer in sorted(kda | mla):
            if not belongs(pattern, layer):
                continue
            if "{E}" in pattern:
                for e in range(experts):
                    keys[pattern.replace("{L}", str(layer)).replace("{E}", str(e))] = "shard"
            else:
                keys[pattern.replace("{L}", str(layer))] = "shard"
    return keys


def test_every_released_tensor_is_mapped_or_explicitly_skipped():
    index = rebuild_index()
    assert len(index) == FIXTURE["_provenance"]["total_tensors"], (
        len(index), FIXTURE["_provenance"]["total_tensors"]
    )
    report = dry_run(index)
    assert report.unmapped == [], report.unmapped[:10]
    assert report.skipped == EXPECTED_SKIPPED
    assert report.mapped + report.skipped == len(index)
    assert report.problems == [], report.problems


def test_layer_kinds_are_inferred_from_the_tensors_that_exist():
    """`self_attn` houses both kinds, so the name cannot tell them apart."""
    kinds = infer_layer_kinds(rebuild_index())
    assert sum(1 for v in kinds.values() if v == "kda") == 69
    assert sum(1 for v in kinds.values() if v == "mla") == 24
    assert kinds[0] == "kda" and kinds[3] == "mla"
    assert kinds[91] == "mla" and kinds[92] == "mla"  # the double-MLA tail


def test_every_moe_layer_has_all_896_experts():
    report = dry_run(rebuild_index())
    assert set(report.experts_seen.values()) == {896}
    assert report.quantized_pairs == 92 * 896 * 3


def test_vision_tensors_are_skipped_by_name_not_by_omission():
    index = rebuild_index()
    vision = [k for k in index if k.startswith(("vision_tower.", "mm_projector."))]
    assert len(vision) == EXPECTED_SKIPPED
    assert all(map_key(k) is None for k in vision)


# --- invariants --------------------------------------------------------------


def test_a_log_is_trimmed_only_when_the_padding_is_zero():
    padded = torch.zeros(128)
    padded[:96] = torch.randn(96)
    torch.testing.assert_close(trim_a_log(padded), padded[:96])
    torch.testing.assert_close(pad_a_log(trim_a_log(padded)), padded)

    corrupt = padded.clone()
    corrupt[100] = 0.5
    with pytest.raises(ConversionError, match="padding is not zero"):
        trim_a_log(corrupt)


@pytest.mark.parametrize("matrix", ["w1", "w2"])
def test_unpaired_mxfp4_tensors_are_rejected(matrix):
    """`w1` is *both* a fused half and MXFP4-packed; `w2` is only the latter.

    Testing only `w2` would have missed a real bug: the packed data and its scale
    were being routed to different buckets for the half-carrying matrices.
    """
    tensors = {
        f"language_model.model.layers.1.block_sparse_moe.experts.0.{matrix}.weight_packed":
            torch.zeros(8, 4, dtype=torch.uint8),
    }
    with pytest.raises(ConversionError, match="packed data and scale"):
        hf_to_mcore(tensors, layer_kinds={1: "kda"})


def test_quantised_expert_halves_are_dequantised_then_fused():
    """The path that was broken: w1 and w3 are each MXFP4 *and* half of fc1."""
    from kimi_k3.moe.k3_qat import quantize_mxfp4

    torch.manual_seed(0)
    gate, up = torch.randn(6, 64), torch.randn(6, 64)
    pre = "language_model.model.layers.1.block_sparse_moe.experts.5."
    tensors = {}
    for name, w in (("w1", gate), ("w3", up)):
        packed, scale = quantize_mxfp4(w)
        tensors[pre + f"{name}.weight_packed"] = packed
        tensors[pre + f"{name}.weight_scale"] = scale

    mcore, _ = hf_to_mcore(tensors, layer_kinds={1: "kda"})
    fused = mcore["decoder.layers.1.mlp.experts.linear_fc1.weight5"]
    assert fused.shape == (12, 64) and fused.dtype == torch.bfloat16
    assert ((fused[:6].float() - gate).norm() / gate.norm()) < 0.2
    assert ((fused[6:].float() - up).norm() / up.norm()) < 0.2


def test_fused_linear_fc1_needs_both_halves():
    tensors = {
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight":
            torch.randn(4, 8),
    }
    with pytest.raises(ConversionError, match="both halves"):
        hf_to_mcore(tensors, layer_kinds={1: "kda"})


# --- round-trip (G31) --------------------------------------------------------


def _synthetic_layer(layer=1, hidden=16, latent=8, inter=8, heads=2, head_dim=4):
    p = heads * head_dim
    a_log = torch.zeros(128)
    a_log[:96] = torch.randn(96)
    pre = f"language_model.model.layers.{layer}."
    return {
        pre + "self_attn.q_proj.weight": torch.randn(p, hidden),
        pre + "self_attn.A_log": a_log,
        pre + "self_attn.dt_bias": torch.randn(p),
        pre + "self_attn.o_proj.weight": torch.randn(hidden, p),
        pre + "input_layernorm.weight": torch.randn(hidden),
        pre + "mlp_res_proj.weight": torch.randn(1, hidden),
        pre + "block_sparse_moe.gate.weight": torch.randn(4, hidden),
        pre + "block_sparse_moe.shared_experts.gate_proj.weight": torch.randn(inter, hidden),
        pre + "block_sparse_moe.shared_experts.up_proj.weight": torch.randn(inter, hidden),
        pre + "block_sparse_moe.shared_experts.down_proj.weight": torch.randn(hidden, inter),
    }


def test_bf16_tensors_round_trip_exactly():
    torch.manual_seed(0)
    hf = {k: v.bfloat16() if v.dtype.is_floating_point else v for k, v in _synthetic_layer().items()}
    hf["language_model.model.layers.1.self_attn.A_log"] = hf[
        "language_model.model.layers.1.self_attn.A_log"
    ].float()

    mcore, skipped = hf_to_mcore(hf, layer_kinds={1: "kda"})
    assert skipped == []
    back = mcore_to_hf(mcore, layer_kinds={1: "kda"})

    for key, original in hf.items():
        assert key in back, f"{key} did not survive the round trip"
        assert torch.equal(back[key], original), key


def test_fused_halves_are_concatenated_gate_first():
    torch.manual_seed(0)
    hf = _synthetic_layer()
    mcore, _ = hf_to_mcore(hf, layer_kinds={1: "kda"})
    fused = mcore["decoder.layers.1.mlp.shared_experts.linear_fc1.weight"]
    gate = hf["language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"]
    up = hf["language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj.weight"]
    assert torch.equal(fused[: gate.shape[0]], gate)
    assert torch.equal(fused[gate.shape[0] :], up)


def test_mxfp4_experts_are_dequantised_on_import():
    from kimi_k3.moe.k3_qat import quantize_mxfp4

    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    packed, scale = quantize_mxfp4(weight)
    pre = "language_model.model.layers.1.block_sparse_moe.experts.3."
    mcore, _ = hf_to_mcore({pre + "w2.weight_packed": packed, pre + "w2.weight_scale": scale},
                           layer_kinds={1: "kda"})
    got = mcore["decoder.layers.1.mlp.experts.linear_fc2.weight3"]
    assert got.dtype == torch.bfloat16
    # dequantised, so close to the original but not equal -- and the report says so
    assert ((got.float() - weight).norm() / weight.norm()) < 0.2
    assert not torch.equal(got.float(), weight)


def test_exporting_experts_refuses_rather_than_writing_the_wrong_thing():
    with pytest.raises(ConversionError, match="MXFP4 packing"):
        mcore_to_hf(
            {"decoder.layers.1.mlp.experts.linear_fc2.weight0": torch.randn(4, 4)},
            layer_kinds={1: "kda"},
        )
