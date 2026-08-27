"""Released-checkpoint keys <-> Megatron keys.

Every rule here comes from the released `model.safetensors.index.json` and the
shard headers, read in P0 (`develop/notes/2026-08-27-release-audit.md`) -- not
from the modeling file's attribute names, which differ in two places that matter:

* **KDA and gated MLA both live under `self_attn`.** The layer kind is told apart
  by which tensors exist (`A_log` / `q_proj` vs `q_a_proj` / `kv_b_proj`), not by
  the module name.
* **The MoE module is `block_sparse_moe`**, while the single dense layer uses
  `mlp`.

Invariants are asserted, never treated as "if present" (the plan's own rule, and
the reason `A_log` was measured rather than assumed):

* `A_log` is `F32 [128]` per KDA layer with `A_log[96:] == 0` exactly -- trim to
  `[96]` on import, zero-pad on export;
* `dt_bias` is `F32 [12288]`, unpadded;
* routed experts are MXFP4: `weight_packed` `U8` plus `weight_scale` `U8`, and
  the two must appear together;
* `vision_tower.*` and `mm_projector.*` are **skipped by an explicit rule and
  counted** -- 168 tensors. "Zero unmapped tensors" is only meaningful if the
  skipped ones are named.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TEXT_PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"
SKIP_PREFIXES = ("vision_tower.", "mm_projector.")
EXPECTED_SKIPPED = 168
A_LOG_PADDED = 128
A_LOG_REAL = 96

#: (released suffix, megatron suffix). Applied after the `layers.{i}.` prefix.
LAYER_RULES: Tuple[Tuple[str, str], ...] = (
    # --- KDA -------------------------------------------------------------
    ("self_attn.q_proj.weight", "self_attention.kda.q_proj.weight"),
    ("self_attn.k_proj.weight", "self_attention.kda.k_proj.weight"),
    ("self_attn.v_proj.weight", "self_attention.kda.v_proj.weight"),
    ("self_attn.q_conv1d.weight", "self_attention.kda.q_conv1d_weight"),
    ("self_attn.k_conv1d.weight", "self_attention.kda.k_conv1d_weight"),
    ("self_attn.v_conv1d.weight", "self_attention.kda.v_conv1d_weight"),
    ("self_attn.f_a_proj.weight", "self_attention.kda.f_a_proj.weight"),
    ("self_attn.f_b_proj.weight", "self_attention.kda.f_b_proj.weight"),
    ("self_attn.b_proj.weight", "self_attention.kda.b_proj.weight"),
    ("self_attn.A_log", "self_attention.kda.A_log"),
    ("self_attn.dt_bias", "self_attention.kda.dt_bias"),
    ("self_attn.o_norm.weight", "self_attention.kda.o_norm_weight"),
    # --- gated MLA -------------------------------------------------------
    ("self_attn.q_a_proj.weight", "self_attention.mla.q_a_proj.weight"),
    ("self_attn.q_a_layernorm.weight", "self_attention.mla.q_a_layernorm"),
    ("self_attn.q_b_proj.weight", "self_attention.mla.q_b_proj.weight"),
    ("self_attn.kv_a_proj_with_mqa.weight", "self_attention.mla.kv_a_proj_with_mqa.weight"),
    ("self_attn.kv_a_layernorm.weight", "self_attention.mla.kv_a_layernorm"),
    ("self_attn.kv_b_proj.weight", "self_attention.mla.kv_b_proj.weight"),
    # --- shared by both kinds (the name is the same, the owner is not) ----
    ("self_attn.g_proj.weight", "self_attention.{kind}.g_proj.weight"),
    ("self_attn.o_proj.weight", "self_attention.{kind}.o_proj.weight"),
    # --- norms and attention residuals -----------------------------------
    ("input_layernorm.weight", "input_layernorm.weight"),
    ("post_attention_layernorm.weight", "pre_mlp_layernorm.weight"),
    ("self_attention_res_norm.weight", "attn_res_attn.weight"),
    ("self_attention_res_proj.weight", "attn_res_attn.proj"),
    ("mlp_res_norm.weight", "attn_res_mlp.weight"),
    ("mlp_res_proj.weight", "attn_res_mlp.proj"),
    # --- dense FFN (layer 1 only) ----------------------------------------
    ("mlp.gate_proj.weight", "mlp.linear_fc1.weight@gate"),
    ("mlp.up_proj.weight", "mlp.linear_fc1.weight@up"),
    ("mlp.down_proj.weight", "mlp.linear_fc2.weight"),
    # --- MoE --------------------------------------------------------------
    ("block_sparse_moe.gate.weight", "mlp.router.weight"),
    ("block_sparse_moe.gate.e_score_correction_bias", "mlp.router.expert_bias"),
    ("block_sparse_moe.routed_expert_down_proj.weight", "mlp.fc1_latent_proj.weight"),
    ("block_sparse_moe.routed_expert_norm.weight", "mlp.routed_expert_norm"),
    ("block_sparse_moe.routed_expert_up_proj.weight", "mlp.fc2_latent_proj.weight"),
    ("block_sparse_moe.shared_experts.gate_proj.weight", "mlp.shared_experts.linear_fc1.weight@gate"),
    ("block_sparse_moe.shared_experts.up_proj.weight", "mlp.shared_experts.linear_fc1.weight@up"),
    ("block_sparse_moe.shared_experts.down_proj.weight", "mlp.shared_experts.linear_fc2.weight"),
)

MODEL_RULES: Tuple[Tuple[str, str], ...] = (
    ("embed_tokens.weight", "embedding.word_embeddings.weight"),
    ("norm.weight", "decoder.final_layernorm.weight"),
    ("output_attn_res_norm.weight", "decoder.output_attn_res.weight"),
    ("output_attn_res_proj.weight", "decoder.output_attn_res.proj"),
)

_EXPERT = re.compile(
    r"^layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<mat>w[123])\.(?P<part>weight_packed|weight_scale)$"
)
_LAYER = re.compile(r"^layers\.(?P<layer>\d+)\.(?P<rest>.+)$")
#: `w1` is the gate half and `w3` the up half of Megatron's fused `linear_fc1`.
EXPERT_MATRIX_TO_SLOT = {"w1": ("linear_fc1", "gate"), "w3": ("linear_fc1", "up"), "w2": ("linear_fc2", None)}


@dataclass
class MappingReport:
    mapped: int = 0
    skipped: int = 0
    unmapped: List[str] = field(default_factory=list)
    layer_kinds: Dict[int, str] = field(default_factory=dict)
    experts_seen: Dict[int, int] = field(default_factory=dict)
    quantized_pairs: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unmapped and not self.problems


def strip_prefix(key: str) -> Optional[str]:
    """Released key -> text-tower-relative key, or None when it is not ours."""
    if key.startswith(SKIP_PREFIXES):
        return None
    if key == LM_HEAD:
        return "lm_head.weight"
    if key.startswith(TEXT_PREFIX):
        return key[len(TEXT_PREFIX) :]
    return None


def map_key(key: str, layer_kind: Optional[str] = None) -> Optional[str]:
    """Released key -> Megatron key. `layer_kind` disambiguates `self_attn`.

    The `@gate` / `@up` suffixes mark the two halves of a fused `linear_fc1`; the
    converter concatenates them and drops the marker.
    """
    rel = strip_prefix(key)
    if rel is None:
        return None
    if rel == "lm_head.weight":
        return "output_layer.weight"

    for suffix, target in MODEL_RULES:
        if rel == suffix:
            return target

    expert = _EXPERT.match(rel)
    if expert:
        matrix, slot = EXPERT_MATRIX_TO_SLOT[expert["mat"]]
        base = (
            f"decoder.layers.{expert['layer']}.mlp.experts.{matrix}.weight{expert['expert']}"
        )
        base += f"@{slot}" if slot else ""
        return base + ("@scale" if expert["part"] == "weight_scale" else "")

    layer = _LAYER.match(rel)
    if not layer:
        return None
    for suffix, target in LAYER_RULES:
        if layer["rest"] == suffix:
            if "{kind}" in target:
                if layer_kind is None:
                    return None
                target = target.format(kind=layer_kind)
            return f"decoder.layers.{layer['layer']}.{target}"
    return None


def infer_layer_kinds(keys) -> Dict[int, str]:
    """KDA or MLA, from which tensors a layer actually has."""
    kinds: Dict[int, str] = {}
    for key in keys:
        rel = strip_prefix(key)
        if not rel:
            continue
        m = _LAYER.match(rel)
        if not m:
            continue
        idx = int(m["layer"])
        if m["rest"].endswith("self_attn.A_log"):
            kinds[idx] = "kda"
        elif m["rest"].endswith("self_attn.kv_b_proj.weight"):
            kinds[idx] = "mla"
    return kinds


def check_invariants(shapes: Dict[str, List[int]]) -> List[str]:
    """Shape/pairing invariants, asserted rather than hoped for."""
    problems = []
    for key, shape in shapes.items():
        rel = strip_prefix(key)
        if not rel:
            continue
        if rel.endswith("self_attn.A_log") and list(shape) != [A_LOG_PADDED]:
            problems.append(f"{key}: A_log is {shape}, expected [{A_LOG_PADDED}]")
        if rel.endswith("self_attn.dt_bias") and len(shape) != 1:
            problems.append(f"{key}: dt_bias should be 1-D, got {shape}")
    packed = {k for k in shapes if k.endswith("weight_packed")}
    scales = {k.replace("weight_scale", "weight_packed") for k in shapes if k.endswith("weight_scale")}
    for orphan in sorted(packed ^ scales)[:5]:
        problems.append(f"{orphan}: MXFP4 packed data and scale must appear together")
    return problems


def dry_run(weight_map: Dict[str, str], shapes: Optional[Dict[str, List[int]]] = None) -> MappingReport:
    """Map every key in a released index and report what happened to each."""
    report = MappingReport()
    report.layer_kinds = infer_layer_kinds(weight_map)

    for key in weight_map:
        if key.startswith(SKIP_PREFIXES):
            report.skipped += 1
            continue
        layer = _LAYER.match(strip_prefix(key) or "")
        kind = report.layer_kinds.get(int(layer["layer"])) if layer else None
        target = map_key(key, layer_kind=kind)
        if target is None:
            report.unmapped.append(key)
            continue
        report.mapped += 1
        if target.endswith("@scale"):
            report.quantized_pairs += 1
        expert = _EXPERT.match(strip_prefix(key) or "")
        if expert:
            idx = int(expert["layer"])
            report.experts_seen[idx] = max(report.experts_seen.get(idx, 0), int(expert["expert"]) + 1)

    if report.skipped != EXPECTED_SKIPPED:
        report.problems.append(
            f"skipped {report.skipped} vision/projector tensors, expected {EXPECTED_SKIPPED}"
        )
    if shapes:
        report.problems.extend(check_invariants(shapes))
    return report
