"""Convert between the released checkpoint and Megatron state dicts.

Shard-wise in both directions. Three things make this more than a rename:

* **`A_log` is padded.** The checkpoint stores `[128]` with the last 32 entries
  zero; we keep `[96]`. Import asserts the padding is really zero before
  trimming, export pads it back. Measured in P0 rather than assumed.
* **Routed experts are MXFP4.** `weight_packed` + `weight_scale` become a bf16
  tensor on import (`dequantize_on_import`). After that the values are
  quantised-then-dequantised, **not** the original weights -- which were never
  released -- so "matches the released model" always means "matches the
  dequantised release".
* **`linear_fc1` is fused.** The release keeps `gate` and `up` apart (`w1`/`w3`,
  or `gate_proj`/`up_proj`); Megatron concatenates them, gate first.

Byte-exact round-trip is claimed only for an untouched checkpoint with its scales
preserved. Once anything has been trained or requantised, the criterion is
dequantised-value equivalence instead.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .mapping import A_LOG_PADDED, A_LOG_REAL, infer_layer_kinds, map_key, strip_prefix


class ConversionError(RuntimeError):
    pass


def _split_marker(target: str) -> Tuple[str, Optional[str]]:
    """`...weight@gate` -> (`...weight`, `gate`)."""
    if "@" in target:
        base, marker = target.rsplit("@", 1)
        return base, marker
    return target, None


def trim_a_log(tensor: torch.Tensor, key: str = "A_log") -> torch.Tensor:
    """`[128]` -> `[96]`, asserting the padding is zero first."""
    if tensor.numel() == A_LOG_REAL:
        return tensor
    if tensor.numel() != A_LOG_PADDED:
        raise ConversionError(f"{key}: expected [{A_LOG_PADDED}] or [{A_LOG_REAL}], got {list(tensor.shape)}")
    tail = tensor[A_LOG_REAL:]
    if bool((tail != 0).any()):
        raise ConversionError(
            f"{key}: padding is not zero (max |tail| = {tail.abs().max().item()}), so trimming "
            "would discard real values"
        )
    return tensor[:A_LOG_REAL].contiguous()


def pad_a_log(tensor: torch.Tensor) -> torch.Tensor:
    """`[96]` -> `[128]`, zero-padded, for export."""
    if tensor.numel() == A_LOG_PADDED:
        return tensor
    out = tensor.new_zeros(A_LOG_PADDED)
    out[: tensor.numel()] = tensor
    return out


def dequantize_expert(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    from ..moe.k3_qat import dequantize_mxfp4

    return dequantize_mxfp4(packed, scale).to(dtype)


def _parse_target(target: str) -> Tuple[str, Optional[str], Optional[str]]:
    """`...weight0@gate@scale` -> (`...weight0`, `gate`, `scale`).

    Routed experts carry *both* markers: `w1` is the gate half of a fused
    `linear_fc1` **and** MXFP4-packed. Splitting on only the last `@` sends the
    packed data and its scale to different buckets, where they never meet -- and
    a `w2`-only test would not notice, because `w2` has no half marker.
    """
    parts = target.split("@")
    base = parts[0]
    slot = next((p for p in parts[1:] if p in ("gate", "up")), None)
    kind = "scale" if "scale" in parts[1:] else None
    return base, slot, kind


def hf_to_mcore(
    tensors: Dict[str, torch.Tensor],
    *,
    layer_kinds: Optional[Dict[int, str]] = None,
    dequantize_on_import: bool = True,
    dtype=torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Convert one shard (or a whole checkpoint) to Megatron keys.

    Returns `(converted, skipped_keys)`. Fused halves and MXFP4 pairs are both
    held until every part has been seen, so a shard boundary between them is
    fine -- and an expert `w1` needs all three: packed data, scale, and its `w3`
    partner.
    """
    layer_kinds = layer_kinds or infer_layer_kinds(tensors)
    out: Dict[str, torch.Tensor] = {}
    plain_halves: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    quant: Dict[Tuple[str, Optional[str]], Dict[str, torch.Tensor]] = defaultdict(dict)
    skipped: List[str] = []

    for key, tensor in tensors.items():
        rel = strip_prefix(key)
        if rel is None:
            skipped.append(key)
            continue
        layer = int(rel.split(".")[1]) if rel.startswith("layers.") else None
        target = map_key(key, layer_kind=layer_kinds.get(layer) if layer is not None else None)
        if target is None:
            raise ConversionError(f"no mapping for {key}")

        base, slot, kind = _parse_target(target)
        if "mlp.experts." in base:
            quant[(base, slot)]["scale" if kind == "scale" else "packed"] = tensor
        elif slot:
            plain_halves[base][slot] = tensor
        elif base.endswith("A_log"):
            out[base] = trim_a_log(tensor, key)
        else:
            out[base] = tensor

    # dequantise the experts, then fuse whatever halves they form
    expert_halves: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for (base, slot), parts in quant.items():
        if set(parts) != {"packed", "scale"}:
            raise ConversionError(
                f"{base}{'@' + slot if slot else ''}: MXFP4 needs packed data and scale, "
                f"got {sorted(parts)}"
            )
        if not dequantize_on_import:
            out[base + (f"@{slot}" if slot else "")] = parts["packed"]
            out[base + (f"@{slot}" if slot else "") + ".scale"] = parts["scale"]
            continue
        value = dequantize_expert(parts["packed"], parts["scale"], dtype)
        if slot:
            expert_halves[base][slot] = value
        else:
            out[base] = value

    for store in (plain_halves, expert_halves):
        for base, parts in store.items():
            if set(parts) != {"gate", "up"}:
                raise ConversionError(
                    f"{base}: fused linear_fc1 needs both halves, got {sorted(parts)}"
                )
            out[base] = torch.cat([parts["gate"], parts["up"]], dim=0)

    return out, skipped


def mcore_to_hf(
    tensors: Dict[str, torch.Tensor],
    *,
    layer_kinds: Dict[int, str],
) -> Dict[str, torch.Tensor]:
    """The inverse, for bf16 tensors: split the fused halves, pad `A_log`.

    Routed experts are **not** requantised here: exporting MXFP4 belongs with the
    QAT path (P10), which owns the packing. Passing an expert tensor raises
    rather than silently writing a bf16 tensor under a `weight_packed` name.
    """
    reverse: Dict[str, str] = {}
    for layer, kind in layer_kinds.items():
        for released, target in _released_pairs(layer, kind):
            reverse[target] = released
    for released, target in _model_pairs():
        reverse[target] = released

    out: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if "mlp.experts." in key:
            raise ConversionError(
                f"{key}: exporting routed experts needs MXFP4 packing, which lands with the QAT "
                "path in P10"
            )
        if key.endswith("linear_fc1.weight"):
            gate_key, up_key = reverse.get(key + "@gate"), reverse.get(key + "@up")
            if gate_key is None or up_key is None:
                raise ConversionError(f"no reverse mapping for the fused halves of {key}")
            half = tensor.shape[0] // 2
            out[gate_key] = tensor[:half].contiguous()
            out[up_key] = tensor[half:].contiguous()
            continue

        released = reverse.get(key)
        if released is None:
            raise ConversionError(f"no reverse mapping for {key}")
        out[released] = pad_a_log(tensor) if key.endswith("A_log") else tensor
    return out


def _model_pairs() -> Iterable[Tuple[str, str]]:
    from .mapping import LM_HEAD, MODEL_RULES, TEXT_PREFIX

    for suffix, target in MODEL_RULES:
        yield TEXT_PREFIX + suffix, target
    yield LM_HEAD, "output_layer.weight"


def _released_pairs(layer: int, kind: str) -> Iterable[Tuple[str, str]]:
    """Every (released key, megatron key) for one layer, using the forward map."""
    from .mapping import LAYER_RULES, TEXT_PREFIX

    for suffix, target in LAYER_RULES:
        released = f"{TEXT_PREFIX}layers.{layer}.{suffix}"
        mapped = map_key(released, layer_kind=kind)
        if mapped is not None:
            yield released, mapped
