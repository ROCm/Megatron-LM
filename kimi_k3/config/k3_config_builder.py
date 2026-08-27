"""Build a KimiK3TransformerConfig without going through core's builder.

``megatron.training.arguments.core_transformer_config_from_args`` overwrites the
caller's ``config_class`` with ``MLATransformerConfig`` when
``args.multi_latent_attention`` is set (arguments.py:1230-1232), so K3 collects
its own fields. The collection loop below mirrors core's, minus the substitution.
"""

import dataclasses
from typing import Any

import torch.nn.functional as F

from .k3_transformer_config import KimiK3TransformerConfig

# Args whose config field name differs from the arg name, or which core derives
# rather than copies. Kept explicit so a core rename shows up as a test failure
# rather than a silently-defaulted field.
_DERIVED = {
    "persist_layer_norm": lambda a: not getattr(a, "no_persist_layer_norm", False),
    "deallocate_pipeline_outputs": lambda a: True,
    "pipeline_dtype": lambda a: getattr(a, "params_dtype", None),
    "batch_p2p_comm": lambda a: not getattr(a, "overlap_p2p_comm", False),
    "num_moe_experts": lambda a: getattr(a, "num_experts", None),
    "rotary_interleaved": lambda a: getattr(a, "rotary_interleaved", False),
}


def k3_config_from_args(args: Any, **overrides) -> KimiK3TransformerConfig:
    """Collect KimiK3TransformerConfig fields from an argparse Namespace."""
    kw = {}
    for f in dataclasses.fields(KimiK3TransformerConfig):
        if hasattr(args, f.name):
            kw[f.name] = getattr(args, f.name)

    for name, fn in _DERIVED.items():
        value = fn(args)
        if value is not None or name in kw:
            kw[name] = value

    if getattr(args, "swiglu", False):
        kw["activation_func"] = F.silu
        kw["gated_linear_unit"] = True

    # K3 is multi-latent + NoPE. rope_type is inherited from MLATransformerConfig
    # and is never applied by the K3 MLA module; keeping the field at its default
    # avoids core's rope validation while the module bypasses rotary entirely.
    kw["multi_latent_attention"] = True

    kw.update(overrides)
    return KimiK3TransformerConfig(**kw)


def config_from_preset(preset: dict, **overrides) -> KimiK3TransformerConfig:
    """Build directly from a preset dict (tests, meta-device construction)."""
    kw = {k: v for k, v in preset.items() if k in _FIELD_NAMES}
    unknown = set(preset) - _FIELD_NAMES
    assert not unknown, f"preset carries non-config keys: {sorted(unknown)}"
    kw.update(overrides)
    return KimiK3TransformerConfig(**kw)


_FIELD_NAMES = {f.name for f in dataclasses.fields(KimiK3TransformerConfig)}
