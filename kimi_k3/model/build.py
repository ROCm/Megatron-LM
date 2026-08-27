"""One call from a preset to a constructed K3 model.

Keeps the assembly order in a single place: build the config with *our* builder
(never core's, which would substitute the config class), build a per-layer block
spec, and construct the model with the block injection in scope.
"""

from typing import Optional

import torch

from ..config.k3_config_builder import config_from_preset
from ..config.presets import preset as get_preset
from ..specs.layer_specs import get_k3_block_spec
from .k3_gpt_model import K3GPTModel


def build_k3_model(
    preset_name: str = "tiny",
    *,
    device: Optional[str] = None,
    pre_process: Optional[bool] = None,
    post_process: Optional[bool] = None,
    vp_stage: Optional[int] = None,
    allow_official: bool = False,
    **config_overrides,
) -> K3GPTModel:
    """Construct a K3 model from a named preset.

    **`device="meta"` does not make this free.** Megatron and TE modules place
    their parameters on `torch.cuda.current_device()` explicitly, ignoring an
    ambient device context: on the tiny preset 93 of 114 parameters land on the
    GPU anyway (only our own AttnRes mixers honour it). So an official preset
    would really try to allocate 215 B or 2.78 T parameters, and this function
    refuses to build one unless the caller says `allow_official=True` and means
    it. Use `tools/mem_budget.py` to inspect official presets (rule R4.3).
    """
    if preset_name != "tiny" and not allow_official:
        raise ValueError(
            f"refusing to construct the {preset_name!r} preset: meta device does not "
            "prevent allocation for TE/Megatron modules, so this would try to "
            "materialise the real parameter count. Use kimi_k3.tools.mem_budget for "
            "analytic inspection, or pass allow_official=True on hardware that fits it."
        )
    spec = get_preset(preset_name)
    config = config_from_preset(spec["config"], **config_overrides)
    block_spec = get_k3_block_spec(config, vp_stage=vp_stage)

    if pre_process is None or post_process is None:
        from megatron.core import parallel_state

        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage()
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage()

    ctx = torch.device(device) if device else torch.device("cpu")
    with ctx:
        return K3GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=spec["model"]["vocab_size"],
            max_sequence_length=spec["model"]["max_sequence_length"],
            position_embedding_type="none",
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
