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

    # Must precede expert construction: TE reads NVTE_USE_CUTLASS_GROUPED_GEMM
    # when it dispatches, so setting it from the config afterwards would be too
    # late. See k3_moe_layer.set_moe_gemm_backend for the measurement.
    if config.num_moe_experts:
        from ..moe.k3_moe_layer import set_moe_gemm_backend

        set_moe_gemm_backend(getattr(config, "k3_moe_ck_grouped_gemm", True))
    if getattr(config, "moe_router_enable_expert_bias", False):
        # Without this, core's finalize_model_grads overwrites the router's
        # quantile-balanced bias with its own sign-step every iteration.
        from .core_patch import install_router_bias_dispatch

        install_router_bias_dispatch()
    block_spec = get_k3_block_spec(config, vp_stage=vp_stage)

    if pre_process is None or post_process is None:
        from megatron.core import parallel_state

        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage()
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage()

    # Default to the device core's own modules force themselves onto. Megatron and
    # TE place parameters on torch.cuda.current_device() regardless of an ambient
    # context (finding A14), so defaulting to CPU here would silently build a
    # mixed-device model whose first matmul fails.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = torch.device(device)
    with ctx:
        model = K3GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=spec["model"]["vocab_size"],
            max_sequence_length=spec["model"]["max_sequence_length"],
            position_embedding_type="none",
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    # QAT is applied after construction rather than through the spec, because it
    # parametrizes weights that core's expert modules own. Doing it here means
    # `--k3-qat-experts` is a config flag like any other instead of something a
    # caller has to remember to invoke -- the state the field was in until now.
    if getattr(config, "k3_qat_experts", False):
        from ..moe.k3_qat_wiring import enable_qat_experts

        enable_qat_experts(model)
    return model
