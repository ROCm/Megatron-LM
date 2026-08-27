"""P8 / gate G32 -- anchored parity against the release, on real weights.

This is the only test that touches released weights, so it skips unless they have
been fetched:

    python -m kimi_k3.tools.fetch_release_tensors \\
        --match layers.0.self_attn --out /tmp/k3_layer0_self_attn.pt

847 MiB by range request, rather than a ~16 GiB shard. Running the *release's
own* `KimiDeltaAttention` alongside ours also needs transformers >= 4.56, which
the container does not have; install it to a `--target` directory and point
`PYTHONPATH` at it rather than changing the environment.
"""

import os
import pathlib

import pytest
import torch

WEIGHTS = pathlib.Path(os.environ.get("K3_REAL_WEIGHTS", "/tmp/k3_layer0_self_attn.pt"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not WEIGHTS.exists(),
        reason=f"real weights not fetched; see this module's docstring ({WEIGHTS})",
    ),
]

PREFIX = "language_model.model.layers.0.self_attn."


def load_real():
    return {k.replace(PREFIX, ""): v for k, v in torch.load(WEIGHTS, weights_only=True).items()}


def test_the_checkpoint_a_log_is_padded_and_the_release_module_is_not():
    """Why the converter's trim rule exists, shown from both sides.

    The checkpoint stores `[128]`; the release's own module declares `[96]`. So
    loading the checkpoint into the release's own code without trimming fails --
    the padding is real, and handling it is not optional.
    """
    from kimi_k3.tools.convert import trim_a_log

    real = load_real()
    assert list(real["A_log"].shape) == [128]
    assert bool((real["A_log"][96:] == 0).all())
    assert list(trim_a_log(real["A_log"]).shape) == [96]


def test_our_module_accepts_the_released_parameter_layout():
    """Zero missing, zero unexpected: the converter is a rename, not a reshape."""
    from kimi_k3.attention.kda import KimiDeltaAttention
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.tools.convert import trim_a_log

    real = load_real()
    module = KimiDeltaAttention(config_from_preset(preset("93L")["config"])).cuda().bfloat16()
    state = {
        "q_proj.weight": real["q_proj.weight"], "k_proj.weight": real["k_proj.weight"],
        "v_proj.weight": real["v_proj.weight"], "o_proj.weight": real["o_proj.weight"],
        "q_conv1d_weight": real["q_conv1d.weight"], "k_conv1d_weight": real["k_conv1d.weight"],
        "v_conv1d_weight": real["v_conv1d.weight"],
        "f_a_proj.weight": real["f_a_proj.weight"], "f_b_proj.weight": real["f_b_proj.weight"],
        "g_proj.weight": real["g_proj.weight"], "b_proj.weight": real["b_proj.weight"],
        "A_log": trim_a_log(real["A_log"]), "dt_bias": real["dt_bias"],
        "o_norm_weight": real["o_norm.weight"],
    }
    missing, unexpected = module.load_state_dict(
        {k: v.cuda() for k, v in state.items()}, strict=False
    )
    assert unexpected == [], unexpected
    assert [m for m in missing if torch.is_tensor(module.state_dict().get(m))] == []
    assert sum(p.numel() for p in module.parameters()) == 443_740_384


@pytest.mark.skipif(
    "K3_REFERENCE_PATH" not in os.environ,
    reason="set K3_REFERENCE_PATH to the directory holding the release's modeling files, "
           "and PYTHONPATH to a transformers>=4.56 install",
)
def test_matches_the_release_module_on_real_weights():
    """The anchored comparison itself: our KDA against the release's own class."""
    import sys

    from kimi_k3.attention.kda import KimiDeltaAttention
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.tests.tolerance import KDA_FWD_BF16, assert_within
    from kimi_k3.tools.convert import trim_a_log

    sys.path.insert(0, os.environ["K3_REFERENCE_PATH"])
    import json

    from k3pkg.modeling_kimi_linear import KimiDeltaAttention as ReleaseKDA

    real = load_real()
    real["A_log"] = trim_a_log(real["A_log"])
    cfg_json = json.load(open(os.path.join(os.environ["K3_REFERENCE_PATH"], "config.json")))
    cfg = type("Cfg", (), dict(cfg_json["text_config"]))()

    reference = ReleaseKDA(cfg, layer_idx=0).cuda().bfloat16().eval()
    reference.load_state_dict({k: v.cuda() for k, v in real.items()}, strict=False)

    ours = KimiDeltaAttention(
        config_from_preset(preset("93L")["config"], k3_kda_backend="fla")
    ).cuda().bfloat16().eval()
    ours.load_state_dict(
        {
            "q_proj.weight": real["q_proj.weight"], "k_proj.weight": real["k_proj.weight"],
            "v_proj.weight": real["v_proj.weight"], "o_proj.weight": real["o_proj.weight"],
            "q_conv1d_weight": real["q_conv1d.weight"], "k_conv1d_weight": real["k_conv1d.weight"],
            "v_conv1d_weight": real["v_conv1d.weight"],
            "f_a_proj.weight": real["f_a_proj.weight"], "f_b_proj.weight": real["f_b_proj.weight"],
            "g_proj.weight": real["g_proj.weight"], "b_proj.weight": real["b_proj.weight"],
            "A_log": real["A_log"], "dt_bias": real["dt_bias"],
            "o_norm_weight": real["o_norm.weight"],
        },
        strict=False,
    )

    torch.manual_seed(1)
    x = torch.randn(1, 128, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
    with torch.no_grad():
        ref_out = reference(hidden_states=x)
        ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        our_out, _ = ours(x)
    assert_within(our_out, ref_out, KDA_FWD_BF16, "KDA vs the release module on real weights")
