"""P4 / gates G17–G18 -- gated MLA with NoPE.

Four things separate K3's MLA from stock MLA, and each of them fails *silently*
if got wrong: NoPE, the `192 ** -0.5` scale, the full-rank sigmoid output gate
before `o_proj`, and LoRA norms at eps 1e-6. Every one gets its own test, because
"the model still trains" is not evidence for any of them.
"""

import pytest
import torch

from kimi_k3.attention.gated_mla import EAGER, SDPA, K3GatedMLA, K3GatedMLASelfAttention
from kimi_k3.attention.gated_mla_eager_fp32 import gated_mla_eager_fp32, softmax_scale
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.tests.tolerance import KDA_FWD_FP32, assert_within, compare

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture()
def mla():
    torch.manual_seed(0)
    cfg = config_from_preset(preset("tiny")["config"])
    return K3GatedMLA(cfg).cuda().float(), cfg


def _x(cfg, b=2, s=32):
    torch.manual_seed(1)
    return torch.randn(b, s, cfg.hidden_size, device="cuda")


# --- G17: the released math -------------------------------------------------


def test_fused_and_eager_agree(mla):
    """Two independent attention implementations, same answer."""
    m, cfg = mla
    x = _x(cfg)
    assert_within(m(x, backend=SDPA), m(x, backend=EAGER), KDA_FWD_FP32, "sdpa vs eager MLA")


def test_scale_is_q_head_dim_not_qk_head_dim():
    """192**-0.5, not 128**-0.5 -- the second-likeliest silent parity bug."""
    assert softmax_scale(128, 64) == pytest.approx(192**-0.5)
    assert softmax_scale(128, 64) != pytest.approx(128**-0.5)
    cfg = config_from_preset(preset("93L")["config"])
    assert K3GatedMLA(cfg).scale == pytest.approx(192**-0.5)


def test_output_gate_is_applied_and_is_full_rank(mla):
    m, cfg = mla
    x = _x(cfg)
    gated = m(x, backend=EAGER)
    weights = dict(m.weights())
    ungated = gated_mla_eager_fp32(
        x, weights, num_heads=m.num_heads, qk_head_dim=m.qk_head_dim,
        qk_pos_emb_head_dim=m.qk_pos_emb_head_dim, v_head_dim=m.v_head_dim,
        lora_norm_eps=m.lora_norm_eps, use_output_gate=False,
    )
    assert not torch.allclose(gated, ungated), "the output gate is not doing anything"
    assert m.g_proj.weight.shape == (m.num_heads * m.v_head_dim, cfg.hidden_size)


def test_gate_multiplies_before_the_output_projection(mla):
    """Order matters: gating after `o_proj` would be a different function."""
    m, cfg = mla
    x = _x(cfg, b=1, s=8)
    with torch.no_grad():
        m.g_proj.weight.zero_()  # sigmoid(0) = 0.5 everywhere
        gated = m(x, backend=EAGER)
        m.config.k3_mla_use_output_gate = False
        m.use_output_gate = False
        plain = m(x, backend=EAGER)
    # a constant 0.5 gate before a linear map scales its output by exactly 0.5
    torch.testing.assert_close(gated, 0.5 * plain, rtol=1e-5, atol=1e-6)


def test_lora_norms_use_1e_6(mla):
    m, cfg = mla
    assert cfg.k3_mla_lora_norm_eps == 1e-6
    assert m.lora_norm_eps == 1e-6
    x = _x(cfg)
    baseline = m(x, backend=EAGER)
    m.lora_norm_eps = 1e-5
    assert not torch.equal(m(x, backend=EAGER), baseline), (
        "the LoRA-norm epsilon has no effect, so the norms are not being applied"
    )


def test_nope_the_rope_dims_are_shared_and_unrotated(mla):
    """`k_rot` is produced once and expanded, so every head sees the same 64 dims."""
    m, cfg = mla
    x = _x(cfg, b=1, s=6)
    compressed = m.kv_a_proj_with_mqa(x)
    _, k_rot = torch.split(compressed, [m.kv_lora_rank, m.qk_pos_emb_head_dim], dim=-1)
    expanded = k_rot.view(1, 1, 6, m.qk_pos_emb_head_dim).expand(1, m.num_heads, 6, -1)
    for head in range(1, m.num_heads):
        assert torch.equal(expanded[0, 0], expanded[0, head])
    # and position never enters: no rotary module exists at all
    assert not hasattr(m, "rotary_emb")


def test_causality(mla):
    m, cfg = mla
    x = _x(cfg, b=1, s=16)
    base = m(x, backend=EAGER)
    perturbed = x.clone()
    perturbed[:, 8:] += 3.0
    after = m(perturbed, backend=EAGER)
    torch.testing.assert_close(base[:, :8], after[:, :8], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(base[:, 8:], after[:, 8:])


def test_parameter_count_matches_the_analytic_model(mla):
    from kimi_k3.tools.mem_budget import mla_layer_params

    m, cfg = mla
    assert sum(p.numel() for p in m.parameters()) == mla_layer_params(cfg)


# --- G18: the fused backend at production head dims -------------------------


def test_fused_attention_handles_the_192_128_head_dim_combination():
    """q/k are 192 wide and v is 128; the release pads V and slices back."""
    cfg = config_from_preset(
        preset("tiny")["config"],
        num_attention_heads=4, num_query_groups=4,
        qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128,
        q_lora_rank=256, kv_lora_rank=128,
    )
    m = K3GatedMLA(cfg).cuda().to(torch.bfloat16)
    assert m.q_head_dim == 192 and m.v_head_dim == 128
    x = torch.randn(1, 64, cfg.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = m(x, backend=SDPA)
    assert out.shape == x.shape
    out.float().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_clip_qk_hook_is_present_for_core_to_find(mla):
    """core's optimizer/qk_clip.py skips layers without this attribute."""
    m, cfg = mla
    attn = K3GatedMLASelfAttention(cfg)
    assert hasattr(attn, "clip_qk") and hasattr(m, "clip_qk")


def test_sequence_first_adapter_keeps_batch_elements_independent(mla):
    _, cfg = mla
    attn = K3GatedMLASelfAttention(cfg).cuda().float()
    s, b = 12, 3
    x = torch.randn(s, b, cfg.hidden_size, device="cuda")
    out, bias = attn(x)
    assert bias is None and out.shape == x.shape
    x2 = x.clone()
    x2[:, 1] += 3.0
    out2, _ = attn(x2)
    torch.testing.assert_close(out[:, 0], out2[:, 0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(out[:, 1], out2[:, 1])


def test_the_te_backend_matches_the_release_workaround(single_rank_world):
    """The TE swap must be numerically free, not merely faster.

    `te` uses TransformerEngine's native asymmetric head dims; `sdpa` pads V to
    192 the way the release does. They must agree, and both must agree with the
    fp32 oracle -- otherwise a 2.93x speedup is buying a different model.
    """
    import torch

    from kimi_k3.attention.gated_mla import K3GatedMLA
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    torch.manual_seed(0)
    module = K3GatedMLA(config_from_preset(preset("tiny")["config"])).cuda()
    x = torch.randn(1, 64, module.config.hidden_size, device="cuda")
    with torch.no_grad():
        te = module(x, backend="te")
        sdpa = module(x, backend="sdpa")
        eager = module(x, backend="eager")

    scale = eager.float().norm()
    assert (te.float() - sdpa.float()).norm() / scale < 5e-3, "the two fused paths disagree"
    assert (te.float() - eager.float()).norm() / scale < 5e-2, "te drifts from the fp32 oracle"


def test_te_is_the_default_and_the_others_stay_reachable(single_rank_world):
    """`te` became the default once TE HEAD fixed the CK backward. See below."""
    from kimi_k3.attention.gated_mla import BACKENDS, TE, K3GatedMLA
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    assert set(BACKENDS) == {"eager", "sdpa", "te"}
    assert K3GatedMLA(config_from_preset(preset("tiny")["config"])).backend == TE
    for name in ("sdpa", "eager"):
        forced = K3GatedMLA(config_from_preset(preset("tiny")["config"], k3_mla_backend=name))
        assert forced.backend == name


def test_the_te_backward_is_finite_on_every_fused_attention_backend(single_rank_world):
    """The defect that kept `te` off by default, now fixed -- and pinned so.

    On TE `2.12.0.dev0+40434cf6` the **CK** fused-attention backward returned NaN
    for `q_a_layernorm` and `q_b_proj` at `hd192_hd128` while `kv_a_layernorm`
    stayed finite; AOTRITON was clean. Deterministic, 6/6. On
    `2.18.0.dev0+8f377e4` all three paths agree at 7.9297e-01.

    This asserts the *fixed* behaviour, so it fails if the environment is rolled
    back to a build carrying the defect -- the direction that now needs catching.
    """
    import torch

    from kimi_k3.attention.gated_mla import K3GatedMLA
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    for backend in ("te", "sdpa"):
        torch.manual_seed(0)
        m = K3GatedMLA(config_from_preset(preset("93L")["config"])).cuda().bfloat16()
        m.zero_grad()
        out = m(torch.randn(1, 128, 7168, device="cuda", dtype=torch.bfloat16) * 0.02,
                backend=backend)
        torch.manual_seed(2)
        out.backward(torch.randn_like(out))
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            grad = getattr(m, name).grad
            assert torch.isfinite(grad).all(), (
                f"{backend}/{name} gradient is not finite -- if TE was rolled back below "
                "2.18.0.dev0+8f377e4 this is the CK hd192_hd128 backward defect returning"
            )
