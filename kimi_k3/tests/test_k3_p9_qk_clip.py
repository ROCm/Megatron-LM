"""P9 / gate G36 -- QK-clip on gated MLA.

Core's `optimizer/qk_clip.py:clip_qk` walks the decoder and reads
`self_attention.core_attention.current_max_attn_logits`, then calls
`self_attention.clip_qk()`. K3's MLA is not core's `MLASelfAttention`, so both
of those are ours; these tests check the contract from core's side, not ours.
"""

import pytest
import torch

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

THRESHOLD = 4.0


def mla(**overrides):
    from kimi_k3.attention.gated_mla import K3GatedMLA

    settings = dict(qk_clip=True, qk_clip_threshold=THRESHOLD)
    settings.update(overrides)
    return K3GatedMLA(config_from_preset(preset("tiny")["config"], **settings)).cuda()


def logits_of(module, hidden):
    """Run a forward and read back what the module recorded."""
    module.core_attention.current_max_attn_logits = None
    with torch.no_grad():
        module(hidden)
    return module.core_attention.current_max_attn_logits


def test_the_recorded_max_logit_is_the_real_causal_max():
    """If the recorded statistic is wrong, everything downstream clips wrongly."""
    module = mla()
    torch.manual_seed(0)
    hidden = torch.randn(1, 24, module.config.hidden_size, device="cuda")
    recorded = logits_of(module, hidden)
    assert recorded.shape == (module.num_heads,)

    # the same thing computed the slow, obvious way
    b, s, _ = hidden.shape
    from kimi_k3.attention.gated_mla_eager_fp32 import rms_norm

    with torch.no_grad():
        q = module.q_b_proj(
            rms_norm(module.q_a_proj(hidden), module.q_a_layernorm, module.lora_norm_eps)
        ).view(b, s, module.num_heads, module.q_head_dim).transpose(1, 2)
        compressed = module.kv_a_proj_with_mqa(hidden)
        k_pass, k_rot = torch.split(
            compressed, [module.kv_lora_rank, module.qk_pos_emb_head_dim], dim=-1
        )
        k_pass = module.kv_b_proj(
            rms_norm(k_pass, module.kv_a_layernorm, module.lora_norm_eps)
        ).view(b, s, module.num_heads, module.qk_head_dim + module.v_head_dim).transpose(1, 2)
        k_pass = k_pass[..., : module.qk_head_dim]
        k_rot = k_rot.view(b, 1, s, module.qk_pos_emb_head_dim).expand(*k_pass.shape[:-1], -1)
        key = torch.cat((k_pass, k_rot), dim=-1)
        scores = (q.float() @ key.float().transpose(-1, -2)) * module.scale
        causal = torch.ones(s, s, device="cuda", dtype=torch.bool).tril()
        expected = scores.masked_fill(~causal, float("-inf")).amax(dim=(0, 2, 3))

    torch.testing.assert_close(recorded, expected, rtol=1e-5, atol=1e-5)


def test_chunking_does_not_change_the_statistic():
    """The chunk size is a memory knob, not a numerical one."""
    torch.manual_seed(1)
    hidden = torch.randn(1, 40, preset("tiny")["config"]["hidden_size"], device="cuda")
    whole, chunked = mla(k3_max_logit_chunk=4096), mla(k3_max_logit_chunk=7)
    chunked.load_state_dict(whole.state_dict())
    torch.testing.assert_close(logits_of(whole, hidden), logits_of(chunked, hidden), rtol=0, atol=0)


def test_clipping_brings_the_max_logit_to_the_threshold():
    """G36: the point of the whole thing."""
    module = mla()
    torch.manual_seed(2)
    # scale the query path up so the logits start well above the threshold
    with torch.no_grad():
        module.q_b_proj.weight.mul_(8.0)
    hidden = torch.randn(1, 24, module.config.hidden_size, device="cuda")

    before = logits_of(module, hidden).clone()
    assert float(before.max()) > THRESHOLD, "test is vacuous unless it starts over threshold"

    module.core_attention.current_max_attn_logits = before.clone()
    module.clip_qk()
    assert module.core_attention.current_max_attn_logits is None, "the statistic must be reset"

    after = logits_of(module, hidden)
    assert float(after.max()) <= THRESHOLD * 1.01, float(after.max())
    # heads that were already under the threshold are left alone
    under = before <= THRESHOLD
    if bool(under.any()):
        torch.testing.assert_close(after[under], before[under], rtol=1e-4, atol=1e-4)


def test_a_run_under_the_threshold_changes_no_weights():
    module = mla(qk_clip_threshold=1e9)
    torch.manual_seed(3)
    hidden = torch.randn(1, 16, module.config.hidden_size, device="cuda")
    module.core_attention.current_max_attn_logits = logits_of(module, hidden)
    before = module.q_b_proj.weight.clone()
    module.clip_qk()
    torch.testing.assert_close(module.q_b_proj.weight, before, rtol=0, atol=0)


def test_the_shared_rope_slice_takes_the_full_correction():
    """K3 is NoPE, but `k_rot` is still shared across heads.

    There is no per-head key weight for that slice, so the whole `eta` has to land
    on the query side. Splitting it like the per-head slice would under-clip, and
    nothing would fail loudly.
    """
    module = mla()
    torch.manual_seed(4)
    with torch.no_grad():
        module.q_b_proj.weight.mul_(8.0)
    hidden = torch.randn(1, 16, module.config.hidden_size, device="cuda")
    before = module.q_b_proj.weight.clone().view(module.num_heads, module.q_head_dim, -1)

    logits = logits_of(module, hidden)
    module.core_attention.current_max_attn_logits = logits.clone()
    module.clip_qk()

    eta = torch.clamp(THRESHOLD / logits, max=1.0)
    after = module.q_b_proj.weight.view(module.num_heads, module.q_head_dim, -1)
    alpha = module.config.qk_clip_alpha
    torch.testing.assert_close(
        after[:, : module.qk_head_dim],
        before[:, : module.qk_head_dim] * eta.pow(alpha).view(-1, 1, 1),
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        after[:, module.qk_head_dim :],
        before[:, module.qk_head_dim :] * eta.view(-1, 1, 1),
        rtol=1e-5, atol=1e-6,
    )


def test_cores_own_helper_drives_it(single_rank_world):
    """The contract that matters: core's `clip_qk(model)`, unmodified."""
    from megatron.core.optimizer.qk_clip import clip_qk

    from kimi_k3.model.build import build_k3_model

    model = build_k3_model("tiny", qk_clip=True, qk_clip_threshold=THRESHOLD)
    torch.manual_seed(5)
    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    model(input_ids=tokens, position_ids=None, attention_mask=None)

    mla_layers = [
        layer for layer in model.decoder.layers
        if hasattr(layer.self_attention, "clip_qk")
    ]
    assert mla_layers, "the tiny preset must contain at least one MLA layer"
    assert all(
        layer.self_attention.core_attention.current_max_attn_logits is not None
        for layer in mla_layers
    )

    # core's helper expects a list of chunks wrapping `.module.module.decoder`
    wrapper = torch.nn.Module()
    wrapper.module = torch.nn.Module()
    wrapper.module.module = model
    reported = clip_qk([wrapper])
    assert reported > 0
    assert all(
        layer.self_attention.core_attention.current_max_attn_logits is None
        for layer in mla_layers
    )
