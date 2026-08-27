"""P0 / gate G6 support -- the AttnRes mixer oracle and the payload protocol.

The mixer here is the FP32 oracle every later AttnRes parity test compares
against, so it is checked against a literal transcription of the released
``_apply_attn_res`` rather than against itself, plus the behavioural properties
that pin the semantics.

The payload tests are the precursor of gate G20: they prove gradients reach
*both* components through the packed tensor, which is the failure a two-tensor
payload would produce silently (review finding A1).
"""

import math

import pytest
import torch

from kimi_k3.block.attn_res import AttnResMixer, attn_res_mix, score_vector
from kimi_k3.block.attn_res_pp import (
    pack,
    payload_bytes,
    payload_multiplier,
    slots_before,
    unpack,
)
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset


def released_apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Verbatim transcription of HF moonshotai/Kimi-K3 `_apply_attn_res`."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


@pytest.fixture()
def toy():
    torch.manual_seed(0)
    t, k, h = 6, 3, 16
    return (
        torch.randn(t, h, dtype=torch.float64),
        torch.randn(t, k, h, dtype=torch.float64),
        torch.ones(h, dtype=torch.float64) * 1.3,
        torch.randn(1, h, dtype=torch.float64) * 0.05,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_the_released_implementation(dtype):
    """Bit-for-bit against the transcription, at the dtypes the release runs on."""
    torch.manual_seed(0)
    t, k, h = 6, 3, 16
    prefix = torch.randn(t, h, dtype=dtype)
    residual = torch.randn(t, k, h, dtype=dtype)
    norm_w = torch.ones(h, dtype=dtype) * 1.3
    proj_w = torch.randn(1, h, dtype=dtype) * 0.05
    ours = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5)
    theirs = released_apply_attn_res(prefix, residual, proj_w, norm_w, 1e-5)
    assert torch.equal(ours, theirs)


def test_zero_projection_gives_a_uniform_mix(toy):
    """Zero scores -> uniform softmax -> the plain mean of the candidates."""
    prefix, residual, norm_w, _ = toy
    proj_w = torch.zeros(1, prefix.shape[-1], dtype=prefix.dtype)
    out = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5)
    expected = torch.cat((residual, prefix.unsqueeze(1)), dim=1).mean(dim=1)
    torch.testing.assert_close(out, expected)


def test_no_slots_is_a_no_op(toy):
    prefix, _, norm_w, proj_w = toy
    empty = prefix.new_zeros(prefix.shape[0], 0, prefix.shape[-1])
    out = attn_res_mix(prefix, empty, norm_w, proj_w, 1e-5)
    assert out is prefix


def test_output_is_a_convex_combination(toy):
    """Softmax weights sum to 1, so every output lies inside the candidate hull."""
    prefix, residual, norm_w, proj_w = toy
    out = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5)
    stack = torch.cat((residual, prefix.unsqueeze(1)), dim=1)
    assert (out <= stack.max(dim=1).values + 1e-9).all()
    assert (out >= stack.min(dim=1).values - 1e-9).all()


def test_norm_gain_and_projection_only_appear_as_a_product(toy):
    """The fold P11's fused kernel relies on: scale one, divide the other."""
    prefix, residual, norm_w, proj_w = toy
    a = attn_res_mix(prefix, residual, norm_w, proj_w, 1e-5)
    b = attn_res_mix(prefix, residual, norm_w * 4.0, proj_w / 4.0, 1e-5)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        score_vector(norm_w, proj_w), score_vector(norm_w * 4.0, proj_w / 4.0)
    )


def test_gradcheck(toy):
    """True autograd, so fp64 gradcheck is valid here (unlike any STE path)."""
    prefix, residual, norm_w, proj_w = toy
    prefix = prefix.clone().requires_grad_(True)
    residual = residual.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p, r: attn_res_mix(p, r, norm_w, proj_w, 1e-5), (prefix, residual)
    )


def test_module_wrapper_defaults_to_a_uniform_mix():
    torch.manual_seed(0)
    mixer = AttnResMixer(hidden_size=8).double()
    prefix = torch.randn(4, 8, dtype=torch.float64)
    residual = torch.randn(4, 2, 8, dtype=torch.float64)
    out = mixer(prefix, residual)
    expected = torch.cat((residual, prefix.unsqueeze(1)), dim=1).mean(dim=1)
    torch.testing.assert_close(out, expected)
    assert mixer(prefix, None) is prefix


# --- payload protocol -------------------------------------------------------


def test_pack_unpack_round_trip():
    torch.manual_seed(0)
    s, b, k, h = 5, 2, 3, 7
    prefix = torch.randn(s, b, h)
    residual = torch.randn(s, b, k, h)
    packed = pack(prefix, residual)
    assert packed.shape == ((1 + k) * s, b, h)
    out_prefix, out_residual = unpack(packed, s, k)
    assert torch.equal(out_prefix, prefix)
    assert torch.equal(out_residual, residual)


def test_pack_with_no_slots_is_the_hidden_state():
    prefix = torch.randn(4, 1, 6)
    empty = prefix.new_zeros(4, 1, 0, 6)
    assert torch.equal(pack(prefix, empty), prefix)
    a, b = unpack(prefix, 4, 0)
    assert torch.equal(a, prefix) and b.shape == (4, 1, 0, 6)


def test_gradients_reach_both_payload_components():
    """The finding-A1 guard in miniature; G20 repeats it across a real PP boundary.

    A two-tensor payload would send `block_residual` forward, receive its gradient
    from the next stage, and drop it -- silently. Packing into one tensor makes a
    single backward reach every component.
    """
    torch.manual_seed(0)
    s, b, k, h = 3, 1, 2, 5
    prefix = torch.randn(s, b, h, requires_grad=True)
    residual = torch.randn(s, b, k, h, requires_grad=True)

    packed = pack(prefix, residual)
    # a downstream stage that touches every slot
    loss = (packed * torch.arange(1, packed.shape[0] + 1).view(-1, 1, 1)).sum()
    loss.backward()

    assert prefix.grad is not None and residual.grad is not None
    assert prefix.grad.abs().sum() > 0
    for slot in range(k):
        assert residual.grad[:, :, slot].abs().sum() > 0, f"slot {slot} got no gradient"


def test_per_stage_multipliers_are_neighbour_consistent():
    cfg = config_from_preset(preset("93L")["config"])
    block = cfg.k3_attn_res_block_size
    layout = [12, 12, 12, 12, 12, 11, 11, 11]
    assert sum(layout) == cfg.num_layers

    first = 0
    prev_send = None
    for count in layout:
        last = first + count - 1
        recv, send = payload_multiplier(first, last, block)
        if prev_send is not None:
            assert recv == prev_send, "a stage must receive what its predecessor sent"
        prev_send = send
        first = last + 1
    assert prev_send == 9  # 1 + 8 slots by the end of the model


def test_slot_schedule_matches_the_release():
    cfg = config_from_preset(preset("93L")["config"])
    block = cfg.k3_attn_res_block_size
    assert [slots_before(l, block) for l in (0, 1, 12, 13, 84, 85, 93)] == [0, 1, 1, 2, 7, 8, 8]
    assert slots_before(cfg.num_layers, block) == math.ceil(93 / 12) == 8


def test_payload_bytes_matches_a_real_tensor():
    prefix = torch.randn(4, 2, 16, dtype=torch.bfloat16)
    residual = torch.randn(4, 2, 3, 16, dtype=torch.bfloat16)
    packed = pack(prefix, residual)
    assert packed.numel() * packed.element_size() == payload_bytes(4, 2, 16, 3)
