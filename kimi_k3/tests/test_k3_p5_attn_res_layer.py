"""P5 / gates G19–G22 -- the layer places its mixes where the release does.

The check that matters is against a **verbatim transcription** of
`KimiDecoderLayer._forward_attn_residual`, driven with the same weights. Anything
weaker -- "the loss falls", "the shapes match" -- would pass with the mixes in
the wrong places, which is exactly the failure this phase exists to exclude.
"""

import pytest
import torch

from kimi_k3.block.attn_res import attn_res_mix
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def released_forward_attn_residual(layer, hidden_states, block_residual):
    """Verbatim from HF moonshotai/Kimi-K3 `KimiDecoderLayer._forward_attn_residual`.

    Written in the release's batch-first `[B, S, H]` terms and with its own flat
    `(T, ...)` reshapes, so that our sequence-first layer has to *agree* rather
    than merely resemble.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    prefix_sum = hidden_states

    def mix(prefix, slots, mixer):
        return attn_res_mix(
            prefix.reshape(-1, hidden_size), slots, mixer.weight, mixer.proj, mixer.eps
        ).view(batch_size, seq_len, hidden_size)

    if block_residual is not None and block_residual.shape[1] > 0:
        hidden_states = mix(prefix_sum, block_residual, layer.attn_res_attn)

    if layer.global_layer_index % layer.block_size == 0:
        block_residual = torch.cat(
            [block_residual, prefix_sum.reshape(-1, hidden_size).unsqueeze(1)], dim=1
        )
        prefix_sum = None

    hidden_states = layer.input_layernorm(hidden_states.transpose(0, 1)).transpose(0, 1)
    attn_out, _ = layer.self_attention(hidden_states.transpose(0, 1))
    attn_out = attn_out.transpose(0, 1)

    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out
    hidden_states = mix(prefix_sum, block_residual, layer.attn_res_mlp)
    hidden_states = layer.pre_mlp_layernorm(hidden_states.transpose(0, 1)).transpose(0, 1)
    mlp_out = layer.mlp(hidden_states.transpose(0, 1))
    if isinstance(mlp_out, tuple):
        mlp_out = mlp_out[0] if mlp_out[1] is None else mlp_out[0] + mlp_out[1]
    mlp_out = mlp_out.transpose(0, 1)
    prefix_sum = prefix_sum + mlp_out
    return prefix_sum, block_residual


@pytest.fixture()
def model(single_rank_world):
    from kimi_k3.model.build import build_k3_model

    torch.manual_seed(0)
    m = build_k3_model("tiny")
    # a zero-initialised projection makes every mix uniform, which would hide a
    # misplaced mix; give the mixers something to say
    with torch.no_grad():
        for layer in m.decoder.layers:
            for mixer in (layer.attn_res_attn, layer.attn_res_mlp):
                mixer.proj.normal_(std=0.05)
                mixer.weight.normal_(mean=1.0, std=0.05)
        m.decoder.output_attn_res.proj.normal_(std=0.05)
    return m


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3])
def test_layer_matches_the_released_placement(model, layer_idx):
    layer = model.decoder.layers[layer_idx]
    s, b, h = 12, 2, model.config.hidden_size
    torch.manual_seed(layer_idx)
    hidden = torch.randn(s, b, h, device="cuda")
    slots_count = layer.global_layer_index // layer.block_size
    slots = torch.randn(s, b, slots_count, h, device="cuda") if slots_count else \
        hidden.new_zeros(s, b, 0, h)

    ours_prefix, ours_slots = layer(hidden, None, block_residual=slots)

    # the release's own shapes: [B, S, H] and [T, K, H]
    theirs_prefix, theirs_slots = released_forward_attn_residual(
        layer,
        hidden.transpose(0, 1).contiguous(),
        slots.permute(1, 0, 2, 3).reshape(b * s, slots_count, h),
    )
    torch.testing.assert_close(
        ours_prefix, theirs_prefix.transpose(0, 1), rtol=1e-5, atol=1e-5
    )
    assert ours_slots.shape[2] == theirs_slots.shape[1]


def test_slot_is_appended_at_block_boundaries_only(model):
    for layer in model.decoder.layers:
        s, b, h = 6, 1, model.config.hidden_size
        hidden = torch.randn(s, b, h, device="cuda")
        before = torch.randn(s, b, 1, h, device="cuda")
        _, after = layer(hidden, None, block_residual=before)
        grew = after.shape[2] - before.shape[2]
        assert grew == (1 if layer.appends_slot else 0), (
            f"layer {layer.global_layer_index}: slots grew by {grew}"
        )


def test_the_prefix_sum_restarts_at_a_block_boundary(model):
    """A layer that appends a slot starts its own stream from its attention
    output, not from its input -- so scaling its input must not scale its output
    the way it does for a non-boundary layer."""
    boundary = next(l for l in model.decoder.layers if l.appends_slot)
    s, b, h = 8, 1, model.config.hidden_size
    hidden = torch.randn(s, b, h, device="cuda")
    slots = hidden.new_zeros(s, b, 0, h)
    out_a, _ = boundary(hidden, None, block_residual=slots)
    out_b, _ = boundary(hidden * 0, None, block_residual=slots)
    # with the stream reset, a zeroed input does not simply zero the output
    assert not torch.allclose(out_a, out_b)


def test_appended_slot_is_the_prefix_before_attention(model):
    boundary = model.decoder.layers[0]
    assert boundary.appends_slot
    s, b, h = 5, 1, model.config.hidden_size
    hidden = torch.randn(s, b, h, device="cuda")
    _, slots = boundary(hidden, None, block_residual=hidden.new_zeros(s, b, 0, h))
    torch.testing.assert_close(slots[:, :, 0], hidden)


def test_gradients_reach_every_mixer_that_has_slots_to_mix(model):
    """A mixer with no slots is a no-op; every other one must learn."""
    s, b = 10, 1
    tokens = torch.randint(0, 4096, (b, s), device="cuda")
    model(input_ids=tokens, position_ids=None, attention_mask=None).float().pow(2).sum().backward()
    for i, layer in enumerate(model.decoder.layers):
        slots_on_entry = layer.global_layer_index // layer.block_size
        for name, mixer in (("attn", layer.attn_res_attn), ("mlp", layer.attn_res_mlp)):
            active = slots_on_entry > 0 or (name == "mlp" and layer.appends_slot)
            if active:
                assert mixer.proj.grad is not None and mixer.proj.grad.abs().sum() > 0, (
                    f"layer {i} {name} mixer got no gradient despite having slots"
                )


def test_forward_is_deterministic(model):
    """The tiny preset sets dropout to 0, so two forwards must agree bitwise.

    With Megatron's default 0.1 they do not, and every determinism check silently
    becomes a measurement of the RNG (the trap recorded in the G7 report).
    """
    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    a = model(input_ids=tokens, position_ids=None, attention_mask=None)
    b = model(input_ids=tokens, position_ids=None, attention_mask=None)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_recompute_is_numerically_invisible(model):
    """G22. AttnRes is recompute-mandatory at production width, so this path has
    to carry **both** state tensors -- core's own checkpointed_forward assumes a
    single hidden state and would drop the block residual."""
    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    model.train()

    model.config.recompute_granularity = None
    plain = model(input_ids=tokens, position_ids=None, attention_mask=None)
    plain.float().pow(2).sum().backward()
    plain_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    model.zero_grad(set_to_none=True)
    model.config.recompute_granularity = "full"
    recomputed = model(input_ids=tokens, position_ids=None, attention_mask=None)
    recomputed.float().pow(2).sum().backward()
    model.config.recompute_granularity = None

    torch.testing.assert_close(recomputed, plain, rtol=1e-6, atol=1e-6)
    for name, want in plain_grads.items():
        got = dict(model.named_parameters())[name].grad
        assert got is not None, f"{name} lost its gradient under recompute"
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}")
