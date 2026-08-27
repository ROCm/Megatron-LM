"""P0 / gate G4 -- block injection with no transient core block.

``GPTModel.__init__`` constructs ``TransformerBlock`` directly. K3 rebinds that
symbol for the duration of construction rather than duplicating the ~190-line
constructor (develop/plan-0/00-review-findings.md finding A5), so this test has
to prove three things: our block is what got built, a core block was never
built, and the rebinding did not leak out of the context manager.
"""

import pytest
import torch

from kimi_k3.block.k3_transformer_block import K3TransformerBlock
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.model.core_patch import assert_pin_contracts, k3_block_class
from kimi_k3.model.k3_gpt_model import K3GPTModel


@pytest.fixture()
def core_block_spy(monkeypatch):
    """Count instantiations of the *core* block class exactly (not subclasses)."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    calls = []
    original = TransformerBlock.__init__

    def counting_init(self, *args, **kwargs):
        if type(self) is TransformerBlock:
            calls.append(type(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TransformerBlock, "__init__", counting_init)
    return calls


def _build_tiny(spec, device="meta"):
    p = preset("tiny")
    cfg = config_from_preset(p["config"])
    with torch.device(device):
        return K3GPTModel(
            config=cfg,
            transformer_layer_spec=spec,
            vocab_size=p["model"]["vocab_size"],
            max_sequence_length=p["model"]["max_sequence_length"],
            position_embedding_type="none",
        )


def test_decoder_is_the_k3_block_and_no_core_block_was_built(
    single_rank_world, tiny_spec, core_block_spy
):
    model = _build_tiny(tiny_spec)
    assert isinstance(model.decoder, K3TransformerBlock)
    assert model.decoder.k3_block is True
    assert core_block_spy == [], (
        f"a core TransformerBlock was constructed {len(core_block_spy)} time(s); "
        "the injection is allocating a transient block"
    )
    assert len(model.decoder.layers) == preset("tiny")["config"]["num_layers"]


def test_rebinding_does_not_leak(single_rank_world, tiny_spec):
    import megatron.core.models.gpt.gpt_model as gm
    from megatron.core.transformer.transformer_block import TransformerBlock

    before = gm.TransformerBlock
    _build_tiny(tiny_spec)
    assert gm.TransformerBlock is before is TransformerBlock

    # and the context manager restores even when construction raises
    class Boom(K3TransformerBlock):
        def __init__(self, *a, **kw):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        with k3_block_class(Boom):
            raise RuntimeError("boom")
    assert gm.TransformerBlock is TransformerBlock


def test_set_input_tensor_takes_exactly_one_packed_payload(single_rank_world, tiny_spec):
    """Core back-props only output_tensor[0], so the payload is one tensor."""
    model = _build_tiny(tiny_spec)
    t = torch.empty(4, 1, preset("tiny")["config"]["hidden_size"], device="meta")
    model.set_input_tensor(t)
    assert model.decoder.input_tensor is t
    model.set_input_tensor([t])
    with pytest.raises(AssertionError):
        model.set_input_tensor([t, t])


def test_pin_contracts_hold():
    checked = assert_pin_contracts()
    assert len(checked) == 7, checked
