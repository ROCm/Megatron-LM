"""P7 / gates G27–G29 -- the model trains, and its checkpoint round-trips.

`train_smoke` runs real steps through the whole stack: KDA, gated NoPE MLA, the
AttnRes layer, LatentMoE with the quantile-balancing router, and a Megatron
optimizer. Two different assertions, because they catch different failures:

* on **fresh random tokens** the loss must sit near `ln(vocab_size)` -- chance.
  Far above means something is broken; far below means the model is seeing its
  labels.
* on a **fixed batch** the loss must fall a long way. That is the difference
  between "the optimizer ran" and "the model learns".
"""

import math

import pytest
import torch

from kimi_k3.config.presets import preset
from kimi_k3.training.pretrain_kimi_k3 import mock_batch, train_smoke

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CHANCE = math.log(preset("tiny")["model"]["vocab_size"])


@pytest.mark.parametrize("optimizer", ["dist_muon", "adam"])
def test_loss_starts_at_chance_and_stays_finite(single_rank_world, optimizer):
    losses = train_smoke(iterations=10, optimizer=optimizer, lr=1e-4)
    assert all(math.isfinite(l) for l in losses), losses
    assert CHANCE - 0.5 < losses[0] < CHANCE + 1.0, (losses[0], CHANCE)


@pytest.mark.parametrize("optimizer", ["dist_muon", "adam"])
def test_the_model_learns_a_fixed_batch(single_rank_world, optimizer):
    """G27: the whole stack, end to end, actually optimises."""
    losses = train_smoke(iterations=30, optimizer=optimizer, lr=1e-3, fixed_batch=True)
    assert losses[-1] < losses[0] - 2.0, losses
    assert all(math.isfinite(l) for l in losses)


def test_checkpoint_round_trips(single_rank_world, tmp_path):
    """G29: save, rebuild from a different init, load, and get the same forward."""
    from kimi_k3.model.build import build_k3_model

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    tokens, _ = mock_batch(preset("tiny")["model"]["vocab_size"], 16, 1)
    before = model(input_ids=tokens, position_ids=None, attention_mask=None)

    path = tmp_path / "k3.pt"
    torch.save({k: v for k, v in model.state_dict().items() if torch.is_tensor(v)}, path)

    torch.manual_seed(1)
    fresh = build_k3_model("tiny")
    assert not torch.allclose(
        fresh(input_ids=tokens, position_ids=None, attention_mask=None), before
    ), "the two inits are identical, so loading proves nothing"
    missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=False)
    assert not unexpected, unexpected
    after = fresh(input_ids=tokens, position_ids=None, attention_mask=None)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_expert_bias_moves_when_the_update_is_called(single_rank_world):
    """The update mutates the bias after real steps have populated the estimator.

    Deliberately *not* a reachability test: it calls `update_expert_bias()`
    directly. Which loops actually reach it is covered by
    `test_k3_p6_moe.py::test_core_bias_update_dispatches_to_the_router`, and the
    earlier name here ("...during_training") claimed a guarantee this body never
    checked -- core's finalize path was in fact bypassing the router entirely.
    """
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, loss_func

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    routers = [m for m in model.modules() if hasattr(m, "update_expert_bias")]
    assert routers, "no quantile-balancing routers found"
    before = routers[0].expert_bias.detach().clone()

    ddp, opt = build_optimizer(model, optimizer="adam", lr=1e-3, bf16=False)
    tokens, labels = mock_batch(preset("tiny")["model"]["vocab_size"], 16, 1)
    for _ in range(3):
        ddp.zero_grad_buffer()
        opt.zero_grad()
        out = ddp(input_ids=tokens, position_ids=None, attention_mask=None)
        loss_func(labels)(out)[0].backward()
        ddp.finish_grad_sync()
        opt.step()
    for r in routers:
        r.update_expert_bias()
    assert not torch.equal(routers[0].expert_bias, before), "the bias never moved"


def test_tokenizer_special_ids_match_the_released_config():
    from kimi_k3.tools.tokenizer import K3Tokenizer

    tok = K3Tokenizer()
    released = {  # config.json, recorded in the P0 release audit
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "media_placeholder_token_id": 163605,
    }
    assert tok.check_ids_against_config(released) == []
    assert tok.vocab_size == 163840
    assert tok.check_ids_against_config({"bos_token_id": 1}) == ["bos: ours 163584 vs config 1"]
