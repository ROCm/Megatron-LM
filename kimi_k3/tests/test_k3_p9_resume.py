"""P9 / gate G37 -- the `dist_muon` resume workaround, and its tripwire.

The measurement is a two-rank probe (`results/pp_resume.md`). What belongs in CI
is the core bug it found: `dist_muon` cannot load its own optimizer state, so
every resume fails as shipped. The workaround is fork-local, and the tripwire
below fails once core fixes it -- so it leaves rather than lingers (rule R4.5).
"""

import pytest
import torch

from kimi_k3.optim.resume import core_layerwise_load_is_broken, load_optimizer_state_dict


def test_cores_layerwise_load_still_cannot_read_a_list():
    """IFU tripwire. When this fails, delete `kimi_k3/optim/resume.py`."""
    assert core_layerwise_load_is_broken(), (
        "core's LayerWiseDistributedOptimizer.load_state_dict has changed; re-check "
        "whether the workaround in kimi_k3/optim/resume.py is still needed"
    )


def test_the_bug_is_where_we_say_it_is():
    """Name the two halves, so a partial upstream fix does not fool the tripwire."""
    import inspect

    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    writes_a_list = inspect.getsource(ChainedOptimizer.state_dict)
    assert "return [optimizer.state_dict() for optimizer in self.chained_optimizers]" in writes_a_list

    reads_a_dict = inspect.getsource(LayerWiseDistributedOptimizer.load_state_dict)
    assert "wrapped_state_dict.values()" in reads_a_dict


def test_the_shim_defers_to_the_optimizer_for_everything_else():
    """It must be a narrow patch, not a replacement for `load_state_dict`."""

    class Plain:
        def __init__(self):
            self.seen = None

        def load_state_dict(self, state):
            self.seen = state

    plain = Plain()
    load_optimizer_state_dict(plain, {"a": 1})
    assert plain.seen == {"a": 1}


def test_the_shim_converts_the_dict_form_back_to_a_list():
    """The conversion core's override intends, checked without a real optimizer."""
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    captured = {}

    class Fake(LayerWiseDistributedOptimizer):
        def __init__(self):  # deliberately not calling super()
            pass

    fake = Fake()
    state = [{"fp32_from_fp16_params": {1: "b", 0: "a"}}, {"fp32_from_fp16_params": ["c"]}]
    original = ChainedOptimizer.load_state_dict
    try:
        ChainedOptimizer.load_state_dict = lambda self, sd: captured.setdefault("sd", sd)
        load_optimizer_state_dict(fake, state)
    finally:
        ChainedOptimizer.load_state_dict = original

    assert captured["sd"][0]["fp32_from_fp16_params"] == ["a", "b"], "must sort by index"
    assert captured["sd"][1]["fp32_from_fp16_params"] == ["c"], "a list is left alone"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_single_rank_dist_muon_round_trips_its_state(single_rank_world):
    """The end-to-end claim at 1 rank; PP=2 is the probe's job."""
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer, mock_batch, loss_func

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    ddp, optimizer = build_optimizer(model, optimizer="dist_muon", lr=1e-4, bf16=False)
    tokens, labels = mock_batch(4096, 16, 1, seed=1)

    for _ in range(2):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        loss, _ = loss_func(labels)(ddp(input_ids=tokens, position_ids=None, attention_mask=None))
        loss.backward()
        ddp.finish_grad_sync()
        optimizer.step()

    load_optimizer_state_dict(optimizer, optimizer.state_dict())
