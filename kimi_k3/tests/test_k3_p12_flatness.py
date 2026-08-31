"""P12 / T12.5 -- the continued-pretrain flatness statistics.

The full evaluation is a human-run job against the released checkpoint (R10.1).
What belongs in CI is the machinery that decides flat from recovering, because the
failure it exists to catch *looks like success*: a run that spikes at step 0 and
re-converges reads as "training is working" on a loss curve, and by the time it
flattens the evidence of the conversion defect is gone.
"""

import pytest
import torch

from kimi_k3.tools.flatness_probe import flatness, verdict

TOL = 0.05


def test_a_flat_window_is_flat():
    stats = flatness([2.50, 2.51, 2.49, 2.50, 2.51, 2.50, 2.49, 2.50])
    assert abs(stats["drift"]) < 0.01
    assert verdict(stats, TOL)["flat"]


def test_the_recovery_signature_is_caught():
    """The one that matters: high arrival, then a climb back down.

    A conversion defect the model trains through produces exactly this. Reading
    the last few steps alone would call it healthy -- they *are* healthy, which is
    the problem.
    """
    losses = [3.40, 3.10, 2.90, 2.75, 2.62, 2.56, 2.52, 2.50]
    stats = flatness(losses)
    assert stats["deceleration"] > 0.5, stats
    result = verdict(stats, TOL)
    assert not result["flat"]
    assert any("recovery" in p for p in result["problems"]), result


def test_a_single_spike_is_caught_even_when_the_window_is_otherwise_flat():
    losses = [2.50] * 4 + [3.20] + [2.50] * 3
    stats = flatness(losses)
    assert abs(stats["drift"]) < 0.01, "drift alone would call this flat"
    assert stats["spike"] == pytest.approx(0.70)
    assert not verdict(stats, TOL)["flat"]


def test_ordinary_descent_is_not_mistaken_for_recovery():
    """Continued pretraining does descend; the threshold must allow it.

    An earlier version of this tested `drift < -tolerance`, which fails every
    healthy run that actually learns -- 0.002 per step over 40 steps is 0.08 of
    drift and entirely normal. Recovery is distinguished by *deceleration*, not by
    the sign of the slope.
    """
    losses = [2.500 - 0.002 * i for i in range(40)]
    stats = flatness(losses)
    assert stats["drift"] < 0, "this window really does descend"
    assert abs(stats["deceleration"]) < 0.01, "and it descends at a steady rate"
    assert verdict(stats, TOL)["flat"], stats


def test_arrival_is_the_first_step_not_the_minimum():
    """Reporting the minimum would flatter a spiking run."""
    stats = flatness([3.40, 2.50, 2.51, 2.49])
    assert stats["arrival"] == 3.40
    assert stats["min"] == 2.49


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_c6_baseline_measures_the_quantisation_round_trip(single_rank_world):
    """C6: a converted checkpoint starts from quantised-then-dequantised weights.

    So a small arrival bump is expected, and this bounds it rather than assuming
    it. The offset must be nonzero -- if it were zero, the baseline would be
    silently measuring nothing and any real bump would look attributable to it.
    """
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.tools.flatness_probe import quantisation_baseline
    from kimi_k3.training.pretrain_kimi_k3 import mock_batch

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    tokens, labels = mock_batch(4096, 32, 1, seed=7)
    base = quantisation_baseline(model, tokens, labels)

    assert base["expert_weights_touched"] > 0, "no expert weights were quantised"
    assert base["c6_offset"] != 0.0, "the round trip changed nothing; baseline is vacuous"
    assert abs(base["c6_offset"]) < 1.0, base
