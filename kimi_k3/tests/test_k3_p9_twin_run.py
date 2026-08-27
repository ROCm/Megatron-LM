"""P9 / gate G34 -- the twin-run statistics and the noise band.

The measurement itself is a tool run (`results/twin_runs.md`); what belongs in CI
is the machinery that decides pass from fail, because a comparison that cannot
fail is worse than no comparison.
"""

import pytest
import torch

from kimi_k3.tools.twin_run import Statistics, compare, widest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_identical_curves_are_zero_on_every_statistic():
    curve = [3.0, 2.0, 1.5, 1.4]
    assert compare(curve, curve) == Statistics(0.0, 0.0, 0.0)


def test_the_final_window_catches_what_the_others_miss():
    """A run that tracks and then walks away.

    Its max and mean are dominated by the long agreeing prefix; only the
    final-quarter statistic sees the divergence. This is the reason there are
    three numbers and not one.
    """
    a = [1.0] * 16
    b = [1.0] * 12 + [1.05, 1.10, 1.15, 1.20]
    stats = compare(a, b)
    assert stats.mean_delta < 0.04
    assert stats.final_delta > 0.1


def test_a_single_spike_survives_averaging():
    a = [1.0] * 20
    b = list(a)
    b[3] = 2.0
    stats = compare(a, b)
    assert stats.max_delta == pytest.approx(1.0)
    assert stats.mean_delta == pytest.approx(0.05)


def test_the_band_is_the_worst_of_each_statistic_independently():
    """Taking the worst *pair* would let one pair's good statistic mask another's."""
    band = widest([Statistics(1.0, 0.1, 0.01), Statistics(0.2, 0.5, 0.02)])
    assert band == Statistics(1.0, 0.5, 0.02)


def test_inside_needs_every_statistic():
    band = Statistics(1.0, 1.0, 1.0)
    assert Statistics(0.9, 0.9, 0.9).inside(band)
    for bad in (Statistics(1.1, 0.9, 0.9), Statistics(0.9, 1.1, 0.9), Statistics(0.9, 0.9, 1.1)):
        assert not bad.inside(band)


def test_mismatched_lengths_are_an_error_not_a_truncation():
    with pytest.raises(AssertionError):
        compare([1.0, 2.0], [1.0])


@pytest.mark.slow
def test_the_recompute_axis_actually_engages(single_rank_world):
    """G34's own guard: the bitwise-zero twin is only meaningful if the flag fires."""
    from kimi_k3.tools.twin_run import axis_engaged

    evidence = axis_engaged("tiny", "recompute")
    assert evidence["differs"], evidence
    assert evidence["checkpoint_calls"][0] == 0
    assert evidence["checkpoint_calls"][1] > 0
