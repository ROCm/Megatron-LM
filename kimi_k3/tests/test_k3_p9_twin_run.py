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


def test_permutation_test_finds_a_real_shift():
    """A clear offset between arms must come out significant."""
    from kimi_k3.tools.twin_run import permutation_test

    a = [[10.0 - 0.1 * i + 0.001 * s for i in range(40)] for s in range(5)]
    b = [[10.5 - 0.1 * i + 0.001 * s for i in range(40)] for s in range(5)]
    r = permutation_test(a, b)
    assert r["exact"], "5 per arm should enumerate exactly"
    assert r["splits"] == 126, r["splits"]          # C(10,5)/2
    assert r["significant_at_05"], r["p_value_holm"]


def test_permutation_test_does_not_flag_pure_noise():
    """Eight runs from one distribution must not look like two populations.

    This is the property the old max-over-pairs band never had: a stated,
    checkable false-positive behaviour rather than a threshold that moved with
    sample size.
    """
    import random

    from kimi_k3.tools.twin_run import permutation_test

    rng = random.Random(7)
    flags = 0
    trials = 20
    for _ in range(trials):
        runs = [[10.0 - 0.1 * i + rng.gauss(0, 0.05) for i in range(40)] for _ in range(10)]
        r = permutation_test(runs[:5], runs[5:])
        flags += bool(r["significant_at_05"])
    # alpha = 0.05 over 20 trials: 0 or 1 is expected, 3+ would mean the test is
    # mis-calibrated rather than unlucky.
    assert flags <= 2, f"{flags}/{trials} false positives; the test is not calibrated"


def test_p_value_can_never_be_zero():
    """The observed labelling is one of the enumerated splits, so p >= 1/splits.

    A p-value of exactly 0 would be a sign the observed split had been left out of
    its own null distribution -- the classic off-by-one in a permutation test.
    """
    from kimi_k3.tools.twin_run import permutation_test

    a = [[float(i) for i in range(40)] for _ in range(5)]
    b = [[float(i) + 99.0 for i in range(40)] for _ in range(5)]
    r = permutation_test(a, b)
    assert min(r["p_value"].values()) >= r["min_attainable_p"] - 1e-12
    assert min(r["p_value"].values()) > 0.0


def test_four_seeds_per_arm_cannot_reach_significance():
    """Below 5 per arm the test is powerless, and must say so rather than pass.

    Holm multiplies the smallest of three p-values by 3, so alpha = 0.05 needs a raw
    p <= 0.0167 and therefore >= 60 splits. 4 per arm gives 35 splits: min raw p
    0.029, which Holm inflates to 0.086. The first version of this tool defaulted to
    4 for exactly the reason this test exists -- the resolution was computed without
    accounting for the correction applied on top of it.
    """
    from kimi_k3.tools.twin_run import permutation_test

    a = [[float(i) for i in range(40)] for _ in range(4)]
    b = [[float(i) + 99.0 for i in range(40)] for _ in range(4)]
    r = permutation_test(a, b)
    assert r["splits"] == 35, r["splits"]
    assert abs(r["min_attainable_p"] - 1 / 35) < 1e-9
    assert not r["significant_at_05"], (
        "4 per arm reaches raw p 0.029 but 0.086 after Holm; it must not be reported "
        "as significant"
    )
