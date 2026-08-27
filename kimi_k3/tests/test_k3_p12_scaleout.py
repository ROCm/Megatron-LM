"""P12 / gate G46 -- 93 L configurations, validated analytically.

No cluster job is launched from here (rule R10.1). What is checked is that the
configurations a human would launch are arithmetically sound: the layouts are
legal, the parameter counts come from the measured breakdown, and the node counts
follow from the **measured** bytes per parameter rather than an estimate.
"""

import pytest

from kimi_k3.config.scaleout import (
    BYTES_PER_PARAM,
    CANDIDATES,
    EP_LADDER,
    aligned_layout,
    boundaries,
    experts_divide,
    layout_problems,
    plan,
    slots_at,
    uniform_layout,
)

BLOCK = 12
LAYERS = 93


def test_the_ep_ladder_divides_the_expert_count():
    assert all(experts_divide(ep) for ep in EP_LADDER), EP_LADDER


@pytest.mark.parametrize("pp", [2, 4, 8])
def test_aligned_layouts_have_no_problems(pp):
    layout = aligned_layout(LAYERS, pp, BLOCK)
    assert sum(layout) == LAYERS
    assert layout_problems(layout, LAYERS, BLOCK) == []
    assert all(b % BLOCK == 0 for b in boundaries(layout))


def test_pp_above_eight_cannot_be_aligned_and_says_so():
    """A real constraint, not a limitation of the helper.

    93 layers at block size 12 give seven whole AttnRes blocks, so there are only
    eight places a stage can start on a block boundary. Any PP above that must
    cut a block, and the planner reports it instead of silently producing an
    illegal layout.
    """
    with pytest.raises(ValueError, match="aligned cut points"):
        aligned_layout(LAYERS, 16, BLOCK)

    problems = plan(next(c for c in CANDIDATES if c.pp == 16))["problems"]
    assert any("aligned cut points" in p for p in problems), problems


def test_a_mid_block_boundary_is_reported_with_the_nearest_legal_one():
    layout = uniform_layout(LAYERS, 4)  # 24, 23, 23, 23 -> boundaries 24, 47, 70
    problems = layout_problems(layout, LAYERS, BLOCK)
    assert any("splits an AttnRes block" in p for p in problems), problems
    assert any("48" in p for p in problems), problems


def test_the_final_mla_pair_is_called_out():
    """Layers 92 and 93 are both MLA -- the one place the 3:1 stride breaks."""
    problems = layout_problems([LAYERS - 1, 1], LAYERS, BLOCK)
    assert any("final MLA pair" in p for p in problems), problems


def test_payload_slot_count_grows_with_depth():
    """What actually crosses a boundary: 1 + slots, and slots = layer / 12."""
    assert [slots_at(b, BLOCK) for b in (12, 24, 36, 84)] == [1, 2, 3, 7]
    assert slots_at(0, BLOCK) == 0


def test_every_candidate_is_arithmetically_consistent():
    for config in CANDIDATES:
        row = plan(config)
        assert row["gpus"] == config.tp * config.pp * config.ep
        assert sum(row["layout"]) == LAYERS
        assert len(row["boundaries"]) == config.pp - 1
        assert row["params_per_gpu"] > 0
        assert row["total_gib"] == pytest.approx(row["state_gib"] + row["headroom_gib"], abs=0.2)


def test_the_node_count_comes_from_the_measured_recipe():
    """If this ever reads an estimate, the whole table is decoration (R9.1)."""
    assert BYTES_PER_PARAM["dist_muon@dp8"] == 7.87  # results/opt_mem.md
    assert BYTES_PER_PARAM["muon"] == 15.17

    row = plan(next(c for c in CANDIDATES if c.pp == 8 and c.ep == 32))
    expected = row["params_per_gpu"] * BYTES_PER_PARAM["dist_muon@dp8"] / 2**30
    assert row["state_gib"] == pytest.approx(expected, abs=0.1)


def test_more_expert_parallelism_always_means_fewer_parameters_per_gpu():
    """A sanity check on the sharding model, not on the numbers."""
    for pp in (4, 8):
        rows = [plan(c) for c in CANDIDATES if c.pp == pp]
        per_gpu = [r["params_per_gpu"] for r in sorted(rows, key=lambda r: r["ep"])]
        assert per_gpu == sorted(per_gpu, reverse=True), per_gpu


def test_at_least_one_candidate_fits_and_at_least_one_does_not():
    """A table where everything fits is not telling you anything."""
    rows = [plan(c) for c in CANDIDATES]
    assert any(r["fits"] for r in rows)
    assert any(not r["fits"] for r in rows)
