"""P6 / gate G26 -- the load statistic the 8-rank smoke depends on.

The multi-rank run is scheduled work. What belongs in CI is the arithmetic it
reports, because the whole point of running eight ranks is to catch a router that
collapsed — and a collapsed router still trains, still descends, and produces a
loss that looks fine.
"""

import pytest
import torch

from kimi_k3.tools.ep_smoke import routing_load


class FakeRouter(torch.nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.register_buffer("local_tokens_per_expert", torch.tensor(counts, dtype=torch.float32))


def model_with(*count_lists):
    model = torch.nn.Module()
    for index, counts in enumerate(count_lists):
        model.add_module(f"layer{index}", FakeRouter(counts))
    return model


def test_a_balanced_router_reports_ratios_near_one():
    load = routing_load(model_with([100.0] * 8))
    assert load["available"]
    assert load["max_over_mean"] == pytest.approx(1.0)
    assert load["min_over_mean"] == pytest.approx(1.0)
    assert load["starved"] == 0


def test_a_collapsed_router_is_visible():
    """Seven experts idle, one taking everything: max/mean = 8, seven starved."""
    load = routing_load(model_with([800.0] + [0.0] * 7))
    assert load["max_over_mean"] == pytest.approx(8.0)
    assert load["starved"] == 7


def test_counts_are_summed_across_layers():
    """One balanced layer must not hide another that collapsed."""
    load = routing_load(model_with([100.0] * 4, [400.0, 0.0, 0.0, 0.0]))
    assert load["total_assignments"] == pytest.approx(800.0)
    assert load["starved"] == 0, "summed, so nothing is starved overall"
    assert load["max_over_mean"] == pytest.approx(2.5)  # (100+400) / 200


def test_no_counter_is_reported_rather_than_guessed():
    """Core only keeps the buffer when `moe_router_enable_expert_bias` is on."""
    assert routing_load(torch.nn.Module()) == {"available": False}


def test_an_untouched_counter_is_not_mistaken_for_balance():
    """All zeros means the forward never ran, not that the load was perfect."""
    assert routing_load(model_with([0.0] * 8)) == {"available": False}
