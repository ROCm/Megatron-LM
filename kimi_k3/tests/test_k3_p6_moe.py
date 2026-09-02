"""P6 / gates G23–G26 -- LatentMoE, the router, and SiTU.

The routing half is checked against a transcription of `KimiMoEGate.forward`.
The balancing half cannot be: Quantile Balancing is named in the K3 report but
its algorithm is unpublished and no reference ships with the release. So it is
gated on internal consistency instead -- exact agreement with an independently
written reference, measured estimator error, and the behaviour it exists to
produce -- and never on a claim of release parity.
"""

import pytest
import torch

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.moe.k3_router import (
    ScoreQuantileEstimator,
    quantile_balancing_bias,
    released_gate_reference,
)
from kimi_k3.moe.situ import SITU_BETA, SITU_LINEAR_BETA, situ_glu


def make_router(**overrides):
    """`TopKRouter` needs process groups, so build it the way core does."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    from kimi_k3.moe.k3_router import QuantileBalancingRouter

    cfg = config_from_preset(preset("tiny")["config"], **overrides)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    return QuantileBalancingRouter(cfg, pg_collection=pg).cuda(), cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# --- G23: SiTU ---------------------------------------------------------------


def test_situ_matches_the_release_and_soft_caps():
    torch.manual_seed(0)
    gate_up = torch.randn(64, 128, device="cuda")
    d = 64
    gate, up = gate_up[:, :d].float(), gate_up[:, d:].float()
    expected = (SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)) * (
        SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    )
    torch.testing.assert_close(situ_glu(gate_up), expected.to(gate_up.dtype))
    huge = situ_glu(torch.full((1, 2), 1e4, device="cuda"))
    assert torch.isfinite(huge).all() and huge.abs().max() <= SITU_BETA * SITU_LINEAR_BETA * 1.001


# --- G24: the layer ----------------------------------------------------------


@pytest.fixture()
def model(single_rank_world):
    from kimi_k3.model.build import build_k3_model

    torch.manual_seed(0)
    return build_k3_model("tiny")


def moe_layers(model):
    return [l.mlp for l in model.decoder.layers if hasattr(l.mlp, "router")]


def test_only_the_first_layer_is_dense(model):
    kinds = [type(l.mlp).__name__ for l in model.decoder.layers]
    assert kinds[0] == "MLP"
    assert all(k == "K3MoELayer" for k in kinds[1:])


def test_latent_norm_exists_and_is_applied_before_the_up_projection(model):
    """Core goes straight from combine to fc2_latent_proj; K3 normalises first
    (review finding A10). The norm's own gain must therefore matter."""
    layer = moe_layers(model)[0]
    assert layer.routed_expert_norm is not None
    assert layer.routed_expert_norm.shape == (model.config.moe_latent_size,)

    tokens = torch.randint(0, 4096, (1, 8), device="cuda")
    baseline = model(input_ids=tokens, position_ids=None, attention_mask=None)
    with torch.no_grad():
        layer.routed_expert_norm.mul_(2.0)
    assert not torch.allclose(
        model(input_ids=tokens, position_ids=None, attention_mask=None), baseline
    ), "the latent norm gain has no effect, so the norm is not in the path"


def test_routed_experts_run_at_the_latent_width_shared_experts_at_hidden(model):
    """Finding B6: only the routed half is 3584 wide."""
    cfg = model.config
    layer = moe_layers(model)[0]
    assert cfg.moe_latent_size == cfg.k3_routed_expert_hidden_size
    assert layer.fc1_latent_proj.weight.shape == (cfg.moe_latent_size, cfg.hidden_size)
    if layer.shared_experts is not None:
        first = layer.shared_experts.linear_fc1.weight
        assert first.shape[-1] == cfg.hidden_size, "shared experts must take the hidden width"


# --- G25: routing and balancing ---------------------------------------------


def test_routing_matches_the_released_gate():
    """Selection uses the bias; the weights come from the UNBIASED scores."""
    torch.manual_seed(0)
    tokens, experts, topk = 64, 8, 2
    hidden = torch.randn(tokens, 16, device="cuda")
    weight = torch.randn(experts, 16, device="cuda")
    bias = torch.randn(experts, device="cuda") * 0.1

    idx, weights = released_gate_reference(hidden, weight, bias, topk)
    logits = torch.nn.functional.linear(hidden.float(), weight.float())
    scores = logits.sigmoid()

    # the weights must be gathered from the unbiased scores, then renormalised
    raw = scores.gather(1, idx)
    torch.testing.assert_close(weights, raw / (raw.sum(-1, keepdim=True) + 1e-20))
    # and selection must follow the biased scores
    biased_top = torch.topk(scores + bias, topk, dim=-1).indices
    assert set(idx[0].tolist()) == set(biased_top[0].tolist())
    # with a large enough bias, selection changes but the weights stay unbiased
    idx2, _ = released_gate_reference(hidden, weight, bias + 10.0 * torch.eye(experts, device="cuda")[0], topk)
    assert not torch.equal(idx.sort(-1).values, idx2.sort(-1).values)


def test_estimator_quantile_error_is_bounded_by_the_bin_width():
    """The accuracy/bin-width trade-off, measured rather than asserted."""
    torch.manual_seed(0)
    experts, tokens = 4, 20000
    scores = torch.rand(tokens, experts, device="cuda")
    for bins, tolerance in ((64, 2e-2), (1024, 2e-3)):
        est = ScoreQuantileEstimator(experts, num_bins=bins, momentum=0.0).cuda()
        est.update(scores)
        got = est.quantile(0.25)
        want = torch.quantile(scores, 0.75, dim=0)
        assert (got - want).abs().max() < tolerance, (bins, (got - want).abs().max().item())


def test_bias_update_is_exactly_the_reference(single_rank_world):
    """The fast path and the plainly-written rule must not drift apart."""
    router, cfg = make_router()
    torch.manual_seed(0)
    router.estimator.update(torch.rand(4096, cfg.num_moe_experts, device="cuda"))

    expected = quantile_balancing_bias(
        router.estimator, cfg.moe_router_topk, cfg.num_moe_experts
    )
    got = router.update_expert_bias()
    torch.testing.assert_close(got.float(), expected.float())


def test_balancing_moves_a_skewed_router_toward_balance():
    """The behaviour the rule exists to produce, on deliberately skewed scores."""
    torch.manual_seed(0)
    experts, topk, tokens = 8, 2, 4096
    # two experts score systematically higher than the rest
    scores = torch.rand(tokens, experts, device="cuda") * 0.5
    scores[:, :2] += 0.4

    def load_ratio(bias):
        chosen = torch.topk(scores + bias, topk, dim=-1).indices
        counts = torch.bincount(chosen.flatten(), minlength=experts).float()
        return (counts.max() / counts.mean()).item()

    est = ScoreQuantileEstimator(experts, num_bins=1024, momentum=0.0).cuda()
    est.update(scores)
    before = load_ratio(torch.zeros(experts, device="cuda"))
    after = load_ratio(quantile_balancing_bias(est, topk, experts))
    assert after < before, (before, after)
    assert after < 1.6, after


def test_router_records_scores_only_while_training(single_rank_world):
    router, cfg = make_router()
    logits = torch.randn(32, cfg.num_moe_experts, device="cuda")
    router.eval()
    router.routing(logits)
    assert not router.estimator.is_populated(), "eval must not disturb the histogram"


def test_balancing_can_be_switched_off(single_rank_world):
    router, _ = make_router(k3_router_quantile_balancing=False)
    assert router.estimator is None
    assert router.update_expert_bias() is None


def test_core_bias_update_dispatches_to_the_router():
    """core's finalize path must reach QuantileBalancingRouter.update_expert_bias.

    `_update_router_expert_bias` collects modules with an `expert_bias` and
    overwrites it with core's own sign-step; it never consults the module. Before
    `install_router_bias_dispatch()` that made quantile balancing dead code under
    `megatron.training.pretrain` -- the bias moved every step, so nothing looked
    wrong, but it was core's value and not ours.
    """
    import importlib

    from kimi_k3.model.build import build_k3_model
    from kimi_k3.model.core_patch import install_router_bias_dispatch

    install_router_bias_dispatch()
    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    model = build_k3_model("tiny")
    model.train()
    routers = [m for m in model.modules() if hasattr(m, "update_expert_bias")]
    assert routers, "no quantile-balancing routers found"

    called = []
    for r in routers:
        r.update_expert_bias = lambda _r=r: called.append(_r)

    fmg._update_router_expert_bias([model], model.config)
    assert len(called) == len(routers), (
        f"core reached {len(called)} of {len(routers)} routers; the dispatch is not installed"
    )
