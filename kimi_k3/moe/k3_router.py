"""Kimi K3's MoE router: released routing, plus quantile balancing.

**Two halves with very different evidence, and they are kept apart deliberately.**

*Routing* is published: `KimiMoEGate.forward` computes fp32 sigmoid scores, adds
`e_score_correction_bias` **for selection only**, takes the top-k, gathers the
weights from the **unbiased** scores, renormalises, and scales by
`routed_scaling_factor = 1.0`. Core's `topk_routing_with_score_function` already
does exactly that, so `QuantileBalancingRouter` inherits it and a test pins the
equivalence against a transcription of the release.

*Balancing* is not published. The K3 report names Quantile Balancing but gives no
algorithm, and no reference implementation ships with the release. What is here
is therefore **our formulation**, and it is gated on internal consistency --
exact agreement with a reference written independently of the fast path, measured
estimator error, and the behaviour it exists to produce -- **not** on any claim
of release parity. Review finding: this is why the plan's load-ratio band is a
reported observation and never a gate.

The rule: an expert should win `topk / num_experts` of the tokens. From a running
histogram of each expert's own scores, take the score at that quantile, and bias
every expert to the pooled threshold. Experts whose scores are systematically low
are raised; systematically high ones are lowered. Core's stock rule
(`get_updated_expert_bias`) instead takes a fixed `sign()` step, which is
scale-free but ignores how far off an expert actually is.
"""

from typing import Optional, Tuple

import torch

from megatron.core.transformer.moe.router import TopKRouter


class ScoreQuantileEstimator(torch.nn.Module):
    """A per-expert histogram of routing scores, and quantiles read off it.

    Sigmoid scores live in ``[0, 1]``, so a fixed-range histogram is exact up to
    the bin width -- no reservoir sampling, no sorting, and it survives being
    updated from many microbatches. The bin-width/accuracy trade-off is measured
    in `test_k3_p6_moe.py` rather than asserted.
    """

    def __init__(self, num_experts: int, num_bins: int = 1024, momentum: float = 0.9):
        super().__init__()
        self.num_experts = num_experts
        self.num_bins = num_bins
        self.momentum = momentum
        self.register_buffer("histogram", torch.zeros(num_experts, num_bins), persistent=True)

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        """``scores`` is ``[..., num_experts]`` in ``[0, 1]``.

        Callers hand this either `[tokens, experts]` or `[s, b, experts]`
        depending on where in the stack they sit, so flatten rather than assume.
        """
        flat = scores.reshape(-1, self.num_experts).float().clamp(0, 1)
        idx = (flat * (self.num_bins - 1)).round().long().t().contiguous()
        counts = torch.zeros_like(self.histogram)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
        self.histogram.mul_(self.momentum).add_(counts, alpha=1 - self.momentum)

    @torch.no_grad()
    def pooled_histogram(self) -> torch.Tensor:
        """The histogram summed over every rank that sees *different* tokens.

        Each rank observes only its own microbatch, so a purely local histogram
        gives each rank a different quantile and therefore a different bias --
        the ranks would then route the same token to different experts, and
        nothing would flag it. Core reduces its own `tokens_per_expert` over
        this same TPxCPxDP group, and for this same reason
        (`moe_utils.py:1201`).

        Reduced into a **copy**, never the persistent buffer: `histogram` is an
        EMA that is re-reduced on every call, so summing in place would
        multiply it by the world size once per step.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self.histogram
        try:
            from megatron.core import parallel_state

            group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
        except (ImportError, AssertionError, RuntimeError):
            return self.histogram  # model-parallel state not initialised (unit tests)
        pooled = self.histogram.clone()
        torch.distributed.all_reduce(pooled, group=group)
        return pooled

    @torch.no_grad()
    def quantile(self, upper_tail: float) -> torch.Tensor:
        """Per-expert score with ``upper_tail`` of that expert's mass above it."""
        histogram = self.pooled_histogram()
        total = histogram.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cdf = torch.cumsum(histogram / total, dim=1)
        target = 1.0 - upper_tail
        idx = torch.searchsorted(cdf.contiguous(), torch.full_like(total, target)).clamp(
            max=self.num_bins - 1
        )
        return idx.squeeze(1).float() / (self.num_bins - 1)

    @torch.no_grad()
    def is_populated(self) -> bool:
        """Populated *anywhere*, not just here.

        A rank whose own histogram is empty must still take part in the
        all-reduce inside `quantile()`; returning False here would make it skip
        the collective and hang every other rank.
        """
        local = torch.tensor(
            [float(self.histogram.sum() > 0)], device=self.histogram.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from megatron.core import parallel_state

                torch.distributed.all_reduce(
                    local,
                    group=parallel_state.get_tensor_and_data_parallel_group(
                        with_context_parallel=True
                    ),
                )
            except (ImportError, AssertionError, RuntimeError):
                pass
        return bool(local.item() > 0)


@torch.no_grad()
def quantile_balancing_bias(
    estimator: ScoreQuantileEstimator, topk: int, num_experts: int, strength: float = 1.0
) -> torch.Tensor:
    """The bias that would put every expert at the same selection threshold.

    Reference implementation: deliberately written as plainly as possible so the
    router can be checked against it exactly.
    """
    target_share = topk / num_experts
    per_expert = estimator.quantile(target_share)
    pooled = per_expert.median()
    return strength * (pooled - per_expert)


class QuantileBalancingRouter(TopKRouter):
    """`TopKRouter` with the released routing and a quantile-balanced bias."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        assert config.moe_router_score_function == "sigmoid", (
            "K3 routes on sigmoid scores; see KimiMoEGate.forward"
        )
        self.quantile_balancing = getattr(config, "k3_router_quantile_balancing", True)
        self.estimator = (
            ScoreQuantileEstimator(config.num_moe_experts, config.k3_qb_num_bins)
            if self.quantile_balancing
            else None
        )

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        probs, routing_map = super().routing(logits, padding_mask=padding_mask)
        if self.estimator is not None and self.training and torch.is_grad_enabled():
            self.estimator.update(torch.sigmoid(logits.detach().float()))
        return probs, routing_map

    @torch.no_grad()
    def update_expert_bias(self, strength: float = 1.0) -> Optional[torch.Tensor]:
        """Replace the stock sign-step update with the quantile-derived one.

        Called from the optimizer step, like core's own bias update. Returns the
        new bias, or None when balancing is off or nothing has been observed yet.
        """
        if self.estimator is None or self.expert_bias is None or not self.estimator.is_populated():
            return None
        bias = quantile_balancing_bias(
            self.estimator, self.config.moe_router_topk, self.config.num_moe_experts, strength
        )
        self.expert_bias.data.copy_(bias.to(self.expert_bias.dtype))
        return self.expert_bias


def released_gate_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verbatim `KimiMoEGate.forward`, for the equivalence test.

    Note what it does *not* do: the bias steers selection but never reaches the
    weights, which come from the unbiased scores. Getting that backwards changes
    what the model optimises and nothing would crash.
    """
    tokens = hidden_states.view(-1, hidden_states.shape[-1])
    logits = torch.nn.functional.linear(tokens.float(), weight.float(), None)
    scores = logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    _, topk_idx = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if topk > 1 and renormalize:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight * routed_scaling_factor
