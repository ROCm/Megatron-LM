# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""DeepSeek V4 MoE Router.

Implements two routing modes:
  1. Hash routing (first num_hash_layers MoE layers): deterministic tid2eid LUT,
     no learned weights, no auxiliary loss.
  2. Learned routing (remaining layers): sqrtsoftplus(logits) scoring with
     decoupled load balancing (expert_bias adjusted outside gradient).

Reference: DeepSeek V4 paper §3.2; Primus PR #698 v4_topk_router.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HashRouter(nn.Module):
    """Deterministic hash-based routing for the first num_hash_layers MoE layers.

    Assigns each token to experts via a fixed lookup table (tid2eid) built once
    from a seeded RNG. No parameters, no gradient, no auxiliary loss.
    """

    def __init__(self, config, moe_layer_idx: int):
        super().__init__()
        n_experts = config.num_moe_experts
        topk = config.moe_router_topk

        # Build the lookup table: for each "token slot" modulo n_experts,
        # which expert indices to select (shape: n_experts, topk).
        rng = torch.Generator()
        rng.manual_seed(config.hash_routing_seed + moe_layer_idx)
        # Permute expert IDs for each slot.
        table = torch.zeros(n_experts, topk, dtype=torch.long)
        for i in range(n_experts):
            perm = torch.randperm(n_experts, generator=rng)
            table[i] = perm[:topk]
        self.register_buffer("tid2eid", table)

        self.n_experts = n_experts
        self.topk = topk

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (tokens, d)

        Returns:
            indices: (tokens, topk) expert indices
            weights: (tokens, topk) uniform weights (1/topk)
        """
        tokens = hidden.shape[0]
        slot = torch.arange(tokens, device=hidden.device) % self.n_experts
        indices = self.tid2eid[slot]                       # (tokens, topk)
        weights = torch.full_like(indices, 1.0 / self.topk, dtype=hidden.dtype)
        return indices, weights


class LearnedRouter(nn.Module):
    """Learned top-k routing with sqrtsoftplus scoring and decoupled load balancing.

    Expert bias is added for selection only (not for routing weights), and
    updated outside gradient flow based on per-batch expert load.

    sqrtsoftplus: sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
        - Positive, monotone, no saturation → stable gradients.

    Decoupled load balancing:
        - expert_bias: updated every step by delta = speed * (overuse ? -1 : +1)
        - sequence-wise balance loss: moe_aux_loss_coeff * sum(f_i * P_i)
    """

    def __init__(self, config):
        super().__init__()
        n_experts = config.num_moe_experts
        d = config.hidden_size
        topk = config.moe_router_topk

        self.n_experts = n_experts
        self.topk = topk
        self.bias_speed = 0.001  # per V4 paper §3.2

        # Router weight matrix.
        self.weight = nn.Linear(d, n_experts, bias=False)

        # Expert bias: not a gradient parameter; updated by load signals.
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def _sqrtsoftplus(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x).sqrt()

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (tokens, d)

        Returns:
            indices: (tokens, topk)
            weights: (tokens, topk) — normalized sqrtsoftplus scores (no bias)
        """
        logits = self.weight(hidden)                      # (tokens, n_experts)
        scores = self._sqrtsoftplus(logits)               # (tokens, n_experts)

        # Selection uses biased scores; weights use unbiased scores.
        biased = scores + self.expert_bias.unsqueeze(0)
        _, indices = biased.topk(self.topk, dim=-1)       # (tokens, topk)

        # Routing weights from unbiased scores (renormalized over selected experts).
        weights = scores.gather(dim=-1, index=indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return indices, weights

    @torch.no_grad()
    def update_bias(self, indices: torch.Tensor):
        """Update expert_bias based on observed load.

        Called after each forward step (outside autograd graph).
        Overused experts get bias decreased; underused get increased.
        """
        tokens = indices.shape[0]
        target = tokens * self.topk / self.n_experts      # ideal load per expert
        count = torch.zeros(self.n_experts, device=indices.device)
        count.scatter_add_(0, indices.view(-1), torch.ones(indices.numel(), device=indices.device))
        delta = torch.where(count > target, -self.bias_speed, self.bias_speed)
        self.expert_bias.add_(delta)


class DeepSeekV4Router(nn.Module):
    """Dispatcher that selects HashRouter or LearnedRouter based on layer index."""

    def __init__(self, config, moe_layer_idx: int):
        super().__init__()
        self.use_hash = moe_layer_idx < config.num_hash_layers
        if self.use_hash:
            self.router = HashRouter(config, moe_layer_idx)
        else:
            self.router = LearnedRouter(config)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.router(hidden)
