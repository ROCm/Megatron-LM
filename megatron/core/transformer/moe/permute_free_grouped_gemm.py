# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for the permute-free route-list grouped GEMM path (TE / AITER on ROCm).

This wires Megatron's MoE expert path to TransformerEngine's route-list gather-GEMM
(``NVTE_PERMUTE_FREE_GROUPED_GEMM``). When active, the expert **FC1** GroupedLinear
takes the raw post-dispatch activations ``[num_recv_tokens, hidden]`` plus a boolean
``routing_map`` and gathers per expert *inside* the GEMM, so the local permute is
never materialized. FC1's forward output is the compact, expert-sorted
``[num_routes, ffn]`` tensor (identical layout to ``permute`` + a regular grouped
GEMM), and FC1's backward scatters the input gradient back to ``[num_recv, hidden]``.

FC2 stays on the default GroupedLinear path (it already receives the expert-sorted
``[num_routes, ffn]`` activations), and the usual ``unpermute`` in ``combine``
scatters the ``[num_routes, hidden]`` output back to the received-token order. This
means the permute-free path differs from the legacy path in exactly one place: the
FC1 gather is fused into the GEMM instead of being a standalone permute.
"""

from __future__ import annotations

import dataclasses
import os

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.moe_routing import PermuteFreeMetadata

    HAVE_TE_ROUTING = hasattr(te, "MoERoutingMetadata")
except ImportError:
    te = None
    PermuteFreeMetadata = None
    HAVE_TE_ROUTING = False


def make_permute_free_metadata(
    routing_map: "torch.Tensor", route_space: bool = False, topk: "int | None" = None
):
    """Build a TE ``PermuteFreeMetadata`` from a boolean routing map.

    ``routing_map`` is ``[num_recv_tokens, num_local_experts]`` (True where a received token
    feeds a local expert). ``route_space=False`` selects the FC1 gather direction;
    ``route_space=True`` selects the FC2 route-in / scatter-to-token direction. The align
    buffers are built lazily inside the GEMM and cached on the returned object, so a single
    metadata can be reused across FC1 and FC2 (see ``as_route_space``).

    ``topk`` is the routing top-k (max experts per token). Passing it tightens the sync-free
    align over-allocation from ``num_recv_tokens * num_experts`` to
    ``num_recv_tokens * min(topk, num_experts)``, which for ``topk << num_experts`` avoids a
    huge padded ``[em_max, out]`` route buffer (and the matching activation-memory blow-up).
    """
    if PermuteFreeMetadata is None:
        raise RuntimeError(
            "moe_permute_free_grouped_gemm is enabled but the installed Transformer Engine "
            "does not expose PermuteFreeMetadata. Please upgrade Transformer Engine."
        )
    return PermuteFreeMetadata(routing_map.bool(), topk=topk, route_space=route_space)


def as_route_space(metadata):
    """Return a route-space (FC2) view of a metadata, sharing its cached align buffers."""
    return dataclasses.replace(metadata, route_space=True)


def build_padded_route_probs(metadata, dispatched_probs: "torch.Tensor") -> "torch.Tensor":
    """Build per-route probs in the padded route layout, sync-free.

    Produces a ``[em_max]`` tensor whose valid compact range ``[0, num_routes)`` holds
    ``dispatched_probs[route_to_token[r], expert(r)]`` (expert-major route order, matching
    FC1's padded output) and whose tail is zero. Uses only the route-list align buffers
    (``route_to_token``, ``route_start``, ``num_routes_dev``) already built on ``metadata``,
    so there is no ``masked_select`` / ``.item()`` device-to-host sync.

    ``metadata`` must have its align buffers built (i.e. after the FC1 forward, or after an
    explicit ``prepare_moe_align``). ``dispatched_probs`` is ``[num_recv_tokens,
    num_local_experts]``.
    """
    route_to_token = metadata.route_to_token
    route_start = metadata.route_start
    if route_to_token is None or route_start is None:
        raise RuntimeError(
            "build_padded_route_probs requires align buffers; call after the FC1 forward "
            "(or prepare_moe_align) so route_to_token/route_start are populated."
        )
    device = route_to_token.device
    num_recv = dispatched_probs.shape[0]
    num_experts = int(route_start.shape[0])
    routes_max = int(route_to_token.shape[0])
    em_max = int(metadata.sorted_slot_ids.shape[0])

    route_start_i64 = route_start.to(torch.int64)
    r_idx = torch.arange(routes_max, device=device)
    # expert(r) = (#experts whose compact start <= r) - 1; handles empty experts (duplicate
    # route_start entries) and clamps the over-allocated tail to the last expert (masked off).
    expert_per_route = torch.searchsorted(route_start_i64, r_idx, right=True) - 1
    expert_per_route = expert_per_route.clamp_(0, num_experts - 1)

    tok = route_to_token.to(torch.int64).clamp_(0, num_recv - 1)
    probs_compact = dispatched_probs[tok, expert_per_route]  # [routes_max]
    # Valid compact routes are [0, num_routes); num_routes = routing_map.sum() is a device
    # scalar, so the compare stays on-GPU (no .item() / masked_select host sync).
    num_routes = metadata.routing_map.sum()
    valid = r_idx < num_routes
    probs_compact = torch.where(valid, probs_compact, torch.zeros_like(probs_compact))

    if em_max >= routes_max:
        return torch.nn.functional.pad(probs_compact, (0, em_max - routes_max))
    return probs_compact[:em_max]


def apply_route_probs(metadata, activation: "torch.Tensor", dispatched_probs: "torch.Tensor"):
    """Fold the per-route gating prob into the FC1 activation via TE (sync-free, fast bwd).

    Replaces ``build_padded_route_probs`` + a separate multiply: TE gathers each route's prob
    ``dispatched_probs[token(r), expert(r)]`` inside a Triton kernel (reusing the FC1 align
    buffers) and scales ``activation[r]`` by it. The backward scatters the prob gradient with
    a bounded, masked atomic add -- avoiding the fp32 ``aten::_index_put_impl_`` whose backward
    dominates the naive advanced-index approach.

    ``metadata`` must have its FC1 align buffers built (i.e. after the FC1 forward).
    ``activation`` is ``[em_max, H]`` (route layout); ``dispatched_probs`` is
    ``[num_recv_tokens, num_local_experts]``.
    """
    from transformer_engine.pytorch.triton_kernels.route_prob import (
        apply_route_probs as _te_apply_route_probs,
    )

    return _te_apply_route_probs(activation, dispatched_probs, metadata)


def is_moe_permute_free_grouped_gemm_active(config: TransformerConfig) -> bool:
    """Return True when Megatron and TE permute-free gather-GEMM are both enabled.

    Training is supported (TE implements the route-list dgrad/wgrad), so this does
    NOT gate on ``torch.is_grad_enabled()``.
    """
    if not getattr(config, "moe_permute_free_grouped_gemm", False):
        return False
    if os.environ.get("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") != "1":
        return False
    if os.environ.get("NVTE_USE_GROUPED_GEMM_TRITON", "0") == "1":
        raise RuntimeError(
            "NVTE_PERMUTE_FREE_GROUPED_GEMM and NVTE_USE_GROUPED_GEMM_TRITON cannot both be "
            "enabled. Unset NVTE_USE_GROUPED_GEMM_TRITON when using permute-free grouped GEMM."
        )
    if not HAVE_TE_ROUTING:
        raise RuntimeError(
            "moe_permute_free_grouped_gemm is enabled but the installed TransformerEngine does "
            "not expose the route-list permute-free API. Please upgrade TransformerEngine."
        )
    return True
