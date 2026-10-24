# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Tests for the MoE token-to-destination dedup plan and kernels."""

import pytest
import torch

from megatron.core.transformer.moe.token_dedup import (
    expand_token_dedup_received,
    pack_token_dedup_payload,
    permute_token_dedup_expert_output,
    reduce_token_dedup_expert_output,
    restore_token_dedup_output,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEDeduplicatingAlltoAllTokenDispatcher,
    _compact_mask_indices,
    _token_destination_hit_map,
)


def test_destination_hit_map_deduplicates_same_rank_experts():
    # Two experts per EP rank: {0, 1} -> rank 0 and {2, 3} -> rank 1.
    indices = torch.tensor(
        [
            [0, 1],  # one wire row to rank 0
            [0, 2],  # one wire row to each rank
            [3, 2],  # one wire row to rank 1
            [1, 1],  # repeated expert is still one wire row
        ],
        dtype=torch.long,
    )

    destinations, hits = _token_destination_hit_map(
        indices, num_local_experts=2, ep_size=2
    )

    assert torch.equal(
        destinations,
        torch.tensor([[0, 0], [0, 1], [1, 1], [0, 0]]),
    )
    assert torch.equal(
        hits,
        torch.tensor(
            [
                [True, True, False, True],
                [False, True, True, False],
            ]
        ),
    )
    # Eight token/expert assignments become five token/destination wire rows.
    assert hits.sum().item() == 5


def test_destination_major_compaction_needs_no_sort():
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, False],
        ]
    )

    compact = _compact_mask_indices(mask, count=4)

    # Flattened destination-major positions: rank0 tokens 0/2, rank1 tokens 1/2.
    assert torch.equal(compact, torch.tensor([0, 2, 5, 6]))


def test_compaction_handles_empty_mask():
    mask = torch.zeros((2, 0), dtype=torch.bool)
    compact = _compact_mask_indices(mask, count=0)
    assert compact.dtype == torch.long
    assert compact.numel() == 0


def test_packed_metadata_uses_same_wire_rows_and_preserves_gradients():
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    probs = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6], [0.125, 0.875]], requires_grad=True
    )
    wire_tokens = torch.tensor([0, 2, 0])
    wire_experts = torch.tensor([[1, -1], [0, 2], [-1, 3]])
    token_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, -1]])
    slot_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, 1]])

    packed = pack_token_dedup_payload(
        hidden,
        probs,
        wire_tokens,
        wire_experts,
        token_wire_rows,
        slot_wire_rows,
    )

    assert packed.shape == (3, 8)
    assert torch.equal(packed[:, :4], hidden.detach().index_select(0, wire_tokens))
    assert torch.equal(packed[:, 4:6].long(), wire_experts)
    assert torch.equal(
        packed[:, 6:8],
        torch.tensor([[0.25, 0.0], [0.125, 0.875], [0.0, 0.75]]),
    )

    packed.sum().backward()
    assert torch.equal(
        hidden.grad,
        torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )
    assert torch.equal(
        probs.grad,
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
    )


def test_received_expansion_and_wire_permutation_preserve_gradients():
    received = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    wire_rows = torch.tensor([0, 2, 1, 0])
    wire_to_expert_rows = torch.tensor([[0, 3], [2, -1], [1, -1]])

    expanded = expand_token_dedup_received(
        received, wire_rows, wire_to_expert_rows
    )
    expanded.sum().backward()
    assert torch.equal(
        received.grad,
        torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    expert_output = torch.arange(16, dtype=torch.float32).view(4, 4)
    expert_output.requires_grad_()
    wire_to_expert_row = torch.tensor([0, 3, 2, 1])
    expert_row_to_wire_rank = torch.tensor([0, 3, 2, 1])
    wire_output = permute_token_dedup_expert_output(
        expert_output, wire_to_expert_row, expert_row_to_wire_rank
    )
    row_weights = torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)
    (wire_output * row_weights).sum().backward()
    assert torch.equal(
        expert_output.grad[:, 0],
        torch.tensor([1.0, 4.0, 3.0, 2.0]),
    )


def test_reduce_and_restore_sum_expert_rows():
    expert_output = torch.arange(8, dtype=torch.float32).view(4, 2).requires_grad_()
    # Wire 0 holds two experts of token 0; wires 1 and 2 hold token 1's experts.
    wire_to_expert_rows = torch.tensor([[0, 1], [2, -1], [-1, 3]])
    expert_wire_rows = torch.tensor([0, 0, 1, 2])
    token_wire_rows = torch.tensor([[0, -1], [1, 2]])
    wire_tokens = torch.tensor([0, 1, 1])

    partials = reduce_token_dedup_expert_output(
        expert_output, expert_wire_rows, wire_to_expert_rows
    )
    output = restore_token_dedup_output(partials, token_wire_rows, wire_tokens)
    e = expert_output.detach()
    assert torch.equal(output, torch.stack((e[0] + e[1], e[2] + e[3])))

    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output.backward(grad)
    assert torch.equal(expert_output.grad, grad[[0, 0, 1, 1]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("ep_size,topk", [(2, 2), (8, 2), (8, 6)])
@pytest.mark.parametrize("weighted_combine", [False, True])
def test_dedup_matches_alltoall_dispatcher(ep_size, topk, weighted_combine):
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer

    container = MoEModelTestContainer(
        tp_size=1, ep_size=ep_size, pp_size=1, num_moe_experts=32, moe_router_topk=topk
    )
    dedup = container.new_moe_layer(
        moe_token_dedup=True, moe_token_dedup_weighted_combine=weighted_combine
    )
    dedup.load_state_dict(container.moe_layer.state_dict())
    assert isinstance(dedup.token_dispatcher, MoEDeduplicatingAlltoAllTokenDispatcher)
    gen = torch.Generator(device="cuda").manual_seed(torch.distributed.get_rank())
    x = torch.randn(64, 2, dedup.config.hidden_size, device="cuda", generator=gen)
    grad = torch.randn(x.shape, device="cuda", generator=gen)
    results = []
    for layer in (container.moe_layer, dedup):
        inp = x.clone().requires_grad_()
        out, _ = layer(inp)
        out.backward(grad)
        results.append((out, inp.grad, layer.router.weight.grad))
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref)
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("n,num_keys", [(5, 3), (1024, 8), (5003, 8), (70000, 16)])
def test_segment_ranks_match_membership_prefix(n, num_keys):
    from megatron.core.transformer.moe.token_dedup import segment_ranks

    gen = torch.Generator(device="cuda").manual_seed(n)
    keys = torch.randint(-1, num_keys, (n,), device="cuda", generator=gen)
    membership = torch.arange(num_keys, device="cuda").unsqueeze(1) == keys.unsqueeze(0)
    reference = (membership.cumsum(1) - 1).gather(0, keys.clamp(min=0).unsqueeze(0)).squeeze(0)
    rank = segment_ranks(keys, num_keys)
    valid = keys >= 0
    assert torch.equal(rank[valid].long(), reference[valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("rows,cols", [(1, 8), (1024, 8), (4097, 8), (3000, 16)])
def test_column_prefix_sum_matches_cumsum(rows, cols):
    from megatron.core.transformer.moe.token_dedup import column_prefix_sum

    gen = torch.Generator(device="cuda").manual_seed(rows)
    hits = torch.rand((rows, cols), device="cuda", generator=gen) < 0.3
    assert torch.equal(column_prefix_sum(hits).long(), hits.long().cumsum(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("tokens,num_experts,ep_size,topk", [(7, 16, 4, 2), (4096, 64, 8, 6), (1000, 32, 4, 4)])
def test_routing_counts_match_torch(tokens, num_experts, ep_size, topk):
    from megatron.core.transformer.moe.token_dedup import column_prefix_sum, routing_counts

    gen = torch.Generator(device="cuda").manual_seed(tokens)
    scores = torch.rand((tokens, num_experts), device="cuda", generator=gen)
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, scores.topk(topk, dim=1).indices, True)
    nle = num_experts // ep_size
    packed, hits, partial = routing_counts(routing_map, nle, ep_size)
    ref_hits = routing_map.view(tokens, ep_size, nle).any(dim=-1)
    assert torch.equal(hits.bool(), ref_hits)
    assert torch.equal(packed[:num_experts], routing_map.sum(0, dtype=torch.long))
    assert torch.equal(packed[num_experts:], ref_hits.sum(0, dtype=torch.long))
    order = column_prefix_sum(hits, partial[:, num_experts:], 256)
    assert torch.equal(order.long(), ref_hits.long().cumsum(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("rows,nle,topk", [(3, 4, 2), (2000, 8, 6), (700, 16, 8)])
def test_recv_index_from_payload_columns(rows, nle, topk):
    from megatron.core.transformer.moe.token_dedup import build_wire_recv_index, segment_ranks

    gen = torch.Generator(device="cuda").manual_seed(rows)
    experts = torch.randint(-1, nle, (rows, topk), device="cuda", generator=gen)
    probs = torch.rand((rows, topk), device="cuda", generator=gen)
    # payload-style strided bf16 columns: [activation(4) | experts | probs | pad]
    payload = torch.zeros((rows, 4 + 2 * topk + 2), dtype=torch.bfloat16, device="cuda")
    payload[:, 4 : 4 + topk] = experts.to(torch.bfloat16)
    payload[:, 4 + topk : 4 + 2 * topk] = probs.to(torch.bfloat16)
    counts = torch.stack([(experts == e).sum() for e in range(nle)]).long()
    num_assignments = int(counts.sum())
    w2e, e2w, eprobs = build_wire_recv_index(
        payload[:, 4 : 4 + topk], counts, num_assignments, payload[:, 4 + topk : 4 + 2 * topk]
    )
    base = counts.cumsum(0) - counts
    rank = segment_ranks(experts.reshape(-1), nle).long().view(rows, topk)
    ref_w2e = torch.where(experts >= 0, base[experts.clamp(min=0)] + rank, torch.full_like(experts, -1))
    assert torch.equal(w2e, ref_w2e)
    valid = experts >= 0
    assert torch.equal(e2w[ref_w2e[valid]], torch.arange(rows, device="cuda").unsqueeze(1).expand(-1, topk)[valid])
    assert torch.equal(eprobs[ref_w2e[valid]], probs.to(torch.bfloat16).float()[valid])
