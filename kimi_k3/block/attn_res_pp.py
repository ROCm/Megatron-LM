"""The AttnRes pipeline payload protocol.

**This module docstring is the specification.** See
develop/notes/2026-08-26-attn-res-pp-transport.md for why it looks like this.

Between layers the decoder carries two tensors:

    prefix_sum      [S, B, H]      the hidden state, reset at each block boundary
    block_residual  [S, B, K, H]   K frozen block outputs

Across a pipeline stage boundary they travel as **one packed tensor**:

    packed          [(1 + K) * S, B, H]

concatenated along the sequence axis so the shape stays the ``(seq, mbs, hidden)``
triple core's ``p2p_communication`` already understands.

Why one tensor and not two: core's ``backward_step`` back-props only
``output_tensor[0]`` with ``output_tensor_grad[0]``
(megatron/core/pipeline_parallel/schedules.py:451-493, with a comment saying it
"can handle at most one skip connection"). A second payload tensor would be sent
forward correctly, its gradient would be received from the next stage, and that
gradient would then be **dropped on the floor** -- no exception, no warning, and
a loss curve that still falls. ``pack`` / ``unpack`` are built from ``cat`` /
``view`` / ``permute``, all differentiable, so a single backward on the packed
tensor reaches ``prefix_sum`` and every slot.

Per-stage shapes come from the local layer range; core applies them through
``adjust_tensor_shapes_fn`` (schedules.py:2019, 2180-2183), which runs on each
rank, so a stage may legally receive a different shape than it sends. Gate G20
is the test that fails if a payload gradient is ever lost.
"""

import math
from typing import Tuple

import torch


def slots_before(layer_idx: int, block_size: int) -> int:
    """Number of block-residual slots visible on entry to 0-indexed ``layer_idx``.

    A slot is appended by every layer whose index is a multiple of ``block_size``,
    so the count on entry to layer ``l`` is ``ceil(l / block_size)``: 0 before
    layer 0, 1 for layers 1..12, 2 for 13..24, and 8 by the end of a 93-layer model.
    """
    return math.ceil(layer_idx / block_size)


def pack(prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
    """``[S,B,H]`` + ``[S,B,K,H]`` -> ``[(1+K)*S, B, H]``."""
    s, b, k, h = block_residual.shape
    assert prefix_sum.shape == (s, b, h), (prefix_sum.shape, block_residual.shape)
    if k == 0:
        return prefix_sum
    slots = block_residual.permute(2, 0, 1, 3).reshape(k * s, b, h)
    return torch.cat([prefix_sum, slots], dim=0)


def unpack(packed: torch.Tensor, seq_len: int, num_slots: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack`. Autograd-transparent."""
    expected = (1 + num_slots) * seq_len
    assert packed.shape[0] == expected, (
        f"packed payload has {packed.shape[0]} rows, expected {expected} "
        f"({1 + num_slots} x {seq_len})"
    )
    prefix_sum = packed[:seq_len]
    if num_slots == 0:
        empty = packed.new_zeros((seq_len, packed.shape[1], 0, packed.shape[2]))
        return prefix_sum, empty
    slots = packed[seq_len:].view(num_slots, seq_len, *packed.shape[1:])
    return prefix_sum, slots.permute(1, 2, 0, 3)


def payload_multiplier(first_layer: int, last_layer: int, block_size: int) -> Tuple[int, int]:
    """``(recv, send)`` sequence-axis multipliers for a stage owning ``[first, last]``.

    Neighbour consistency is automatic: stage *s* sends with
    ``1 + slots_before(last + 1)`` and stage *s+1* receives with
    ``1 + slots_before(first')`` where ``first' == last + 1``.
    """
    return 1 + slots_before(first_layer, block_size), 1 + slots_before(last_layer + 1, block_size)


def payload_bytes(seq_len: int, micro_batch: int, hidden: int, slots: int, dtype_size: int = 2) -> int:
    """Bytes on the wire for one packed payload."""
    return (1 + slots) * seq_len * micro_batch * hidden * dtype_size
