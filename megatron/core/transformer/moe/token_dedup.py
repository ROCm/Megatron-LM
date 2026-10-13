# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Fused payload and route-metadata packing for MoE token dedup."""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# RCCL is substantially slower for the otherwise 16,392-byte BF16 wire rows
# produced by hidden=8192/topk=2. Round every row to its next 16-byte boundary.
_PAYLOAD_ALIGNMENT_BYTES = 16


_PRIMUS_GROUPED_GEMM = None
_PRIMUS_CHECKED = False


PRIMUS_GROUPED_GEMM_BACKEND_ENV = "PRIMUS_TURBO_GROUPED_GEMM_BACKEND"
PRIMUS_GROUPED_GEMM_DEFAULT_BACKEND = "TRITON"


def primus_grouped_gemm():
    """Primus-Turbo's grouped GEMM (``primus_turbo.pytorch.ops.grouped_gemm``) or None.

    Imported lazily on first use so runs without token dedup never load the package. When the
    package is missing the dedup expert batch falls back to the legacy grouped_gemm backend and
    a warning is printed once.

    Primus-Turbo's own default backend (CK) converts its accumulator to bf16 with round-toward-zero,
    which shrinks every expert output by ~0.28% per GEMM and shows up as a systematic loss /
    grad-norm offset against the stock dispatcher. The TRITON backend accumulates in fp32, rounds
    to nearest like grouped_gemm, and is faster than CK on MI300X, so it is selected here unless
    PRIMUS_TURBO_GROUPED_GEMM_BACKEND is already set in the environment.
    """
    global _PRIMUS_GROUPED_GEMM, _PRIMUS_CHECKED
    if not _PRIMUS_CHECKED:
        _PRIMUS_CHECKED = True
        try:
            os.environ.setdefault(PRIMUS_GROUPED_GEMM_BACKEND_ENV, PRIMUS_GROUPED_GEMM_DEFAULT_BACKEND)
            from primus_turbo.pytorch.ops import grouped_gemm

            _PRIMUS_GROUPED_GEMM = grouped_gemm
        except ImportError:
            import warnings

            warnings.warn(
                "primus_turbo is not importable: --moe-token-dedup runs the expert grouped GEMMs "
                "on the legacy grouped_gemm backend (pass --no-moe-token-dedup-primus-gemm to "
                "silence this)."
            )
    return _PRIMUS_GROUPED_GEMM


def dedup_payload_width(hidden_size: int, topk: int, element_size: int) -> int:
    """Return the activation-plus-route-metadata width with RCCL-safe alignment."""
    # Each route slot carries one exactly-representable local expert ID and one
    # routing probability.  For hidden=8192/topk=2/BF16 these four elements fit
    # inside the padding already required by the 16-byte RCCL alignment.
    payload_bytes = (hidden_size + 2 * topk) * element_size
    aligned_bytes = (
        (payload_bytes + _PAYLOAD_ALIGNMENT_BYTES - 1)
        // _PAYLOAD_ALIGNMENT_BYTES
        * _PAYLOAD_ALIGNMENT_BYTES
    )
    return (aligned_bytes + element_size - 1) // element_size


if triton is not None:

    @triton.jit
    def _pack_token_dedup_kernel(
        hidden_ptr,
        prob_ptr,
        token_ptr,
        expert_ptr,
        output_ptr,
        hidden_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        output_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        output_mask = columns < output_width
        hidden_mask = columns < hidden_size

        token = tl.load(token_ptr + row)
        hidden = tl.load(
            hidden_ptr + token * hidden_stride + columns,
            mask=hidden_mask,
            other=0.0,
        )

        expert_column = columns - hidden_size
        expert_mask = (columns >= hidden_size) & (
            columns < hidden_size + topk
        )
        expert = tl.load(
            expert_ptr + row * topk + expert_column,
            mask=expert_mask,
            other=0,
        )
        prob_column = columns - hidden_size - topk
        prob_mask = (columns >= hidden_size + topk) & (
            columns < hidden_size + 2 * topk
        )
        prob_expert = tl.load(
            expert_ptr + row * topk + prob_column,
            mask=prob_mask,
            other=-1,
        )
        prob = tl.load(
            prob_ptr + token * topk + prob_column,
            mask=prob_mask & (prob_expert >= 0),
            other=0.0,
        )
        metadata = tl.where(expert_mask, expert.to(tl.float32), prob)
        value = tl.where(hidden_mask, hidden, metadata)
        tl.store(output_ptr + row * output_width + columns, value, mask=output_mask)

    @triton.jit
    def _unpack_token_dedup_payload_grad_kernel(
        grad_output_ptr,
        token_wire_rows_ptr,
        slot_wire_rows_ptr,
        grad_hidden_ptr,
        grad_prob_ptr,
        grad_output_stride_0,
        grad_output_stride_1,
        num_grad_output_rows,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid_wire_row = (wire_row >= 0) & (
                wire_row < num_grad_output_rows
            )
            safe_wire_row = tl.where(valid_wire_row, wire_row, 0)
            value = tl.load(
                grad_output_ptr
                + safe_wire_row * grad_output_stride_0
                + columns * grad_output_stride_1,
                mask=hidden_mask & valid_wire_row,
                other=0.0,
            )
            accumulator += value

        tl.store(
            grad_hidden_ptr + token * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )

        # Every token/route slot occurs in exactly one wire row.  Have the first
        # hidden tile also recover those tiny metadata gradients so packing adds
        # no separate backward kernel.
        prob_mask = (block == 0) & (columns < topk)
        slot_wire_row = tl.load(
            slot_wire_rows_ptr + token * topk + columns,
            mask=prob_mask,
            other=0,
        )
        prob_grad = tl.load(
            grad_output_ptr
            + slot_wire_row * grad_output_stride_0
            + (hidden_size + topk + columns) * grad_output_stride_1,
            mask=prob_mask & (slot_wire_row >= 0) & (slot_wire_row < num_grad_output_rows),
            other=0.0,
        )
        tl.store(
            grad_prob_ptr + token * topk + columns,
            prob_grad,
            mask=prob_mask,
        )

    @triton.jit
    def _unpack_token_dedup_grad_kernel(
        grad_output_ptr,
        token_wire_rows_ptr,
        grad_hidden_ptr,
        grad_output_stride_0,
        grad_output_stride_1,
        num_grad_output_rows,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Gather and sum bounded row gradients without metadata handling."""
        token = tl.program_id(0)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid_wire_row = (wire_row >= 0) & (
                wire_row < num_grad_output_rows
            )
            safe_wire_row = tl.where(valid_wire_row, wire_row, 0)
            value = tl.load(
                grad_output_ptr
                + safe_wire_row * grad_output_stride_0
                + columns * grad_output_stride_1,
                mask=hidden_mask & valid_wire_row,
                other=0.0,
            )
            accumulator += value

        tl.store(
            grad_hidden_ptr + token * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )

    @triton.jit
    def _weighted_reduce_token_dedup_kernel(
        expert_output_ptr,
        wire_to_expert_rows_ptr,
        wire_prob_ptr,
        output_ptr,
        wire_prob_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        wire_row = tl.program_id(0)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            expert_row = tl.load(
                wire_to_expert_rows_ptr + wire_row * topk + slot
            )
            valid = expert_row >= 0
            safe_expert_row = tl.where(valid, expert_row, 0)
            expert_value = tl.load(
                expert_output_ptr + safe_expert_row * hidden_size + columns,
                mask=hidden_mask & valid,
                other=0.0,
            )
            probability = tl.load(
                wire_prob_ptr + wire_row * wire_prob_stride + slot,
                mask=valid,
                other=0.0,
            )
            # Match the existing eager BF16 expression: round each weighted
            # product to the activation dtype before the top-k reduction.
            product = (expert_value * probability).to(
                output_ptr.dtype.element_ty
            )
            accumulator += product.to(tl.float32)

        tl.store(
            output_ptr + wire_row * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )

    @triton.jit
    def _weighted_reduce_token_dedup_backward_kernel(
        grad_output_ptr,
        expert_output_ptr,
        wire_to_expert_rows_ptr,
        wire_prob_ptr,
        grad_expert_ptr,
        grad_prob_ptr,
        wire_prob_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        wire_row = assignment // topk
        slot = assignment % topk
        expert_row = tl.load(wire_to_expert_rows_ptr + assignment)
        valid = expert_row >= 0
        safe_expert_row = tl.where(valid, expert_row, 0)
        probability = tl.load(
            wire_prob_ptr + wire_row * wire_prob_stride + slot,
            mask=valid,
            other=0.0,
        )
        accumulator = tl.zeros((block_size,), dtype=tl.float32)
        columns = tl.arange(0, block_size)

        for block in tl.static_range(
            0, (hidden_size + block_size - 1) // block_size
        ):
            offsets = block * block_size + columns
            hidden_mask = offsets < hidden_size
            grad_output = tl.load(
                grad_output_ptr + wire_row * hidden_size + offsets,
                mask=hidden_mask & valid,
                other=0.0,
            )
            expert_value = tl.load(
                expert_output_ptr + safe_expert_row * hidden_size + offsets,
                mask=hidden_mask & valid,
                other=0.0,
            )
            grad_expert = (grad_output * probability).to(
                grad_expert_ptr.dtype.element_ty
            )
            tl.store(
                grad_expert_ptr + safe_expert_row * hidden_size + offsets,
                grad_expert,
                mask=hidden_mask & valid,
            )
            prob_product = (grad_output * expert_value).to(
                grad_prob_ptr.dtype.element_ty
            )
            accumulator += prob_product.to(tl.float32)

        tl.store(
            grad_prob_ptr + assignment,
            tl.sum(accumulator, axis=0),
            mask=valid,
        )

    @triton.jit
    def _build_wire_send_index_kernel(
        routing_map_ptr,  # uint8 [T, E], slot-space routing bits
        order_ptr,  # int64 [T, EP], inclusive per-destination token prefix of hits
        base_ptr,  # int64 [EP], exclusive per-destination wire-row base
        indices_ptr,  # int64 [T, K] out: set-bit expert ids, ascending
        slot_wire_rows_ptr,  # int64 [T, K] out: wire row of every route slot
        token_wire_rows_ptr,  # int64 [T, K] out: wire row at its first slot, else -1
        wire_tokens_ptr,  # int64 [Wmax] out: source token of each wire row
        wire_experts_ptr,  # int64 [Wmax, K] out: local expert per slot (prefilled -1)
        num_experts: tl.constexpr,
        experts_pow2: tl.constexpr,
        num_local_experts: tl.constexpr,
        ep_size: tl.constexpr,
        topk: tl.constexpr,
    ):
        """One program per token builds every send-side dedup index in one pass.

        Replaces the ~28-launch torch composition in ``_build_send_buffers``
        (compactions, remainders, index_copy chains). Wire-row numbering is
        destination-major with tokens ascending inside a destination -- the
        exact order ``_compact_mask_indices`` produced -- via
        ``base[dest] + order[token, dest] - 1``. Slot order within a token is
        ascending expert id (torch.topk's tie order was arbitrary; any
        consistent order is valid because the probabilities ride in the same
        slots).
        """
        token = tl.program_id(0)
        cols = tl.arange(0, experts_pow2)
        emask = cols < num_experts
        bits = tl.load(
            routing_map_ptr + token * num_experts + cols, mask=emask, other=0
        ).to(tl.int32)
        set_mask = emask & (bits != 0)
        # Conditional top-1 leaves unused slots in the fixed-width top-2
        # payload. Initialize only those slots; their backward maps must be -1.
        ks = tl.arange(0, triton.next_power_of_2(topk))
        unused = (ks < topk) & (ks >= tl.sum(set_mask.to(tl.int32), axis=0))
        tl.store(indices_ptr + token * topk + ks, 0, mask=unused)
        tl.store(slot_wire_rows_ptr + token * topk + ks, -1, mask=unused)
        tl.store(token_wire_rows_ptr + token * topk + ks, -1, mask=unused)
        slot = tl.cumsum(bits, axis=0) - 1
        slot_ok = set_mask & (slot < topk)
        dest = cols // num_local_experts
        local_expert = (cols % num_local_experts).to(tl.int64)
        order_t = tl.load(order_ptr + token * ep_size + dest, mask=slot_ok, other=0)
        base_d = tl.load(base_ptr + dest, mask=slot_ok, other=0)
        wire = base_d + order_t - 1

        tl.store(indices_ptr + token * topk + slot, cols.to(tl.int64), mask=slot_ok)
        tl.store(slot_wire_rows_ptr + token * topk + slot, wire, mask=slot_ok)

        # A slot is the wire row's representative iff it is the token's FIRST
        # set bit within that destination block (destinations are e // nle, so
        # blocks are contiguous in expert id).
        inblk_pref = tl.zeros((experts_pow2,), dtype=tl.int32)
        for d in tl.static_range(ep_size):
            blk = tl.where(dest == d, bits, 0)
            inblk_pref += tl.where(dest == d, tl.cumsum(blk, axis=0), 0)
        rep = slot_ok & (inblk_pref == 1)

        tl.store(
            token_wire_rows_ptr + token * topk + slot,
            tl.where(rep, wire, -1),
            mask=slot_ok,
        )
        tl.store(wire_tokens_ptr + wire, token.to(tl.int64), mask=rep)
        tl.store(wire_experts_ptr + wire * topk + slot, local_expert, mask=slot_ok)

    @triton.jit
    def _build_wire_recv_index_kernel(
        experts_ptr,  # int64 [R, K], received local experts (-1 inactive)
        order_ptr,  # int64 [nle, R*K], inclusive per-expert membership prefix
        base_ptr,  # int64 [nle], exclusive expert-major row base
        wire_to_expert_rows_ptr,  # int64 [R, K] out: expert row per slot, else -1
        expert_wire_rows_ptr,  # int64 [A] out: expert-major row -> wire row
        total_flat,  # R * K
        topk: tl.constexpr,
        topk_pow2: tl.constexpr,
    ):
        """One program per received wire row builds the receive-side indices.

        Replaces the ~19-launch torch composition in ``_expand_received``.
        Expert-major numbering: ``base[e] + order[e, flat_slot] - 1`` reproduces
        the membership-compaction order (rows ascending within an expert).
        """
        row = tl.program_id(0)
        ks = tl.arange(0, topk_pow2)
        kmask = ks < topk
        expert = tl.load(experts_ptr + row * topk + ks, mask=kmask, other=-1)
        valid = kmask & (expert >= 0)
        safe_expert = tl.where(valid, expert, 0)
        flat = row * topk + ks
        order_e = tl.load(order_ptr + safe_expert * total_flat + flat, mask=valid, other=0)
        base_e = tl.load(base_ptr + safe_expert, mask=valid, other=0)
        expert_row = base_e + order_e - 1
        tl.store(
            wire_to_expert_rows_ptr + flat,
            tl.where(valid, expert_row, -1),
            mask=kmask,
        )
        tl.store(expert_wire_rows_ptr + expert_row, row.to(tl.int64), mask=valid)

    @triton.jit
    def _restore_token_dedup_kernel(
        returned_ptr,
        token_wire_rows_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid = wire_row >= 0
            safe_wire_row = tl.where(valid, wire_row, 0)
            value = tl.load(
                returned_ptr + safe_wire_row * hidden_size + columns,
                mask=hidden_mask & valid,
                other=0.0,
            )
            accumulator += value.to(tl.float32)

        tl.store(
            output_ptr + token * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )


class _PackTokenDedup(torch.autograd.Function):
    """Pack activations and routes while preserving hidden/probability gradients."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        wire_tokens: torch.Tensor,
        wire_experts: torch.Tensor,
        token_wire_rows: torch.Tensor,
        slot_wire_rows: torch.Tensor,
    ) -> torch.Tensor:
        num_wire_rows = wire_tokens.numel()
        hidden_size = hidden_states.shape[1]
        topk = wire_experts.shape[1]
        output_width = dedup_payload_width(
            hidden_size, topk, hidden_states.element_size()
        )
        ctx.hidden_shape = hidden_states.shape
        ctx.prob_shape = probs.shape
        ctx.prob_dtype = probs.dtype
        ctx.save_for_backward(wire_tokens, token_wire_rows, slot_wire_rows)

        if num_wire_rows == 0:
            return hidden_states.new_empty((0, output_width))

        if triton is None or not hidden_states.is_cuda:
            payload = torch.cat(
                (
                    hidden_states.index_select(0, wire_tokens),
                    wire_experts.to(dtype=hidden_states.dtype),
                    torch.where(
                        wire_experts >= 0,
                        probs.index_select(0, wire_tokens),
                        torch.zeros((), dtype=probs.dtype, device=probs.device),
                    ).to(dtype=hidden_states.dtype),
                ),
                dim=1,
            )
            if payload.shape[1] != output_width:
                payload = torch.nn.functional.pad(
                    payload, (0, output_width - payload.shape[1])
                )
            return payload

        output = hidden_states.new_empty((num_wire_rows, output_width))
        # Hidden=8192 otherwise launches 33 programs per wire row and reloads
        # the row's token/expert metadata in every program. A wider tile cuts
        # that launch/grid and metadata overhead while retaining enough rows to
        # saturate the device for the dispatcher workloads.
        block_size = 1024
        grid = (
            num_wire_rows,
            triton.cdiv(output_width, block_size),
        )
        _pack_token_dedup_kernel[grid](
            hidden_states,
            probs,
            wire_tokens,
            wire_experts,
            output,
            hidden_states.stride(0),
            hidden_size=hidden_size,
            topk=topk,
            output_width=output_width,
            block_size=block_size,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        wire_tokens, token_wire_rows, slot_wire_rows = ctx.saved_tensors
        num_tokens, hidden_size = ctx.hidden_shape
        if triton is None or not grad_output.is_cuda:
            grad_hidden = grad_output.new_zeros((num_tokens, hidden_size))
            grad_hidden.index_add_(0, wire_tokens, grad_output[:, :hidden_size])
            topk = ctx.prob_shape[1]
            grad_prob = torch.empty(
                ctx.prob_shape,
                dtype=ctx.prob_dtype,
                device=grad_output.device,
            )
            slots = torch.arange(topk, device=grad_output.device).view(1, topk)
            # Inactive slots carry -1; clamp for the gather and zero them after, matching
            # the masked store of the triton backward.
            grad_prob.copy_(
                grad_output[
                    slot_wire_rows.clamp(min=0),
                    hidden_size + topk + slots,
                ]
            )
            grad_prob.mul_((slot_wire_rows >= 0).to(dtype=grad_prob.dtype))
            return grad_hidden, grad_prob, None, None, None, None

        grad_hidden = grad_output.new_empty((num_tokens, hidden_size))
        grad_prob = torch.empty(
            ctx.prob_shape,
            dtype=ctx.prob_dtype,
            device=grad_output.device,
        )
        block_size = 256
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))
        _unpack_token_dedup_payload_grad_kernel[grid](
            grad_output,
            token_wire_rows,
            slot_wire_rows,
            grad_hidden,
            grad_prob,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.shape[0],
            hidden_size=hidden_size,
            topk=token_wire_rows.shape[1],
            block_size=block_size,
        )
        return grad_hidden, grad_prob, None, None, None, None


class _ExpandTokenDedup(torch.autograd.Function):
    """Expand wire rows while gathering their gradients without atomics."""

    @staticmethod
    def forward(
        ctx,
        received: torch.Tensor,
        wire_rows: torch.Tensor,
        wire_to_expert_rows: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = received.shape
        ctx.save_for_backward(wire_rows, wire_to_expert_rows)
        return received.index_select(0, wire_rows)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        wire_rows, wire_to_expert_rows = ctx.saved_tensors
        num_wire_rows, hidden_size = ctx.input_shape
        grad_received = _expanded_token_dedup_grad(
            grad_output,
            wire_to_expert_rows,
            num_wire_rows,
            hidden_size,
            wire_rows,
        )
        return grad_received, None, None


def _expanded_token_dedup_grad(
    grad_output: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
    num_wire_rows: int,
    hidden_size: int,
    wire_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather one expert-major gradient back to its deduplicated wire rows."""
    if triton is None or not grad_output.is_cuda:
        if wire_rows is None:
            active = wire_to_expert_rows >= 0
            wire_rows = torch.nonzero(active, as_tuple=False)[:, 0]
        grad_received = grad_output.new_zeros((num_wire_rows, hidden_size))
        grad_received.index_add_(0, wire_rows, grad_output)
        return grad_received

    grad_received = grad_output.new_empty((num_wire_rows, hidden_size))
    # As in forward payload packing, hidden=8192 benefits from fewer, wider
    # programs per wire row.  Every output row is written by exactly one
    # program, so multiple peer sources can be gathered without atomics.
    block_size = 1024
    grid = (num_wire_rows, triton.cdiv(hidden_size, block_size))
    _unpack_token_dedup_grad_kernel[grid](
        grad_output,
        wire_to_expert_rows,
        grad_received,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.shape[0],
        hidden_size=hidden_size,
        topk=wire_to_expert_rows.shape[1],
        block_size=block_size,
    )
    return grad_received


class _PermuteTokenDedup(torch.autograd.Function):
    """Apply a known row permutation with a gather-only backward."""

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        wire_to_expert_row: torch.Tensor,
        expert_row_to_wire_rank: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(expert_row_to_wire_rank)
        return expert_output.index_select(0, wire_to_expert_row)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (expert_row_to_wire_rank,) = ctx.saved_tensors
        grad_expert_output = grad_output.index_select(
            0, expert_row_to_wire_rank
        )
        return grad_expert_output, None, None


class _WeightedReduceTokenDedup(torch.autograd.Function):
    """Weight expert outputs and reduce them to one row per destination."""

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        wire_to_expert_rows: torch.Tensor,
        wire_probs: torch.Tensor,
    ) -> torch.Tensor:
        num_wire_rows, topk = wire_to_expert_rows.shape
        hidden_size = expert_output.shape[1]
        ctx.save_for_backward(expert_output, wire_to_expert_rows, wire_probs)
        if num_wire_rows == 0:
            return expert_output.new_empty((0, hidden_size))

        if triton is None or not expert_output.is_cuda:
            safe_rows = wire_to_expert_rows.clamp_min(0)
            gathered = expert_output.index_select(0, safe_rows.reshape(-1)).view(
                num_wire_rows, topk, hidden_size
            )
            weighted = gathered * wire_probs.unsqueeze(-1)
            weighted = weighted.masked_fill(
                (wire_to_expert_rows < 0).unsqueeze(-1), 0
            )
            return weighted.sum(dim=1)

        output = expert_output.new_empty((num_wire_rows, hidden_size))
        block_size = 1024
        grid = (num_wire_rows, triton.cdiv(hidden_size, block_size))
        _weighted_reduce_token_dedup_kernel[grid](
            expert_output,
            wire_to_expert_rows,
            wire_probs,
            output,
            wire_probs.stride(0),
            hidden_size=hidden_size,
            topk=topk,
            block_size=block_size,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        expert_output, wire_to_expert_rows, wire_probs = ctx.saved_tensors
        num_wire_rows, topk = wire_to_expert_rows.shape
        if triton is None or not grad_output.is_cuda:
            safe_rows = wire_to_expert_rows.clamp_min(0)
            valid = wire_to_expert_rows >= 0
            gathered = expert_output.index_select(0, safe_rows.reshape(-1)).view(
                num_wire_rows, topk, expert_output.shape[1]
            )
            grad_expanded = grad_output.unsqueeze(1) * wire_probs.unsqueeze(-1)
            grad_expert = torch.zeros_like(expert_output)
            grad_expert.index_add_(
                0,
                safe_rows[valid],
                grad_expanded[valid],
            )
            grad_prob = (grad_output.unsqueeze(1) * gathered).sum(dim=-1)
            grad_prob.masked_fill_(~valid, 0)
            return grad_expert, None, grad_prob

        grad_expert = torch.empty_like(expert_output)
        grad_prob = torch.zeros_like(wire_probs)
        block_size = 1024
        grid = (num_wire_rows * topk,)
        _weighted_reduce_token_dedup_backward_kernel[grid](
            grad_output,
            expert_output,
            wire_to_expert_rows,
            wire_probs,
            grad_expert,
            grad_prob,
            wire_probs.stride(0),
            hidden_size=expert_output.shape[1],
            topk=topk,
            block_size=block_size,
            num_warps=8,
        )
        return grad_expert, None, grad_prob


class _RestoreTokenDedup(torch.autograd.Function):
    """Sum destination partials into tokens with a gather-only backward."""

    @staticmethod
    def forward(
        ctx,
        returned: torch.Tensor,
        token_wire_rows: torch.Tensor,
        wire_tokens: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, topk = token_wire_rows.shape
        hidden_size = returned.shape[1]
        ctx.save_for_backward(wire_tokens)
        if num_tokens == 0:
            return returned.new_empty((0, hidden_size))

        if triton is None or not returned.is_cuda:
            output = returned.new_zeros((num_tokens, hidden_size))
            output.index_add_(0, wire_tokens, returned)
            return output

        output = returned.new_empty((num_tokens, hidden_size))
        block_size = 1024
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))
        _restore_token_dedup_kernel[grid](
            returned,
            token_wire_rows,
            output,
            hidden_size=hidden_size,
            topk=topk,
            block_size=block_size,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (wire_tokens,) = ctx.saved_tensors
        return grad_output.index_select(0, wire_tokens), None, None


def pack_token_dedup_payload(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    wire_tokens: torch.Tensor,
    wire_experts: torch.Tensor,
    token_wire_rows: torch.Tensor,
    slot_wire_rows: torch.Tensor,
) -> torch.Tensor:
    """Return contiguous ``[activation | expert IDs | probabilities]`` rows."""
    return _PackTokenDedup.apply(
        hidden_states,
        probs,
        wire_tokens,
        wire_experts,
        token_wire_rows,
        slot_wire_rows,
    )


def expand_token_dedup_received(
    received: torch.Tensor,
    wire_rows: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
) -> torch.Tensor:
    """Fan wire rows out to experts with a bounded gather-sum backward."""
    return _ExpandTokenDedup.apply(received, wire_rows, wire_to_expert_rows)


def permute_token_dedup_expert_output(
    expert_output: torch.Tensor,
    wire_to_expert_row: torch.Tensor,
    expert_row_to_wire_rank: torch.Tensor,
) -> torch.Tensor:
    """Restore wire order without the atomic backward of ``index_select``."""
    return _PermuteTokenDedup.apply(
        expert_output, wire_to_expert_row, expert_row_to_wire_rank
    )


def weighted_reduce_token_dedup_expert_output(
    expert_output: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
    wire_probs: torch.Tensor,
) -> torch.Tensor:
    """Return one probability-weighted expert-output row per wire row."""
    return _WeightedReduceTokenDedup.apply(
        expert_output, wire_to_expert_rows, wire_probs
    )


def restore_token_dedup_output(
    returned: torch.Tensor,
    token_wire_rows: torch.Tensor,
    wire_tokens: torch.Tensor,
) -> torch.Tensor:
    """Sum returned destination rows into the original local token order."""
    return _RestoreTokenDedup.apply(returned, token_wire_rows, wire_tokens)


def fused_index_build_available(device: torch.device) -> bool:
    """True when the single-kernel (Triton) index builders can run on ``device``;
    otherwise the equivalent multi-op torch composition is used."""
    return triton is not None and device.type == "cuda"


def build_wire_send_index(
    routing_map: torch.Tensor,
    order: torch.Tensor,
    base: torch.Tensor,
    num_local_experts: int,
    ep_size: int,
    topk: int,
    max_wire_rows: int,
):
    """Build every send-side dedup index tensor with one kernel launch.

    ``order`` is ``hits.cumsum(0)`` ([tokens, ep_size] inclusive per-destination
    token prefix) and ``base`` the exclusive cumsum of the per-destination wire
    counts, both pure GPU tensors -- so this needs NO host split values and can
    run while the metadata all-gather is still in flight. The wire tensors are
    sized ``max_wire_rows`` (an upper bound); the caller narrows them to the
    real count once the splits have landed on the host.

    Returns ``(indices, slot_wire_rows, token_wire_rows, wire_tokens,
    wire_experts)`` with the same semantics the torch path in
    ``_build_send_buffers`` produces, except slot order is ascending expert id
    rather than torch.topk's tie order (equivalent: probabilities and expert
    ids ride in the same slots).
    """
    num_tokens, num_experts = routing_map.shape
    device = routing_map.device
    indices = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    slot_wire_rows = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    token_wire_rows = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    wire_tokens = torch.empty((max_wire_rows,), dtype=torch.long, device=device)
    # Inactive slots of a wire row must read -1 (the pack kernel masks
    # probabilities on them); the kernel writes active slots only.
    wire_experts = torch.full((max_wire_rows, topk), -1, dtype=torch.long, device=device)
    if num_tokens:
        _build_wire_send_index_kernel[(num_tokens,)](
            routing_map.contiguous().view(torch.uint8),
            order,
            base,
            indices,
            slot_wire_rows,
            token_wire_rows,
            wire_tokens,
            wire_experts,
            num_experts=num_experts,
            experts_pow2=triton.next_power_of_2(num_experts),
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            topk=topk,
        )
    return indices, slot_wire_rows, token_wire_rows, wire_tokens, wire_experts


def build_wire_recv_index(
    received_experts: torch.Tensor,
    tokens_per_local_expert: torch.Tensor,
    num_assignments: int,
):
    """Build the receive-side expert-major indices with one kernel launch.

    Returns ``(wire_to_expert_rows, expert_wire_rows)``: the former maps each
    received (wire row, slot) to its expert-major row (-1 inactive), the latter
    maps each expert-major row back to its wire row -- exactly what the torch
    membership-compaction path in ``_expand_received`` produced.
    """
    num_rows, topk = received_experts.shape
    device = received_experts.device
    flat_experts = received_experts.reshape(-1)
    num_local_experts = tokens_per_local_expert.shape[0]
    membership = (
        torch.arange(num_local_experts, device=device).unsqueeze(1)
        == flat_experts.unsqueeze(0)
    )
    order = membership.cumsum(1)
    base = tokens_per_local_expert.cumsum(0) - tokens_per_local_expert
    wire_to_expert_rows = torch.empty((num_rows, topk), dtype=torch.long, device=device)
    expert_wire_rows = torch.empty((num_assignments,), dtype=torch.long, device=device)
    if num_rows:
        _build_wire_recv_index_kernel[(num_rows,)](
            received_experts.contiguous(),
            order,
            base,
            wire_to_expert_rows,
            expert_wire_rows,
            flat_experts.numel(),
            topk=topk,
            topk_pow2=triton.next_power_of_2(topk),
        )
    return wire_to_expert_rows, expert_wire_rows
