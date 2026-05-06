# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE


try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import torch

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""

        # x -> [num_tokens, hidden_dim]
        # token_indices -> [num_tokens, topk]
        # token_probs -> [num_tokens, topk]
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # recv_x -> [num_recv_tokens, hidden_dim]
        # recv_token_indices -> [num_recv_tokens, topk]
        # recv_token_probs -> [num_recv_tokens, topk]
        # num_recv_tokens_per_expert_list -> [num_experts]
        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    import mori
    import mori.ops
    import mori.shmem

    HAVE_MORI = True
except ImportError:
    HAVE_MORI = False


_mori_op = None

# Process-wide CUDA stream dedicated to MORI dispatch/combine kernel launches.
# Mirrors DeepEP's `allocate_on_comm_stream` pattern so that op.dispatch() and
# op.combine() can run concurrently with non-dependent work on the default
# (compute) stream when the caller passes async_finish=True. The host wait at
# `recv_num_token[0].item()` inside `MoriDispatch.forward` still blocks until
# the dispatch kernel drains, so the practical wall-time win from this alone
# is limited until that .item() is deferred — see TODO in MoriDispatch.forward.
_mori_comm_stream = None


def _get_mori_comm_stream():
    """Return a process-wide CUDA stream dedicated to MORI launches.

    Lazily allocated on the active CUDA device; cached for the lifetime of
    the process. Priority is set to high (-1) so MORI's communication kernels
    aren't preempted by lower-priority compute work on the default stream.
    """
    global _mori_comm_stream
    if _mori_comm_stream is None and torch.cuda.is_available():
        _mori_comm_stream = torch.cuda.Stream(
            device=torch.cuda.current_device(), priority=-1
        )
    return _mori_comm_stream


def _run_mori_op_on_stream(fn, async_finish: bool, allocate_on_comm_stream: bool):
    """Execute a MORI op (dispatch/combine) on the comm stream when requested.

    Honors the (async_finish, allocate_on_comm_stream) contract from
    `_MoriManager.dispatch` / `combine` and matches the DeepEP-side semantics:
    when both flags are True we move the launch to the dedicated comm stream,
    bracket it with the appropriate `wait_stream()` calls so the comm stream
    sees the producer of `fn`'s inputs and the compute stream sees the
    op's outputs, then return whatever `fn()` returned. Otherwise we run
    `fn()` synchronously on the current stream (legacy behavior).
    """
    if not (async_finish and allocate_on_comm_stream and torch.cuda.is_available()):
        return fn()
    comm_stream = _get_mori_comm_stream()
    if comm_stream is None:
        return fn()
    current_stream = torch.cuda.current_stream()
    # Comm stream must wait for inputs (e.g., the contiguous() copy of x and
    # the prior layer's writes to token_indices/probs) to be visible.
    comm_stream.wait_stream(current_stream)
    with torch.cuda.stream(comm_stream):
        result = fn()
    # Compute stream must wait for the dispatch/combine output before any
    # downstream op consumes it (slicing, .item(), bincount, ...).
    current_stream.wait_stream(comm_stream)
    return result




def get_mori_op(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    num_local_experts: int,
    router_topk: int,
    max_num_tokens_per_rank: int,
    data_type: torch.dtype = torch.bfloat16,
    kernel_type=None,
    fp8_dispatch: bool = False,
    use_standard_api: bool = False,
):
    """Get or create the MORI EpDispatchCombineOp.

    Lazily creates the operator on first call. Subsequent calls return the
    cached operator.

    Args:
        group: Process group for EP communication.
        hidden_dim: Hidden dimension of token embeddings.
        num_local_experts: Number of experts per rank.
        router_topk: Number of experts selected per token.
        max_num_tokens_per_rank: Maximum input tokens per rank.
        data_type: Token data type.
        kernel_type: MORI kernel type. Auto-selected if None.
        fp8_dispatch: Whether dispatch uses FP8.
        use_standard_api: When True, downstream callers will use MORI's
            standard MoE APIs (dispatch_standard_moe / combine_standard_moe).
            Those kernels are only implemented for IntraNode and
            InterNodeV1LL, so kernel-type auto-selection is constrained
            accordingly when this flag is set.
    """
    global _mori_op

    if _mori_op is not None:
        return _mori_op

    world_size = group.size()
    rank = torch.distributed.get_rank(group)

    if kernel_type is None:
        if world_size <= 8:
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        elif use_standard_api:
            # dispatch_standard_moe / combine_standard_moe only support
            # IntraNode + InterNodeV1LL (see mori/ops/dispatch_combine.py
            # `dispatch_standard_moe only supports IntraNode/InterNodeV1LL`).
            # Force InterNodeV1LL for multi-node std-API runs.
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        else:
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1

    if use_standard_api:
        # Hard constraint, not a recommendation: MORI only compiles the
        # `_stdmoe` kernel variants for IntraNode and InterNodeV1LL (gated
        # on ENABLE_STANDARD_MOE_ADAPT=ON). Any other kernel_type fails at
        # the C++ launch with `ValueError: dispatch_standard_moe only
        # supports IntraNode/InterNodeV1LL` (mori/ops/dispatch_combine.py
        # raises this in both dispatch_standard_moe and
        # combine_standard_moe). Fail fast here with a Megatron-side
        # message instead of letting the launch fail mid-iteration.
        _STD_API_KERNELS = (
            mori.ops.EpDispatchCombineKernelType.IntraNode,
            mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
        )
        if kernel_type not in _STD_API_KERNELS:
            raise ValueError(
                "use_standard_api=True requires kernel_type in "
                "(IntraNode, InterNodeV1LL); got "
                f"{getattr(kernel_type, 'name', kernel_type)}. "
                "MORI's standard MoE adaptor only compiles _stdmoe kernel "
                "variants for those two transports. Either drop "
                "--moe-mori-kernel-type to use Megatron's auto-selection "
                "(IntraNode for world_size<=8, InterNodeV1LL otherwise), "
                "or pick one of IntraNode / InterNodeV1LL explicitly."
            )

    dispatch_dtype = torch.float8_e4m3fnuz if fp8_dispatch else data_type
    scale_dim = hidden_dim // 128 if fp8_dispatch else 0

    config = mori.ops.EpDispatchCombineConfig(
        data_type=dispatch_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
        max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
        max_num_inp_token_per_rank=max_num_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        kernel_type=kernel_type,
    )

    _mori_op = mori.ops.EpDispatchCombineOp(config)
    return _mori_op


def reset_mori_op():
    """Reset the MORI operator state between iterations."""
    global _mori_op
    if _mori_op is not None:
        _mori_op.reset()


class MoriDispatch(torch.autograd.Function):
    """Fused dispatch using MORI EP backend.

    Performs inter-rank all-to-all via op.dispatch(). Returns the received
    tokens, their routing indices, probs, and per-expert token counts.
    The local permutation (grouping by expert) is handled separately in
    the token dispatcher layer.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Forward pass: dispatch tokens to correct ranks via MORI."""
        hidden_dim = x.shape[1]
        op = get_mori_op(
            group=group,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            max_num_tokens_per_rank=max_num_tokens_per_rank,
            data_type=x.dtype,
            fp8_dispatch=fp8_dispatch,
        )

        # x -> [num_tokens, hidden_dim]
        # token_probs -> [num_tokens, router_topk]
        # token_indices -> [num_tokens, router_topk]
        # scales=None: BF16 path uses scale_dim=0, so MORI ignores this arg
        # (see dispatch_combine.py:477-479). For an FP8 dispatch we'd need a
        # real [num_tokens, scale_dim] tensor in float8_e4m3fnuz.
        # When async_finish=True the launch is moved onto a dedicated comm
        # stream so it can overlap with non-dependent compute. The .item()
        # below still serializes on the host, however — eliminating that is
        # tracked separately (would require returning the un-sliced full
        # buffer + on-device row_valid mask to downstream consumers).
        dispatch_out, dispatch_weights, dispatch_scales, dispatch_indices, recv_num_token = (
            _run_mori_op_on_stream(
                lambda: op.dispatch(
                    x, token_probs.float(), None, token_indices.to(torch.int32)
                ),
                async_finish,
                allocate_on_comm_stream,
            )
        )

        # TODO(mori-overlap): defer this .item() out of the autograd Function
        # so dispatch can actually overlap with compute. See comment above and
        # the comm-stream notes in docs/design/mori_ep_integration.md.
        total_recv = recv_num_token[0].item()
        recv_x = dispatch_out[:total_recv]
        recv_token_indices_global = dispatch_indices[:total_recv]
        recv_token_probs = dispatch_weights[:total_recv]

        # MORI's dispatch returns the original GLOBAL expert ids in [0, num_experts)
        # for every topk slot of every received token, with no -1 sentinel for
        # non-local slots. Megatron's downstream pipeline (specifically
        # `_MoriManager._indices_to_multihot`) follows the DeepEP contract, which
        # expects values in the LOCAL expert space [0, num_local_experts) with
        # `-1` marking non-local slots. Without the rebase below, the advanced-
        # indexing assignment in `_indices_to_multihot` writes out of bounds and
        # triggers `HSA_STATUS_ERROR_EXCEPTION 0x1016` from the underlying
        # `at::native::index_put_kernel`.
        my_rank = torch.distributed.get_rank(group)
        local_id_start = my_rank * num_local_experts
        local_id_end = local_id_start + num_local_experts
        is_local = (recv_token_indices_global >= local_id_start) & (
            recv_token_indices_global < local_id_end
        )
        recv_token_indices = (recv_token_indices_global - local_id_start).to(torch.int64)
        recv_token_indices = torch.where(
            is_local,
            recv_token_indices,
            torch.full_like(recv_token_indices, -1),
        )

        # Per-local-expert token counts (matches DeepEP's
        # num_recv_tokens_per_expert_list contract). Filter -1 sentinels via the
        # `is_local` mask before bincount so we don't accidentally count global
        # expert ids that fall in [0, num_local_experts) but belong to other ranks.
        local_ids_flat = recv_token_indices[is_local]
        tokens_per_expert = torch.bincount(
            local_ids_flat, minlength=num_local_experts
        )[:num_local_experts]

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        # Stash comm-stream flags for backward — backward can't receive new
        # arguments via apply(), so we have to carry them through ctx.
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        # Save the RECEIVER-side topk probs (not sender-side `token_probs`) for
        # backward. The dispatch-backward calls `op.combine`, whose IntraNode
        # kernel reads `weights` indexed by totalRecvTokenNum (~43k) — see
        # intranode.hpp:281-286. Passing sender-side probs of shape [N, topk]
        # would re-introduce the same ~1.1 MiB OOB read that caused the
        # forward-side `Memory access fault by GPU`, just inside backward.
        ctx.save_for_backward(token_indices, recv_token_probs)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, None)

    @staticmethod
    def backward(ctx, grad_output, grad_indices, grad_probs, grad_tpe, grad_handle):
        """Backward pass: combine gradients back using MORI."""
        token_indices, recv_token_probs = ctx.saved_tensors
        op = get_mori_op(
            group=ctx.group,
            hidden_dim=grad_output.shape[1],
            num_local_experts=ctx.num_local_experts,
            router_topk=ctx.router_topk,
            max_num_tokens_per_rank=ctx.max_num_tokens_per_rank,
            data_type=grad_output.dtype,
            fp8_dispatch=ctx.fp8_dispatch,
        )
        num_tokens = token_indices.shape[0]
        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine(
                grad_output.contiguous(),
                recv_token_probs.float(),
                token_indices.to(torch.int32),
                call_reset=True,
            ),
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )
        # See MoriCombine.forward: op.combine() returns the full
        # [max_num_inp_token_per_rank, hidden_dim] buffer; slice to the
        # sender-side row count so the gradient matches `x`'s shape.
        combined_x = combined_x[:num_tokens]
        return (
            combined_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MoriCombine(torch.autograd.Function):
    """Fused combine operation using MORI EP backend.

    Performs the reverse all-to-all to send expert outputs back to
    their original ranks via op.combine().
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        token_indices,
        token_probs,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Forward pass: combine expert outputs back to original ranks via MORI."""
        op = get_mori_op(
            group=group,
            hidden_dim=x.shape[1],
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            max_num_tokens_per_rank=max_num_tokens_per_rank,
            data_type=x.dtype,
            fp8_dispatch=fp8_dispatch,
        )
        # Sender-side num_tokens — needed both as ctx for backward and to
        # slice op.combine()'s output (which is shaped to the full
        # [max_num_inp_token_per_rank, hidden_dim] capacity buffer).
        num_tokens = token_indices.shape[0]

        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine(
                x.contiguous(),
                token_probs.float(),
                token_indices.to(torch.int32),
                call_reset=True,
            ),
            async_finish,
            allocate_on_comm_stream,
        )
        # op.combine() returns the full [max_num_inp_token_per_rank, hidden_dim]
        # buffer view. Slice down to the actual sender-side row count so
        # combine_postprocess()'s view(self.hidden_shape) is correct whenever
        # max_num_inp_token_per_rank != num_tokens.
        combined_x = combined_x[:num_tokens]

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        # Stash comm-stream flags for backward — backward can't receive new
        # arguments via apply(), so we have to carry them through ctx.
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        # Stash the receiver-side row count for backward. By the time we get
        # here, x.shape[0] is the same `total_recv` that MoriDispatch.forward
        # already paid the .item() sync for — it's the row count of `recv_x`
        # that propagated through unpermute → expert outputs → here. Saving
        # it on ctx lets backward skip the redundant
        # `recv_num_token[0].item()` host-sync after the grad dispatch.
        # Since routing is deterministic given the same indices, this is the
        # exact total_recv backward's op.dispatch will produce.
        ctx.total_recv = x.shape[0]
        ctx.save_for_backward(token_indices, token_probs)
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: dispatch gradients using MORI."""
        token_indices, token_probs = ctx.saved_tensors
        op = get_mori_op(
            group=ctx.group,
            hidden_dim=grad_output.shape[1],
            num_local_experts=ctx.num_local_experts,
            router_topk=ctx.router_topk,
            max_num_tokens_per_rank=ctx.max_num_tokens_per_rank,
            data_type=grad_output.dtype,
            fp8_dispatch=ctx.fp8_dispatch,
        )
        dispatch_out, _, _, _, _ = _run_mori_op_on_stream(
            lambda: op.dispatch(
                grad_output.contiguous(),
                token_probs.float(),
                None,
                token_indices.to(torch.int32),
            ),
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )
        # Reuse the total_recv saved in forward instead of calling
        # `recv_num_token[0].item()` here — that .item() forces a host sync
        # waiting on the dispatch kernel and was the ~1ms-per-layer cost
        # diagnosed in the MoRI EP profiling notes.
        return (
            dispatch_out[: ctx.total_recv],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MoriDispatchStandard(torch.autograd.Function):
    """Fused dispatch using MORI's standard MoE API (3-D pre-binned output).

    Calls `op.dispatch_standard_moe`, which performs the all-to-all and the
    expert-bin packing in a single kernel launch. Returns:

    - `packed_recv_x` of shape `[num_local_experts, max_tokens_per_expert,
      hidden_dim]`, with rows `[e, :counts[e], :]` containing the actual
      tokens routed to local expert `e`. Slots beyond `counts[e]` are
      undefined and must be masked out by the consumer.
    - `packed_recv_count` of shape `[num_local_experts]`, dtype `int32`,
      containing per-expert receive counts. This is a non-owning view into
      MORI's internal state and is cloned here so it survives across
      iterations.

    Routing weights are NOT applied to data inside this kernel. They are
    applied at combine time by `op.combine_standard_moe`. The forward
    therefore preserves the same token-wise math as the raw 2-D path —
    `output_token = sum_e prob_{token,e} * expert_e(x_token)` — by
    deferring the weight multiplication to combine. Consumers feeding the
    expert MLP should pass `permuted_probs = ones_like(...)` (or skip the
    output-weighting step) so weights are not applied twice.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Forward pass: dispatch tokens via MORI's standard 3-D API."""
        hidden_dim = x.shape[1]
        op = get_mori_op(
            group=group,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            max_num_tokens_per_rank=max_num_tokens_per_rank,
            data_type=x.dtype,
            fp8_dispatch=fp8_dispatch,
            use_standard_api=True,
        )

        # `dispatch_standard_moe` requires MORI built with
        # ENABLE_STANDARD_MOE_ADAPT=ON. Surface a clearer error than the
        # bare `RuntimeError("dispatch_standard_moe is not available...")`
        # raised by the Python wrapper.
        if not hasattr(op, "dispatch_standard_moe"):
            raise RuntimeError(
                "MORI's dispatch_standard_moe is unavailable on this build. "
                "Rebuild MORI with ENABLE_STANDARD_MOE_ADAPT=ON, or disable "
                "--moe-mori-use-standard-api."
            )

        packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range = (
            _run_mori_op_on_stream(
                lambda: op.dispatch_standard_moe(
                    x, token_probs.float(), None, token_indices.to(torch.int32)
                ),
                async_finish,
                allocate_on_comm_stream,
            )
        )

        # `packed_recv_count` is a non-owning view into MORI's internal
        # symmetric heap (see `from_gpu_ptr` in
        # mori/ops/dispatch_combine.py:get_standard_moe_packed_recv_count_ptr).
        # The next iteration's reset will clobber it; clone so the value
        # survives across the boundary that downstream Python code crosses.
        packed_recv_count = packed_recv_count.clone()

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        # Save sender-side indices/probs for backward; these are what
        # combine_standard_moe needs to reverse the routing.
        ctx.save_for_backward(token_indices, token_probs)

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    @staticmethod
    def backward(
        ctx,
        grad_packed_x,
        grad_count,
        grad_src_info,
        grad_layout_range,
    ):
        """Backward pass: route grads back via combine_standard_moe."""
        token_indices, token_probs = ctx.saved_tensors
        op = get_mori_op(
            group=ctx.group,
            hidden_dim=grad_packed_x.shape[2],
            num_local_experts=ctx.num_local_experts,
            router_topk=ctx.router_topk,
            max_num_tokens_per_rank=ctx.max_num_tokens_per_rank,
            data_type=grad_packed_x.dtype,
            fp8_dispatch=ctx.fp8_dispatch,
            use_standard_api=True,
        )
        num_tokens = token_indices.shape[0]
        # combine_standard_moe applies `weights` to data while reducing.
        # Pass token_probs to mirror the existing raw-path backward, which
        # invokes `op.combine(grad_output, weights=recv_token_probs, ...)`
        # — semantics: gradient of weighted sum-combine is the dispatch of
        # weighted grads back to the source position.
        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine_standard_moe(
                grad_packed_x.contiguous(),
                token_probs.float(),
                token_indices.to(torch.int32),
                call_reset=True,
            ),
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )
        # combine_standard_moe returns a non-owning view sized to
        # [max_num_inp_token_per_rank, hidden_dim]; slice + clone so the
        # gradient survives MORI's reset and matches `x`'s shape.
        combined_x = combined_x[:num_tokens].clone()
        return (
            combined_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MoriCombineStandard(torch.autograd.Function):
    """Fused combine using MORI's standard MoE API (3-D input layout).

    Inverse of `MoriDispatchStandard`. Accepts a 3-D
    `[num_local_experts, max_tokens_per_expert, hidden_dim]` buffer where
    rows `[e, :counts[e], :]` are the expert outputs for local expert `e`,
    and routes them back to the original sender ranks. Routing weights
    `token_probs` (sender-side `[num_tokens, topk]`) are applied by the
    kernel as part of the weighted reduction.
    """

    @staticmethod
    def forward(
        ctx,
        x,  # [num_local_experts, max_tokens_per_expert, hidden_dim]
        group,
        token_indices,
        token_probs,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Forward pass: combine 3-D expert outputs back to sender ranks."""
        hidden_dim = x.shape[2]
        op = get_mori_op(
            group=group,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            max_num_tokens_per_rank=max_num_tokens_per_rank,
            data_type=x.dtype,
            fp8_dispatch=fp8_dispatch,
            use_standard_api=True,
        )

        if not hasattr(op, "combine_standard_moe"):
            raise RuntimeError(
                "MORI's combine_standard_moe is unavailable on this build. "
                "Rebuild MORI with ENABLE_STANDARD_MOE_ADAPT=ON, or disable "
                "--moe-mori-use-standard-api."
            )

        num_tokens = token_indices.shape[0]
        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine_standard_moe(
                x.contiguous(),
                token_probs.float(),
                token_indices.to(torch.int32),
                call_reset=True,
            ),
            async_finish,
            allocate_on_comm_stream,
        )
        # See MoriDispatchStandard.backward: output is a non-owning view
        # sized to the full max_num_inp_token_per_rank capacity. Slice and
        # clone so combine_postprocess()'s `view(self.hidden_shape)` is
        # correct and the buffer survives MORI's reset.
        combined_x = combined_x[:num_tokens].clone()

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.save_for_backward(token_indices, token_probs)
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: dispatch grads via dispatch_standard_moe (3-D)."""
        token_indices, token_probs = ctx.saved_tensors
        op = get_mori_op(
            group=ctx.group,
            hidden_dim=grad_output.shape[1],
            num_local_experts=ctx.num_local_experts,
            router_topk=ctx.router_topk,
            max_num_tokens_per_rank=ctx.max_num_tokens_per_rank,
            data_type=grad_output.dtype,
            fp8_dispatch=ctx.fp8_dispatch,
            use_standard_api=True,
        )
        packed_grad_x, _, _, _ = _run_mori_op_on_stream(
            lambda: op.dispatch_standard_moe(
                grad_output.contiguous(),
                token_probs.float(),
                None,
                token_indices.to(torch.int32),
            ),
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )
        return (
            packed_grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


if HAVE_MORI:

    def mori_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Perform fused dispatch using MORI EP backend.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Total number of experts
            group: Process group
            num_local_experts: Experts per rank
            router_topk: Top-K experts per token
            max_num_tokens_per_rank: Max tokens per rank for buffer sizing
            fp8_dispatch: Whether to use FP8 dispatch
            async_finish: When True (and allocate_on_comm_stream=True),
                MORI's op.dispatch is launched on a dedicated CUDA comm
                stream so it can overlap with non-dependent compute on the
                default stream. Mirrors DeepEP's flag of the same name.
            allocate_on_comm_stream: See `async_finish`.

        Returns:
            Tuple of (recv_x, recv_indices, recv_probs, tokens_per_expert, handle)
        """
        return MoriDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            num_local_experts,
            router_topk,
            max_num_tokens_per_rank,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
        )

    def mori_combine(
        x,
        group,
        token_indices,
        token_probs,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Perform fused combine using MORI EP backend.

        Args:
            x: Input tensor from expert computation
            group: Process group
            token_indices: Original token routing indices
            token_probs: Original token routing probabilities
            num_local_experts: Experts per rank
            router_topk: Top-K experts per token
            max_num_tokens_per_rank: Max tokens per rank for buffer sizing
            fp8_dispatch: Whether to use FP8 dispatch
            async_finish: When True (and allocate_on_comm_stream=True),
                MORI's op.combine is launched on a dedicated CUDA comm
                stream so it can overlap with non-dependent compute on the
                default stream. Mirrors DeepEP's flag of the same name.
            allocate_on_comm_stream: See `async_finish`.

        Returns:
            Combined output tensor
        """
        return MoriCombine.apply(
            x,
            group,
            token_indices,
            token_probs,
            num_local_experts,
            router_topk,
            max_num_tokens_per_rank,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
        )

    def mori_dispatch_standard(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Standard-API dispatch: returns 3-D pre-binned packed_recv_x.

        Wraps `MoriDispatchStandard.apply`. The dispatcher consumes
        `packed_recv_x [num_local_experts, max_tokens_per_expert, hidden_dim]`
        + `packed_recv_count [num_local_experts]` directly, replacing the
        Python-side `_indices_to_multihot` + `permute` chain with a single
        mask-and-gather. See `MoriDispatchStandard` for the trade-off.

        Returns:
            Tuple of (packed_recv_x, packed_recv_count, packed_recv_src_info,
            packed_recv_layout_range).
        """
        return MoriDispatchStandard.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            num_local_experts,
            router_topk,
            max_num_tokens_per_rank,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
        )

    def mori_combine_standard(
        x,  # [num_local_experts, max_tokens_per_expert, hidden_dim]
        group,
        token_indices,
        token_probs,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Standard-API combine: takes 3-D expert outputs, returns 2-D.

        Wraps `MoriCombineStandard.apply`. Routing weights `token_probs`
        (sender-side) are applied by the kernel during the weighted
        reduction, so the upstream expert MLP should NOT pre-multiply
        outputs by per-token-per-expert probs.
        """
        return MoriCombineStandard.apply(
            x,
            group,
            token_indices,
            token_probs,
            num_local_experts,
            router_topk,
            max_num_tokens_per_rank,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
        )

else:
    mori_dispatch = None
    mori_combine = None
    mori_dispatch_standard = None
    mori_combine_standard = None