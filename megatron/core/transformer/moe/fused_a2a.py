# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from megatron.core.utils import internal_api

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
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    '''
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


def reset_hybrid_ep_buffer():
    '''
    Reset the HybridEP buffer
    '''
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x, probs=grad_probs, handle=handle, pad_multiple=ctx.pad_multiple
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None


@internal_api
class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
        '''
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None



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


def _ensure_combine_weights_non_empty(
    weights: torch.Tensor, router_topk: int, device: torch.device
) -> torch.Tensor:
    """Workaround MORI's `weights.size(0) != 0` gate in op.combine().

    MORI's python wrapper sets ``weight_ptr=0`` (→ kernel's
    ``args.weightsBuf=nullptr``) when the passed weights tensor has
    ``size(0) == 0``. On ranks that received zero tokens during dispatch
    (highly imbalanced routing), the receiver-side probs slice is exactly
    that 0-row tensor — and the resulting null weightsBuf breaks the
    combine kernel's slot-1 contribution path for those ranks' senders,
    producing only PE 0's contribution instead of the sum across all
    unique destinations. Reproduced standalone in
    ``mori_zero_recv_repro.py``.

    This helper substitutes a 1-row dummy of zeros when the input is
    empty so ``weight_ptr`` stays non-null. The kernel only reads the
    first ``totalRecvTokenNum`` rows, so the dummy is never dereferenced
    on the data path.
    """
    if weights.size(0) > 0:
        return weights
    return torch.zeros(
        (1, router_topk), dtype=weights.dtype, device=device
    )


def get_mori_op(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    num_local_experts: int,
    router_topk: int,
    max_num_tokens_per_rank: int,
    data_type: torch.dtype = torch.bfloat16,
    kernel_type=None,
    fp8_dispatch: bool = False,
):
    """Return the process-wide :class:`EpDispatchCombineOp`, creating it once.

    Dispatch and combine in the same forward pass share this instance; do not
    call :func:`reset_mori_op` between dispatch and combine.

    Call :func:`reset_mori_op` when MoE layout or sizing changes (e.g. pytest
    teardown between parametrized ``(tp_size, ep_size)`` cases) so the next
    call builds a new op with the right ``num_experts_per_rank`` / buffers.

    Args:
        group: Process group for EP communication.
        hidden_dim: Hidden dimension of token embeddings.
        num_local_experts: Number of experts per rank.
        router_topk: Number of experts selected per token.
        max_num_tokens_per_rank: Maximum input tokens per rank.
        data_type: Token data type.
        kernel_type: MORI kernel type. Auto-selected if None.
        fp8_dispatch: Whether dispatch uses FP8.
    """
    global _mori_op

    if max_num_tokens_per_rank is None:
        raise ValueError(
            "max_num_tokens_per_rank must not be None for MORI EP. "
            "Set --moe-mori-max-tokens-per-rank (e.g. to micro_batch_size * seq_length)."
        )

    if _mori_op is not None:
        return _mori_op

    world_size = group.size()
    rank = torch.distributed.get_rank(group)

    resolved_kernel_type = kernel_type
    if resolved_kernel_type is None:
        if world_size <= 8:
            resolved_kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        else:
            resolved_kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1

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
        kernel_type=resolved_kernel_type,
    )

    _mori_op = mori.ops.EpDispatchCombineOp(config)
    return _mori_op


def reset_mori_op():
    """Clear the global MORI op so the next :func:`get_mori_op` allocates fresh.

    Calls :meth:`~mori.ops.EpDispatchCombineOp.reset` when present, then drops
    the reference. Use after tests or when changing EP/TP layout so config
    (e.g. ``num_experts_per_rank``) cannot mismatch the cached handle.
    """
    global _mori_op
    if _mori_op is not None:
        _mori_op.reset()
    _mori_op = None


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
        # Save the RECEIVER-side topk probs and the RECEIVER-side global indices
        # (not sender-side `token_indices`) for backward. The dispatch-backward
        # calls `op.combine`, whose IntraNode kernel expects the `indices`
        # argument to be the receiver-side global indices [total_recv, topk]
        # (matches MORI's own dispatch_combine_test_utils.py). Passing
        # sender-side indices [N, topk] makes the kernel only sum one source
        # PE's contribution rather than all unique destinations, producing
        # `0.5x` instead of `x` for topk=2 routing.
        ctx.save_for_backward(token_indices, recv_token_probs, recv_token_indices_global)

        # 5th return is recv_token_indices_global (receiver-side, GLOBAL expert
        # ids in [0, num_experts) without -1 sentinels). Downstream
        # `_MoriManager.combine` needs this to call `op.combine()` correctly —
        # MORI's combine kernel expects receiver-side indices, not the rebased
        # local `recv_token_indices` we send through the regular pipeline.
        return (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            recv_token_indices_global,
        )

    @staticmethod
    def backward(ctx, grad_output, grad_indices, grad_probs, grad_tpe, grad_handle):
        """Backward pass: combine gradients back using MORI."""
        token_indices, recv_token_probs, recv_token_indices_global = ctx.saved_tensors
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
        # MORI's op.combine() expects RECEIVER-side global indices — see
        # the comment in MoriDispatch.forward where we save them and the
        # matching pattern in dispatch_combine_test_utils.py:427 which
        # passes the receiver-side `dispatch_indices` to `op.combine()`.
        combine_weights = _ensure_combine_weights_non_empty(
            recv_token_probs.float(), ctx.router_topk, grad_output.device
        )
        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine(
                grad_output.contiguous(),
                combine_weights,
                recv_token_indices_global.to(torch.int32),
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
        recv_token_indices,
        token_probs,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
    ):
        """Forward pass: combine expert outputs back to original ranks via MORI.

        Args:
            x: Receiver-side hidden states [total_recv, hidden_dim].
            token_indices: SENDER-side topk indices [num_sender_tokens, topk].
                Used only to (a) slice op.combine()'s output to sender row
                count and (b) re-dispatch grads in backward.
            recv_token_indices: RECEIVER-side global topk indices
                [total_recv, topk]. This is what op.combine() expects (matches
                dispatch_combine_test_utils.py:427). Passing sender-side
                indices instead breaks the kernel's per-unique-PE summation
                and produces `0.5x` for topk=2 routing.
            token_probs: RECEIVER-side topk probs [total_recv, topk].
        """
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

        combine_weights = _ensure_combine_weights_non_empty(
            token_probs.float(), router_topk, x.device
        )
        combined_x, _ = _run_mori_op_on_stream(
            lambda: op.combine(
                x.contiguous(),
                combine_weights,
                recv_token_indices.to(torch.int32),
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
        # Return one None per non-tensor input arg (10 inputs after ctx:
        # x, group, token_indices, recv_token_indices, token_probs,
        # num_local_experts, router_topk, max_num_tokens_per_rank,
        # fp8_dispatch, async_finish, allocate_on_comm_stream).
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
        recv_token_indices,
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
            x: Input tensor from expert computation [total_recv, hidden_dim]
            group: Process group
            token_indices: SENDER-side topk indices [num_sender_tokens, topk].
                Used for output slicing and the backward dispatch.
            recv_token_indices: RECEIVER-side global topk indices
                [total_recv, topk]. This is what MORI's op.combine() expects
                — see MoriCombine.forward docstring for details.
            token_probs: RECEIVER-side topk probabilities [total_recv, topk].
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
            recv_token_indices,
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