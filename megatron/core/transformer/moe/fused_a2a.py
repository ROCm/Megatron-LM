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
    """
    global _mori_op

    if _mori_op is not None:
        return _mori_op

    world_size = group.size()
    rank = torch.distributed.get_rank(group)

    if kernel_type is None:
        if world_size <= 8:
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        else:
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1

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

        scales = torch.empty(x.shape[0], 0, dtype=torch.float8_e4m3fnuz, device=x.device)

        dispatch_out, dispatch_weights, dispatch_scales, dispatch_indices, recv_num_token = (
            op.dispatch(x, token_probs.float(), scales, token_indices.to(torch.int32))
        )

        total_recv = recv_num_token[0].item()
        recv_x = dispatch_out[:total_recv]
        recv_token_indices = dispatch_indices[:total_recv]
        recv_token_probs = dispatch_weights[:total_recv]

        # Compute per-expert token counts
        all_expert_ids = recv_token_indices.flatten()
        valid = all_expert_ids >= 0
        tokens_per_expert = torch.bincount(
            all_expert_ids[valid], minlength=num_local_experts
        )[:num_local_experts]

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        ctx.save_for_backward(token_indices, token_probs)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, None)

    @staticmethod
    def backward(ctx, grad_output, grad_indices, grad_probs, grad_tpe, grad_handle):
        """Backward pass: combine gradients back using MORI."""
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
        combined_x, _ = op.combine(
            grad_output.contiguous(),
            token_probs.float(),
            token_indices.to(torch.int32),
            call_reset=True,
        )
        return combined_x, None, None, None, None, None, None, None, None


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
        combined_x, _ = op.combine(
            x.contiguous(),
            token_probs.float(),
            token_indices.to(torch.int32),
            call_reset=True,
        )
        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
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
        scales = torch.empty(
            grad_output.shape[0], 0, dtype=torch.float8_e4m3fnuz, device=grad_output.device
        )
        dispatch_out, _, _, _, recv_num_token = op.dispatch(
            grad_output.contiguous(),
            token_probs.float(),
            scales,
            token_indices.to(torch.int32),
        )
        total_recv = recv_num_token[0].item()
        return dispatch_out[:total_recv], None, None, None, None, None, None, None


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
        )

else:
    mori_dispatch = None
    mori_combine = None