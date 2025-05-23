import contextlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Tuple, Union

import torch
from torch import Tensor
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel

from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, make_viewless_tensor


# Types
Shape = Union[List[int], torch.Size]


def make_viewless(e):
    """make_viewless util func"""
    e = make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)
    return e


@contextmanager
def stream_acquire_context(stream, event):
    event.wait(stream)
    try:
        yield
    finally:
        event.record(stream)


class ScheduleNode:
    """base node for fine-grained schedule"""

    def __init__(
        self,
        forward_func,
        stream,
        event,
        backward_func=None,
        free_inputs=False,
        name="schedule_node",
    ):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.stream = stream
        self.event = event
        self.free_inputs = free_inputs
        self.inputs = None
        self.outputs = None

    def default_backward_func(self, outputs, output_grad):
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        return output_grad

    def forward(self, inputs=()):
        """schedule node forward"""

        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} forward")
            with torch.cuda.stream(self.stream):
                self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
                for i, input in enumerate(self.inputs):
                    if input is not None:
                        input.requires_grad = inputs[i].requires_grad

                data = tuple(self.inputs)
                data = self.forward_func(*data)

                if not isinstance(data, tuple):
                    data = make_viewless(data)
                else:
                    data = tuple([make_viewless(e) if isinstance(e, Tensor) else e for e in data])

                self.output = data
            torch.cuda.nvtx.range_pop()

        if self.free_inputs:
            for input in inputs:
                input.record_stream(self.stream)
                input.untyped_storage().resize_(0)

        return self.output

    def get_output(self):
        """get the forward output"""
        return self.output

    def backward(self, output_grad):
        """schedule node backward"""
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} backward")
            with torch.cuda.stream(self.stream):
                outputs = self.output
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                assert len(outputs) == len(
                    output_grad
                ), f"{len(outputs)} of {type(outputs[0])} vs {len(output_grad)} of {type(output_grad[0])}"
                if self.backward_func is not None:
                    output_grad = self.backward_func(outputs, output_grad)
                else:
                    output_grad = self.default_backward_func(outputs, output_grad)
            torch.cuda.nvtx.range_pop()

        # output_grad maybe from another stream
        for g in output_grad:
            g.record_stream(self.stream)

        return self.get_grad()

    def get_grad(self):
        """get the grad of inputs"""
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # clear state
        self.inputs = None
        self.output = None
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad


class AbstractSchedulePlan(ABC):
    """to use combined 1f1b, model must implement build_schedule_plan while take the same
    signature as model forward but return an instance of AbstractSchedulePlan"""

    @classmethod
    @abstractmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """forward_backward is the protocol between our schedule logic and model"""
        ...


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    """model level 1f1b fine-grained schedule"""
    return type(f_schedule_plan or b_schedule_plan).forward_backward(
        f_schedule_plan,
        b_schedule_plan,
        grad=grad,
        f_context=f_context,
        b_context=b_context,
        pre_forward=pre_forward,
        pre_backward=pre_backward,
        post_forward=post_forward,
        post_backward=post_backward,
    )


def schedule_chunk_forward(schedule_plan):
    """model level fine-grained forward schedule"""
    f_input = schedule_chunk_1f1b(schedule_plan, None, None)
    return f_input


def schedule_chunk_backward(schedule_plan, grad):
    """model level fine-grained backward schedule"""
    tmp = schedule_chunk_1f1b(None, schedule_plan, grad)


_COMP_STREAM = None
_COM_STREAM = None


def set_streams(comp_stream=None, com_stream=None):
    """set the streams for communication and computation"""
    global _COMP_STREAM
    global _COM_STREAM
    if _COMP_STREAM is not None:
        return

    if comp_stream is None:
        comp_stream = torch.cuda.Stream(device="cuda")
    if com_stream is None:
        com_stream = torch.cuda.Stream(device="cuda")

    assert _COMP_STREAM is None
    assert _COM_STREAM is None
    _COMP_STREAM = comp_stream
    _COM_STREAM = com_stream


def get_comp_stream():
    """get the stream for computation"""
    global _COMP_STREAM
    return _COMP_STREAM


def get_com_stream():
    """get the stream for communication"""
    global _COM_STREAM
    return _COM_STREAM


class VppContextManager:
    """a reusable context manager for switch vpp stage"""

    def __init__(self, vpp_rank):
        self.vpp_rank = vpp_rank

    def __enter__(self):
        self.origin_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.vpp_rank)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.origin_vpp_rank)


def forward_backward_step(
    forward_step_func,
    data_iterator,
    f_model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    b_model,
    b_input_tensor,
    b_output_tensor,
    b_output_tensor_grad,
    config,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    from .schedules import set_current_microbatch

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    # forward preprocess
    unwrap_output_tensor = False
    if f_model is not None:
        with f_context:
            if is_first_microbatch and hasattr(f_model, 'set_is_first_microbatch'):
                f_model.set_is_first_microbatch()
            if current_microbatch is not None:
                set_current_microbatch(f_model, current_microbatch)
            if not isinstance(input_tensor, list):
                input_tensor = [input_tensor]
                unwrap_output_tensor = True

            set_input_tensor = get_attr_wrapped_model(f_model, "set_input_tensor")
            set_input_tensor(input_tensor)

            with context_manager:
                if checkpoint_activations_microbatch is None:
                    output_tensor, loss_func = forward_step_func(data_iterator, f_model)
                else:
                    output_tensor, loss_func = forward_step_func(
                        data_iterator, f_model, checkpoint_activations_microbatch
                    )
                assert isinstance(
                    output_tensor, AbstractSchedulePlan
                ), "first output of forward_step_func must be one instance of AbstractSchedulePlan"

    # backward preprocess
    unwrap_input_tensor_grad = False
    b_schedule_plan = None
    if b_model is not None:
        # Retain the grad on the input_tensor.
        if not isinstance(b_input_tensor, list):
            b_input_tensor = [b_input_tensor]
            unwrap_input_tensor_grad = True
        for x in b_input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(b_output_tensor, list):
            b_output_tensor = [b_output_tensor]
        if not isinstance(b_output_tensor_grad, list):
            b_output_tensor_grad = [b_output_tensor_grad]

        # Backward pass for loss function
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        if b_output_tensor_grad[0] is None and config.grad_scale_func is not None:
            # backward schedule plan
            loss_node = b_output_tensor[0].loss_func
            b_output_tensor[0].loss_func = None
            b_output_tensor[0] = config.grad_scale_func(b_output_tensor[0])
            torch.autograd.backward(b_output_tensor[0], grad_tensors=b_output_tensor_grad[0])
            b_output_tensor_grad[0] = loss_node.get_grad()

    f_schedule_plan = output_tensor if f_model else None
    grad = b_output_tensor_grad[0] if b_model else None
    with context_manager:
        # schedule forward and backward
        output_tensor = schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    # forward post process
    num_tokens = None
    if f_model:
        with f_context:
            num_tokens = torch.tensor(0, dtype=torch.int)
            if parallel_state.is_pipeline_last_stage():
                if not collect_non_loss_data:
                    loss_node = ScheduleNode(
                        loss_func,
                        torch.cuda.current_stream(),
                        f_schedule_plan.event,
                        name="loss_func",
                    )
                    loss_func = loss_node.forward
                    outputs = loss_func(output_tensor)
                    if len(outputs) == 3:
                        output_tensor, num_tokens, loss_reduced = outputs
                        if not config.calculate_per_token_loss:
                            output_tensor /= num_tokens
                            output_tensor /= num_microbatches
                    else:
                        # preserve legacy loss averaging behavior
                        # (ie, over the number of microbatches)
                        assert len(outputs) == 2
                        output_tensor, loss_reduced = outputs
                        output_tensor = output_tensor / num_microbatches
                    forward_data_store.append(loss_reduced)

                    # attach loss_func on output_tensor
                    output_tensor.loss_func = loss_node
                else:
                    data = loss_func(output_tensor, non_loss_data=True)
                    forward_data_store.append(data)
            # attach schedule plan on output tensor
            output_tensor.schedule_plan = f_schedule_plan
            if config.timers is not None:
                config.timers('forward-compute').stop()

            # Set the loss scale for the auxiliary loss of the MoE layer.
            # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
            # explicitly.
            if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
                # Calculate the loss scale based on the grad_scale_func if available,
                # else default to 1.
                loss_scale = (
                    config.grad_scale_func(torch.ones(1, device=output_tensor.device))
                    if config.grad_scale_func is not None
                    else torch.tensor(1.0)
                )
                # Set the loss scale
                MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

            if not unwrap_output_tensor:
                output_tensor, num_tokens = [output_tensor], num_tokens
    # backward post process
    input_tensor_grad = None
    if b_model is not None:
        input_tensor_grad = [None]
        if b_input_tensor is not None:
            input_tensor_grad = []
            for x in b_input_tensor:
                if x is None:
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)

        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]

    return output_tensor, num_tokens, input_tensor_grad

def get_default_cls_for_unwrap():
    cls = (DistributedDataParallel, Float16Module)
    try:
        # legacy should not be used in core, but for backward compatibility, we support it here
        from megatron.legacy.model import Float16Module as LegacyFloat16Module
        cls = cls + (LegacyFloat16Module,)
    except:
        pass
    return cls

def unwrap_model(model, module_instances=get_default_cls_for_unwrap()):
    """unwrap_model DistributedDataParallel and Float16Module wrapped model"""
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def wrap_forward_func(config, forward_step_func):
    """wrap the input to forward_step_func, to make forward_step_func return schedule plan"""

    def wrapped_func(data_iterator, model):
        return forward_step_func(data_iterator, unwrap_model(model).build_schedule_plan)

    if config.combined_1f1b and config.combined_1f1b_recipe == "ep_a2a":
        return wrapped_func
    else:
        return forward_step_func
