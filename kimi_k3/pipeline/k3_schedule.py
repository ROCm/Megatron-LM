"""Bind the AttnRes payload shapes into core's 1F1B schedule.

Core computes one recv-shape list and one send-shape list per rank and then hands
them to an optional ``adjust_tensor_shapes_fn(recv, send)``
(megatron/core/pipeline_parallel/schedules.py:2019, 2180-2183). Because it runs
per rank, a stage may legally declare a different recv shape than its send shape
— exactly what a monotonically growing ``block_residual`` needs.

So K3 does **not** re-implement a schedule. It supplies that function and binds
it. Two ways to bind:

* pass ``adjust_tensor_shapes_fn=`` directly to the callable returned by
  ``get_forward_backward_func()`` — what our own harnesses do;
* :func:`install`, for the stock ``megatron.training`` loop, whose ``train_step``
  only ever passes the hook for NVIDIA-modelopt distillation
  (training.py:1815-1822, 1853).

**The binding must be conditional on PP > 1.** Both the no-pipelining schedule
(``schedules.py:631``) and the interleaved/VPP schedule (``:949``) assert the hook
is ``None``, which is also why VPP stays descoped for AttnRes.
"""

import contextlib
from typing import Callable, List, Optional, Tuple

Shape = Tuple[int, ...]


def _scale_seq(shape: Shape, multiplier: int) -> Shape:
    return (shape[0] * multiplier,) + tuple(shape[1:])


def make_adjust_tensor_shapes_fn(recv_multiplier: int, send_multiplier: int) -> Callable:
    """Build the hook for a stage that receives ``1+K_in`` and sends ``1+K_out``."""

    def adjust(recv_shapes: List[Shape], send_shapes: List[Shape]):
        return (
            [_scale_seq(s, recv_multiplier) for s in recv_shapes],
            [_scale_seq(s, send_multiplier) for s in send_shapes],
        )

    return adjust


def adjust_fn_for_block(block) -> Optional[Callable]:
    """Hook for a `K3TransformerBlock`, or None when there is nothing to adjust."""
    recv_mult, send_mult = block.payload_multipliers()
    if recv_mult == 1 and send_mult == 1:
        return None
    return make_adjust_tensor_shapes_fn(recv_mult, send_mult)


def resolve(model, pipeline_model_parallel_size: int) -> Optional[Callable]:
    """The hook this rank should use, or None when the schedule forbids one.

    Returns None for PP == 1 (`schedules.py:631` asserts the hook is None there),
    so a single-GPU run and a pipelined run can share one call site.
    """
    if pipeline_model_parallel_size <= 1:
        return None
    block = getattr(model, "decoder", model)
    return adjust_fn_for_block(block)


@contextlib.contextmanager
def install(model, pipeline_model_parallel_size: int):
    """Make the stock training loop pass our hook.

    ``megatron.training.training`` imports ``get_forward_backward_func`` into its
    own namespace, so rebinding that name there is enough — no file under
    ``megatron/**`` is modified (rule R2.2). Scoped, and asserted by the pin
    contracts.
    """
    import functools

    import megatron.training.training as training

    adjust = resolve(model, pipeline_model_parallel_size)
    if adjust is None:
        yield
        return

    original = training.get_forward_backward_func

    def patched():
        fn = original()
        return functools.partial(fn, adjust_tensor_shapes_fn=adjust)

    training.get_forward_backward_func = patched
    try:
        yield
    finally:
        training.get_forward_backward_func = original
