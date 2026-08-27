"""KDA backend dispatch: the FP32 oracle, or fla's kernels.

Precedence and defaults are rule R8.1 / R5.3: **eager is the default** and stays
so until the fla parity gate (G15) is green at production shapes. Compiling is
not the same as being correct, and fla's KDA backward has a bug history
(#807, #785) plus, at our pin, only compiles at all on triton 3.7.1.

A `flydsl` slot is reserved and unimplemented in v1.
"""

from typing import Callable, Optional, Tuple

import torch

from .kda_eager_fp32 import chunk_kda_eager_fp32, released_call_kwargs

EAGER = "eager"
FLA = "fla"
BACKENDS = (EAGER, FLA)


def _fla_chunk_kda() -> Callable:
    from fla.ops.kda import chunk_kda  # deferred: fla is an optional dependency

    return chunk_kda


def fla_available() -> bool:
    try:
        _fla_chunk_kda()
        return True
    except Exception:
        return False


def kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    backend: str = EAGER,
    config=None,
    **overrides,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run KDA through the selected backend with the released kwargs.

    Both backends receive exactly the kwarg set `KimiDeltaAttention.forward`
    passes -- one definition, in `released_call_kwargs()` -- so a backend swap
    cannot quietly change the semantics.
    """
    assert backend in BACKENDS, f"unknown KDA backend {backend!r}; expected one of {BACKENDS}"
    kwargs = {**released_call_kwargs(config), **overrides}

    if backend == EAGER:
        return chunk_kda_eager_fp32(
            q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
            initial_state=initial_state, **kwargs,
        )

    return _fla_chunk_kda()(
        q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
        initial_state=initial_state, **kwargs,
    )
