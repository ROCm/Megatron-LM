"""SiTU-GLU activation.

Transcribed from the released `SituAndMul` (HF moonshotai/Kimi-K3). Both branches
are tanh-limited, which is why the release ships no Hadamard rotation for QAT --
SiTU is the outlier control. The math runs in fp32 and casts back at the end
(rule R7.3).

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)     # beta = 4.0
    up     = linear_beta * tanh(up / linear_beta)         # linear_beta = 25.0
    y      = situ_a * up
"""

import torch

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def situ_glu(
    gate_up: torch.Tensor, beta: float = SITU_BETA, linear_beta: float | None = SITU_LINEAR_BETA
) -> torch.Tensor:
    """``gate_up`` is the concatenated ``[gate | up]`` of width ``2 * inter``."""
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(gate_up.dtype)
