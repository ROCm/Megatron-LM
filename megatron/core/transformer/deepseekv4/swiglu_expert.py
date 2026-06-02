# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""SwiGLU expert FFN with optional linear-component clamping.

V4 paper §3.2: linear component clamped to [-10, 10], gate upper-bounded to 10.
This eliminates activation outliers without loss of expressiveness in practice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_utils import column_parallel, row_parallel


class SwiGLUExpert(nn.Module):
    """Single expert FFN: SwiGLU(x) = (clamp(W1·x) * silu(W3·x)) · W2.

    Standard SwiGLU: gate = silu(W3·x), linear = W1·x.
    V4 clamping: linear = clamp(W1·x, -limit, limit); gate = clamp(W3·x, max=limit).
    """

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        h = config.moe_intermediate_size
        self.limit = config.swiglu_limit  # 0.0 = no clamp; 10.0 for V4

        self.w1 = column_parallel(d, h, config)
        self.w3 = column_parallel(d, h, config)
        self.w2 = row_parallel(h, d, config, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (tokens, d) -> (tokens, d)"""
        linear, _ = self.w1(x)   # (tokens, h/TP)
        gate, _ = self.w3(x)     # (tokens, h/TP)

        if self.limit > 0:
            linear = linear.clamp(-self.limit, self.limit)
            gate = gate.clamp(max=self.limit)

        out, _ = self.w2(F.silu(gate) * linear)
        return out
