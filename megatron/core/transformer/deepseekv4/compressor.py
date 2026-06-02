# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""HCA Compressor: softmax-gated pooling that compresses hidden states into
a smaller sequence for use as compressed KV tokens in Hyper Compressed Attention.

compress_ratio=128  -> non-overlap mode (V4 HCA): coff=1, straight 128-token pooling
compress_ratio=4    -> overlap mode (V3 CSA): coff=2, groups overlap, cross-group mixing

Reference: DeepSeek V4 paper §3.1; reference impl:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Compressor(nn.Module):
    """Softmax-gated temporal pooling.

    Groups the sequence into windows of `compress_ratio` tokens and produces
    one compressed token per group via learned gating weights.

    Overlap mode (compress_ratio == 4):
        - Window size = 2 * compress_ratio (=8), stride = compress_ratio (=4)
        - Each group overlaps with its neighbours.
        - An overlap_transform mixes adjacent groups.

    Non-overlap mode (compress_ratio == 128):
        - Window = compress_ratio, stride = compress_ratio (no overlap).
        - coff = 1 (no cross-group mixing).
    """

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        r = config.compress_ratios[0]  # primary ratio

        self.compress_ratio = r
        self.overlap = (r == 4)
        self.coff = 2 if self.overlap else 1
        window = self.coff * r  # tokens per group

        # Gating weights: one scalar per token in the window.
        self.weight = nn.Linear(d, window, bias=False)
        # Output projection to keep dimension consistent.
        self.out_proj = nn.Linear(d * self.coff, d, bias=False)

        if self.overlap:
            # Mix adjacent compressed tokens.
            self.overlap_transform = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, s, d) input hidden states

        Returns:
            c: (b, s_c, d) compressed tokens, s_c = ceil(s / compress_ratio)
        """
        b, s, d = x.shape
        r = self.compress_ratio

        # Pad sequence so s is divisible by r.
        pad = (r - s % r) % r
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # pad along seq dim
        s_padded = x.shape[1]
        n_groups = s_padded // r

        if self.overlap:
            return self._forward_overlap(x, b, s_padded, n_groups, d)
        else:
            return self._forward_nonoverlap(x, b, s_padded, n_groups, d)

    def _forward_nonoverlap(self, x, b, s, n_groups, d):
        # x: (b, s, d), reshape to (b, n_groups, r, d)
        x_groups = x.view(b, n_groups, self.compress_ratio, d)
        # Gate: (b, n_groups, r)
        gates = F.softmax(self.weight(x_groups.view(b * n_groups, self.compress_ratio, d)), dim=1)
        # gates: (b*n_groups, r, 1) for broadcasting
        gates = gates.view(b, n_groups, self.compress_ratio, 1)
        # Weighted sum: (b, n_groups, d)
        c = (x_groups * gates).sum(dim=2)
        return c  # (b, n_groups, d)

    def _forward_overlap(self, x, b, s, n_groups, d):
        r = self.compress_ratio
        window = 2 * r
        # Build overlapping windows: group i covers tokens [i*r - r, i*r + r)
        # Pad left by r for first group.
        x_padded = F.pad(x, (0, 0, r, 0))  # (b, s+r, d)
        chunks = []
        for i in range(n_groups):
            chunk = x_padded[:, i * r: i * r + window, :]  # (b, 2r, d)
            chunks.append(chunk)
        windows = torch.stack(chunks, dim=1)  # (b, n_groups, 2r, d)

        gates = F.softmax(
            self.weight(windows.view(b * n_groups, window, d)), dim=1
        )  # (b*n_groups, 2r)
        gates = gates.view(b, n_groups, window, 1)
        c = (windows * gates).sum(dim=2)  # (b, n_groups, d)

        # Cross-group mixing via overlap_transform.
        c = c + self.overlap_transform(c)
        return c  # (b, n_groups, d)
