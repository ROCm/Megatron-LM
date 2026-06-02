# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class DeepSeekV4Config(MLATransformerConfig):
    """Configuration for DeepSeek-V4 / V4-Pro.

    Extends MLATransformerConfig with HCA compressor, mHC, grouped-O projection,
    hash-based routing, and decoupled load balancing.
    """

    # --- Hyper-Connections (mHC) ---
    hc_mult: int = 1
    """Number of residual streams. Set to 4 for V4. 1 = disabled (standard residual)."""
    hc_sinkhorn_iters: int = 20
    """Sinkhorn-Knopp iterations for doubly-stochastic projection. 20 per V4 paper."""
    hc_eps: float = 1e-6
    """Epsilon for Sinkhorn numerical stability."""

    # --- HCA Compressor ---
    compress_ratios: Tuple[int, ...] = (128,)
    """Compression ratio(s) for HCA compressor. 128 = non-overlap (V4 default). 4 = overlap (CSA)."""
    compress_rope_theta: float = 160000.0
    """RoPE base for compressed token positions (DualRoPE)."""

    # --- Sliding window & attention sink ---
    attn_sliding_window: int = 128
    """Local attention window size for HCA. 0 = disabled."""
    attn_sink: bool = False
    """Whether to include attention sink tokens."""
    attn_sink_size: int = 1
    """Number of attention sink tokens."""

    # --- Lightning Indexer (CSA, compress_ratio=4 only) ---
    index_topk: int = 0
    """Top-k for lightning indexer. 0 = no indexer (HCA path)."""
    index_head_dim: int = 128
    """Head dimension for indexer scoring."""
    index_n_heads: int = 64
    """Number of heads used in indexer."""

    # --- Grouped low-rank output projection ---
    o_groups: int = 1
    """Number of groups for grouped output projection. 8 for V4. 1 = standard projection."""
    o_lora_rank: int = 0
    """Intermediate rank per group for grouped-O. 0 = no low-rank factorization."""

    # --- Hash-based routing ---
    num_hash_layers: int = 0
    """Number of early MoE layers to use hash routing instead of learned routing. 3 for V4."""
    hash_routing_seed: int = 0
    """RNG seed for constructing the tid2eid lookup table."""

    # --- MoE gating ---
    moe_intermediate_size: int = 0
    """Hidden dimension of each MoE expert. Separate from ffn_hidden_size for V4."""
    swiglu_limit: float = 0.0
    """Clamp bound for SwiGLU linear component. 0 = no clamping. 10.0 for V4."""

    # --- Vocab ---
    vocab_size: int = 129280
    padded_vocab_size: int = 129280

    def __post_init__(self):
        super().__post_init__()

        # Normalize compress_ratios if passed as a comma-separated string (YAML-friendly).
        if isinstance(self.compress_ratios, str):
            self.compress_ratios = tuple(int(r) for r in self.compress_ratios.split(","))
        elif isinstance(self.compress_ratios, (list, tuple)):
            self.compress_ratios = tuple(int(r) for r in self.compress_ratios)

        if self.moe_intermediate_size == 0:
            # Fall back to standard ffn_hidden_size if not explicitly set.
            self.moe_intermediate_size = self.ffn_hidden_size

        if self.hc_mult < 1:
            raise ValueError(f"hc_mult must be >= 1, got {self.hc_mult}")
