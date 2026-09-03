# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE recipe helpers for quantized MORI token dispatch.

Quantize sender tokens with Transformer Engine Quantizers, pack a per-row
payload + scales for MORI, and reconstruct a QuantizedTensor after recv so
expert GEMMs skip a second input quantize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611
    import transformer_engine_torch as tex
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Quantizer,
        Float8Tensor,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    QuantizedTensor = ()  # type: ignore[misc, assignment]
    Quantizer = Any  # type: ignore[misc, assignment]


_BLOCKWISE_BLOCK_LEN = 128
_MXFP8_BLOCK_LEN = 32


@dataclass
class DispatchQuantMeta:
    """Layout needed to wrap MORI recv buffers back into a TE tensor."""

    quantizer: Any
    fake_dtype: torch.dtype
    kind: str  # "blockwise" | "mxfp8" | "per_tensor"
    te_scale_shape: Optional[Tuple[int, ...]] = None
    scales_need_transpose: bool = False
    payload_dtype: torch.dtype = torch.uint8


def is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Return True if ``tensor`` is a TE QuantizedTensor."""
    return HAVE_TE and isinstance(tensor, QuantizedTensor)


def is_quantized_dispatch_context_active() -> bool:
    """True when TE autocast is enabled (FP8 or FP4 recipes share this flag)."""
    if not HAVE_TE:
        return False
    try:
        return bool(FP8GlobalStateManager.is_fp8_enabled())
    except Exception:  # pylint: disable=broad-except
        return False


def should_quantize_dispatch(config) -> bool:
    """Whether MORI dispatch should quantize using the live TE recipe."""
    if config is None:
        return False
    if not getattr(config, "moe_quantized_dispatch", True):
        return False
    if getattr(config, "moe_flex_dispatcher_backend", None) != "mori":
        return False
    if not is_quantized_dispatch_context_active():
        return False
    recipe = current_dispatch_recipe()
    if recipe is None:
        return False
    if _is_fp4_recipe(recipe):
        return False
    return True


def current_dispatch_recipe():
    """Recipe from the active TE autocast, or None."""
    if not HAVE_TE:
        return None
    if not is_quantized_dispatch_context_active():
        return None
    return FP8GlobalStateManager.get_fp8_recipe()


def _is_fp4_recipe(recipe) -> bool:
    if not HAVE_TE:
        return False
    nvfp4 = getattr(te_recipe, "NVFP4BlockScaling", None)
    mxfp4 = getattr(te_recipe, "MXFP4BlockScaling", None)
    if nvfp4 is not None and isinstance(recipe, nvfp4):
        return True
    if mxfp4 is not None and isinstance(recipe, mxfp4):
        return True
    return False


def _e4m3_dtype():
    return tex.DType.kFloat8E4M3


def dispatch_block_size(recipe) -> int:
    """Hidden-dim alignment for the recipe's rowwise scales."""
    if not HAVE_TE or recipe is None:
        return 1
    if isinstance(recipe, te_recipe.MXFP8BlockScaling):
        return _MXFP8_BLOCK_LEN
    if isinstance(recipe, te_recipe.Float8BlockScaling):
        return _BLOCKWISE_BLOCK_LEN
    return 1


def mori_scale_dim(recipe, hidden_dim: int) -> int:
    """Per-token scale count for MORI ``scale_dim``."""
    block = dispatch_block_size(recipe)
    if block <= 1:
        return 1
    if hidden_dim % block != 0:
        raise ValueError(
            f"hidden_size ({hidden_dim}) must be divisible by {block} for quantized "
            f"MORI dispatch with recipe {type(recipe).__name__}."
        )
    return hidden_dim // block


def mori_scale_type_size(recipe) -> int:
    """Bytes per MORI scale element."""
    if HAVE_TE and isinstance(recipe, te_recipe.MXFP8BlockScaling):
        return torch.uint8.itemsize
    return torch.float32.itemsize


def mori_payload_dtype(recipe) -> torch.dtype:
    """Torch dtype of the quantized payload MORI communicates."""
    del recipe
    if hasattr(torch, "float8_e4m3fnuz"):
        try:
            _ = torch.empty(0, dtype=torch.float8_e4m3fnuz)
            return torch.float8_e4m3fnuz
        except Exception:  # pylint: disable=broad-except
            pass
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return torch.uint8


def view_payload_for_mori(data: torch.Tensor) -> torch.Tensor:
    """Reinterpret TE uint8 FP8 storage as a MORI kernel dtype (no copy)."""
    if data.dtype in (torch.uint8, torch.int8):
        return data.view(mori_payload_dtype(None))
    return data


def view_payload_for_te(data: torch.Tensor, meta: DispatchQuantMeta) -> torch.Tensor:
    """Reinterpret MORI recv payload as TE QuantizedTensor storage dtype."""
    target = meta.payload_dtype
    if data.dtype == target:
        return data
    return data.view(target)


def make_dispatch_quantizer(recipe, device: torch.device) -> Quantizer:
    """Build a rowwise-only TE quantizer from ``recipe`` (not Linear fp8_meta)."""
    if not HAVE_TE:
        raise RuntimeError("Transformer Engine is required for quantized MORI dispatch.")
    if _is_fp4_recipe(recipe):
        raise NotImplementedError("FP4 MORI dispatch is not supported yet.")

    fp8_dtype = _e4m3_dtype()
    if isinstance(recipe, te_recipe.Float8BlockScaling):
        return Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=1,
            amax_epsilon=getattr(recipe, "amax_epsilon", 0.0),
            force_pow_2_scales=getattr(recipe, "w_force_pow_2_scales", True)
            if hasattr(recipe, "w_force_pow_2_scales")
            else True,
        )
    if isinstance(recipe, te_recipe.MXFP8BlockScaling):
        return MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=False)
    if isinstance(recipe, te_recipe.Float8CurrentScaling):
        return Float8CurrentScalingQuantizer(
            fp8_dtype=fp8_dtype,
            device=device,
            rowwise=True,
            columnwise=False,
        )
    if isinstance(recipe, te_recipe.DelayedScaling):
        scale = torch.ones(1, dtype=torch.float32, device=device)
        amax = torch.zeros(1, dtype=torch.float32, device=device)
        return Float8Quantizer(
            scale=scale,
            amax=amax,
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,
        )
    if isinstance(recipe, te_recipe.CustomRecipe):
        factory = getattr(recipe, "qfactory", None)
        if factory is None:
            raise ValueError("CustomRecipe for quantized dispatch requires qfactory.")
        quantizer = factory("input") if callable(factory) else factory
        if hasattr(quantizer, "rowwise_usage"):
            quantizer.rowwise_usage = True
            quantizer.columnwise_usage = False
        return quantizer
    raise ValueError(f"Unsupported TE recipe for MORI dispatch: {type(recipe).__name__}")


def unpack_rowwise(quantized: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, DispatchQuantMeta]:
    """Extract MORI ``[T, H]`` payload and ``[T, scale_dim]`` scales."""
    fake_dtype = quantized.dtype
    T = quantized.shape[0]
    if hasattr(quantized, "_rowwise_data") and quantized._rowwise_data is not None:
        data = quantized._rowwise_data
        scales = quantized._rowwise_scale_inv
        kind = "mxfp8" if isinstance(quantized, MXFP8Tensor) else "blockwise"
    else:
        data = quantized._data
        scales = quantized._scale_inv
        kind = "per_tensor"

    te_scale_shape = tuple(scales.shape)
    scales_need_transpose = False
    if kind == "per_tensor":
        if scales.numel() == 1:
            scales = scales.reshape(1).expand(T).reshape(T, 1).contiguous()
        else:
            scales = scales.reshape(-1)[:1].expand(T).reshape(T, 1).contiguous()
    elif kind == "mxfp8":
        scales = scales.reshape(scales.shape[0], -1)[:T]
    else:
        if scales.dim() == 1:
            scales = scales.reshape(T, -1)
        elif scales.shape[0] == T:
            scales = scales.reshape(T, -1)[:T]
        else:
            # TE 1x128 rowwise scales are [scale_dim, T] (possibly padded on T).
            scales_need_transpose = True
            token_dim = scales.shape[-1]
            scales = scales.reshape(-1, token_dim)[:, :T].transpose(0, 1).contiguous()

    meta = DispatchQuantMeta(
        quantizer=getattr(quantized, "_quantizer", None),
        fake_dtype=fake_dtype,
        kind=kind,
        te_scale_shape=te_scale_shape,
        scales_need_transpose=scales_need_transpose,
        payload_dtype=data.dtype,
    )
    return data, scales.contiguous(), meta


def wrap_rowwise(
    data: torch.Tensor,
    scales: torch.Tensor,
    meta: DispatchQuantMeta,
) -> torch.Tensor:
    """Rebuild a TE QuantizedTensor from MORI recv payload + per-row scales."""
    if not HAVE_TE:
        raise RuntimeError("Transformer Engine is required to wrap quantized dispatch recv.")
    data = view_payload_for_te(data, meta)
    quantizer = meta.quantizer
    fake_dtype = meta.fake_dtype
    T = data.shape[0]
    if meta.kind == "per_tensor":
        scale_inv = scales.reshape(-1)[:1].to(torch.float32).contiguous()
        return Float8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            data=data,
            fp8_scale_inv=scale_inv,
            fp8_dtype=quantizer.dtype,
            requires_grad=False,
            data_transpose=None,
            quantizer=quantizer,
            device=data.device,
        )
    te_scales = scales
    if meta.scales_need_transpose:
        te_scales = scales.transpose(0, 1).contiguous()
    elif meta.kind == "mxfp8" and meta.te_scale_shape is not None:
        padded = torch.zeros(meta.te_scale_shape, dtype=scales.dtype, device=scales.device)
        sl0 = min(T, padded.shape[0])
        sl1 = min(scales.shape[1], padded.shape[1]) if padded.dim() > 1 else scales.numel()
        if padded.dim() == 2:
            padded[:sl0, :sl1] = scales[:sl0, :sl1]
            te_scales = padded
    if meta.kind == "mxfp8":
        return MXFP8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            fp8_dtype=quantizer.dtype,
            rowwise_data=data,
            rowwise_scale_inv=te_scales,
            columnwise_data=None,
            columnwise_scale_inv=None,
            quantizer=quantizer,
            requires_grad=False,
        )
    return Float8BlockwiseQTensor(
        shape=data.shape,
        dtype=fake_dtype,
        rowwise_data=data,
        rowwise_scale_inv=te_scales,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=quantizer.dtype,
        quantizer=quantizer,
        is_2D_scaled=False,
        requires_grad=False,
    )


def quantize_hidden_for_dispatch(
    hidden: torch.Tensor, recipe
) -> Tuple[torch.Tensor, torch.Tensor, DispatchQuantMeta]:
    """Quantize ``[T, H]`` hidden states; return payload, per-row scales, meta."""
    if hidden.numel() == 0:
        scale_dim = mori_scale_dim(recipe, hidden.shape[-1] if hidden.dim() == 2 else 1)
        empty_data = torch.empty(
            (0, hidden.shape[-1] if hidden.dim() == 2 else 0),
            dtype=torch.uint8,
            device=hidden.device,
        )
        empty_scales = torch.empty(
            (0, scale_dim),
            dtype=(
                torch.uint8
                if HAVE_TE and isinstance(recipe, te_recipe.MXFP8BlockScaling)
                else torch.float32
            ),
            device=hidden.device,
        )
        quantizer = make_dispatch_quantizer(recipe, hidden.device)
        kind = "mxfp8" if isinstance(recipe, te_recipe.MXFP8BlockScaling) else (
            "blockwise" if isinstance(recipe, te_recipe.Float8BlockScaling) else "per_tensor"
        )
        meta = DispatchQuantMeta(
            quantizer=quantizer,
            fake_dtype=hidden.dtype,
            kind=kind,
            scales_need_transpose=kind == "blockwise",
            payload_dtype=empty_data.dtype,
        )
        return empty_data, empty_scales, meta

    quantizer = make_dispatch_quantizer(recipe, hidden.device)
    quantized = quantizer(hidden.contiguous())
    data, scales, meta = unpack_rowwise(quantized)
    meta.quantizer = quantizer
    meta.fake_dtype = hidden.dtype
    return data, scales, meta


def wrap_dispatched_quantized(recv_data, recv_scales, meta: DispatchQuantMeta):
    """Autograd-safe wrap of MORI recv payload into a TE QuantizedTensor."""

    class _WrapDispatchedQuantized(torch.autograd.Function):
        @staticmethod
        def forward(ctx, data, scales):
            return wrap_rowwise(data, scales, meta)

        @staticmethod
        def backward(ctx, grad_output):
            if grad_output is None:
                return None, None
            return grad_output.contiguous(), None

    return _WrapDispatchedQuantized.apply(recv_data, recv_scales)
