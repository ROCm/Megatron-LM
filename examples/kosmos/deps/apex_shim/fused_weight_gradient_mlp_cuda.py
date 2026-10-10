###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Stand-in for APEX's fused_weight_gradient_mlp_cuda extension (not installable here: no APEX source).

Same contract as APEX: main_grad[out, in] += grad_output[tokens, out]^T @ input[tokens, in], bf16/fp16 operands,
fp32 accumulation, written in place. Implemented with TransformerEngine's general_gemm (hipBLASLt), i.e. the same
call TE's own modules use for their fused wgrad accumulation. Only Megatron's non-TE ColumnParallelLinear (the GPT
output layer) calls it.
"""
import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm


def _accum(total_input: torch.Tensor, grad_output: torch.Tensor, main_grad: torch.Tensor) -> None:
    assert total_input.dim() == 2 and grad_output.dim() == 2
    assert main_grad.shape == (grad_output.shape[1], total_input.shape[1])
    general_gemm(
        total_input,
        grad_output,
        out_dtype=main_grad.dtype,
        accumulate=True,
        layout="NT",
        out=main_grad,
        grad=True,
    )


def wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
    assert main_grad.dtype == torch.float32
    _accum(total_input, grad_output, main_grad)


def wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
    assert main_grad.dtype in (torch.float16, torch.bfloat16)
    _accum(total_input, grad_output, main_grad)
