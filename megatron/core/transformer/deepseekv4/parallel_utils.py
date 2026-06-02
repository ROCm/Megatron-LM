# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Thin wrappers that fill required keyword args for parallel linear layers.

Megatron's ColumnParallelLinear and RowParallelLinear have `init_method` and
`skip_bias_add` as required kwargs (no defaults). When constructing these layers
directly (not via ModuleSpec), callers must supply them explicitly.
These helpers pull init_method from the config, matching Megatron's spec builder.
"""
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


def column_parallel(
    input_size: int,
    output_size: int,
    config,
    *,
    bias: bool = False,
    gather_output: bool = False,
    is_expert: bool = False,
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        bias=bias,
        gather_output=gather_output,
        skip_bias_add=False,
        is_expert=is_expert,
    )


def row_parallel(
    input_size: int,
    output_size: int,
    config,
    *,
    bias: bool = False,
    input_is_parallel: bool = True,
    is_expert: bool = False,
) -> RowParallelLinear:
    return RowParallelLinear(
        input_size,
        output_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=bias,
        input_is_parallel=input_is_parallel,
        skip_bias_add=False,
        is_expert=is_expert,
    )
