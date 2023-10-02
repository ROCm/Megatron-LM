# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderCrossAttention,
    RetroEncoderBiasDropoutAdd,
    RetroEncoderLayerNorm,
)
from megatron.core.transformer import (
    ModuleSpec,
    TransformerBlock,
    TransformerBlockSubmodules,
    TransformerConfig,
)
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules


def get_retro_encoder_layer_spec() -> ModuleSpec:
    """
    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.cross_attention=ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={
            "attn_mask_type" : AttnMaskType.padding,
        },
        submodules=CrossAttentionSubmodules(
            linear_q=TELayerNormColumnParallelLinear,
            linear_kv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        )
    )
    spec.submodules.cross_attn_bda=ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm=ModuleSpec(module=RetroEncoderLayerNorm)
    spec.submodules.mlp=ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    return spec


def get_retro_encoder_block_spec(config: TransformerConfig) -> ModuleSpec:

    """
    The retro encoder block consists of one customized Retro encoder layer
    (layer 1), and all of the following layers are standard GPT layers.
    """

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    retro_layer_spec = get_retro_encoder_layer_spec()
    for spec in (gpt_layer_spec, retro_layer_spec):
        spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.padding

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(layer_specs=layer_specs),
    )

    return block_spec
