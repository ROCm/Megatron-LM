# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseRetroCrossAttention(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs,
    ):
        super().__init__(config=config)

        self.attn = CrossAttention(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        self.retro_num_neighbors = config.retro_num_neighbors
        self.retro_chunk_length = config.retro_preprocess.retro_gpt_chunk_length
        self.retro_retrieved_length = config.retro_preprocess.retro_gpt_retrieved_length
