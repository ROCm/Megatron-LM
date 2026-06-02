# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""DeepSeek-V4 language model.

Parallel to megatron/core/models/deepseekv2/model.py.
Uses DeepSeekV4TransformerLayer instead of standard TransformerLayer.
"""
import logging
from typing import Literal, Optional, Union

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.dist_checkpointing.mapping import ShardedStateDict

from .config import DeepSeekV4Config
from megatron.core.transformer.deepseekv4.transformer_layer import DeepSeekV4TransformerLayer


logger = logging.getLogger(__name__)


class DeepSeekV4TransformerBlock(torch.nn.Module):
    """Stack of DeepSeekV4TransformerLayer.

    Handles pipeline parallelism (pre_process / post_process flags) and
    assigns moe_layer_idx for MoE-capable layers based on config.
    """

    def __init__(self, config: DeepSeekV4Config, pre_process: bool, post_process: bool):
        super().__init__()
        self.config = config

        # Determine which global layers this pipeline stage owns.
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        total_layers = config.num_layers
        layers_per_stage = total_layers // pp_size
        start_layer = pp_rank * layers_per_stage  # 0-based global layer index
        end_layer = start_layer + layers_per_stage

        # MoE layer schedule: every layer with index >= first_moe_layer is MoE.
        # Adjust first_moe_layer to account for non-MoE dense layers at the front.
        first_moe = getattr(config, "first_moe_layer", 0)

        layers = []
        moe_counter = 0
        for global_idx in range(start_layer, end_layer):
            layer_number = global_idx + 1  # 1-based
            if global_idx >= first_moe and config.num_moe_experts and config.num_moe_experts > 0:
                layer = DeepSeekV4TransformerLayer(config, layer_number, moe_layer_idx=moe_counter)
                moe_counter += 1
            else:
                layer = DeepSeekV4TransformerLayer(config, layer_number, moe_layer_idx=None)
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)

        # Final norm on last stage only.
        if post_process:
            from megatron.legacy.model.rms_norm import RMSNorm
            self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.final_layernorm = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_params=None,
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, inference_params)

        # Collapse stream dimension before final norm.
        if hidden_states.dim() == 4:
            # (s, b, N, d) -> average or take stream 0 (index 0 is the "main" stream).
            hidden_states = hidden_states[:, :, 0, :]  # (s, b, d)

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class DeepSeekV4Model(LanguageModule):
    """DeepSeek-V4 causal language model.

    Compatible with Megatron-LM pipeline + tensor + expert parallelism.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ) -> None:
        super().__init__(config=config)

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_type = ModelType.encoder_or_decoder

        if pre_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=config.padded_vocab_size,
                max_sequence_length=config.max_position_embeddings,
                position_embedding_type="none",  # RoPE applied inside attention
            )

        self.decoder = DeepSeekV4TransformerBlock(config, pre_process, post_process)

        if post_process:
            self.output_layer = torch.nn.Linear(
                config.hidden_size, config.padded_vocab_size, bias=False
            )
            if share_embeddings_and_output_weights:
                self.output_layer.weight = self.embedding.word_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inference_params=None,
    ) -> Union[Tensor, dict]:
        if decoder_input is not None:
            hidden_states = decoder_input
        elif self.pre_process:
            hidden_states = self.embedding(input_ids=input_ids, position_ids=position_ids)
            # Megatron embeddings return (b, s, d); transpose to (s, b, d).
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            raise RuntimeError("decoder_input required when pre_process=False")

        # Expand hidden states to multi-stream shape if mHC is active.
        if self.config.hc_mult > 1:
            # (s, b, d) -> (s, b, N, d)
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.config.hc_mult, -1
            ).contiguous()

        hidden_states = self.decoder(hidden_states, attention_mask, inference_params)
        # hidden_states: (s, b, d) after decoder collapses stream dim

        if not self.post_process:
            return hidden_states

        # Logits: (b, s, vocab)
        logits = self.output_layer(hidden_states.transpose(0, 1))

        if labels is None:
            return logits.float()

        loss = self.compute_language_model_loss(labels, logits)
        return loss
