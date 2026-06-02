# Copyright (c) 2025, ROCm/Megatron-LM contributors. All rights reserved.
"""Pretrain DeepSeek-V4.

Minimal training script that reuses pretrain_gpt.py's forward_step, loss_func,
and dataset provider, but substitutes DeepSeekV4Model via deepseekv4_builder.

Usage:
    torchrun ... pretrain_deepseekv4.py [standard megatron args] [v4 args]
"""
import time
_PROGRAM_START_TIME = time.time()

import os
from functools import partial
from typing import List, Optional

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer

from model_provider import model_provider
from deepseekv4_builder import deepseekv4_model_builder

stimer = StragglerDetector()


# ---------------------------------------------------------------------------
# V4-specific config overrides — edit these for your run.
# Standard MLA / MoE dims come from CLI args; only V4 extras go here.
# ---------------------------------------------------------------------------
V4_CONFIG_OVERRIDES = dict(
    hc_mult=4,
    hc_sinkhorn_iters=20,
    compress_ratios=(128,),
    compress_rope_theta=160000.0,
    attn_sliding_window=128,
    o_groups=8,
    num_hash_layers=3,
    swiglu_limit=10.0,
)


def get_batch(data_iterator, vp_stage=None):
    """Reuse pretrain_gpt get_batch logic (without MTP/CP complexity)."""
    args = get_args()
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens = batch.get('text')
    if tokens is None:
        return None, None, None, None, None, None

    labels = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()
    loss_mask = torch.ones(labels.shape, dtype=torch.float, device=labels.device)
    attention_mask = None   # DeepSeekV4 builds its own causal+window mask
    position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand_as(tokens)
    return tokens, labels, loss_mask, attention_mask, position_ids, None


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    return loss, num_tokens, {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}


def forward_step(data_iterator, model):
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage", default=None)
    tokens, labels, loss_mask, attention_mask, position_ids, _ = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()

    try:
        tokenizer = build_tokenizer(args)
    except Exception:
        tokenizer = None

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    data_cfg = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )

    dataset_type = MockGPTDataset if args.mock_data else GPTDataset
    print_rank_0('> building train, validation, and test datasets for DeepSeek-V4 ...')
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, lambda: True, data_cfg
    ).build()
    print_rank_0('> finished creating datasets.')
    return train_ds, valid_ds, test_ds


def extra_args_provider(parser):
    """No new CLI args needed — V4 extras are in V4_CONFIG_OVERRIDES above."""
    return parser


if __name__ == '__main__':
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    builder = partial(deepseekv4_model_builder, V4_CONFIG_OVERRIDES)

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'NullTokenizer'},
        extra_args_provider=extra_args_provider,
    )
