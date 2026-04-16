#!/bin/bash

set -ex

COMMON="MODEL_SIZE=8 TP=1 MBS=1 BS=4 SEQ_LENGTH=2048 \
  TOTAL_ITERS=5000 \
  EVAL_INTERVAL=250 EVAL_ITERS=10 \
  DATA_PATH=./wikitext103_train_text_document \
  LOG_INTERVAL=10 \
  SAVE_CKPT_PATH=/checkpoints \
  NNODES=1"

# BF16 baseline
bash examples/llama/train_llama3.sh $COMMON \
  TE_FP8=0 TE_FP4=0 EXP_NAME=bf16_baseline

# FP8 tensorwise/current scaling
bash examples/llama/train_llama3.sh $COMMON \
  TE_FP8=1 TE_FP8_RECIPE=tensorwise TE_FP4=0 EXP_NAME=fp8_tensorwise

# NVFP4
bash examples/llama/train_llama3.sh $COMMON \
  TE_FP8=0 TE_FP4=1 \
  FP4_SELECTIVE_BF16=1 FP4_BF16_START=2 FP4_BF16_END=8 \
  EXP_NAME=nvfp4_selective
