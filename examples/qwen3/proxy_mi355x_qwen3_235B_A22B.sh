# Smaller Qwen3-235B-style MoE proxy for single-node / single-GPU smoke runs.
# Full architecture remains HIDDEN_SIZE / MOE dims from train_qwen3.sh (235B preset);
# shrink depth and expert count here.
#
# Primus reference:
#   https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-pretrain.yaml
#
# Usage (from repo root):
#   source examples/qwen3/1P1G_AAI_proxy_mi355x_qwen3_235B_A22B.sh
#   bash examples/qwen3/train_qwen3.sh
#
# tunables (defaults are a light proxy; override before sourcing or edit below):
#   NUM_LAYERS   — decoder layers (full model: 94)
#   NUM_EXPERTS  — MoE experts (full model: 128); must be divisible sensibly by EP
#   ROUTER_TOPK  — experts per token (default 8); must be <= NUM_EXPERTS

export MODEL_SIZE=235B_A22B

export NUM_LAYERS="${NUM_LAYERS:-24}"
export NUM_EXPERTS="${NUM_EXPERTS:-128}"
# Must satisfy ROUTER_TOPK <= NUM_EXPERTS (lower TOPK if you shrink experts below 8).
export ROUTER_TOPK="${ROUTER_TOPK:-8}"

# --- hyperparameters (Primus overrides) ---
export TRAIN_ITERS=20
export MICRO_BATCH_SIZE=2
export GLOBAL_BATCH_SIZE=16
export SEQ_LENGTH=4096
export MAX_POSITION_EMBEDDINGS=4096
export LR=1.0e-4
export MIN_LR=1.0e-5
export LR_WARMUP_ITERS=2
export LR_DECAY_ITERS=320000
export WEIGHT_DECAY=0.1

# --- parallelism (PP4 + uneven layout / VPP from layout string) ---
export TP=1
export PP=1
export EP=8

# sequence_parallel: 1 in yaml; with TP=1, train_qwen3 does not enable --sequence-parallel

# --- data ---
export MOCK_DATA=1

# --- checkpoint / eval ---
export EVAL_ITERS=0
export SAVE_INTERVAL=20000
export CKPT_FORMAT=torch

# --- activation checkpointing (recompute_granularity full, num_layers 3) ---
export AC=full
export RECOMPUTE_METHOD=block
export RECOMPUTE_NUM_LAYERS=3

# --- optimizer (use_precision_aware_optimizer + bf16 states) ---
export USE_PRECISION_AWARE_OPTIMIZER=true
export MAIN_GRADS_DTYPE=bf16
export EXP_AVG_DTYPE=bf16
export EXP_AVG_SQ_DTYPE=bf16

# --- rope fusion + experimental ---
export APPLY_ROPE_FUSION=true

# --- MoE / DeepEP / grouped GEMM ---
export ENABLE_DEEP_EP=false
export ENABLE_MORI=true
export USE_GROUPED_GEMM=true
export FORCE_BALANCE=true

# --- cross-entropy fusion ---
export CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"

# --- distributed optimizer ---
export DO=true

export PROFILE_START=12
export PROFILE_END=13
export PROFILE=true