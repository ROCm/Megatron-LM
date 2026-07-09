# DeepSeek-V3 671B MI355X proxy — source before train_deepseekv3.sh.
# Defines MOE_LAYER_FREQ and related Primus-style defaults (override as needed).
#
# Usage (repo root):
#   source examples/deepseek_v3/proxy_mi355x_deepseekv3_671B.sh
#   bash examples/deepseek_v3/train_deepseekv3.sh

export MODEL_SIZE=671B

# --- MoE layer pattern only for shallow proxy (3 layers: 1 dense + 2 MoE) ---
export NUM_LAYERS=3
export MOE_LAYER_FREQ='([0]*1+[1]*2)'

# --- hyperparameters (Primus overrides) ---
export TRAIN_ITERS=20
export MICRO_BATCH_SIZE=2
export GLOBAL_BATCH_SIZE=16
export SEQ_LENGTH=4096
export MAX_POSITION_EMBEDDINGS=4096
export LR=1.0e-5
export MIN_LR=0.0
export LR_WARMUP_ITERS=2
export LR_DECAY_ITERS=null
export WEIGHT_DECAY=0.1

# --- parallelism (PP4 + uneven layout / VPP from layout string) ---
export TP=1
export PP=1
export EP=8

export MOCK_DATA=1


export APPLY_ROPE_FUSION=true

# recompute: full AC + block method (Megatron has no recompute_layer_ids CLI in this tree)
export AC="${AC:-full}"

# --- optimizer (use_precision_aware_optimizer + bf16 states) ---
export USE_PRECISION_AWARE_OPTIMIZER=true
export MAIN_GRADS_DTYPE=bf16
export EXP_AVG_DTYPE=bf16
export EXP_AVG_SQ_DTYPE=bf16

export MOE_ROUTER_FUSION=true
export MOE_PERMUTE_FUSION=true
export MOE_SHARED_EXPERT_OVERLAP=false

# --- MoE / DeepEP / grouped GEMM ---
export ENABLE_DEEP_EP=false
export ENABLE_MORI=true
export MOE_USE_LEGACY_GROUPED_GEMM=false
export USE_GROUPED_GEMM=true
export FORCE_BALANCE=true

export CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"
export GA_FUSION=true

export PROFILE_START=12
export PROFILE_END=13
export PROFILE=true
