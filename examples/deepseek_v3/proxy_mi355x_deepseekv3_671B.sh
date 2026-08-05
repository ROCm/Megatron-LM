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

# --- activation checkpointing ---
# Selective recompute across all applicable modules. Excluded on purpose:
#   * moe          -> covers the graphed MoE scope (would trigger moe_layer_recompute over the
#                     graphed region). Recompute must not *cover* the CUDA graph scope.
#   * moe_act      -> recomputing it disables the fused FC1 activation + router-prob epilogue
#                     (fc1_fused_act requires not activation_recompute).
export AC=sel
export RECOMPUTE_MODULES="core_attn layernorm mlp mla_up_proj shared_experts"
export RECOMPUTE_METHOD=block
export RECOMPUTE_NUM_LAYERS=3

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

# --- CUDA graph ---
# Selective TE-scoped graph for the WHOLE MoE layer: captures MoRI dispatch + expert
# GroupedGEMM + MoRI combine. The permute-free + MoRI path provides static shapes
# (fixed max_num_tokens_per_rank symmetric buffers + static per-expert count list, no host
# DtoH sync), so drop-padding is not required (see transformer_config.py pf_mori_static).
# NOTE: scope "moe" cannot be combined with "moe_router"/"moe_preprocess" (config assert).
# Requires GPT_LAYER_IN_TE=true (default).
export ENABLE_CUDA_GRAPH=true
export CUDA_GRAPH_IMPL=transformer_engine
export CUDA_GRAPH_SCOPE="moe"

export CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"
export GA_FUSION=true

export PROFILE_START=12
export PROFILE_END=13
export PROFILE=true
export AITER_USE_SYSTEM_TRITON=1
export NVTE_PERMUTE_FREE_MOE_AUTOTUNE=1
export AITER_MOE_FLYDSL_V3=1