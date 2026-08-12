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

# Force HF offline: the Qwen3-235B-A22B tokenizer (tokenizer.json/vocab/merges/config) is
# already in HF_HOME. Without offline mode, from_pretrained blocks on an HF Hub network call
# and hangs at "building HuggingFaceTokenizer tokenizer" when the network is unreachable.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NUM_LAYERS="${NUM_LAYERS:-24}"
export NUM_EXPERTS="${NUM_EXPERTS:-128}"
# Must satisfy ROUTER_TOPK <= NUM_EXPERTS (lower TOPK if you shrink experts below 8).
export ROUTER_TOPK="${ROUTER_TOPK:-8}"

# --- hyperparameters (Primus overrides) ---
export TRAIN_ITERS=20
export MICRO_BATCH_SIZE=2
# 1F1B EP overlap needs >=2 microbatches: num_micro = GBS/(MBS*DP), DP=world/(TP*PP)=8,
# MBS=2 -> GBS=64 gives 4 microbatches (matches train_qwen_ck_overlap_og2.log).
# GBS=16 collapses to 1 microbatch (16/(2*8)) -> NO 1F1B overlap possible. Use >=32 for 2.
export GLOBAL_BATCH_SIZE=32
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

# --- activation checkpointing ---
# og2 used --recompute-activations (AC=sel, no RECOMPUTE_MODULES).
export AC=sel
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
export MOE_USE_LEGACY_GROUPED_GEMM=false
export USE_GROUPED_GEMM=true
export FORCE_BALANCE=true
# CK grouped GEMM (TE CUTLASS path). Inert for the permute-free FC1/FC2 (they take the
# route-list gather GEMM once permute_free_metadata is passed); affects only any non-PF GEMM.
export NVTE_USE_CUTLASS_GROUPED_GEMM=0

# --- Permute-free grouped GEMM ---
# Enables --moe-permute-free-grouped-gemm (train_qwen3.sh). arguments.py auto-sets
# NVTE_PERMUTE_FREE_GROUPED_GEMM=1. Keeps flex+mori, bf16, no-bias (all satisfied above).
export MOE_PERMUTE_FREE_GG=true
# Sync permute-free (opt-in): size the route-ordered activation buffers to the EXACT routed-
# token count (routing_map.sum()) instead of the sync-free worst-case num_recv*min(topk,E)
# bound. Drastically cuts MoE activation memory (the win scales with how sparse each rank's
# local routing_map is), at the cost of one device->host sync per expert layer and
# data-dependent buffer shapes -> INCOMPATIBLE with CUDA graphs (none used here). Sets
# --moe-permute-free-exact-routes; the config hard-errors if CUDA graphs are enabled.
export MOE_PERMUTE_FREE_EXACT_ROUTES=false

# --- EP all-to-all 1F1B overlap (matches og2) ---
# Overlap ON: validates the MoriCombine.backward perm-free clone fix. Root cause was combine's
# backward returning a raw view into MORI's reusable symm buffer (no fused-unpermute reader on
# the PF path); a sibling microbatch's op overwrote it under 1F1B -> NaN wgrad. PF alone (overlap
# off) already trains clean for 20 iters, so a finite grad norm here confirms the fix.
export EP_A2A_OVERLAP=true
export DELAY_WGRAD_COMPUTE=true
# --- cross-entropy fusion ---
export CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"

# --- distributed optimizer ---
export DO=true

export PROFILE_START=12
export PROFILE_END=13
export PROFILE=true
export AITER_USE_SYSTEM_TRITON=1
# NOTE: the var name is GPU_MAX_HW_QUEUES (plural). The old `GPU_MAX_HW_QUEUE=4`
# was a typo that set nothing, so train_qwen3.sh fell back to its default of 2 ->
# comm and compute HIP streams round-robined onto the same physical HW queue and
# serialized (trace showed ~4% comm/compute overlap). 8 gives each critical stream
# its own queue and restores 1F1B a2a overlap.
export GPU_MAX_HW_QUEUES=4