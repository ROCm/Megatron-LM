#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3 pretraining launcher (dense + MoE), structured like train_deepseekv3.sh.
#
# Usage:
#   cd /path/to/Megatron-LM && bash examples/qwen3/train_qwen3.sh
#
# Primus MI355X parity (optional): source one of the env files first, then run:
#   source examples/qwen3/primus_mi355x_qwen3_30B_A3B_pretrain.env.sh
#   source examples/qwen3/primus_mi355x_qwen3_235B_A22B_pretrain.env.sh
#
# Model sizes (MODEL_SIZE):
#   235B_A22B | 235B   — MoE (128 experts, top-8), defaults aligned with Primus qwen3_235B_A22B
#   30B_A3B | 30B      — MoE, Primus qwen3_30B_A3B
#   32B                — dense, FSDP2 + full recompute defaults
#   4B | 8B | 14B      — dense
#
# Override any default via environment, e.g. SEQ_LENGTH, EP, TP, LR, TRAIN_ITERS.
# MoE: MOE_PERMUTE_FUSION=false disables --moe-permute-fusion (fused token permute/unpermute).
# For 235B proxy runs: export NUM_LAYERS / NUM_EXPERTS (and optionally ROUTER_TOPK <= NUM_EXPERTS) before launch.
# FP8: PR=fp8 and FP8_RECIPE=delayed|tensorwise|mxfp8|blockwise (mxfp8 sets NVTE_ROCM_ENABLE_MXFP8=1;
#   mxfp8 MoE adds --moe-router-padding-for-fp8 automatically).
#################################################################################

set -e

EXPERIMENT="qwen3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH=$(dirname "$(dirname "${CURRENT_DIR}")")
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}
echo "EXPERIMENT: $EXPERIMENT - $TIMESTAMP"
echo "MEGATRON_PATH: ${MEGATRON_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# --- network / RCCL (same baseline as train_deepseekv3.sh) ---
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
NCCL_IB_HCA_LIST=$(rdma link -j 2>/dev/null | python3 -c "import json, sys
try:
    links = json.load(sys.stdin)
    print(*[links[i][\"ifname\"] for i in range(min(8, len(links)))], sep=',')
except Exception:
    pass") || NCCL_IB_HCA_LIST=""
export NCCL_IB_HCA=${NCCL_IB_HCA:-$NCCL_IB_HCA_LIST}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NVTE_FLASH_ATTN=0

GPUS_PER_NODE=$(python3 -c "import torch; print(torch.cuda.device_count())")
RUN_ENV="${RUN_ENV:-cluster}"
if [ "$RUN_ENV" = "cluster" ]; then
    MASTER_ADDR=${MASTER_ADDR:-localhost}
    MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
elif [ "$RUN_ENV" = "slurm" ]; then
    MASTER_ADDR=${SLURM_MASTER_ADDR}
    MASTER_PORT=${SLURM_MASTER_PORT}
    NNODES=$SLURM_NNODES
    NODE_RANK=${SLURM_NODEID}
fi
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo ""

if [ "${NNODES:-1}" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

# --- model selection ---
MODEL_SIZE=${MODEL_SIZE:-30B_A3B}
HF_MODEL_CKPT=${HF_MODEL_CKPT:-}
export HF_HOME=${HF_HOME:-"../huggingface"}

echo "MODEL_SIZE: $MODEL_SIZE"
echo "HF_HOME: $HF_HOME"
echo ""

# --- data ---
DATA_DIR=${DATA_DIR:-"../data"}
MOCK_DATA="${MOCK_DATA:-1}"
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"../.cache"}

if [ "$MOCK_DATA" -eq 1 ]; then
    echo "Using mock data."
    data_args="--mock-data --data-cache-path ${DATA_CACHE_PATH}"
else
    echo "Using data from ${DATA_DIR}"
    data_args="--train-data-path ${DATA_DIR}/mmap_qwen3_text_document \
        --valid-data-path ${DATA_DIR}/mmap_qwen3_text_document \
        --test-data-path ${DATA_DIR}/mmap_qwen3_text_document"
fi

# --- training / parallel defaults (Primus-style; override with env) ---
PR=${PR:-bf16}
# When PR=fp8: delayed (default), tensorwise, mxfp8 (TE ROCm: NVTE_ROCM_ENABLE_MXFP8), blockwise
FP8_RECIPE="${FP8_RECIPE:-delayed}"
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
ETP=${ETP:-1}
SP=${SP:-true}
TRAIN_ITERS=${TRAIN_ITERS:-50}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2}
PAD_LEN=${PAD_LEN:-8192}
SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
EVAL_ITERS=${EVAL_ITERS:--1}

GEMM_TUNING="${GEMM_TUNING:-1}"
USE_GROUPED_GEMM="${USE_GROUPED_GEMM:-true}"
MOE_USE_LEGACY_GROUPED_GEMM="${MOE_USE_LEGACY_GROUPED_GEMM:-false}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-true}"
NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-1}"
GPT_LAYER_IN_TE="${GPT_LAYER_IN_TE:-true}"

ENABLE_DEEP_EP="${ENABLE_DEEP_EP:-false}"
PROFILE=${PROFILE:-false}
PROFILE_SYNC=${PROFILE_SYNC:-false}
PROFILE_START=${PROFILE_START:-3}
PROFILE_END=${PROFILE_END:-4}
FORCE_BALANCE=${FORCE_BALANCE:-false}

OPTIMIZER_OFFLOAD=false
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-none}

# tokenizer + common Qwen3 flags
MAKE_VOCAB_DIV=${MAKE_VOCAB_DIV:-1187}
ROPE_THETA=${ROPE_THETA:-1000000}
KV_CHANNELS=128

IS_MOE=0
EMBED_OPT=""
DO=true
USE_FSDP2=false
CKPT_FORMAT=${CKPT_FORMAT:-torch}
GA_FUSION=true
CE_FUSION_ARGS=""
AC=${AC:-none}
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

# Architecture + size-specific defaults
case $MODEL_SIZE in
235B_A22B | 235B)
    IS_MOE=1
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-235B-A22B}"
    NUM_LAYERS=${NUM_LAYERS:-94}
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=12288
    NUM_ATTN_HEADS=64
    NUM_QUERY_GROUPS=4
    NUM_EXPERTS=${NUM_EXPERTS:-128}
    MOE_INTERMEDIATE_SIZE=1536
    ROUTER_TOPK=${ROUTER_TOPK:-8}
    MOE_AUX_LOSS=1e-3
    EMBED_OPT="--untie-embeddings-and-output-weights"

    EP=${EP:-8}
    SEQ_LEN="${SEQ_LENGTH:-${SEQ_LEN:-2048}}"
    MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
    LR=${LR:-1e-4}
    MIN_LR=${MIN_LR:-1e-5}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-320000}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    TP=${TP:-1}
    PP=${PP:-1}
    USE_FSDP2=false
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    GA_FUSION=true
    PAD_LEN=$SEQ_LEN
    ;;
30B_A3B | 30B)
    IS_MOE=1
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-30B-A3B}"
    NUM_LAYERS=48
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=6144
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=4
    NUM_EXPERTS=128
    MOE_INTERMEDIATE_SIZE=768
    ROUTER_TOPK=8
    MOE_AUX_LOSS=1e-3
    EMBED_OPT="--untie-embeddings-and-output-weights"

    EP=${EP:-${EP:-8}}
    TP=${TP:-${TP:-1}}
    PP=${PP:-${PP:-1}}
    SEQ_LEN="${SEQ_LENGTH:-${SEQ_LEN:-4096}}"
    MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-4096}"
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    LR=${LR:-1e-5}
    MIN_LR=${MIN_LR:-0.0}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    DO=true
    USE_FSDP2=false
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    GA_FUSION=true
    EVAL_ITERS=${EVAL_ITERS:-0}
    PAD_LEN=$SEQ_LEN
    ;;
32B)
    IS_MOE=0
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-32B}"
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    INTERMEDIATE_SIZE=25600
    NUM_ATTN_HEADS=64
    NUM_QUERY_GROUPS=8
    EMBED_OPT="--untie-embeddings-and-output-weights"

    EP=1
    SEQ_LEN="${SEQ_LENGTH:-${SEQ_LEN:-2048}}"
    MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
    LR=${LR:-1e-5}
    MIN_LR=${MIN_LR:-0.0}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    DO=false
    USE_FSDP2=true
    CKPT_FORMAT=${CKPT_FORMAT:-torch_dist}
    GA_FUSION=false
    AC=${AC:-full}
    RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-64}
    CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"
    PAD_LEN=$SEQ_LEN
    ;;
4B)
    IS_MOE=0
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-4B}"
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    INTERMEDIATE_SIZE=9728
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=8
    EMBED_OPT=""

    EP=1
    SEQ_LEN=${SEQ_LEN:-8192}
    MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-8192}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
    LR=${LR:-1e-5}
    MIN_LR=${MIN_LR:-0.0}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    DO=true
    USE_FSDP2=false
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    GA_FUSION=false
    CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"
    PAD_LEN=$SEQ_LEN
    ;;
8B)
    IS_MOE=0
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-8B}"
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=12288
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=8
    EMBED_OPT="--untie-embeddings-and-output-weights"

    EP=1
    SEQ_LEN=${SEQ_LEN:-8192}
    MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-8192}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
    LR=${LR:-1e-5}
    MIN_LR=${MIN_LR:-0.0}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    DO=true
    USE_FSDP2=false
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    GA_FUSION=false
    PAD_LEN=$SEQ_LEN
    ;;
14B)
    IS_MOE=0
    TOKENIZER_MODEL="${HF_MODEL_CKPT:-Qwen/Qwen3-14B}"
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    INTERMEDIATE_SIZE=17408
    NUM_ATTN_HEADS=40
    NUM_QUERY_GROUPS=8
    EMBED_OPT="--untie-embeddings-and-output-weights"

    EP=1
    SEQ_LEN="${SEQ_LENGTH:-${SEQ_LEN:-2048}}"
    MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
    LR=${LR:-1e-5}
    MIN_LR=${MIN_LR:-0.0}
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
    DO=true
    USE_FSDP2=false
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    GA_FUSION=false
    CE_FUSION_ARGS="--cross-entropy-fusion-impl te --cross-entropy-loss-fusion"
    PAD_LEN=$SEQ_LEN
    ;;
*)
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE"
    echo "Choose one of: 235B_A22B, 235B, 30B_A3B, 30B, 32B, 4B, 8B, 14B"
    exit 1
    ;;
esac

# torch FSDP2: see megatron training/arguments.py (not compatible with overlap-param-gather w/o dist optim)
if [ "$USE_FSDP2" = true ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
fi

if [ "$ETP" -eq 1 ]; then
    if [ "$TP" != "$EP" ] && [ "$TP" -ne 1 ] && [ "$IS_MOE" -eq 1 ]; then
        echo "Note: for MoE with ETP=1, TP and EP are typically equal when TP>1 (see train_deepseekv3)."
    fi
fi

echo "TOKENIZER_MODEL: $TOKENIZER_MODEL"
echo "TRAIN_ITERS: $TRAIN_ITERS LR_WARMUP_ITERS: $LR_WARMUP_ITERS LR_DECAY_ITERS: $LR_DECAY_ITERS"
echo "SEQ_LEN: $SEQ_LEN GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "TP: $TP PP: $PP EP: $EP IS_MOE: $IS_MOE USE_FSDP2: $USE_FSDP2"
echo "PR: $PR"
if [ "$PR" = fp8 ]; then
    echo "FP8_RECIPE: $FP8_RECIPE"
fi
echo ""

if [ "$NVTE_CK_USES_BWD_V3" -eq 1 ]; then
    export NVTE_CK_USES_BWD_V3=1
    export NVTE_CK_V3_ATOMIC_FP32=0
    export NVTE_CK_V3_SPEC=1
else
    export NVTE_CK_USES_BWD_V3=0
    export NVTE_CK_V3_ATOMIC_FP32=1
    export NVTE_CK_V3_SPEC=0
fi

if [ "$GPT_LAYER_IN_TE" = true ]; then
    TRANSFORMER_IMPL=transformer_engine
else
    TRANSFORMER_IMPL=local
fi

if [ -n "${PP_LAYOUT:-}" ] && [ -n "${MP_VP:-}" ]; then
    echo "Error: PP_LAYOUT and MP_VP are mutually exclusive (see megatron --pipeline-model-parallel-layout)."
    exit 1
fi

if [ -z "${MP_VP:-}" ]; then
    vp_options=""
else
    vp_options=" --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

pp_layout_suffix=""
if [ -n "${PP_LAYOUT:-}" ]; then
    pp_layout_suffix=" --pipeline-model-parallel-layout ${PP_LAYOUT}"
fi

pao_options=""
if [ "${USE_PRECISION_AWARE_OPTIMIZER:-false}" = true ] || [ "${USE_PRECISION_AWARE_OPTIMIZER:-0}" = 1 ]; then
    MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-bf16}
    EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-bf16}
    EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-bf16}
    pao_options=" --use-precision-aware-optimizer --main-grads-dtype ${MAIN_GRADS_DTYPE} --exp-avg-dtype ${EXP_AVG_DTYPE} --exp-avg-sq-dtype ${EXP_AVG_SQ_DTYPE}"
fi

# FP8 MoE requires TE grouped GEMM; legacy grouped_gemm does not implement FP8 expert matmuls.
if [ "$PR" = fp8 ]; then
    if [ "${MOE_USE_LEGACY_GROUPED_GEMM:-false}" = true ]; then
        echo "[INFO] PR=fp8: disabling legacy grouped GEMM (TE grouped GEMM required for FP8 MoE)."
    fi
    MOE_USE_LEGACY_GROUPED_GEMM=false
fi

# MoE options (Qwen3 MoE; DeepSeek-specific MLA flags are not used)
if [ "$IS_MOE" -eq 1 ]; then
    if [ "$MOE_PERMUTE_FUSION" != false ]; then
        moe_permute_fusion_options=" --moe-permute-fusion"
    else
        moe_permute_fusion_options=""
    fi
    moe_options=" \
        --num-experts ${NUM_EXPERTS} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-topk ${ROUTER_TOPK} \
        --moe-router-dtype fp32 \
        --moe-aux-loss-coeff ${MOE_AUX_LOSS} \
        --moe-router-load-balancing-type aux_loss \
        --expert-model-parallel-size ${EP} \
        --expert-tensor-parallel-size ${ETP} \
        ${moe_permute_fusion_options} \
    "
    if [ "$USE_GROUPED_GEMM" = true ]; then
        moe_options="${moe_options} --moe-grouped-gemm"
    fi
    if [ $MOE_USE_LEGACY_GROUPED_GEMM = true ]; then
        moe_options="${moe_options} --moe-use-legacy-grouped-gemm"
    else
        # disable gemm tuning when using TE Group GEMM.
        GEMM_TUNING=0
        echo "[WARN] GEMM tuning is disabled when using TransformerEngine Group GEMM."
    fi

    if [ "$ENABLE_DEEP_EP" = true ]; then
        moe_options="${moe_options} --moe-token-dispatcher-type flex --moe-enable-deepep"
    else
        moe_options="${moe_options} --moe-token-dispatcher-type alltoall"
    fi

    # MXFP8: pad per-expert token counts for MXFP8 grouped GEMM (see --moe-router-padding-for-fp8 in arguments.py).
    if [ "$PR" = fp8 ] && [ "${FP8_RECIPE:-delayed}" = mxfp8 ]; then
        moe_options="${moe_options} --moe-router-padding-for-fp8"
        echo "[INFO] MXFP8 MoE: --moe-router-padding-for-fp8"
    fi
else
    moe_options=""
fi

TP_COMM_OVERLAP=0
# --overlap-param-gather requires distributed optimizer (or megatron FSDP), not MCore FSDP2 alone.
if [ "$USE_FSDP2" = true ]; then
    comm_overlap_option=" \
        --overlap-grad-reduce \
        --ddp-bucket-size 629145600"
else
    comm_overlap_option=" \
        --overlap-grad-reduce \
        --ddp-bucket-size 629145600 \
        --overlap-param-gather"
fi

if [ "$AC" = full ]; then
    activation_checkpoint_options=" \
        --recompute-method ${RECOMPUTE_METHOD} \
        --recompute-granularity full \
        --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [ "$AC" = sel ]; then
    activation_checkpoint_options=" --recompute-activations"
else
    activation_checkpoint_options=""
fi

if [ "$GEMM_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
    unset TE_HIPBLASLT_TUNING_RUN_COUNT
    unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi

if [ "$PR" = fp16 ]; then
    pr_options=" --fp16 --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ "$PR" = bf16 ]; then
    pr_options=" --bf16"
elif [ "$PR" = fp8 ]; then
    TRANSFORMER_IMPL=transformer_engine
    case "$FP8_RECIPE" in
    delayed)
        pr_options=" \
        --bf16 \
        --fp8-recipe delayed \
        --fp8-format hybrid \
        --fp8-margin 0 \
        --fp8-interval 1 \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --attention-softmax-in-fp32"
        ;;
    tensorwise)
        pr_options=" \
        --bf16 \
        --fp8-recipe tensorwise \
        --fp8-format hybrid"
        ;;
    mxfp8)
        pr_options=" \
        --bf16 \
        --fp8-recipe mxfp8 \
        --fp8-format e4m3"
        export NVTE_ROCM_ENABLE_MXFP8=1
        ;;
    blockwise)
        pr_options=" \
        --bf16 \
        --fp8-recipe blockwise \
        --fp8-format hybrid"
        ;;
    *)
        echo "Unsupported FP8_RECIPE=${FP8_RECIPE} (use delayed, tensorwise, mxfp8, or blockwise)."
        exit 1
        ;;
    esac
fi

if [ "$DO" = true ]; then
    do_options=" --use-distributed-optimizer"
else
    do_options=""
fi

if [ "$SP" = true ] && [ "$TP" -gt 1 ]; then
    sp_options=" --sequence-parallel"
else
    sp_options=""
fi

if [ "$PRETRAIN_CHECKPOINT_PATH" != none ]; then
    load_options=" --load $PRETRAIN_CHECKPOINT_PATH"
else
    load_options=""
fi

if [ "$OPTIMIZER_OFFLOAD" = static ]; then
    offload_option=" --optimizer hybridadam --optimizer-offload-policy static --optimizer-offload-fraction 1.0"
elif [ "$OPTIMIZER_OFFLOAD" = auto ]; then
    offload_option=" --optimizer hybridadam --optimizer-offload-policy auto"
else
    offload_option=""
fi

if [ "$GA_FUSION" = true ]; then
    ga_fusion_opt=""
else
    ga_fusion_opt=" --no-gradient-accumulation-fusion"
fi

fsdp_option=""
if [ "$USE_FSDP2" = true ]; then
    fsdp_option=" --use-torch-fsdp2"
fi

sft_option="--train-mode pretrain"

FP8_RUN_SUFFIX=""
if [ "$PR" = fp8 ]; then
    FP8_RUN_SUFFIX="-fp8recipe-${FP8_RECIPE}"
fi
NAME="${RUN_ENV}-qwen3-${MODEL_SIZE}-lr-${LR}-bs-${MICRO_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}${FP8_RUN_SUFFIX}-tp-${TP}-pp-${PP}-ep-${EP}-ac-${AC}-${TIMESTAMP}"
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"output/${EXPERIMENT}-${NAME}"}
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/"
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint"
TRAIN_LOG=${OUTPUT_BASEPATH}/log/${EXPERIMENT}-${NAME}.log
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
mkdir -p "${TENSORBOARD_DIR}"
echo "OUTPUT_BASEPATH: $OUTPUT_BASEPATH"
echo "TRAIN_LOG: $TRAIN_LOG"
echo ""

# RoPE fusion: Primus enables apply_rope_fusion + enable_experimental; default here stays --no-rope-fusion unless APPLY_ROPE_FUSION is set.
if [ "${APPLY_ROPE_FUSION:-false}" = true ] || [ "${APPLY_ROPE_FUSION:-0}" = 1 ]; then
    qwen_rope_experimental_opts=" --enable-experimental"
else
    qwen_rope_experimental_opts=" --no-rope-fusion"
fi

qwen_base_options=" \
    --use-mcore-models \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --make-vocab-size-divisible-by ${MAKE_VOCAB_DIV} \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --position-embedding-type rope \
    ${qwen_rope_experimental_opts} \
    --qk-layernorm \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --kv-channels ${KV_CHANNELS} \
    --rotary-percent 1.0 \
    --rotary-base ${ROPE_THETA} \
    --rotary-seq-len-interpolation-factor 1 \
    --no-bias-swiglu-fusion \
    ${EMBED_OPT} \
"

megatron_options=" \
    --log-throughput \
    --no-async-tensor-model-parallel-allreduce \
    ${ga_fusion_opt} \
    ${data_args} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --init-method-std 0.008 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataloader-type cyclic \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --max-padding-length ${PAD_LEN} \
    --log-interval 1 \
    --eval-interval 10000 \
    --eval-iters ${EVAL_ITERS} \
    --save-interval ${SAVE_INTERVAL} \
    --ckpt-format ${CKPT_FORMAT} \
    --tensorboard-queue-size 1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-workers 8 \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --distributed-timeout-minutes 60 \
    --eod-mask-loss \
    --attention-backend fused \
    ${qwen_base_options} \
    ${fsdp_option} \
    ${CE_FUSION_ARGS} \
    ${pao_options} \
    ${pp_layout_suffix} \
"

if [ "$PROFILE" = true ]; then
    if [ "$PROFILE_SYNC" = true ]; then
        export HIP_LAUNCH_BLOCKING=1
    fi
    profile_options=" \
        --profile \
        --profile-ranks 0 \
        --use-pytorch-profiler \
        --profile-step-start ${PROFILE_START} \
        --profile-step-end ${PROFILE_END} \
        --moe-router-force-load-balancing"
elif [ "$FORCE_BALANCE" = true ]; then
    profile_options=" --moe-router-force-load-balancing"
else
    profile_options=""
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_PROJECT=${WANDB_PROJECT:-"Qwen3_Pretrain"}
    LOGGING_ARGS="--wandb-project=${WANDB_PROJECT} \
        --wandb-exp-name=qwen3_${MODEL_SIZE} \
        --wandb-save-dir logs/wandb"
else
    LOGGING_ARGS=""
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/pretrain_gpt.py \
    ${megatron_options} ${pr_options} ${load_options} ${activation_checkpoint_options} \
    ${do_options} ${sp_options} ${moe_options} ${offload_option} ${comm_overlap_option} \
    ${sft_option} ${vp_options} ${profile_options} ${LOGGING_ARGS}"

run_cmd="$run_cmd | tee $TRAIN_LOG"
echo "${run_cmd}"
eval "${run_cmd}"
set +x

if [ "$RUN_ENV" = "cluster" ] || { [ "$RUN_ENV" = "slurm" ] && [ "$SLURM_NODEID" = "$((NNODES - 1))" ]; }; then
    cat <<'PY' >"${MEGATRON_PATH}/mean_log_value.py"
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[1:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)
PY

    echo '============================================================================================================'
    grep -Eo 'throughput per GPU [^|]*' "$TRAIN_LOG" | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' >tmp.txt || true
    echo "throughput per GPU: $(python3 "${MEGATRON_PATH}/mean_log_value.py" tmp.txt 2>/dev/null || echo n/a)" | tee -a "$TRAIN_LOG"
    rm -f tmp.txt

    echo '============================================================================================================'
    grep -Eo 'elapsed time per iteration [^|]*' "$TRAIN_LOG" | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' >tmp.txt || true
    echo "elapsed time per iteration: $(python3 "${MEGATRON_PATH}/mean_log_value.py" tmp.txt 2>/dev/null || echo n/a)" | tee -a "$TRAIN_LOG"

    TIME_PER_ITER=$(python3 "${MEGATRON_PATH}/mean_log_value.py" tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
    rm -f tmp.txt
    PERFORMANCE=$(awk -v bs="$GLOBAL_BATCH_SIZE" -v sl="$SEQ_LEN" -v tpi="$TIME_PER_ITER" -v ws="$((NNODES * GPUS_PER_NODE))" 'BEGIN { if (tpi+0 > 0 && ws+0 > 0) printf "%.6f", bs * sl * 1000/ (tpi * ws); else print "n/a" }')
    echo "tokens/GPU/s: $PERFORMANCE" | tee -a "$TRAIN_LOG"
fi
