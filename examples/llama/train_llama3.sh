#!/bin/bash
###############################################################################
# Copyright (c) 2024 - 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
#set -x

# set envs 
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
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${EXP_NAME:-perf}"

TEE_OUTPUT="${TEE_OUTPUT:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`

# single node config, Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "${NNODES:-1}" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MAX_POSITION_EMBEDDINGS=131072
TOTAL_ITERS="${TOTAL_ITERS:-12}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
TE_FP8="${TE_FP8:-0}"  # 0: disable FP8, 1: enable FP8
TE_FP8_RECIPE="${TE_FP8_RECIPE:-delayed}" # Options: delayed, tensorwise, mxfp8
TE_FP4="${TE_FP4:-0}"  # 0: disable FP4, 1: enable FP4
TE_FP4_RECIPE="${TE_FP4_RECIPE:-nvfp4}" # Options: nvfp4, mxfp4
FP4_PARAM_GATHER="${FP4_PARAM_GATHER:-0}"
# nvfp4 keeps first/last layers in BF16 (NVFP4 paper recipe); mxfp4 uses all layers in FP4 (MLPerf recipe).
if [ "$TE_FP4_RECIPE" == "mxfp4" ]; then
    FP4_SELECTIVE_BF16="${FP4_SELECTIVE_BF16:-0}"  # mxfp4: all layers in FP4 (set 1 to keep first/last in BF16)
else
    FP4_SELECTIVE_BF16="${FP4_SELECTIVE_BF16:-1}"  # nvfp4: keep first/last layers in BF16 (NVFP4 paper recipe)
fi
FP4_BF16_START="${FP4_BF16_START:-2}"    # Number of layers at start in BF16 (paper: 2)
FP4_BF16_END="${FP4_BF16_END:-8}"        # Number of layers at end in BF16 (paper: 8)

# --- Fusion / data-parallel-comm perf toggles (ON by default; ~+3.5-7% across bf16/fp8/fp4). Set 0 to disable. ---
GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-1}"  # fuse wgrad accumulation into the GEMM
DDP_AVERAGE_IN_COLLECTIVE="${DDP_AVERAGE_IN_COLLECTIVE:-1}"        # fold gradient averaging into the DP collective
CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-1}"        # fused cross-entropy loss
CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-te}"       # native | te (TE fused vocab-parallel cross-entropy)
FUSED_SINGLE_QKV_ROPE="${FUSED_SINGLE_QKV_ROPE:-1}"               # fused QKV+RoPE kernel (TE; asserts if config unsupported, e.g. QK-layernorm)
# NOTE: GRAD_REDUCE_IN_BF16 all-reduces gradients in BF16 instead of the FP32 default, halving DP-reduce
# bandwidth at the cost of some gradient-comm precision. Convergence-neutral: a controlled 2x2 on C4
# (LLaMA3-8B, BS=32, lr 8e-4) showed bf16 converges with this ON or OFF — convergence was gated by adam-eps,
# NOT this toggle. The Megatron default adam-eps=1e-8 plateaus at lr 8e-4; use 1e-5 (the MLPerf llama3.1-8B
# config primus/.../MI355X/llama3.1_8B-pretrain-FP8.yaml sets adam_eps: 1.0e-5). Kept ON as a perf win.
# mxfp4 REQUIRES it ON (a4w4 asm wgrad emits BF16 only) — auto-forced by the guard below.
GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-1}"
# mxfp4 needs a BF16 gradient buffer: the aiter a4w4 asm wgrad GEMM only emits BF16 output, so when wgrad is
# fused into the grad buffer (gradient_accumulation_fusion=1) that buffer must be BF16. If you set
# GRAD_REDUCE_IN_BF16=0 (FP32 buffer) the run crashes ("gemm_a4w4_asm only support BFloat16 output now!").
# Re-force it on for mxfp4; set gradient_accumulation_fusion=0 instead if you must keep an FP32 grad buffer.
if [ "$TE_FP4" -eq 1 ] && [ "$TE_FP4_RECIPE" == "mxfp4" ] && [ "$GRADIENT_ACCUMULATION_FUSION" -eq 1 ] && [ "$GRAD_REDUCE_IN_BF16" -ne 1 ]; then
    echo "mxfp4 + gradient_accumulation_fusion requires GRAD_REDUCE_IN_BF16=1 (a4w4 asm wgrad emits BF16 only); forcing it on."
    GRAD_REDUCE_IN_BF16=1
fi

GEMM_TUNING="${GEMM_TUNING:-1}"
MCORE="${MCORE:-1}"
OPTIMIZER="${OPTIMIZER:-adam}"
FSDP="${FSDP:-0}"
RECOMPUTE="${RECOMPUTE:-0}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-NousResearch/Meta-Llama-3-8B}"
ROPE_FUSION="${ROPE_FUSION:-1}" # 1: use rope-fusion, 0: no-rope-fusion
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_ITERS="${EVAL_ITERS:-'-1'}"
CKPT_FORMAT="${CKPT_FORMAT:-torch}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/root/cache}"
MEGATRON_FSDP="${MEGATRON_FSDP:-0}"
FP8_PARAM_GATHER="${FP8_PARAM_GATHER:-0}"
FP8_TRANSPOSE_CACHE="${FP8_TRANSPOSE_CACHE:-0}"
ENABLE_HSDP="${ENABLE_HSDP:-0}"
HSDP_NUM_REPLICAS="${HSDP_NUM_REPLICAS:-2}"

if [ "$TE_FP8" -eq 1 ] && [ "$TE_FP4" -eq 1 ]; then
    echo "Error: FP8 and FP4 cannot be used simultaneously. Please choose one."
    exit 1
fi

if [ "$FSDP" -eq 1 ] || [ "$MEGATRON_FSDP" -eq 1 ]; then
    unset CUDA_DEVICE_MAX_CONNECTIONS
    # Gradient accumulation fusion is incompatible with FSDP: torch-FSDP2 hard-asserts against it
    # in arguments.py, and Megatron-FSDP crashes at runtime with an fsdp_grads "No buffer found for
    # bucket_id" assertion. Force it off so the default-on toggle does not break the FSDP suites.
    if [ "$GRADIENT_ACCUMULATION_FUSION" -eq 1 ]; then
        echo "FSDP is incompatible with gradient accumulation fusion; disabling fusion (GRADIENT_ACCUMULATION_FUSION=0)."
        GRADIENT_ACCUMULATION_FUSION=0
    fi
    if [ "$TP" -gt 1 ]; then
        echo "It is not recommended to use FSDP and TP together. Disabling TP."
        TP=1
        echo "Resetting TP=$TP"
    fi
fi

if [ "$ENABLE_HSDP" -eq 1 ]; then
    if [ "$MEGATRON_FSDP" -ne 1 ]; then
        echo "Error: HSDP requires MEGATRON_FSDP=1"
        exit 1
    fi
    
    if [ "$HSDP_NUM_REPLICAS" -lt 2 ]; then
        echo "Error: HSDP_NUM_REPLICAS must be >= 2 when ENABLE_HSDP=1."
        exit 1
    fi
fi

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR
# Precision tag for the log directory: bf16 | fp8 | fp8_tensorwise | mxfp8 | mxfp4 | nvfp4
if [ "$TE_FP4" -eq 1 ]; then
    PREC_TAG="$TE_FP4_RECIPE"            # mxfp4 | nvfp4
elif [ "$TE_FP8" -eq 1 ]; then
    case "$TE_FP8_RECIPE" in
        delayed)    PREC_TAG="fp8" ;;
        tensorwise) PREC_TAG="fp8_tensorwise" ;;
        mxfp8)      PREC_TAG="mxfp8" ;;
        *)          PREC_TAG="fp8_${TE_FP8_RECIPE}" ;;
    esac
else
    PREC_TAG="bf16"
fi
DEFAULT_LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_cp${CP}_iter${TOTAL_ITERS}/${PREC_TAG}/${TIME_STAMP}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
TRAIN_LOG="${LOG_DIR}/output_${EXP_NAME}.log"
mkdir -p $LOG_DIR
echo $TRAIN_LOG

# gemm tuning
if [ "$GEMM_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
fi

if [ "$SEQ_LENGTH" -le 8192 ]; then
    ds_works=8
else
    ds_works=24
fi

if [[ $MODEL_SIZE -eq 8 ]]; then #llama3.1-8B
    HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
    FFN_HIDDEN_SIZE=14336 # e.g. llama-13b: 13824
    NUM_LAYERS=32 # e.g. llama-13b: 40
    NUM_HEADS=32 # e.g. llama-13b: 40
    NUM_KV_HEADS=8
elif [[ $MODEL_SIZE -eq 70 ]]; then
    HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
    FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
    NUM_LAYERS=80 # e.g. llama-13b: 40
    NUM_HEADS=64 # e.g. llama-13b: 40
    NUM_KV_HEADS=8 # llama3 70B uses GQA
else
    echo "Model size not supported."
    exit 1
fi

GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))

PROFILING_DIR="${LOG_DIR}/trace_${EXP_NAME}"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --swiglu \
    --use-te-activation-func \
    --no-bias-swiglu-fusion \
    --disable-bias-linear \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --train-iters $TOTAL_ITERS \
    --bf16 \
    --no-masked-softmax-fusion \
"

if [ "$RECOMPUTE" -eq 1 ]; then
    GPT_ARGS="$GPT_ARGS --recompute-num-layers $NUM_LAYERS \
        --recompute-granularity full \
        --recompute-method block \
        "
fi
if [ "$ROPE_FUSION" -eq 0 ]; then
    GPT_ARGS="$GPT_ARGS --no-rope-fusion"
fi

# Fusion / mixed-precision toggles -> GPT_ARGS, matching their arguments.py groups (transformer
# config, training, mixed precision) — same buckets as --no-bias-swiglu-fusion, --train-iters, --bf16.
if [ "$GRADIENT_ACCUMULATION_FUSION" -eq 0 ]; then
    GPT_ARGS="$GPT_ARGS --no-gradient-accumulation-fusion"
fi
if [ "$CROSS_ENTROPY_LOSS_FUSION" -eq 1 ]; then
    GPT_ARGS="$GPT_ARGS --cross-entropy-loss-fusion --cross-entropy-fusion-impl $CROSS_ENTROPY_FUSION_IMPL"
fi
if [ "$FUSED_SINGLE_QKV_ROPE" -eq 1 ]; then
    GPT_ARGS="$GPT_ARGS --fused-single-qkv-rope"
fi
if [ "$GRAD_REDUCE_IN_BF16" -eq 1 ]; then
    GPT_ARGS="$GPT_ARGS --grad-reduce-in-bf16"
fi

TRAIN_ARGS="--lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1.0e-1 \
    --clip-grad 1.0 \
"
if [ "$OPTIMIZER" == "adam" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        "
else
    TRAIN_ARGS="$TRAIN_ARGS --optimizer sgd \
        "
fi

DATA_ARGS="
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type cyclic \
    --save-interval 200000 \
    --tensorboard-dir $LOG_DIR \
    --log-interval 1 \
    --eval-interval 320000 \
    --eval-iters 10 \
    --num-workers $ds_works \
"
if [ -z ${DATA_PATH+x} ]; then
    DATA_ARGS="$DATA_ARGS --mock-data"
    echo "Using Mock data"
else
    DATA_ARGS="$DATA_ARGS --data-path $DATA_PATH"
    echo "Using ${DATA_PATH} data"
fi
if [ "$NNODES" -gt 1 ]; then
    # For multi-node runs DATA_CACHE_PATH should exist and should point to a common
    # path accessible by all the nodes (for example, a NFS directory)"
    DATA_ARGS="$DATA_ARGS --data-cache-path $DATA_CACHE_PATH"
fi

OUTPUT_ARGS="
    --log-interval $LOG_INTERVAL \
    --log-throughput \
    --no-save-optim \
    --no-save-rng \
    --eval-iters $EVAL_ITERS
"
if [ -n "$SAVE_CKPT_PATH" ]; then
    OUTPUT_ARGS="$OUTPUT_ARGS \
        --save-interval $SAVE_INTERVAL \
        --eval-interval $EVAL_INTERVAL \
        --ckpt-format $CKPT_FORMAT \
        --save $SAVE_CKPT_PATH
    "
fi

CKPT_LOAD_ARGS=""
if [ -n "$LOAD_CKPT_PATH" ]; then
    CKPT_LOAD_ARGS="$CKPT_LOAD_ARGS \
        --exit-on-missing-checkpoint \
        --no-load-optim \
        --no-load-rng \
        --use-checkpoint-args \
        --load ${LOAD_CKPT_PATH}"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --distributed-backend nccl \
    --distributed-timeout-minutes 120 \
    --overlap-grad-reduce \
    --rerun-mode disabled \
"

if [ "$FSDP" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-torch-fsdp2"
    if [ "$SEQ_PARALLEL" -eq 1 ]; then
        echo "Warning: Sequence Parallelism and FSDP2 have conflicting CUDA_MAX_CONNECTIONS requirements. It is recommended not to use them together."
        echo "FSDP2 and sequence parallel are on. Disabling sequence parallel."
        SEQ_PARALLEL=0
    fi
else
    if [ "$OPTIMIZER" == "adam" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --use-distributed-optimizer --overlap-param-gather"
    fi
fi

if [ "$ENABLE_PROFILING" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler --tensorboard-dir $LOG_DIR"
fi

if [ "$USE_FLASH_ATTN" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
else
    EXTRA_ARGS="$EXTRA_ARGS --attention-backend fused"
fi

if [ "$SEQ_PARALLEL" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --sequence-parallel"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$MCORE" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-mcore-models"
fi

if [ "$TE_FP8" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
"

    if [ "$TE_FP8_RECIPE" == "delayed" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp8-recipe=delayed \
            --fp8-format=hybrid \
            --fp8-margin=0 \
            --fp8-interval=1 \
            --fp8-amax-history-len=1024 \
            --fp8-amax-compute-algo=max \
            --attention-softmax-in-fp32 \
        "
    elif [ "$TE_FP8_RECIPE" == "mxfp8" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp8-recipe=mxfp8 \
            --fp8-format=e4m3 \
        "
        # TE does not enable mxfp8 by default
        export NVTE_ROCM_ENABLE_MXFP8=1
    elif [ "$TE_FP8_RECIPE" == "tensorwise" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp8-recipe=tensorwise \
            --fp8-format=hybrid \
        "
    else
        echo "$TE_FP8_RECIPE is not supported"
        exit
    fi

    if [ "$FP8_PARAM_GATHER" -eq 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp8-param-gather"

        # MXFP8 + DDP path: TE does not implement `replace_raw_data` for MXFP8Tensor,
        # so the default `_ParamAndGradBuffer` storage swap fails. Reusing the grad
        # buffer for the MXFP8 param all-gather sidesteps that path. The FSDP and
        # Megatron-FSDP paths handle MXFP8 param all-gather natively in TE and do
        # not need this workaround.
        if [ "$TE_FP8_RECIPE" == "mxfp8" ] && [ "$FSDP" -ne 1 ] && [ "$MEGATRON_FSDP" -ne 1 ]; then
            EXTRA_ARGS="$EXTRA_ARGS --reuse-grad-buf-for-mxfp8-param-ag"
        fi
    fi

    if [ "$FP8_TRANSPOSE_CACHE" -eq 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --keep-fp8-weight-transpose-cache-te \
            --keep-fp8-transpose-cache \
        " 
    fi
fi

if [ "$TE_FP4" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
        --fp4-format=e2m1 \
    "

    if [ "$TE_FP4_RECIPE" == "nvfp4" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp4-recipe=nvfp4"
    elif [ "$TE_FP4_RECIPE" == "mxfp4" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp4-recipe=mxfp4"
    else
        echo "$TE_FP4_RECIPE is not supported"
        exit 1
    fi

    if [ "$FP4_SELECTIVE_BF16" -eq 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --first-last-layers-bf16 \
            --num-layers-at-start-in-bf16 $FP4_BF16_START \
            --num-layers-at-end-in-bf16 $FP4_BF16_END \
        "
        echo "FP4 ($TE_FP4_RECIPE): Keeping first $FP4_BF16_START and last $FP4_BF16_END layers in BF16"
    fi
    if [ "$FP4_PARAM_GATHER" -eq 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fp4-param-gather"
    fi
fi

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS="--wandb-project=LLama \
        --wandb-exp-name=LLama_${MODEL_SIZE}B \
        --wandb-save-dir logs/wandb \
    "
else
   LOGGING_ARGS=""
fi

if [ "$MEGATRON_FSDP" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-megatron-fsdp --ckpt-format fsdp_dtensor --data-parallel-sharding-strategy optim_grads_params --fsdp-double-buffer"
    
    if [ "$ENABLE_HSDP" -eq 1 ]; then
        echo "Megatron HSDP is enabled with $HSDP_NUM_REPLICAS DP outer replicas"
        EXTRA_ARGS="$EXTRA_ARGS --num-distributed-optimizer-instances $HSDP_NUM_REPLICAS"
    fi
fi

# DP-comm toggle -> EXTRA_ARGS ('distributed' arg group in arguments.py, alongside --overlap-grad-reduce)
if [ "$DDP_AVERAGE_IN_COLLECTIVE" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --ddp-average-in-collective"
fi

run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        $TRAIN_ARGS \
        $LOGGING_ARGS \
        $CKPT_LOAD_ARGS
"

if [ "$TEE_OUTPUT" -eq 0 ]; then 
    run_cmd="$run_cmd >& $TRAIN_LOG"
else
    run_cmd="$run_cmd |& tee $TRAIN_LOG"
fi

TRAIN_RC=0
if [ "$NO_TRAINING" -eq 0 ]; then
    # pipefail so the run's `... |& tee $TRAIN_LOG` (TEE_OUTPUT=1) returns torchrun's status, not tee's
    # (~always 0). Without it a training crash is masked and the script would exit 0. ($? after `eval`
    # of the pipeline, since `eval` collapses PIPESTATUS to a single element = tee's status.)
    set -o pipefail
    eval $run_cmd
    TRAIN_RC=$?
    set +o pipefail
    if [ "$TRAIN_RC" -ne 0 ]; then
        echo "ERROR: training (torchrun pretrain_gpt.py) failed with exit code $TRAIN_RC" |& tee -a $TRAIN_LOG
    fi
fi


echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[2:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
PERFORMANCE=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
ETPI=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG

# Extract memory usage
grep -Eo 'mem usages: [^|]*' "$TRAIN_LOG" | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a "$TRAIN_LOG"
rm tmp.txt

# Propagate the training exit status (the perf-parse above must not mask a torchrun failure).
exit $TRAIN_RC
