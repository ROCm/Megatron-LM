#!/bin/bash
# Single-GPU smoke test for DeepSeek-V4 training integration.
# Uses mock data and a tiny model config — no tokenizer, no dataset files needed.
# Validates forward + backward + optimizer step without a cluster.
#
# Usage:
#   bash examples/deepseek_v4/train_deepseekv4_smoke.sh
#   TRAIN_ITERS=20 SEQ_LEN=256 bash examples/deepseek_v4/train_deepseekv4_smoke.sh

set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH="$( dirname $( dirname ${CURRENT_DIR} ) )"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# --- Single-GPU distributed setup ---
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

# --- Smoke-test model dims (tiny, CPU-runnable if no GPU) ---
# Keeping dims small so the test finishes in seconds.
HIDDEN_SIZE=256
NUM_LAYERS=2
NUM_ATTN_HEADS=4
FFN_HIDDEN_SIZE=512
MOE_INTERMEDIATE_SIZE=128
Q_LORA_RANK=64
KV_LORA_RANK=64
QK_NOPE_HEAD_DIM=32
QK_ROPE_HEAD_DIM=16
V_HEAD_DIM=32
NUM_EXPERTS=4
ROUTER_TOPK=2
SEQ_LEN=${SEQ_LEN:-128}
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2
TRAIN_ITERS=${TRAIN_ITERS:-5}
VOCAB_SIZE=512

OUTPUT_DIR="${MEGATRON_PATH}/output/dsv4_smoke"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "DeepSeek-V4 smoke test"
echo "  MEGATRON_PATH : ${MEGATRON_PATH}"
echo "  TRAIN_ITERS   : ${TRAIN_ITERS}"
echo "  SEQ_LEN       : ${SEQ_LEN}"
echo "============================================================"

torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "${MEGATRON_PATH}/pretrain_deepseekv4.py" \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --kv-channels ${V_HEAD_DIM} \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --multi-latent-attention \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --bf16 \
    --mock-data \
    --data-cache-path "${OUTPUT_DIR}/.cache" \
    --vocab-size ${VOCAB_SIZE} \
    --tokenizer-type NullTokenizer \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-rope-fusion \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --lr-decay-iters $((TRAIN_ITERS - 2)) \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --no-gradient-accumulation-fusion \
    --log-interval 1 \
    --eval-interval 100 \
    --eval-iters 0 \
    --save-interval 10000 \
    --split 900,50,50 \
    --seed 42 \
    --distributed-backend nccl \
    2>&1 | tee "${OUTPUT_DIR}/smoke.log"

echo "============================================================"
echo "Smoke test complete. Log: ${OUTPUT_DIR}/smoke.log"
echo "============================================================"
