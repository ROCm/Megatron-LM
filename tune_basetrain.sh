#!/bin/bash

# Prepare the tuning code
# git clone https://github.com/ROCm/pytorch_afo_testkit.git
# cd pytorch_afo_testkit && pip install -e . && cd ..

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

TEE_OUTPUT="${TEE_OUTPUT:-1}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DEVICES_IDS=`python -c "print(' '.join([str(a) for a in range($GPUS_PER_NODE)]))"`

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TOTAL_ITERS="${TOTAL_ITERS:-8}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
TE_FP16="${TE_FP16:-0}"
NORM="${NORM:-RMSNorm}" # RMSNorm gives errors on H100 LayerNorm
BASHFILE="${BASHFILE:-train70b_throughput.sh}"

LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_optim_${OPTIMIZER}_iter${TOTAL_ITERS}/nocompile${NO_TORCH_COMPILE}_TE_FP16_${TE_FP16}/${TIME_STAMP}"
mkdir -p $LOG_DIR

rm -f *.csv

ROCBLAS_DIR=$LOG_DIR
ROCBLAS_FILE="${LOG_DIR}/rocblas.yaml"
ROCBLAS_LOG="${LOG_DIR}/rocblas.log"

echo $LOG_DIR

echo "Start tuning"
TEE_OUTPUT=$TEE_OUTPUT TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_LAYER=4 TOTAL_ITERS=8 MODEL_SIZE=$MODEL_SIZE \
        TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH SEQ_PARALLEL=$SEQ_PARALLEL PYTORCH_TUNABLEOP_ENABLED=0 \
        NO_TORCH_COMPILE=$NO_TORCH_COMPILE USE_FLASH_ATTN=$USE_FLASH_ATTN NO_TRAINING=$NO_TRAINING ENABLE_PROFILING=$ENABLE_PROFILING \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=$NODE_RANK WORLD_SIZE=$WORLD_SIZE \
        LOG_DIR=$LOG_DIR CONTI_PARAMS=$CONTI_PARAMS OPTIMIZER=$OPTIMIZER TE_FP16=$TE_FP16 NORM=$NORM EXP_NAME=gemm_tuning \
        bash $BASHFILE 2>&1 | grep "\- { rocblas_function:" | uniq > $ROCBLAS_FILE
echo "Tuning stopped"

python pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py $ROCBLAS_FILE --cuda_device $DEVICES_IDS >& $ROCBLAS_LOG 

mkdir -p $ROCBLAS_DIR
mv full_tuned*.csv $ROCBLAS_DIR


# =============== search =============== #

rm -f *.csv
sleep 30
TRAIN_LOG="${LOG_DIR}/test_run.log"
TORCH_BLAS_PREFER_HIPBLASLT=0 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH \
        PYTORCH_TUNABLEOP_FILENAME=$ROCBLAS_DIR/full_tuned%d.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_ENABLED=1 NO_TORCH_COMPILE=$NO_TORCH_COMPILE \
        ENABLE_PROFILING=$ENABLE_PROFILING SEQ_PARALLEL=$SEQ_PARALLEL USE_FLASH_ATTN=$USE_FLASH_ATTN NO_TRAINING=$NO_TRAINING  \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=$NODE_RANK WORLD_SIZE=$WORLD_SIZE \
        LOG_DIR=$LOG_DIR CONTI_PARAMS=$CONTI_PARAMS OPTIMIZER=$OPTIMIZER TE_FP16=$TE_FP16 NORM=$NORM \
        bash $BASHFILE > $TRAIN_LOG 2>&1 
