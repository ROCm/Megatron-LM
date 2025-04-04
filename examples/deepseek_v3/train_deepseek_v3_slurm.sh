#!/bin/bash

echo "get first node"
# Get the list of nodes and the first node (master node)
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
node_array=(${node_list})
master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_node
export SLURM_MASTER_PORT=29508

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"
# Define the Docker image
export DOCKER_IMAGE="rocm/megatron-lm:latest"
# Pull docker image
docker pull $DOCKER_IMAGE
# # Define the mount points
export HOST_MOUNT="/path/to/Megatron-LM" # Before run, change it to your own path where Megatron-LM locats
export CONTAINER_MOUNT="/workspace/dev"

# Run the Docker container with the script
bash -c 'docker stop $(docker ps -q); \
  module load rocm ;\
  docker run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env "SLURM_PROCID=$SLURM_PROCID" \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date); \
 cd $CONTAINER_MOUNT/Megatron-LM; \
 pip install $CONTAINER_MOUNT/data/transformer_engine-1.12.0.dev0+f6072c4-cp310-cp310-linux_x86_64.whl; \
 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 \
FORCE_BALANCE=true \
RUN_ENV=slurm \
HF_HOME=/workspace/dev/huggingface \
DATA_DIR=/workspace/dev/data \
MODEL_SIZE=671B \
TRAIN_ITERS=10 \
NUM_LAYERS=61 \
SEQ_LEN=4096 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 \
PR=bf16 \
AC=full \
TP=2 PP=16 ETP=1 EP=4 \
GEMM_TUNING=0 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee log_deepseek-v3.txt; \
echo $(date)"'


