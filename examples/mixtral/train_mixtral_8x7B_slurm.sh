#!/bin/bash

export batch_size_per_node=32 # Set the batch size per node, change it to your own value
export GBS=$(( SLURM_NNODES * batch_size_per_node ))
echo "GBS:" $GBS
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

export NETWORK_INTERFACE="bond0" # Change this to your network interface name

# # Define the mount points
export HOST_MOUNT="/path/to/Megatron-LM" # Before run, change it to your own path
export CONTAINER_MOUNT="/workspace/dev" # Before run, change it to your own path
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
 NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE} GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE} \
 RECOMPUTE_NUM_LAYERS=0 \
 TEE_OUTPUT=1 MBS=2 GBS=${GBS} TP_SIZE=1 PP_SIZE=1 AC=none \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true \
 RUN_ENV=slurm MODEL_SIZE=8x7B bash examples/mixtral/train_mixtral_moe.sh 2>&1 | tee result_8X7B.log; \
 echo $(date)"'


