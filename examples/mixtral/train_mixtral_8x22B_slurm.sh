#!/bin/bash
#SBATCH --job-name=mixtral-8X22B-train
#SBATCH --output=logs/slurm/mixtral-8X22B-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=02:00:00                      # Adjust as necessary
#SBATCH --partition=<your_partition_name> # modify based on your reservation settings

BATCH_SIZE_PER_NODE=$1
export batch_size_per_node=$BATCH_SIZE_PER_NODE # Set the batch size per node, change it to your own value
export GBS=$(( SLURM_NNODES * batch_size_per_node ))
echo "GBS:" $GBS
echo "get first node"
# Get the list of nodes and the first node (master node)
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
node_array=(${node_list})
master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_node
export SLURM_MASTER_PORT="${SLURM_MASTER_PORT:-29475}"


# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"
# Define the Docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}


echo "Trying 'docker ps'..."
if docker ps; then
    echo "Docker is working."
    export container_command=docker
else
    echo "'docker ps' failed. Trying 'podman ps'..."
    if podman ps; then
        echo "Podman is working."
        export container_command=podman
    else
        echo "Both 'docker ps' and 'podman ps' failed."
        exit 1
    fi
fi

sudo amd-smi set -g all -p 1
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Pull docker image
${container_command} pull $DOCKER_IMAGE        # change podman command to docker command if necessary

export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"bond0"} # Can be get by run `ip a` 

MEGATRON_DIR=${PWD}
# Define the dataset path. Before each run, change the following paths accordingly
export HOST_MOUNT=${HOME}                  # change the path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${HOME}           # change the path to workspace developing path inside the docker
export MEGATRON_DIR=${MEGATRON_DIR:-"${MEGATRON_DIR}"} # change the path to Megatron-LM inside the docker
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${CONTAINER_MOUNT}/tokenizer/tokenizer.model"}   # change the tokenizer path accordingly
export DATA_DIR=${DATA_DIR:-"${CONTAINER_MOUNT}/dataset"}                # change the path to dataset location
export WANDB_API_KEY=${WANDB_API_KEY:-}

# if using broadcom network, uncomment the following line to mount the necessary directories
# -v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib \
if [ -d "/etc/libibverbs.d" ]; then
  echo "/etc/libibverbs.d exists and using broadcom."
  export IB_MOUNT_OPTIONS="-v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib"
  
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi

# Run the Docker container with the script
srun bash -c '${container_command} run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env "SLURM_PROCID=$SLURM_PROCID" \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --env WANDB_API_KEY=${WANDB_API_KEY} \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 ${IB_MOUNT_OPTIONS} \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date); \
 cd $MEGATRON_DIR; \
 TOKENIZER_MODEL=${TOKENIZER_MODEL} \
 DATA_DIR=${DATA_DIR} \
 TRAIN_ITERS=20 \
 NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE} GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE} \
 RECOMPUTE_NUM_LAYERS=50 \
 GEMM_TUNING=0 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
 TEE_OUTPUT=1 CP_SIZE=1 MBS=1 GBS=${GBS} TP_SIZE=1 PP_SIZE=1 AC=full MOE_PERMUTE_FUSION=true \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=8192 FORCE_BALANCE=true \
 NVTE_CK_USES_BWD_V3=1 \
 RUN_ENV=slurm MODEL_SIZE=8x22B bash examples/mixtral/train_mixtral_moe.sh 2>&1 | tee result_8X22B.log; \
 echo $(date)"'
