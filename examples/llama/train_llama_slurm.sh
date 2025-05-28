#!/bin/bash
#SBATCH --job-name=llama-train
#SBATCH --output=logs/slurm/multinode-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=128                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --partition=byte # modify based on your reservation settings

# Determine MASTER_ADDR and MASTER_PORT
MASTER_ADDR=$(srun --ntasks=1 hostname | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29475}"

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

srun bash -c '${container_command} stop $(${container_command} ps -aq)'
srun bash -c 'rocm-smi'

# Setting the output directory
OUTPUT_DIR="$HOME/output"
mkdir -p $OUTPUT_DIR
mkdir -p $HOME/cache

# This is the directory on the host machine which has ROCE libs
ROCELIB_DIR=$(dirname $(find /opt -type f -name "libbnxt_re-*tar.gz" 2>/dev/null | head -n 1))

# Install required packages
# For broadcom check for the ROCE library on the host and then compile the same
# For Mellanox, compiling ROCE library is not required
echo '
#apt install iproute2 -y
#apt install -y linux-headers-"$(uname -r)" libelf-dev
#apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev

if lspci | grep -i "ethernet" | grep -qi "broadcom"; then
    echo -e "\n\n============Compiling RoCE Lib now============\n\n"
    cd '${ROCELIB_DIR}'
    tar -xf libbnxt_re-*.tar.gz
    export LIBNXT_DIR=$(find . -type d -name "libbnxt*" 2>/dev/null)
    cd $LIBNXT_DIR
    sh autogen.sh
    ./configure
    make
    find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \;
    make install all
    sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
    ldconfig
    cp -f bnxt_re.driver /etc/libibverbs.d/
    find . -name "*.so" -exec md5sum {} \;
    BUILT_MD5SUM=$(find . -name "libbnxt_re-rdmav*.so" -exec md5sum {} \; | cut -d " " -f 1)
    echo -e "\n\nmd5sum of the built libbnxt_re is $BUILT_MD5SUM"
    echo -e "\n\n===================RoCE userlib compile complete===================\n\n"
    export NCCL_IB_GID_INDEX=3
fi

' > $OUTPUT_DIR/install_packages.sh

# Environment variables
echo 'export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH' > $OUTPUT_DIR/megatron_env.sh

# We need to mount the dir of roce libs on to the docker
if [ -z $ROCELIB_DIR ]; then
    mount_roce=""
else
    mount_roce="-v$ROCELIB_DIR:$ROCELIB_DIR"
fi


export CONTAINER_NAME="training_env"
export IMAGE=${IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"} # change the docker name accordingly 

export HOST_MOUNT=${HOST_MOUNT:=${HOME}}               # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:=${HOME}}     # change this path to development workspace path inside the docker
export MEGATRON_DIR=${PWD}                             # change this path to Megatron-LM inside the docker
export CONTAINER_DIR=${HOME}
export DATA_DIR=${DATA_DIR:-"${HOME}/.cache/data"}
export HF_TOKEN="${HF_TOKEN:-hf_xxxx}"  
export MODEL_NAME=${MODEL_NAME:-"llama2"}
export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"ens51np0"} # Can be get by run `ip a` 
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Build and launch the Docker container, change podmand command to docker command if the system is using docker instead of podman
srun bash -c '
    ${container_command} pull $IMAGE
    ${container_command} rm $CONTAINER_NAME
    ${container_command} images
    ibdev2netdev
    ${container_command} run -d --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband \
      --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged '${mount_roce}' \
      -v $HOST_MOUNT:$CONTAINER_MOUNT --shm-size 128G --name $CONTAINER_NAME $IMAGE tail -f /dev/null
'

MODEL_SIZE=$1
MBS=$2
BATCH_SIZE_PER_NODE=$3
export BS=$(( SLURM_NNODES * BATCH_SIZE_PER_NODE ))
SEQ_LENGTH=$4
TOTAL_ITERS=$5
FSDP=$6
RECOMPUTE=$7
RECOMPUTE_NUM_LAYERS=$8

echo "MODEL_SIZE: $MODEL_SIZE"
echo "MBS: $MBS"
echo "BS: $BS"
echo "SEQ_LENGTH: $SEQ_LENGTH"
echo "TOTAL_ITERS: $TOTAL_ITERS"
echo "FSDP: $FSDP"
echo "RECOMPUTE: $RECOMPUTE"
echo "RECOMPUTE_NUM_LAYERS: $RECOMPUTE_NUM_LAYERS"


# Execute the training inside the Docker container
srun bash -c '
  ${container_command} exec \
    -e NNODES=$SLURM_JOB_NUM_NODES \
    -e NODE_RANK=$SLURM_NODEID \
    -e MASTER_ADDR='"$MASTER_ADDR"' \
    -e MASTER_PORT='"$MASTER_PORT"' \
    -e NUM_PROCESSES='"$NUM_PROCESSES"' \
    -e MBS='"$MBS"' \
    -e BS='"$BS"' \
    -e SEQ_LENGTH='"$SEQ_LENGTH"' \
    -e MODEL_SIZE='"$MODEL_SIZE"' \
    -e TOTAL_ITERS='"$TOTAL_ITERS"' \
    -e RECOMPUTE='"$RECOMPUTE"' \
    -e RECOMPUTE_NUM_LAYERS='"$RECOMPUTE_NUM_LAYERS"' \
    -e FSDP='"$FSDP"' \
    -e WANDB_API_KEY='"$WANDB_API_KEY"' \
    '"$CONTAINER_NAME"' \
    bash -c "
      echo Inside container: NODE_RANK=\$NODE_RANK
      echo MASTER_ADDR=\$MASTER_ADDR
      echo MASTER_PORT=\$MASTER_PORT
      echo NUM_PROCESSES=\$NUM_PROCESSES
      echo SEQ_LENGTH=\$SEQ_LENGTH
      echo FSDP=\$FSDP
      echo RECOMPUTE=\$RECOMPUTE
      NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE}
      GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE}
      cd $HOME; \
      pwd; \
      source output/install_packages.sh; \
      source output/megatron_env.sh; \
      sudo amd-smi set -g all -p 1
      echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      cd ${MEGATRON_DIR}
      pwd
      ibv_devices
      export NVTE_USE_CAST_TRANSPOSE_TRITON=1
      export NVTE_USE_RMSNORM_TRITON=1
      export OMP_NUM_THREADS=4
      DATA_CACHE_PATH=${CONTAINER_DIR}/cache \
      DATA_DIR=${DATA_DIR} \
      HF_TOKEN=${HF_TOKEN} \
      NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME NODE_RANK=\$NODE_RANK NNODES=\$NNODES MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT MODEL_SIZE=\$MODEL_SIZE TOTAL_ITERS=\$TOTAL_ITERS \
      TEE_OUTPUT=1 MBS=\$MBS BS=\$BS \
      RECOMPUTE=\$RECOMPUTE RECOMPUTE_NUM_LAYERS=\$RECOMPUTE_NUM_LAYERS TE_FP8=0 FSDP=\$FSDP TP=1 \
      SEQ_LENGTH=\$SEQ_LENGTH bash examples/llama/train_${MODEL_NAME}.sh
    "
' 
