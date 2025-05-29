#!/bin/bash
#SBATCH --job-name=mixtral-8X7B-train
#SBATCH --output=logs/slurm/mixtral-8X7B-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=128 
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --partition=byte                     # Adjust to your partition

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
export SLURM_MASTER_PORT="${SLURM_MASTER_PORT:-29475}"


# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"
# Define the Docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-"rocm/megatron-lm:v25.5_py310"}

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

srun bash -c 'docker stop $(docker ps -aq)'
srun bash -c 'rocm-smi'

# Setting the output directory
OUTPUT_DIR="$HOME/output"
mkdir -p $OUTPUT_DIR

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


# Pull docker image
${container_command} pull $DOCKER_IMAGE

export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"ens51np0"} # Can be get by run `ip a` 

MEGATRON_DIR=${PWD}
# Define the dataset path. Before each run, change the following paths accordingly
export HOST_MOUNT=${HOME}                  # change the path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${HOME}           # change the path to workspace developing path inside the docker
export MEGATRON_DIR=${MEGATRON_DIR:-"${MEGATRON_DIR}"} # change the path to Megatron-LM inside the docker
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${CONTAINER_MOUNT}/tokenizer/tokenizer.model"}   # change the tokenizer path accordingly
export DATA_DIR=${DATA_DIR:-"${CONTAINER_MOUNT}/dataset"}                # change the path to dataset location
export WANDB_API_KEY=${WANDB_API_KEY:-}
export OVERLAP_MOE_A2A=${OVERLAP_MOE_A2A:-false}
export PP_SIZE=1
if [ $OVERLAP_MOE_A2A != false ]; then
    echo "Setting OVERLAP_MOE_A2A to true enables pipeline parallelism, with default PP_SIZE=2"
    PP_SIZE=2
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
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 '${mount_roce}' \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date); \
 cd $HOME; \
 pwd; \
 source output/install_packages.sh; \
 source output/megatron_env.sh; \
 cd $MEGATRON_DIR; \
 pwd; \
 export NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1; \
 TOKENIZER_MODEL=${TOKENIZER_MODEL} \
 DATA_DIR=${DATA_DIR} \
 NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE} GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE} \
 RECOMPUTE_NUM_LAYERS=0 \
 TEE_OUTPUT=1 MBS=2 GBS=${GBS} TP_SIZE=1 PP_SIZE=${PP_SIZE} AC=none \
 MOE_PERMUTE_FUSION=true OVERLAP_MOE_A2A=${OVERLAP_MOE_A2A} \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true \
 RUN_ENV=slurm MODEL_SIZE=8x7B bash examples/mixtral/train_mixtral_moe.sh 2>&1 | tee result_8X7B.log; \
 echo $(date)"'
