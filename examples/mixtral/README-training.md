# Mixtral 8x7B and 8X22B MoE Model pretraining

## 1. Prepare dataset
```
mkdir -p /path/to/dataset/
cd /path/to/dataset/
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx
```

## 2. Prepare the tokenizer
```
mkdir -p /path/to/tokenizer/mixtral-8x7B
cd /path/to/tokenizer/mixtral-8x7B
# download tokenizer.model from https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/tokenizer.model
```

## 3. Start the training
Start the docker
```
docker run \
 -d \
 --name=mixtral_pretrain \
 --ipc=host \
 --network=host \
 --device=/dev/kfd \
 --device=/dev/dri \
 --cap-add=SYS_PTRACE \
 --cap-add=CAP_SYS_ADMIN \
 --security-opt seccomp=unconfined \
 --group-add video \
 --privileged \
 --device=/dev/infiniband \
 --entrypoint /bin/bash \
 -it docker.io/rocm/megatron-lm:latest sleep infinity
 
```
Enter into the container
```
docker exec -it mixtral_pretrain bash 
```
Start the training script in the docker
```
 RECOMPUTE_NUM_LAYERS=0 \
 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=none \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true \
 RUN_ENV=localhost MODEL_SIZE=8x7B bash examples/mixtral/train_mixtral_moe.sh
```

## 4. Start multinode training in slurm environment

With slurm environment, the pretraining can be launched with the following script.

Note: Before the run, please modify the $HOST_MOUNT, $CONTAINER_MOUNT, $MEGATRON_DIR, $TOKENIZER_MODEL and $DATA_DIR variables, as well as the SBATCH arguments accordingly in the slurm script.     

Mixtral 8X7B
```
  sbatch examples/mixtral/train_mixtral_8x7B_slurm.sh
```

Mixtral 8X22B
```
  sbatch examples/mixtral/train_mixtral_8x22B_slurm.sh
```