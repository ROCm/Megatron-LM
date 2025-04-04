
# DeepSeek-v3 Megatron Training

##  1. Prepare tokenizer
```shell
export HF_HOME=/path/to/huggingface
python download_tokenizer.py
```

download_tokenizer.py
```python
import os
from transformers import AutoTokenizer

access_token = "your_huggingface_token"
model_name = "deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
```

## 2.Prepare datasets
```
export DATA_DIR=/path/to/data
mkdir -p ${DATA_DIR}/deepseek-datasets
cd ${DATA_DIR}/deepseek-datasets

# Get pretrain data
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json

# Process the json file to get idx and bin file
bash tools/run_make_pretraining_dataset_megatron.sh \
/path-to/deepseek-datasets/SlimPajama.json \
DeepSeekV3Tokenizer \
text \
/path-to/deepseek-datasets/ \
/path-to/DeepSeek-V3-Tokenizer 
```


## 3. Prepare docker image
```
docker pull rocm/megatron-lm:latest

docker run -d \
  --name=train_deepseek_v3 \
  --network=host\
  --device /dev/dri \
  --device=/dev/kfd \
  --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /path/to/Megatron-LM:/workspace/Megatron-LM \
  rocm/megatron-lm:latest sleep infinity 

docker exec -it train_deepseek_v3 bash

```

## 4. Run DeepSeek-v3 pretraining
It is suggested to use 16 Nodes like MI300X for full deepseek_v3 model, but it is also possible to reduce NUM_LAYERS to fit the model with fewer number of GPUs. 

```
cd path_to_your_megatron

FORCE_BANLANCE=true \
RUN_ENV=cluster \
HF_HOME=/path/to/huggingface \
DATA_DIR=/path/to/data \
MODEL_SIZE=671B \
TRAIN_ITERS=100000 \
SEQ_LEN=4096 \
NUM_LAYERS=61 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 \
PR=bf16 \
TP=1 PP=1 ETP=1 EP=8 \
GEMM_TUNING=1 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee log.txt
```

## 5. Pretraining DeepSeek-v3 with multinode
We provide an example scipt to enable training at scale under slurm environment.
For example, to run the training on 16 nodes, one can use the following command

```
srun -N 16 --cpus-per-task=$(nproc) --ntasks-per-node=1 bash examples/deepseek_v3/train_deepseek_v3_slurm.sh
```