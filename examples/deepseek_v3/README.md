
# How to run DeepseekV3 using Megatron LM

## 1.Download datasets
```
export DATA_DIR=/home/azureuser/tas-public/data
mkdir -p ${DATA_DIR}/deepseek-datasets
cd ${DATA_DIR}/deepseek-datasets
# get raw data
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
```

## 2. Pull docker image and run docker container interactively 
```
docker pull rocm/megatron-lm:latest
docker run -d \
  --name=test_deepseek_v3 \
  --network=host\
  --device /dev/dri \
  --device=/dev/kfd \
  --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /path/to/Megatron-LM:/workspace/Megatron-LM \
  -v /home/azureuser/tas-public:/home/azureuser/tas-public \
  rocm/megatron-lm:latest sleep infinity 

docker ps
docker exec -it test_deepseek_v3 bash

```


##  3. Download DeepSeekV3Tokenizer tokenizer (Onetime only)
```shell
export HF_HOME=/home/azureuser/tas-public/huggingface
python download_tokenizer.py
```

download_tokenizer.py
```python
import os
from transformers import AutoTokenizer

access_token = "your_huggingface_access_token"
model_name = "deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

```

## 4. test Model training
```   
FORCE_BALANCE=true \
RUN_ENV=cluster \
HF_HOME=/home/azureuser/tas-public/huggingface \
DATA_DIR=/home/azureuser/tas-public/data \
MODEL_SIZE=16B \
TRAIN_ITERS=10 \
SEQ_LEN=4096 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=16 \
PR=bf16 \
TP=1 PP=1 ETP=1 EP=8 \
GEMM_TUNING=1 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee log.txt
```

## 5. Grouped gemm libraries (Onetime only)
If the grouped gemm has not been installed in the docker images. We can directly install it. Currently there are three ways of grouped gemm implementation.

### 5.1 Grouped gemm-CK from Soli.AI
CK backend should work with the current megatron main branch. 
CK should also be the default, but to be sure we can set HIP_BACKEND=CK as the main branch already has the multi-backend support merged in. 

CK grouped GEMM was referenced / tested in https://github.com/AMD-AIG-AIMA/megatron-lm-grok/issues/14
  https://github.com/AMD-AIG-AIMA/megatron-lm-grok/pull/21/files  . I would check those. 
  We have used the following make target:
```
git clone https://<yourGithubToken>@github.com/AMD-AIG-AIMA/grouped-gemm-ck.git
cd ../grouped-gemm-ck && \
pip install --no-build-isolation -v -e .
```

### 5.2 Grouped gemm third party library support rocm and hipblasLt from Alibaba
```
git clone https://github.com/caaatch22/grouped_gemm.git &&\
    cd grouped_gemm &&\
    git checkout rocm &&\
    git submodule update --init --recursive &&\
    pip install .

or 

git clone https://github.com/caaatch22/grouped_gemm.git &&\
    cd grouped_gemm &&\ 
    git checkout hipblaslt &&\
    git submodule update --init --recursive &&\
    pip install .
Currently hipblast branch has memory access fault
```
### 5.3 Grouped gemm in TE based on hipblasLt
[HIPBLASLT_GROUPED_GEMM](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-5.7.0/usage.html)
[code](https://github.com/ROCm/hipBLASLt/blob/0200ac211b4f080ae41be771d046bbec5b902b13/library/include/hipblaslt-ext.hpp#L42)
[TE group gemm related branch and dockers](https://confluence.amd.com/display/AIG/Codes+and+Docker+Images)


## 6. FAv3 backward kerenel on ROCm
[Link](https://github.com/ROCm/TransformerEngine?tab=readme-ov-file#fa-v3-backward-kernels-in-ck-backend)
ROCm TE provides experimental support for flash-attention v3 bwd kernels using the ck backend for limited fused attention configs (currently only for hdim=128). To enable FA v3 kernels, the following environment variables can be used:

NVTE_CK_USES_BWD_V3 - by default 0, if set to 1, some cases will call the bwd v3 dqdkdv kernel;
NVTE_CK_IS_V3_ATOMIC_FP32 - by default 1, if set to 0 will use atomic fp16/bf16(w/o convert_dq kernel) when NVTE_CK_USES_BWD_V3 is set to 1;
NVTE_CK_IS_V3_SPEC - by default 0, if set to 1 will call the specialized v3 kernel when NVTE_CK_USES_BWD_V3 is set to 1;
NVTE_CK_HOW_V3_BF16_CVT - by default 1, float to bf16 convert type when bwd_v3 is set to 1, 0:RTNE; 1:RTNA; 2:RTZ.

The runing command examples are as follows
```
# disable fav3 backward
NVTE_CK_USES_BWD_V3=0 NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=1 PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg_gt-nofa3back.log

# enable fav3 backward with bf16
NVTE_CK_USES_BWD_V3=1 NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=1 PR=bf16 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_bf16_dryrun_legacygg_gt-fa3back.log

# enable fav3 backward with fp8
NVTE_CK_USES_BWD_V3=1 NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=1 PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg_gt-fa3back.log


```

## 7. Inside the container, you can run deepseek_v3 pre-training with the following command.
```
Note that setting PROFILE=true if you want to profile your training process.  

# Use TE layers + moe grouped gemm (TE version by default) + TE
cd /workspace/Megatron-LM-ROCm; && \
NUM_OF_LAYERS=4 RUN_ENV=localhost PR=bf16 PP=1 EP=8 USE_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_bf16_dryrun_tegg.log

NUM_OF_LAYERS=4 RUN_ENV=localhost PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_tegg.log

# Use TE layers + third party library moe grouped gemm
cd /workspace/Megatron-LM-ROCm; && \
NUM_OF_LAYERS=4 RUN_ENV=localhost PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg.log

cd /workspace/Megatron-LM-ROCm; && \
TP=2 NUM_OF_LAYERS=4 RUN_ENV=localhost PR=fp8 PP=1 EP=4 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg.log

# gemm tuning 

NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=1 PR=bf16 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_bf16_dryrun_legacygg_gt.log


NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=1 PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg_gt.log

# no gemm tuning 

NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=0 PR=bf16 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_bf16_dryrun_legacygg_nogt.log

NUM_OF_LAYERS=4 RUN_ENV=localhost GEMM_TUNING=0 PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true GPT_LAYER_IN_TE=true bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun_legacygg_nogt.log


# use legeacy Mcore layers instead of TE layers
NUM_OF_LAYERS=4 RUN_ENV=localhost PR=fp8 PP=1 EP=8 USE_GROUPED_GEMM=false GPT_LAYER_IN_TE=false bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee output/train_dsv3_fp8_dryrun.log
```

 
## 8. To further tune the performance, we can play with these flags and other ENV variables to tune proformance:
```
export NCCL_IB_TC=41
export NCCL_IB_SL=0
export NCCL_SOCKET_IFNAME=ens51f0np0
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
```