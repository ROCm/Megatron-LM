
# Qwen3 Megatron Training

This directory contains `train_qwen3.sh`, a launcher for **Qwen3** dense and MoE pretraining with Megatron Core. It follows the same layout as [`examples/deepseek_v3/train_deepseekv3.sh`](../deepseek_v3/train_deepseekv3.sh): RCCL/NCCL-friendly defaults, `torchrun`, mock or mmap data, and optional DeepEP for MoE.

Supported `MODEL_SIZE` values: `235B_A22B` / `235B`, `30B_A3B` / `30B` (MoE), `32B`, `4B`, `8B`, `14B` (dense). See the script header for architecture sources and default hyperparameters. The `14B` preset matches the common Primus `qwen3_14B` pretrain defaults (e.g. micro batch 4, global batch 32, seq 2048, distributed optimizer, TE cross-entropy fusion).

## 1. Prepare tokenizer

Use the Hugging Face tokenizer for the checkpoint you train or resume from (defaults match each `MODEL_SIZE` preset).

```shell
export HF_HOME=/path/to/huggingface
```

Example: download tokenizer only with Python:

```python
import os
from transformers import AutoTokenizer

access_token = os.environ.get("HF_TOKEN")  # optional, if the repo is gated
model_name = "Qwen/Qwen3-30B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
tokenizer.save_pretrained(os.path.join(os.environ["HF_HOME"], "Qwen3-30B-A3B-tokenizer"))
```

Point training at the model id or local path with `HF_MODEL_CKPT` if it differs from the script default.

## 2. Prepare datasets

For real data, set `MOCK_DATA=0` when launching the script. By default `MOCK_DATA=1` (synthetic data).

Build mmap datasets compatible with your Megatron preprocessing pipeline, then align paths with the script. With `MOCK_DATA=0`, the launcher expects (by default):

```text
${DATA_DIR}/mmap_qwen3_text_document
```

Set `DATA_DIR` to the directory that contains your processed `.bin` / `.idx` prefixes (same stems for train / valid / test as in the script).

## 3. Prepare docker image

Same pattern as DeepSeek-v3:

```shell
docker pull rocm/megatron-lm:latest

docker run -d \
  --name=train_qwen3 \
  --network=host \
  --device /dev/dri \
  --device=/dev/kfd \
  --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /path/to/Megatron-LM:/workspace/Megatron-LM \
  rocm/megatron-lm:latest sleep infinity

docker exec -it train_qwen3 bash
```

## 4. Run Qwen3 pretraining (single node)

Run from the **Megatron-LM repository root** so `pretrain_gpt.py` and `PYTHONPATH` resolve correctly.

### Example: MoE 30B-A3B with mock data

```shell
cd /path/to/Megatron-LM

RUN_ENV=cluster \
MODEL_SIZE=30B_A3B \
TRAIN_ITERS=50 \
MOCK_DATA=1 \
PR=bf16 \
TP=1 PP=1 EP=8 \
GEMM_TUNING=1 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/qwen3/train_qwen3.sh 2>&1 | tee log.txt
```

### Example: MoE 235B-A22B-style preset

```shell
cd /path/to/Megatron-LM

RUN_ENV=cluster \
MODEL_SIZE=235B_A22B \
TRAIN_ITERS=50 \
MOCK_DATA=1 \
FORCE_BALANCE=true \
bash examples/qwen3/train_qwen3.sh 2>&1 | tee log.txt
```

### Example: dense 14B (Primus-style defaults)

```shell
cd /path/to/Megatron-LM

MODEL_SIZE=14B \
TRAIN_ITERS=50 \
MOCK_DATA=1 \
bash examples/qwen3/train_qwen3.sh 2>&1 | tee log.txt
```

### Example: dense 32B (uses `--use-torch-fsdp2`)

The script raises `CUDA_DEVICE_MAX_CONNECTIONS` for the 32B preset (FSDP2 requires a value greater than 1). Override only if you know your stack requires a different setting.

```shell
cd /path/to/Megatron-LM

MODEL_SIZE=32B \
TRAIN_ITERS=50 \
MOCK_DATA=1 \
bash examples/qwen3/train_qwen3.sh 2>&1 | tee log.txt
```

Useful overrides (non-exhaustive): `LR`, `SEQ_LEN`, `PRIMUS_SEQ_LENGTH`, `GLOBAL_BATCH_SIZE`, `EP`, `TP`, `PP`, `ENABLE_DEEP_EP=true`, `WANDB_API_KEY` + `WANDB_PROJECT`, `PRETRAIN_CHECKPOINT_PATH` (set to a checkpoint path instead of `none` to load).

## 5. Multinode training

Set `RUN_ENV=slurm` or keep `RUN_ENV=cluster` and export `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and `NODE_RANK` on each node so `torchrun` can initialize the process group.

For Slurm, you can adapt [`examples/deepseek_v3/train_deepseek_v3_slurm.sh`](../deepseek_v3/train_deepseek_v3_slurm.sh): replace the invoked script with `examples/qwen3/train_qwen3.sh` and pass the same environment variables you use locally.

Adjust `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` for your cluster if multi-node communication fails (the script sets defaults for multi-node runs).
