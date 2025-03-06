# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script
to train Llama2 or Llama3 models.

---

## 1. Environment Setup

Start a Docker container by running

```
docker run \
    -it --rm \
    --device /dev/dri --device /dev/kfd \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v .:/root/Megatron-LM \
    --shm-size 64G \
    rocm/pytorch-training:latest bash
```

from ROCm/Megatron-LM repository root.

**Note** that it is recommended to use `rocm/pytorch-training:latest` like images which
have most requirements setup, for example `PyTorch >= 2.5.0` is needed for full support
of FSDP-v2.

Run

```
pip install .
```

in `/root/Megatron-LM` to install megatron package.

---

## 2. How to Run

### 2.1 Single Node Training
To run training on a single MI300X node, go to ROCm/Megatron-LM repository root and run
the command

```bash
./examples/llama/train_llama2.sh
```

and similarly for `examples/llama/train_llama3.sh`. 

For either script, to run training with non-default options, for example `FSDP-v2`
enabled, simply add (in this case) `FSDP=1` argument as shown in

```bash
FSDP=1 ./examples/llama/train_llama3.sh
```

**Note** that it is suggested to use `TP=1` when FSDP is enabled, for higher throughput.
And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's
distributed optimizer, gradient accumulation fusion and fp16.

### 2.2 Multi-node Training
To manually run training on N nodes: launch a container on each node, setup the
required network-related environment variables (see Section 3.1) and run

- **On the Master Node:**

  ```bash
  MASTER_ADDR=address NNODES=N NODE_RANK=0 ./examples/llama/train_llama2.sh
  ```

- **On Worker Node i:**

  ```bash
  MASTER_ADDR=address NNODES=N NODE_RANK=i ./examples/llama/train_llama2.sh
  ```

where `address` is the master node ip address.

## 3. Configurations in Scripts

### 3.1 Network Interface
Update the network interface in the training scripts to match your system’s network
interface. To find your network interface, run

```bash
ip a
```

on host and update

```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
export GLOO_SOCKET_IFNAME=ens50f0np0
```

in the training scripts based on the output.

### 3.2 Dataset
You can use either mock data or real data for training.

- **Mock Data:**  
  Mock data is used when no `DATA_PATH` argument is passed. 

- **Downloading real data:**  
  Set argument `DATASET` to the dataset you would like to use: three datasets
  `bookcorpus`, `fineweb` and `wiki` are supported. For example, use the
  following command to download and preprocess the bookcorpus dataset:

  ```bash
  DATASET=bookcorpus DATA_DIR=bookcorpus TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf ./examples/llama/prepare_dataset.sh
  ```

  where `TOKENIZER_MODEL` can be any accessible HuggingFace tokenizer. Remember to
  either pre-download the tokenizer or setup HuggingFace access otherwise when needed.

- **Real Data:**  
  When training, real data is retrieved from `DATA_PATH` argument, for example
  bookcorpus data can be used with

  ```bash
  DATA_PATH=bookcorpus/data_text_document TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf ./examples/llama/train_llama2.sh 
  ```

  **Note** that when training you need to set `DATA_PATH` to the specific file name
  prefix that is pointing to `.bin` or `.idx` file. Remember also to be consistent with
  the choice of the tokenizer.

---

## 4. Key Variables to Pay Attention To

- **TE_FP8:**  
  `0` for BP16 (default), `1` for FP8.

- **GEMM_TUNING:**  
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**  
  `1` to enable Flash Attention.

- **FSDP:**  
  `1` to enable torch fsdp-v2. 
  
  Note that if FSDP is enabled, `--use-distributed-optimizer`, `--overlap-param-gather`, `--sequence-parallel` will be automatically set off. 

- **ENABLE_PROFILING:**  
  `1` to enable PyTorch profiling for performance analysis.

- **MODEL_SIZE:**  
  Set to `7` or `70` for Llama2, and `8` or `70` for Llama3/3.1.

- **TOTAL_ITERS:**  
  Set the total number of iterations (default: 10).

--- 
