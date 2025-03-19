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
    -v .:/workspace/Megatron-LM \
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

in `/workspace/Megatron-LM` to install megatron package.

**Note** that it is also possible to use `rocm/megatron-lm:latest` like images, which
have ROCm/Megatron-LM already installed. If doing so, the bind mount is not required,
there is no need to install anything and please make sure to follow the README inside
the container to run these examples.

---

## 2. How to Run

### 2.1 Single Node Training
To run training on a single node, go to ROCm/Megatron-LM repository root and run
the command

```bash
./examples/llama/train_llama2.sh
```

and similarly for `examples/llama/train_llama3.sh`. 

For either script, to run training with non-default options, simply add arguments as
shown in

```bash
MBS=2 BS=16 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=8192 ./examples/llama/train_llama3.sh
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

where `NNODES` sets the number of nodes, `NODE_RANK` is a variable indicating the rank
of the node (with zero reserved for the master node) and `MASTER_ADDR=address` sets the
master node ip address to `address`. 

**Note** that for multi-node runs, remember to properly setup a bind mount, with the
default mount point `/root/cache` inside the container, to a host directory accessible
to all of the nodes, for example a NFS directory. For non-default mount points, set
`DATA_CACHE_PATH` appropriately and pass it to the training scripts.

## 3. Configurations

### 3.1 Network
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

**Note** that for multi-node runs, make sure that the correct network drivers are
available. Either install the drivers inside the docker container or pass the network
drivers from the host (e.g. with `--device /dev/infiniband`) when launching the
container.

Specify which RDMA interfaces to use for communication. Update 

```bash
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
```

in your training script to match available interfaces.

### 3.2 Dataset
You can use either mock data or real data for training.

When preparing a real dataset, a tokenizer is required. The scripts support tokenizers
which are fully specified with choices of `TOKENIZER_TYPE` and `TOKENIZER_MODEL`. With
the exception of Llama 2 training script, the default `TOKENIZER_TYPE` is
`HuggingFaceTokenizer` and for it, only a valid `TOKENIZER_MODEL` is needed. For
example, after obtaining a permission, run

```bash
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/special_tokens_map.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/special_tokens_map.json
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer.json
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer.model https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/original/tokenizer.model
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer_config.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer_config.json
```

with a valid `HF_TOKEN` to download Llama 3.1 tokenizer and pass the path of
`tokenizer` as `TOKENIZER_MODEL` to use it.

**Note** that while the training scripts support default tokenizers, the user is
adviced to be explicit about their tokenizer choice.

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

## 4. Key Variables to Pay Attention To

- **BS:**  
  Sets the global batch size (default: 8)

- **MBS:**  
  Sets the micro batch size (default: 1)

- **SEQ_LENGTH:**  
  Sets the sequence length

- **TP:**  
  Tensor parallel (1, 2, 4, 8). Note `TP` is disabled with `FSDP`.

- **TE_FP8:**  
  `0` for B16 (default), `1` for FP8.

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
  Set to `7` or `70` for Llama2, and `8` or `70` for Llama3/3.1 (default: 70).

- **TOTAL_ITERS:**  
  Sets the total number of iterations.

--- 