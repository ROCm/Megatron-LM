#!/bin/bash
###############################################################################
# Lumen — Llama2-7B pretrain (BF16 or FP8 delayed/hybrid), native launcher
#
# Run inside a Lumen Docker container (or any env with Megatron + Lumen installed).
# No docker run — invokes torchrun directly.
#
# Usage:
#   PRECISION=bf16 bash run_pretrain_llama2_7b_native.sh
#   PRECISION=fp8  bash run_pretrain_llama2_7b_native.sh   # default
#
#   # Backend toggle (same config, different library) for comparison:
#   LUMEN_BACKEND=lumen bash run_pretrain_llama2_7b_native.sh   # default
#   LUMEN_BACKEND=te    bash run_pretrain_llama2_7b_native.sh   # TransformerEngine
#
#   # Megatron e2e profiling (torch.profiler → tensorboard traces):
#   ENABLE_PROFILING=1 bash run_pretrain_llama2_7b_native.sh
#   ENABLE_PROFILING=1 PROFILE_STEP_START=3 PROFILE_STEP_END=6 PROFILE_DIR=/tmp/prof bash ...
#
# Override MBS / GBS / SEQ_LEN / TRAIN_STEPS / SEED / NGPU via env.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# Shared Megatron entry point lives under examples/llama31.
EX_DIR="${LUMEN_DIR}/examples/llama31"

PRECISION="${PRECISION:-fp8}"          # bf16 | fp8
# Model library: lumen (default) or te (TransformerEngine) — read by
# pretrain_llama31.py to pick the model provider. Same training config either way.
LUMEN_BACKEND="${LUMEN_BACKEND:-lumen}"
export LUMEN_BACKEND
TOKENIZER_DIR="${TOKENIZER_DIR:-${LUMEN_DIR}/examples/llama2/tokenizer}"
# This launcher may live outside the Lumen tree; fall back to a sibling Lumen checkout.
if [ ! -d "${TOKENIZER_DIR}" ] && [ -d /workspace/Lumen/examples/llama2/tokenizer ]; then
    TOKENIZER_DIR=/workspace/Lumen/examples/llama2/tokenizer
fi
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
DATA_DIR="${DATA_DIR:-${RESULTS_DIR}/mock_data}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/mock_train.jsonl}"
LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/${LUMEN_BACKEND}_llama2_7b_${PRECISION}.log}"

MBS="${MBS:-4}"
GBS="${GBS:-256}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
NGPU="${NGPU:-8}"

mkdir -p "${RESULTS_DIR}" "${DATA_DIR}"

# ---- Runtime env (mirrors docker -e flags) -----------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export USE_HIPBLASLT="${USE_HIPBLASLT:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# Lumen fusion flags change transformer numerics/kernels. Keep them OFF for the
# TransformerEngine comparison run so its internals stay 100% TE.
if [ "${LUMEN_BACKEND}" = "lumen" ]; then
    export LUMEN_PREFER_HIPBLASLT="${LUMEN_PREFER_HIPBLASLT:-1}"
    export LUMEN_FUSED_SWIGLU="${LUMEN_FUSED_SWIGLU:-1}"
    export LUMEN_FUSED_RESIDUAL_NORM="${LUMEN_FUSED_RESIDUAL_NORM:-1}"
    export LUMEN_FUSED_RES_BWD="${LUMEN_FUSED_RES_BWD:-1}"
    export LUMEN_SKIP_BACKEND_SYNC="${LUMEN_SKIP_BACKEND_SYNC:-1}"

    if [ "${PRECISION}" = "fp8" ]; then
        export LUMEN_FUSED_QUANT_TRANSPOSE_CPP="${LUMEN_FUSED_QUANT_TRANSPOSE_CPP:-1}"
        export LUMEN_FUSED_QUANT_AMAX="${LUMEN_FUSED_QUANT_AMAX:-1}"
        export LUMEN_FUSED_QUANT_SCALE="${LUMEN_FUSED_QUANT_SCALE:-1}"
        export LUMEN_FUSED_CAST_TRANSPOSE="${LUMEN_FUSED_CAST_TRANSPOSE:-1}"
        export LUMEN_FUSED_CAST_TRANSPOSE_V2="${LUMEN_FUSED_CAST_TRANSPOSE_V2:-1}"
        export LUMEN_FUSED_SWIGLU_QUANT="${LUMEN_FUSED_SWIGLU_QUANT:-1}"
        export LUMEN_FUSED_NORM_QUANT="${LUMEN_FUSED_NORM_QUANT:-1}"
        export LUMEN_FUSED_NORM_QUANT_V2="${LUMEN_FUSED_NORM_QUANT_V2:-1}"
        export LUMEN_TRANSPOSE_CACHE="${LUMEN_TRANSPOSE_CACHE:-1}"
        export LUMEN_FAST_QUANT_DISPATCH="${LUMEN_FAST_QUANT_DISPATCH:-1}"
        export LUMEN_WEIGHT_QUANT_ONCE="${LUMEN_WEIGHT_QUANT_ONCE:-1}"
    fi
fi

# TransformerEngine attention backend. With no standalone flash-attn package
# installed, TE's FlashAttention backend is unavailable and it falls back to the
# CK FusedAttention backend (slow ck_tile FMHA backward). Setting
# NVTE_FLASH_ATTN_AITER=1 makes TE import AITER's Triton flash-attention and
# masquerade it as flash-attn 2.7.1; on ROCm TE selects FlashAttention over
# FusedAttention, so this routes attention through AITER. Set TE_AITER_ATTN=0 to
# keep stock TE CK attention.
if [ "${LUMEN_BACKEND}" = "te" ]; then
    export NVTE_FLASH_ATTN_AITER="${NVTE_FLASH_ATTN_AITER:-${TE_AITER_ATTN:-1}}"
fi

# Make the in-repo `lumen` package importable regardless of CWD (the Docker
# image baked this in via ENV PYTHONPATH=/workspace/Lumen).
export PYTHONPATH="${LUMEN_DIR}:${PYTHONPATH:-}"

# AITER lives in third_party/aiter; Docker installs it editable at build time.
AITER_DIR="${AITER_DIR:-${LUMEN_DIR}/third_party/aiter}"
if [ -d "${AITER_DIR}/aiter" ]; then
    export PYTHONPATH="${AITER_DIR}:${PYTHONPATH:-}"
fi

export MBS GBS SEQ_LEN TRAIN_STEPS SEED

# ---- Backend-specific + FP8 flags -------------------------------------------
# Lumen-only flags (attention backend + Lumen FP8 pipeline). In TE mode we drop
# these and rely on Megatron's native --fp8-format to drive TransformerEngine FP8.
BACKEND_ARGS=()
FP8_ARGS=()
if [ "${LUMEN_BACKEND}" = "lumen" ]; then
    BACKEND_ARGS=(--lumen-attn-backend csrc)
    if [ "${PRECISION}" = "fp8" ]; then
        FP8_ARGS=(
            --linear-fp8
            --fp8-format hybrid
            --linear-fp8-scaling delayed
            --linear-fp8-amax-algo max
            --linear-fp8-amax-history 1024
        )
    fi
else
    # TransformerEngine: use Megatron's native FP8 knobs only.
    if [ "${PRECISION}" = "fp8" ]; then
        FP8_ARGS=(
            --transformer-impl transformer_engine
            --fp8-format hybrid
        )
        # TE_AITER_ATTN=0 → CK fused / ASM v3 FMHA (the te_asm baseline).
        # Megatron's current default is AttnBackend.auto, which can miss fused.
        if [ "${TE_AITER_ATTN:-1}" = "0" ]; then
            FP8_ARGS+=(--attention-backend fused)
        fi
    else
        BACKEND_ARGS=(--transformer-impl transformer_engine)
    fi
    # Route SwiGLU through TE's fused activation kernels. This is mutually
    # exclusive with Megatron's bias+swiglu fusion, so disable that.
    if [ "${TE_ACT_FUSE:-1}" = "1" ]; then
        BACKEND_ARGS+=(--use-te-activation-func --no-bias-swiglu-fusion)
    fi
fi

# ---- Megatron e2e profiling (torch.profiler → tensorboard) -------------------
# Opt-in via ENABLE_PROFILING=1. Captures CPU+CUDA activity on profile_ranks
# for iterations [profile_step_start, profile_step_end). Defaults skip warmup
# iters 1-2 and profile steps 3-5 (3 active steps).
PROFILE_ARGS=()
if [ "${ENABLE_PROFILING:-0}" = "1" ]; then
    PROFILE_DIR="${PROFILE_DIR:-${RESULTS_DIR}/profile}"
    PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
    PROFILE_STEP_END="${PROFILE_STEP_END:-6}"
    PROFILE_RANKS="${PROFILE_RANKS:-0}"
    mkdir -p "${PROFILE_DIR}"
    # shellcheck disable=SC2206
    _profile_ranks_arr=(${PROFILE_RANKS})
    PROFILE_ARGS=(
        --profile
        --use-pytorch-profiler
        --profile-step-start "${PROFILE_STEP_START}"
        --profile-step-end "${PROFILE_STEP_END}"
        --profile-ranks "${_profile_ranks_arr[@]}"
        --tensorboard-dir "${PROFILE_DIR}"
    )
fi

# Resolve a FULL Megatron-LM checkout (one that provides megatron/training).
# The pip-installed `megatron-core` only ships megatron/core, so `megatron.training`
# is unavailable unless a full Megatron-LM is on PYTHONPATH.
if [ -z "${MEGATRON_ROOT:-}" ]; then
    for _cand in /workspace/megatron_lm /workspace/Megatron-LM; do
        if [ -d "${_cand}/megatron/training" ]; then
            MEGATRON_ROOT="${_cand}"
            break
        fi
    done
fi
if [ -z "${MEGATRON_ROOT:-}" ] || [ ! -d "${MEGATRON_ROOT}/megatron/training" ]; then
    echo "ERROR: full Megatron-LM not found (need megatron/training)."
    echo "  Set MEGATRON_ROOT to a Megatron-LM checkout, e.g.:"
    echo "  MEGATRON_ROOT=/workspace/Megatron-LM bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

# Put the full checkout first so megatron.training / megatron.core both resolve
# from MEGATRON_ROOT (the tree the RMSNorm patch below targets).
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH:-}"
echo "[megatron] using ${MEGATRON_ROOT}"

# Lumen ships a layer-spec patch; baseline Megatron (te) doesn't need it.
# Only apply it for the Lumen backend, and only if the patch script is present.
_PATCH="${LUMEN_DIR}/examples/llama2/scripts/patch_gpt_layer_specs.py"
if [ "${LUMEN_BACKEND}" = "lumen" ] && [ -f "${_PATCH}" ]; then
    python "${_PATCH}" "${MEGATRON_ROOT}"
fi

python - <<PYEOF
import json
import os
import random

seq = int(os.environ["SEQ_LEN"])
gbs = int(os.environ["GBS"])
steps = int(os.environ["TRAIN_STEPS"])
need_chunks = gbs * (steps + 5)
need_tokens = int(need_chunks * (seq + 1) * 1.2)
random.seed(int(os.environ["SEED"]))
path = "${TRAIN_JSONL}"
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 31999)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] wrote {docs} docs to {path}")
PYEOF

echo "================================================================"
echo "Llama2-7B pretrain — native (inside container)"
echo "  Library:   ${LUMEN_BACKEND}"
echo "  Precision: ${PRECISION}"
echo "  GPUs:      ${NGPU}"
echo "  Batch:     MBS=${MBS} GBS=${GBS} seq_len=${SEQ_LEN}"
echo "  Steps:     ${TRAIN_STEPS}  seed=${SEED}"
echo "  Tokenizer: ${TOKENIZER_DIR}"
echo "  Log:       ${LOG_FILE}"
if [ "${ENABLE_PROFILING:-0}" = "1" ]; then
    echo "  Profiling: steps ${PROFILE_STEP_START:-3}-${PROFILE_STEP_END:-6}, rank(s) ${PROFILE_RANKS:-0}"
    echo "  Profile:   ${PROFILE_DIR:-${RESULTS_DIR}/profile}"
fi
echo "================================================================"

cd "${MEGATRON_ROOT}"
set -x
torchrun --nproc_per_node="${NGPU}" --nnodes=1 pretrain_gpt.py \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --use-rotary-position-embeddings \
    --rotary-base 10000 \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_STEPS}" \
    --lr 1.0e-5 --min-lr 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_DIR}" \
    --mock-data \
    --split 98,1,1 \
    --seed "${SEED}" \
    --eval-iters 1 \
    --eval-interval "${TRAIN_STEPS}" \
    --save-interval 1000000 \
    --log-interval 1 \
    "${BACKEND_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "[DONE] log: ${LOG_FILE}"
