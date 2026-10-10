#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/manual.sh: prints the steps left to the user (clones, images, containers, network installs). Runs nothing.
# Everything else (KOSMOS, TE, Primus-Turbo, DeepEP, the MegaMoE shims, FlyDSL) is built by setup --build --build-deps.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
FLAGS="--network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
--device /dev/kfd --device /dev/dri --shm-size 64g $CT_MOUNTS"
PT_URL=https://github.com/AMD-AGI/Primus-Turbo.git
echo "# Sources:"
manual "git clone --recursive <KOSMOS remote> $KOSMOS_DIR && git -C $KOSMOS_DIR checkout $KOSMOS_BRANCH && git -C $KOSMOS_DIR submodule update --init"
manual "git clone <TransformerEngine remote> $TE_DIR && git -C $TE_DIR checkout $TE_BRANCH && git -C $TE_DIR submodule update --init --recursive"
manual "git -C $TPSP_MLM_DIR checkout $TPSP_MLM_BRANCH   # TP+SP tree (this kit)"
[ "$A2A_MLM_DIR" != "$TPSP_MLM_DIR" ] &&
    manual "git -C $TPSP_MLM_DIR worktree add $A2A_MLM_DIR $A2A_MLM_BRANCH   # A2A tree, a second worktree (README \"Megatron trees\")"
manual "git clone $PT_URL $PT_DIR && git -C $PT_DIR checkout $PT_COMMIT && git -C $PT_DIR submodule update --init --depth 1"
[ "$PT_MX_DIR" != "$PT_DIR" ] &&
    manual "git clone $PT_URL $PT_MX_DIR && git -C $PT_MX_DIR checkout $PT_MX_COMMIT && git -C $PT_MX_DIR submodule update --init --depth 1   # MXFP8 MegaMoE"
manual "git clone https://github.com/AMD-AGI/Primus.git $PRIMUS_DIR && git -C $PRIMUS_DIR checkout $PRIMUS_COMMIT && git -C $PRIMUS_DIR submodule update --init third_party/Megatron-LM   # Primus-workflow arm only"
echo "# Containers. This directory, the trees, DEPS_ROOT and OUT_ROOT must be mounted at the same path in both:"
manual "$DOCKER pull $KOSMOS_IMAGE"
manual "$DOCKER pull $PRIMUS_IMAGE"
manual "$DOCKER run -d --name $KOSMOS_CT $FLAGS $KOSMOS_IMAGE sleep infinity"
manual "$DOCKER run -d --name $PRIMUS_CT $FLAGS $PRIMUS_IMAGE sleep infinity"
manual "$DOCKER exec $KOSMOS_CT bash -lc 'pip install transformers==5.17.0 tensorstore==0.1.85'   # Megatron's HF tokenizers; not in the TE CI image"
manual "$DOCKER exec $KOSMOS_CT bash -lc 'apt-get update && apt-get install -y numactl'   # numa ablation only"
echo "# Tokenizers (the scripts also fetch them on first use):"
manual "$DOCKER exec -e HF_HOME=$MOE_HF_HOME $KOSMOS_CT bash -lc \"hf download Qwen/Qwen3-235B-A22B --include '*token*' --include '*.json' --exclude '*.safetensors'\""
manual "$DOCKER exec -e HF_HOME=$MOE_HF_HOME $KOSMOS_CT bash -lc \"hf download deepseek-ai/DeepSeek-V3 --include '*token*' --include '*.json' --exclude '*.safetensors'\""
manual "$DOCKER exec ${LLAMA_HF_HOME:+-e HF_HOME=$LLAMA_HF_HOME }$KOSMOS_CT bash -lc \"hf download NousResearch/Meta-Llama-3-8B --include '*token*' --include '*.json'\""
echo "# Then: run_all.sh setup --build --build-deps (KOSMOS make all python, TE in place against it + editable install,"
echo "# Primus-Turbo $PT_COMMIT and $PT_MX_COMMIT, DeepEP, the MegaMoE shims, FlyDSL 0.2.4), run_all.sh setup until it"
echo "# reports OK, and the GPU steps."
