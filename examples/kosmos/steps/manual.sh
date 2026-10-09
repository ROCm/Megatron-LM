#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/manual.sh: prints the steps left to the user (images, containers, clones, network installs). Runs nothing.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
FLAGS="--network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
--device /dev/kfd --device /dev/dri --shm-size 64g $CT_MOUNTS"
echo "# Containers. This directory, the three trees, DEPS_ROOT and OUT_ROOT must be mounted at the same path in both:"
manual "$DOCKER pull $KOSMOS_IMAGE"
manual "$DOCKER pull $PRIMUS_IMAGE"
manual "$DOCKER run -d --name $KOSMOS_CT $FLAGS $KOSMOS_IMAGE sleep infinity"
manual "$DOCKER run -d --name $PRIMUS_CT $FLAGS $PRIMUS_IMAGE sleep infinity"
manual "$DOCKER exec $KOSMOS_CT bash -lc 'pip install --no-build-isolation --no-deps -e $TE_DIR'   # TE editable, once per container"
manual "$DOCKER exec $KOSMOS_CT bash -lc 'apt-get update && apt-get install -y numactl'   # numa ablation only"
echo "# Sources:"
manual "git clone --recursive <KOSMOS remote> $KOSMOS_DIR && git -C $KOSMOS_DIR checkout $KOSMOS_BRANCH"
manual "git clone <TransformerEngine remote> $TE_DIR && git -C $TE_DIR checkout $TE_BRANCH && git -C $TE_DIR submodule update --init --recursive"
manual "git -C $MLM_DIR checkout $MLM_BRANCH"
manual "git clone https://github.com/AMD-AGI/Primus-Turbo.git $PT_DIR && git -C $PT_DIR checkout $PT_COMMIT && git -C $PT_DIR submodule update --init --depth 1"
manual "git clone https://github.com/AMD-AGI/Primus.git $PRIMUS_DIR && git -C $PRIMUS_DIR checkout 4969fd51   # Primus-workflow arm only"
manual "$DOCKER exec $KOSMOS_CT bash -lc 'apt-get update && apt-get install -y libpci-dev libibverbs-dev libdrm-dev'"
echo "# MegaMoE in the KOSMOS container: Primus-Turbo's python tree without its torch-2.10 _C imports, plus FlyDSL 0.2.4:"
manual "mkdir -p $MEGAMOE_SHIM && cp -as $PT_DIR/primus_turbo $MEGAMOE_SHIM/   # then replace primus_turbo/pytorch/{,core/,ops/,ops/moe/}__init__.py by files that import only what fused_mega_moe needs"
manual "mkdir -p $MEGAMOE_PYENV/lib && $DOCKER exec $PRIMUS_CT bash -lc 'cp -a /opt/venv/lib/python3.12/site-packages/flydsl* $MEGAMOE_PYENV/' && ln -sfn $CT_ROCM_CORE_LIB/libamdhip64.so.7 $MEGAMOE_PYENV/lib/libamdhip64.so"
echo "# Other inputs under DEPS_ROOT: APEX_SHIM_DIR (fused_weight_gradient_mlp_cuda on TE general_gemm),"
echo "# DEEPEP_DIR (setup --build-deps builds it), NUMA_BIND_SCRIPT. Tokenizers (the scripts also fetch them on first use):"
manual "$DOCKER exec -e HF_HOME=$MOE_HF_HOME $KOSMOS_CT bash -lc \"huggingface-cli download Qwen/Qwen3-235B-A22B --include '*token*' '*.json' --exclude '*.safetensors'\""
manual "$DOCKER exec -e HF_HOME=$MOE_HF_HOME $KOSMOS_CT bash -lc \"huggingface-cli download deepseek-ai/DeepSeek-V3 --include '*token*' '*.json' --exclude '*.safetensors'\""
manual "$DOCKER exec ${LLAMA_HF_HOME:+-e HF_HOME=$LLAMA_HF_HOME }$KOSMOS_CT bash -lc \"huggingface-cli download NousResearch/Meta-Llama-3-8B --include '*token*' '*.json'\""
echo "# Then: run_all.sh setup --build [--build-deps], and the GPU steps."
