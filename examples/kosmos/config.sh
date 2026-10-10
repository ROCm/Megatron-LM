#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# config.sh: every knob of the run set. Sourced on the host by every step. Any variable can be set in the environment
# or in the site file KOSMOS_SITE (default ~/.config/kosmos/site.sh), which is read first and holds node-local paths.

KOSMOS_SITE=${KOSMOS_SITE:-$HOME/.config/kosmos/site.sh}
[ -f "$KOSMOS_SITE" ] && . "$KOSMOS_SITE"
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

# ---- node (read from the node unless set) -----------------------------------------------------------------------------
node_gpu_name() {  # "AMD Instinct MI350 OAM" -> MI350X, "AMD Instinct MI355X OAM" -> MI355X
    local p
    p=$(cat /sys/class/drm/card*/device/product_name 2>/dev/null | head -1 | sed -e 's/^AMD Instinct //' -e 's/ OAM$//' -e 's/ /_/g')
    [[ $p =~ ^MI3[0-9][0-9]$ ]] && p=${p}X
    echo "${p:-GPU}"
}
node_gpu_arch() {
    local v
    v=$(awk '/^gfx_target_version/ && $2 != 0 { print $2; exit }' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null)
    [ -n "$v" ] && printf 'gfx%d%x%x\n' $((v / 10000)) $((v / 100 % 100)) $((v % 100))
}
node_ngpu()    { ls /sys/class/drm/card*/device/product_name 2>/dev/null | wc -l; }
node_power_w() { awk '{ printf "%d", $1 / 1000000; exit }' /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null; }

GPU_NAME=${GPU_NAME:-$(node_gpu_name)}
GPU_ARCH=${GPU_ARCH:-$(node_gpu_arch)}
GPU_ARCH=${GPU_ARCH:-gfx950}
NGPU=${NGPU:-$(node_ngpu)}
POWER_CAP_W=${POWER_CAP_W:-$(node_power_w)}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((${NGPU:-8} - 1)))}
GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}

# ---- repositories: used on their branches as they are; setup verifies them ------------------------------------------
# A *_COMMIT is the oldest accepted commit: the tree's HEAD must be it or a descendant of it.
MLM_ONE=${MLM_DIR:+1}                            # MLM_DIR set by the caller: one Megatron tree for every step
MLM_DIR=${MLM_DIR:-$(cd "$KIT/../.." && pwd)}   # this Megatron-LM tree
MLM_BRANCH=${MLM_BRANCH:-cosmic_crisp}
MLM_COMMIT=${MLM_COMMIT:-96cb60ec0}           # cosmic_crisp rebased on rocm_dev b0695a047 (native MORI flex backend)
# Megatron tree per step (README "Megatron trees"): TP+SP on cosmic_crisp (this tree), A2A on alemagro/kosmos_oldbase,
# a second worktree next to it. With MLM_DIR set, both default to MLM_DIR / MLM_BRANCH / MLM_COMMIT.
TPSP_MLM_DIR=${TPSP_MLM_DIR:-$MLM_DIR}
TPSP_MLM_BRANCH=${TPSP_MLM_BRANCH:-$MLM_BRANCH}
TPSP_MLM_COMMIT=${TPSP_MLM_COMMIT:-$MLM_COMMIT}
if [ -n "$MLM_ONE" ]; then
    A2A_MLM_DIR=${A2A_MLM_DIR:-$MLM_DIR}
    A2A_MLM_BRANCH=${A2A_MLM_BRANCH:-$MLM_BRANCH}
    A2A_MLM_COMMIT=${A2A_MLM_COMMIT:-$MLM_COMMIT}
else
    A2A_MLM_DIR=${A2A_MLM_DIR:-$(dirname "$MLM_DIR")/Megatron-LM-oldbase}
    A2A_MLM_BRANCH=${A2A_MLM_BRANCH:-alemagro/kosmos_oldbase}
    A2A_MLM_COMMIT=${A2A_MLM_COMMIT:-bc4a530dc}   # the KOSMOS integration on c6bf60ce7 (pushed as alemagro/kosmos_oldbase)
fi
TE_DIR=${TE_DIR:-$(dirname "$MLM_DIR")/TransformerEngine}
TE_BRANCH=${TE_BRANCH:-alemagro/kosmos_final}   # cosmic_crisp 4e331aff0 + MXFP8 KOSMOS dispatch: the TE of the MI350X results
TE_COMMIT=${TE_COMMIT:-4e331aff0}             # dev 9bd66f096 + KOSMOS backend + fp32 wgrad accumulate
# The no-overlap baseline (TP_COMM_OVERLAP=0) runs on TE_BASE_DIR, by default the KOSMOS TE build (TE_DIR).
TE_BASE_DIR=${TE_BASE_DIR:-$TE_DIR}
TE_MAX_JOBS=${TE_MAX_JOBS:-64}
KOSMOS_DIR=${KOSMOS_DIR:-$(dirname "$MLM_DIR")/KOSMOS}
KOSMOS_BRANCH=${KOSMOS_BRANCH:-main}
KOSMOS_COMMIT=${KOSMOS_COMMIT:-1f00ba1}       # + MXFP8 MoE megakernels, python/kosmos.cpp quantize_mxfp8
KOSMOS_BUILD=${KOSMOS_BUILD:-$KOSMOS_DIR/build}
KOSMOS_PYTHON=${KOSMOS_PYTHON:-$KOSMOS_BUILD/python}   # `make python` output; the Megatron adapter imports kosmos from it
HK_COMMIT=${HK_COMMIT:-e4fa1cc3}              # KOSMOS 3rdparty/hipkittens

# ---- dependencies (built once, outside the three trees) --------------------------------------------------------------
DEPS_ROOT=${DEPS_ROOT:-$HOME/kosmos_deps}
APEX_SHIM_DIR=${APEX_SHIM_DIR:-$KIT/deps/apex_shim}         # fused_weight_gradient_mlp_cuda stand-in (TE general_gemm)
DEEPEP_DIR=${DEEPEP_DIR:-$DEPS_ROOT/deepep}                 # Primus-Turbo intranode DeepEP as `deep_ep` (tools/deepep.sh)
PT_DIR=${PT_DIR:-$DEPS_ROOT/Primus-Turbo}                   # Primus-Turbo, built in place in PRIMUS_CT
PT_COMMIT=${PT_COMMIT:-4b0f01b9}
PT_MX_DIR=${PT_MX_DIR:-$DEPS_ROOT/Primus-Turbo-mx}          # Primus-Turbo of the MXFP8 MegaMoE arms, built in place
PT_MX_COMMIT=${PT_MX_COMMIT:-30e6de87}                       # >= #525 and #529: the staged MXFP8 op, system-scope combine
MEGAMOE_SHIM=${MEGAMOE_SHIM:-$DEPS_ROOT/megamoe/pt_shim}    # PT_DIR's python tree importable in KOSMOS_CT (tools/megamoe_shim.sh)
MEGAMOE_SHIM_MX=${MEGAMOE_SHIM_MX:-$DEPS_ROOT/megamoe/pt_shim_mx}   # the same from PT_MX_DIR (MXFP8 arm)
MEGAMOE_PYENV=${MEGAMOE_PYENV:-$DEPS_ROOT/megamoe/pyenv}    # FlyDSL 0.2.4 copied from the Primus image
MEGAMOE_FLYDSL_CACHE=${MEGAMOE_FLYDSL_CACHE:-$DEPS_ROOT/megamoe/flydsl_autotune}
NUMA_BIND_SCRIPT=${NUMA_BIND_SCRIPT:-$KIT/deps/numa_bind.sh}    # per-rank numactl wrapper (numa ablation only)
MOE_HF_HOME=${MOE_HF_HOME:-$DEPS_ROOT/huggingface}          # Qwen3-235B-A22B, DeepSeek-V3 tokenizers
MOE_DATA_CACHE=${MOE_DATA_CACHE:-$DEPS_ROOT/data_cache}
LLAMA_HF_HOME=${LLAMA_HF_HOME:-}                             # empty: the container default
PRIMUS_DIR=${PRIMUS_DIR:-$DEPS_ROOT/Primus}                 # AMD-AGI/Primus (Primus-workflow arm only)
PRIMUS_COMMIT=${PRIMUS_COMMIT:-4969fd51}
PRIMUS_HF_HOME=${PRIMUS_HF_HOME:-$DEPS_ROOT/primus_hf}
PRIMUS_FLYDSL_CACHE=${PRIMUS_FLYDSL_CACHE:-$DEPS_ROOT/primus_flydsl_autotune}
MICRO_FLYDSL_SEED=${MICRO_FLYDSL_SEED:-}                     # MegaMoE autotune cache to start from; empty: tune afresh

# ---- containers (created by hand: run_all.sh manual) ----------------------------------------------------------------
DOCKER=${DOCKER:-docker}
KOSMOS_CT=${KOSMOS_CT-kosmos}        # TheRock ROCm 7.14, torch 2.12, editable TE = TE_DIR, Open MPI; empty: the host
KOSMOS_IMAGE=${KOSMOS_IMAGE:-<TE CI image: TheRock ROCm 7.14, torch 2.12, Open MPI>}
PRIMUS_CT=${PRIMUS_CT:-primus}         # MegaMoE microbenchmark (MICRO_MEGA), Primus-Turbo build, Primus-workflow arm
PRIMUS_IMAGE=${PRIMUS_IMAGE:-rocm/primus:v26.3}
CT_ROCM=${CT_ROCM:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel}
CT_ROCM_CORE_LIB=${CT_ROCM_CORE_LIB:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib}
CT_OMPI=${CT_OMPI:-/opt/ompi}
CT_PYTHON=${CT_PYTHON:-/opt/venv/bin/python3}  # PyTorch for the KOSMOS microbenchmarks' baselines (KOSMOS PYTHON)
PRIMUS_CT_ROCM=${PRIMUS_CT_ROCM:-/opt/rocm}
PRIMUS_CT_SITE=${PRIMUS_CT_SITE:-/opt/venv/lib/python3.12/site-packages}   # FlyDSL 0.2.4 in the Primus image
CT_MOUNTS=${CT_MOUNTS:-"-v $HOME:$HOME"}   # docker run volumes: must cover KIT, the trees, DEPS_ROOT and OUT_ROOT at the same path

# ---- outputs (outside the trees) --------------------------------------------------------------------------------------
OUT_ROOT=${OUT_ROOT:-$HOME/kosmos_results}
OUT_ROOT=${OUT_ROOT%/}
[ "${OUT_ROOT##*/}" = "$GPU_NAME" ] || OUT_ROOT=$OUT_ROOT/$GPU_NAME   # one tree per GPU model

# ---- run hygiene ----------------------------------------------------------------------------------------------------
LOAD_MAX=${LOAD_MAX:-20}               # wait for the host 1-min loadavg below this before every GPU run (0: off)
LOAD_POLL_S=${LOAD_POLL_S:-60}
LOAD_WAIT_MAX_S=${LOAD_WAIT_MAX_S:-0}  # 0: wait as long as needed; else start anyway after this long (logged)
LOAD_SAMPLE_S=${LOAD_SAMPLE_S:-30}     # loadavg sampled during every step, for the per-run load column
COLLECT_LOAD_MAX=${COLLECT_LOAD_MAX:-0}   # collect: valid only if the mean sampled host load was below this (0: off; a run's own 8 ranks put it at 12-23)
GPU_IDLE_CHECK=${GPU_IDLE_CHECK:-1}    # first wait until no process holds a GPU context (/sys/class/kfd/kfd/proc)
GPU_IDLE_WAIT_S=${GPU_IDLE_WAIT_S:-3600}
LOCK_FILE=${LOCK_FILE:-}               # optional: one flock per GPU invocation
PORT0=${PORT0:-29800}                  # MASTER_PORT base; every run takes the next free port
SEED=${SEED:-$(date +%s)}              # arm-order shuffle seed (recorded in each step's directory)
POST_RUN_SLEEP_S=${POST_RUN_SLEEP_S:-10}
FIXOWN=${FIXOWN:-1}                    # chown files the containers created as root back to the caller

# ---- step: micro (KOSMOS bench/run_all.sh) ------------------------------------------------------------------------
MICRO_REPS=${MICRO_REPS:-3}            # bench/moe/run.sh and bench/tpsp/run.sh REPS: time = median over the reps
MICRO_CHECK_SHAPES=${MICRO_CHECK_SHAPES:-gate8 dsv3 qwen35 dsv4pro kimi27 glm52 qwen235 qwen30}
MICRO_DISTS=${MICRO_DISTS:-0 1}       # routing: 0 balanced, 1 skewed (bench dist 1); separate tables per dist
MICRO_RESULTS_IN_TREE=${MICRO_RESULTS_IN_TREE:-0}   # 1: write KOSMOS results/<GPU>.md in the KOSMOS tree
MICRO_ABLATIONS=${MICRO_ABLATIONS-unfused}   # unfused (default): the KOSMOS unfused arm (MoE and TP+SP --unfused); empty: off
MICRO_PRECISIONS=${MICRO_PRECISIONS:-bf16 mxfp8}   # run_all.sh PRECISIONS: gate, checks, MoE, MegaMoE, TP+SP
MICRO_MEGA=${MICRO_MEGA:-0}            # 1: also MegaMoE (internal only), run_all.sh mega in PRIMUS_CT
# MI350X minutes, estimates (these runners are not yet timed): gate + checks; MoE fwd + bwd per routing (7 shapes,
# 3 reps; the ablation adds an arm); TP+SP every config, BF16 + MXFP8, 3 reps; MegaMoE per sweep.
EST_MICRO_CHECK_MIN=${EST_MICRO_CHECK_MIN:-3}
EST_MICRO_MOE_MIN=${EST_MICRO_MOE_MIN:-21.5}
EST_MICRO_MOE_UNF_MIN=${EST_MICRO_MOE_UNF_MIN:-3}
EST_MICRO_TPSP_MIN=${EST_MICRO_TPSP_MIN:-23}
EST_MICRO_TPSP_UNF_MIN=${EST_MICRO_TPSP_UNF_MIN:-4.5}
EST_MICRO_MEGA_SWEEP_MIN=${EST_MICRO_MEGA_SWEEP_MIN:-0.21}

# ---- E2E protocol ---------------------------------------------------------------------------------------------------
E2E_ITERS=${E2E_ITERS:-10}             # TP+SP TOTAL_ITERS, A2A TRAIN_ITERS, Primus --train_iters
E2E_REPS=${E2E_REPS:-3}                # runs per (config, arm); collect reports the median of the valid quiet runs

# ---- step: tpsp (Llama 3 TP+SP E2E, train_llama3.sh, BF16, SEQ 8192, TP8) --------------------------------------------
# MODEL:MBS:GBS:LAYERS:HARD_TIMEOUT_S:EST_MIN:EST_NOOVL_MIN. EST = MI350X minutes per completed run at E2E_ITERS=10,
# for the overlapped arms and for base_noovl: the published 20-iteration runs' wall from the first log line to
# iteration 10, plus their teardown after the last iteration, plus 0.6 min of runner overhead (exec, TE import, poll,
# sleep). Timeout = 600 s + 3 x the run's wall if it completes; 405B never completed on MI350X, so its EST is that wall,
# (TIMEOUT - 600) / 3, plus 0.5 min (base_noovl x 1.2); its published failures are in TPSP_EST_FAIL.
TPSP_CONFIGS=${TPSP_CONFIGS:-"8:1:64:32:1450:2.6:3.0 8:2:64:32:1300:2.3:2.5 8:4:64:32:1250:2.2:2.4 8:8:64:32:1250:2.1:2.4
 70:1:64:80:2100:8.4:10.0 70:2:64:80:1950:7.6:9.2 70:4:64:60:1750:6.0:8.0 70:8:64:50:1850:5.4:6.9
 405:1:64:24:4100:13.5:17.7 405:2:64:24:3600:12.0:15.0 405:4:64:24:3300:12.7:15.4 405:8:64:24:2800:12.4:14.9"}
TPSP_RECOMPUTE_MODELS=${TPSP_RECOMPUTE_MODELS:-405}   # RECOMPUTE=1 for these model sizes (as the published 405B runs)
TPSP_GA=${TPSP_GA:-0}                  # gradient-accumulation fusion: 0 off (default); "0 1" adds the GA-on set (TPSP_GA1_CONFIGS)
TPSP_GA1_CONFIGS=${TPSP_GA1_CONFIGS:-"8:1 70:1 70:2 70:4 70:8"}   # MODEL:MBS run with GA fusion on (as published)
TPSP_ARMS=${TPSP_ARMS:-"base_noovl kos_tpsp"}     # no overlap vs KOSMOS
TPSP_PRECISIONS=${TPSP_PRECISIONS:-"bf16 mxfp8"}  # mxfp8: TE_FP8=1 TE_FP8_RECIPE=mxfp8 (every TE linear MXFP8)
TPSP_REPS_GA0=${TPSP_REPS_GA0:-$E2E_REPS}
TPSP_REPS_GA1=${TPSP_REPS_GA1:-$E2E_REPS}
# MODEL:ARM:N overrides GPU_MAX_HW_QUEUES for that model and arm: the 405B baselines hang in their first backward on
# MI350X with more than one HW queue (the KOSMOS arms do not).
TPSP_HWQ=${TPSP_HWQ:-"405:base_noovl:1"}
TPSP_STALL_S=${TPSP_STALL_S:-600}
TPSP_STALL_S_405=${TPSP_STALL_S_405:-1200}   # stall watchdog of the 405B runs (GBS 64: iterations ~4-8 min; 31-60 s at GBS 8-16)
# 1: a stall (rc 86), hard timeout or out-of-memory / launch-resource failure of a (model, arm, GA) skips its remaining
# repeats and every larger MBS (recorded as skipped rows); 0: every run is attempted.
TPSP_HANG_SKIP=${TPSP_HANG_SKIP:-1}
# Published MI350X failures, MODEL:MBS:ARM:GA:MINUTES, used only by the DRYRUN plan (the run follows real outcomes):
# 405B base_noovl hung at MBS1 with 2 HW queues (fixed by TPSP_HWQ); kos_tpsp crashed at MBS4 after 8 iterations
# (launch resources).
TPSP_EST_FAIL=${TPSP_EST_FAIL:-"405:4:kos_tpsp:0:4.6"}

# ---- step: a2a (Megatron example scripts, proxies, BF16 and MXFP8, EP8, PROFILE=false) -------------------------------
A2A_MODELS=${A2A_MODELS:-"qwen dsv3"}       # qwen: Qwen3-235B proxy, 24 layers, recompute 3; dsv3: DSv3 proxy
DSV3_MOE_LAYER_FREQ=${DSV3_MOE_LAYER_FREQ:-1}  # 1 = 3 MoE layers (published); empty = proxy default 1 dense + 2 MoE
A2A_ARMS=${A2A_ARMS:-"kosmos grouped deepep megamoe"}   # megamoe: internal table only (never published)
# Precisions: bf16 (PR=bf16) and mxfp8 (PR=fp8 FP8_RECIPE=mxfp8, every layer FP8). An mxfp8 run is made only for the
# arms of A2A_MXFP8_ARMS that are also in A2A_ARMS; megamoe MXFP8 is Primus-Turbo's staged op (MEGAMOE_SHIM_MX). Ablations and the Primus workflow are bf16 only.
A2A_PRECISIONS=${A2A_PRECISIONS:-"bf16 mxfp8"}
A2A_MXFP8_ARMS=${A2A_MXFP8_ARMS:-"kosmos grouped deepep megamoe"}
# mxfp8 expert padding. 0 (default): the runs drop the scripts' --moe-router-padding-for-quantization (a derived copy,
# tools/gen_scripts.sh), so TE's Fp8Padding pads each expert's rows after the dispatch, to 256 under
# NVTE_USE_HIPKITTENS_GROUPED_GEMM=1 on gfx950, which the HipKittens grouped GEMM needs. 1: the scripts as they are
# (router padding to Megatron's MXFP8 alignment, 128: odd multiples of 128 make HipKittens decline, CK MX runs them).
A2A_MXFP8_ROUTER_PAD=${A2A_MXFP8_ROUTER_PAD:-0}
A2A_REPS_KOSMOS=${A2A_REPS_KOSMOS:-$E2E_REPS}
A2A_REPS=${A2A_REPS:-$E2E_REPS}             # per baseline arm
PRIMUS_REPS=${PRIMUS_REPS:-$E2E_REPS}
# Ablations, arm:variant, each run A2A_REPS_KOSMOS times. Variants: rtroff (KOSMOS_ROUTER=0), gateoff
# (KOSMOS_DDP_GATE=0), noovl (DDP_OVERLAP=false), numa (per-rank NUMA binding; any arm). e.g. "kosmos:rtroff kosmos:noovl"
A2A_ABLATIONS=${A2A_ABLATIONS:-}
A2A_PRIMUS=${A2A_PRIMUS:-0}                 # 1: also the Primus-workflow MegaMoE arm (PRIMUS_CT, internal table only)
KOSMOS_WGCAP=${KOSMOS_WGCAP:-256}
# Routing: balanced (FORCE_BALANCE=true: Megatron's force-load-balancing, as published), skewed, or both (ablations
# and the Primus workflow are balanced only). Skewed = natural: FORCE_BALANCE=false, Megatron's router with its aux
# loss from initialization (the fused router for KOSMOS). Its skew is not known before a run; MOE_ROUTING_STATS
# records it in every skewed run.
A2A_ROUTING=${A2A_ROUTING:-both}
# KOSMOS arena per (routing kind, model): KOSMOS_CAPACITY_FACTOR, rows per rank = factor * T * k (empty: the adapter's
# 1.5). natural: measured (collect prints each run's need); an overflow makes the run invalid. qwen 4.1: the measured need
# is at most 4.07 over 10 iterations (the routing collapses onto one expert by iteration 3, the same every run), at mem
# usage ~0.97 (MI350X, 270.6 GB); 4.5 runs out of memory at iteration 3 in BF16 (2026-10-09); the ceiling, every
# token's experts on one rank, is 8.07. dsv3 3.0 (needs 1.27).
KOSMOS_CF_balanced_qwen=${KOSMOS_CF_balanced_qwen:-}
KOSMOS_CF_balanced_dsv3=${KOSMOS_CF_balanced_dsv3:-}
KOSMOS_CF_natural_qwen=${KOSMOS_CF_natural_qwen:-4.1}
KOSMOS_CF_natural_dsv3=${KOSMOS_CF_natural_dsv3:-3.0}
# MegaMoE MXFP8 symmetric pool per (routing kind, model), in mean rows per rank (MEGAMOE_POOL_MULT; empty: Primus's 2,
# which a qwen natural run, up to 4.07x on one rank, overflows: the combine times out).
MEGAMOE_POOL_natural_qwen=${MEGAMOE_POOL_natural_qwen:-5}
MEGAMOE_POOL_natural_dsv3=${MEGAMOE_POOL_natural_dsv3:-}
# MegaMoE MXFP8 combine CU split per (routing kind, model), "L1:L2" (PT_MEGA_FP8_L1_COMBINE_CU / L2): Primus's online
# tuner's winners from a tuning run of the same configuration (2026-10-09). Unpinned, the tuner's compiles fall into the
# timed iterations of the 3-MoE-layer DSv3 proxy (iteration 3: 9.7 s instead of 0.29 s).
MEGAMOE_CU_balanced_qwen=${MEGAMOE_CU_balanced_qwen:-32:48}
MEGAMOE_CU_natural_qwen=${MEGAMOE_CU_natural_qwen:-64:48}
MEGAMOE_CU_balanced_dsv3=${MEGAMOE_CU_balanced_dsv3:-32:32}
MEGAMOE_CU_natural_dsv3=${MEGAMOE_CU_natural_dsv3:-48:32}
# GPU_MAX_HW_QUEUES for every A2A run (empty: the training scripts' 2). On the rebased cosmic_crisp Megatron, DeepEP and
# MegaMoE hung with 2 (and the qwen natural grouped runs at iteration 5); on the old base (c6bf60ce7) they did not.
A2A_HWQ_ALL=${A2A_HWQ_ALL:-}
A2A_BF16_CK=${A2A_BF16_CK-0}        # NVTE_USE_CK_GROUPED_GEMM for the BF16 grouped / deepep arms: 0 = TE default; CK (1) is 2.4-3.5x slower (2026-10-09)
A2A_GA_FUSION=${A2A_GA_FUSION:-false}
A2A_HWQ=${A2A_HWQ:-}
# MegaMoE MXFP8's first iteration compiles its kernels (about 3 min, log silent), which the stall watchdog covers.
A2A_TMO=${A2A_TMO:-1800}
A2A_STALL_S=${A2A_STALL_S:-600}
PRIMUS_TMO=${PRIMUS_TMO:-900}
# MI350X minutes per run at 10 iterations: the published runs' wall to iteration 10 plus teardown, plus 0.5 min of
# runner overhead. Primus DSv3: the 2026-09-30 all-MoE run.
EST_A2A_MIN_qwen=${EST_A2A_MIN_qwen:-1.4}
EST_A2A_MIN_dsv3=${EST_A2A_MIN_dsv3:-1.4}
EST_A2A_MX_MIN_qwen=${EST_A2A_MX_MIN_qwen:-$EST_A2A_MIN_qwen}   # MXFP8 runs: not yet timed, the BF16 estimate
EST_A2A_MX_MIN_dsv3=${EST_A2A_MX_MIN_dsv3:-$EST_A2A_MIN_dsv3}
EST_PRIMUS_MIN_qwen=${EST_PRIMUS_MIN_qwen:-1.6}
EST_PRIMUS_MIN_dsv3=${EST_PRIMUS_MIN_dsv3:-1.8}
