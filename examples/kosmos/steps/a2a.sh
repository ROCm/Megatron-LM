#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# steps/a2a.sh: MoE (A2A) E2E through Megatron's example scripts, BF16 and MXFP8, E2E_ITERS (10) iterations, EP8: the Qwen3-235B
# proxy (24 layers, recompute 3) and the DSv3 proxy with MOE_LAYER_FREQ=1 (3 MoE layers); GA fusion off, GEMM_TUNING=0
# and ENABLE_MORI=false in every arm (the proxies export GA_FUSION / ENABLE_MORI true).
# Every arm of A2A_ARMS, plus A2A_ABLATIONS and (A2A_PRIMUS=1) Primus's own workflow.
# Each round runs every (model, arm, routing, precision) once, in a shuffled order. Precision (A2A_PRECISIONS): bf16
# (PR=bf16) and mxfp8 (PR=fp8 FP8_RECIPE=mxfp8) for the arms of A2A_MXFP8_ARMS; the TE grouped GEMM arms (grouped,
# deepep) run it on the HipKittens MXFP8 grouped GEMM (NVTE_USE_HIPKITTENS_GROUPED_GEMM=1, CK off), with fallbacks
# printed (NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1, also on in the bf16 grouped arms); kosmos KOSMOS_MXFP8=1; megamoe Primus-Turbo's staged MXFP8 op (MEGAMOE_SHIM_MX). GPU_MAX_HW_QUEUES: the scripts' 2
# (A2A_HWQ_ALL / A2A_HWQ override it). A2A_MXFP8_ROUTER_PAD=0: mxfp8 runs use the derived scripts without
# --moe-router-padding-for-quantization (TE's Fp8Padding pads each expert to 256 for HipKittens). Routing (A2A_ROUTING): balanced (the proxies'
# FORCE_BALANCE=true) and natural skew (FORCE_BALANCE=false); natural runs record their routing (MOE_ROUTING_STATS) and
# set the KOSMOS arena (KOSMOS_CF_<routing>_<model>).
#   kosmos   MOE_EXPERTS=kosmos, KOSMOS_WGCAP, DDP gate, fused router (the adapter's defaults, set explicitly)
#   megamoe  MOE_EXPERTS=megamoe (Primus-Turbo through MEGAMOE_SHIM, MXFP8 MEGAMOE_SHIM_MX, and MEGAMOE_PYENV)
#   grouped  TE grouped GEMM + all-to-all (the proxy default with ENABLE_MORI=false)
#   deepep   ENABLE_DEEP_EP=true (Primus-Turbo intranode DeepEP)
# Runs: $OUT_ROOT/a2a/runs/<tag>/train_<bf16|mxfp8>.log (tag suffix _mxfp8 for MXFP8, _natural for skewed routing);
# list $OUT_ROOT/a2a/runs.tsv.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
O=$OUT_ROOT/a2a
need_ct "$KOSMOS_CT"
[ "$A2A_PRIMUS" = 1 ] && need_ct "$PRIMUS_CT"
run mkdir -p "$O/runs"
log "Megatron: $A2A_MLM_DIR ($A2A_MLM_BRANCH)"
dry || echo "SEED=$SEED GPU=$GPU_NAME $(date '+%F %T')" >> "$O/RUN.txt"
for p in $A2A_PRECISIONS; do
    case $p in bf16 | mxfp8) ;; *) die "A2A_PRECISIONS: unknown precision $p (bf16 mxfp8)" ;; esac
done
for a in $A2A_MXFP8_ARMS; do
    case $a in
        kosmos | grouped | deepep | megamoe) ;;
        *) die "A2A_MXFP8_ARMS: unknown arm $a" ;;
    esac
done
mx_router_off() { [ "$1" = mxfp8 ] && [ "$A2A_MXFP8_ROUTER_PAD" = 0 ]; }   # mx_router_off PREC
if [[ " $A2A_ABLATIONS " == *":noovl "* ]] || { [[ " $A2A_PRECISIONS " == *" mxfp8 "* ]] && mx_router_off mxfp8; }; then
    for m in $A2A_MODELS; do
        if dry; then echo "[dryrun] bash $KIT/tools/gen_scripts.sh $m $A2A_MLM_DIR $O"
        else bash "$KIT/tools/gen_scripts.sh" "$m" "$A2A_MLM_DIR" "$O" > /dev/null || die "cannot derive the e2e scripts"; fi
    done
fi
sampler_start "$O/loadavg.log"
PORT=$((PORT0 + 500))   # apart from the tpsp ports
case $A2A_ROUTING in
    balanced) KINDS=balanced ;;
    skewed)   KINDS=natural ;;
    both)     KINDS="balanced natural" ;;
    *)        die "A2A_ROUTING=$A2A_ROUTING: balanced, skewed or both" ;;
esac
if [ "$KINDS" != balanced ]; then
    grep -q MOE_ROUTING_STATS "$A2A_MLM_DIR/megatron/core/transformer/moe/moe_utils.py" &&
        grep -q _report_status "$A2A_MLM_DIR/megatron/core/transformer/moe/kosmos_moe.py" ||
        { dry && warn "$A2A_MLM_DIR has no MOE_ROUTING_STATS (moe_utils.py) or KOSMOS status report" ||
            die "$A2A_MLM_DIR has no MOE_ROUTING_STATS (moe_utils.py) or KOSMOS status report"; }
fi
rt_tag() { case $1 in balanced) ;; natural) echo _natural ;; esac; }
pr_tag() { case $1 in bf16) ;; mxfp8) echo _mxfp8 ;; esac; }

arm_env() {  # arm_env MODEL ARM VARIANT RUN ROUTING PRECISION: exports applied after the proxy
    local m=$1 arm=$2 var=$3 run=$4 rt=$5 pr=$6 pre=$A2A_MLM_DIR/pretrain_gpt.py cf
    echo "export PYTHONPATH=$TE_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
    case $pr in
        bf16)  echo "export PR=bf16" ;;
        mxfp8) echo "export PR=fp8 FP8_RECIPE=mxfp8"
               mx_router_off "$pr" && echo "export MOE_ROUTER_PADDING_FOR_QUANT=false" ;;
    esac
    # TE grouped GEMM arms: print every grouped-GEMM fallback; MXFP8 on HipKittens (CK off: CK=1 would turn HK off).
    case $arm in
        grouped | deepep)
            echo "export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1"
            # BF16: A2A_BF16_CK (default 0, TE default; CK is 2.4-3.5x slower).
            if [ "$pr" = mxfp8 ]; then echo "export NVTE_USE_HIPKITTENS_GROUPED_GEMM=1 NVTE_USE_CK_GROUPED_GEMM=0"
            elif [ -n "$A2A_BF16_CK" ]; then echo "export NVTE_USE_CK_GROUPED_GEMM=$A2A_BF16_CK"; fi ;;
    esac
    echo "export TRAIN_ITERS=$E2E_ITERS PROFILE=false GA_FUSION=$A2A_GA_FUSION"
    [ -n "$A2A_HWQ_ALL" ] && echo "export GPU_MAX_HW_QUEUES=$A2A_HWQ_ALL"
    for x in $A2A_HWQ; do [ "${x%:*}" = "$m:$arm" ] && echo "export GPU_MAX_HW_QUEUES=${x##*:}"; done
    # The proxies export ENABLE_MORI=true; every arm turns it off. hipBLASLt tuning off in every arm (the scripts
    # force it off under TE grouped GEMM, so 0 is the only setting all arms can share).
    echo "export GEMM_TUNING=0 ENABLE_MORI=false"
    [ "$m" = dsv3 ] && [ -n "$DSV3_MOE_LAYER_FREQ" ] && echo "export MOE_LAYER_FREQ=$DSV3_MOE_LAYER_FREQ"
    case $arm in
        kosmos)  echo "export MOE_EXPERTS=kosmos KOSMOS_PYTHON=$KOSMOS_PYTHON KOSMOS_WGCAP=$KOSMOS_WGCAP KOSMOS_DDP_GATE=1 KOSMOS_ROUTER=1"
                 [ "$pr" = mxfp8 ] && echo "export KOSMOS_MXFP8=1" ;;
        megamoe) echo "export MOE_EXPERTS=megamoe PYTHONPATH=\$PYTHONPATH:$([ "$pr" = mxfp8 ] && echo "$MEGAMOE_SHIM_MX" ||
                     echo "$MEGAMOE_SHIM"):$MEGAMOE_PYENV"
                 echo "export LD_LIBRARY_PATH=$CT_ROCM_CORE_LIB:$MEGAMOE_PYENV/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
                 echo "export FLYDSL_AUTOTUNE_CACHE_DIR=$MEGAMOE_FLYDSL_CACHE" ;;
        grouped) ;;
        deepep)  echo "export ENABLE_DEEP_EP=true PYTHONPATH=$DEEPEP_DIR/pylib:\$PYTHONPATH" ;;
        *)       die "unknown A2A arm $arm" ;;
    esac
    case $var in
        -)       ;;
        rtroff)  echo "export KOSMOS_ROUTER=0" ;;
        gateoff) echo "export KOSMOS_DDP_GATE=0" ;;
        noovl)   echo "export DDP_OVERLAP=false" ;;
        numa)    echo "export PRETRAIN_SCRIPT='--no-python $NUMA_BIND_SCRIPT $pre' NUMA_BIND_LOG=$run/numa_bind.log" ;;
        *)       die "unknown A2A variant $var" ;;
    esac
    case $rt in
        balanced) ;;
        natural)  echo "export FORCE_BALANCE=false MOE_ROUTING_STATS=$run/routing" ;;
    esac
    cf=$(eval echo "\${KOSMOS_CF_${rt}_$m:-}")
    if [ "$arm" = kosmos ] && [ -n "$cf" ]; then echo "export KOSMOS_CAPACITY_FACTOR=$cf"; fi
    pm=$(eval echo "\${MEGAMOE_POOL_${rt}_$m:-}")
    if [ "$arm" = megamoe ] && [ "$pr" = mxfp8 ] && [ -n "$pm" ]; then echo "export MEGAMOE_POOL_MULT=$pm"; fi
    cu=$(eval echo "\${MEGAMOE_CU_${rt}_$m:-}")
    if [ "$arm" = megamoe ] && [ "$pr" = mxfp8 ] && [ -n "$cu" ]; then
        echo "export PT_MEGA_FP8_L1_COMBINE_CU=${cu%:*} PT_MEGA_FP8_L2_COMBINE_CU=${cu#*:}"
    fi
}

primus_args() {  # the Primus-workflow overrides on Primus's MI355X yaml; DSv3 with 3 MoE layers as the other arms
    local c="--micro_batch_size 2 --global_batch_size 16 --seq_length 4096 --max_position_embeddings 4096"
    c="$c --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 --pipeline_model_parallel_layout null"
    c="$c --expert_model_parallel_size 8 --mock_data True --moe_router_force_load_balancing True"
    c="$c --moe_router_force_load_balancing_type uniform --enable_primus_turbo True --use_turbo_mega_moe True --train_iters $E2E_ITERS"
    case $1 in
        qwen) echo "--config examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-pretrain.yaml --num_layers 24 \
--recompute_granularity full --recompute_method block --recompute_num_layers 3 $c" ;;
        dsv3) echo "--config examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml --num_layers 3 --mtp_num_layers 0 \
--moe_layer_freq 1 --recompute_granularity full --recompute_method block --recompute_num_layers 1 \
--recompute_layer_ids null $c" ;;
    esac
}

# Specs of one round: model:arm:variant:routing:precision:reps
SPECS=()
for m in $A2A_MODELS; do
    for k in $KINDS; do
        for p in $A2A_PRECISIONS; do
            for a in $A2A_ARMS; do
                [ "$p" = mxfp8 ] && [[ " $A2A_MXFP8_ARMS " != *" $a "* ]] && continue
                n=$A2A_REPS; [ "$a" = kosmos ] && n=$A2A_REPS_KOSMOS
                SPECS+=("$m:$a:-:$k:$p:$n")
            done
        done
        [ "$k" = balanced ] || continue
        for s in $A2A_ABLATIONS; do SPECS+=("$m:${s%%:*}:${s#*:}:$k:bf16:$A2A_REPS_KOSMOS"); done
        [ "$A2A_PRIMUS" = 1 ] && SPECS+=("$m:primus:-:$k:bf16:$PRIMUS_REPS")
    done
done
maxr=1
for s in "${SPECS[@]}"; do [ "${s##*:}" -gt "$maxr" ] && maxr=${s##*:}; done

for r in $(seq 1 "$maxr"); do
    for s in $(shuffle "$SEED-a2a-$r" "${SPECS[@]}"); do
        IFS=: read -r m arm var rt pr n <<< "$s"
        [ "$r" -le "$n" ] || continue
        tag=${m}_${arm}$([ "$var" = - ] || echo "_$var")$(pr_tag "$pr")$(rt_tag "$rt")_r$r
        RUN=$O/runs/$tag
        if done_log "$RUN/rc" 'rc=0 '; then log "$tag done already"; continue; fi
        next_port
        if [ "$arm" = primus ]; then
            { echo "export PYTHONPATH=$PT_DIR:\$PYTHONPATH"
              echo "export LD_LIBRARY_PATH=$PT_DIR/primus_turbo/lib:\$LD_LIBRARY_PATH"
              echo "export FLYDSL_AUTOTUNE_CACHE_DIR=$PRIMUS_FLYDSL_CACHE"; } | wfile "$RUN/primus_turbo.env"
            { echo "export MASTER_PORT=$PORT HF_HOME=$PRIMUS_HF_HOME HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
              echo "PRIMUS_DIR=$PRIMUS_DIR TAG=$tag TMO=$PRIMUS_TMO STALL_S=$A2A_STALL_S ITERS=$E2E_ITERS"
              echo "PRIMUS_ARGS='$(primus_args "$m")'"; } | wfile "$RUN/run.env"
            plan_add "$(eval echo \$EST_PRIMUS_MIN_$m)"
            gpu "$PRIMUS_CT" "$tag" "bash $KIT/tools/primus_run.sh $RUN > $RUN/runner.log 2>&1"
            rc=$?
            log_f=$RUN/train_primus.log
        else
            train=examples/qwen3/train_qwen3.sh proxy=examples/qwen3/proxy_mi355x_qwen3_235B_A22B.sh
            [ "$m" = dsv3 ] && train=examples/deepseek_v3/train_deepseekv3.sh proxy=examples/deepseek_v3/proxy_mi355x_deepseekv3_671B.sh
            if [ "$var" = noovl ] || mx_router_off "$pr"; then train=$O/e2e_$(basename "$train"); fi
            { echo "export MASTER_PORT=$PORT HF_HOME=$MOE_HF_HOME DATA_CACHE_PATH=$MOE_DATA_CACHE HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
              echo "export ROCM_PATH=$CT_ROCM HSA_DISABLE_COREDUMP_ON_EXCEPTION=1"
              echo "MLM_DIR=$A2A_MLM_DIR PROXY=$proxy TRAIN=$train TMO=$A2A_TMO STALL_S=$A2A_STALL_S PREC=$pr"; } | wfile "$RUN/run.env"
            arm_env "$m" "$arm" "$var" "$RUN" "$rt" "$pr" | wfile "$RUN/arm.env"
            if [ "$pr" = mxfp8 ]; then plan_add "$(eval echo \$EST_A2A_MX_MIN_$m)"; else plan_add "$(eval echo \$EST_A2A_MIN_$m)"; fi
            gpu "$KOSMOS_CT" "$tag" "bash $KIT/tools/moe_run.sh $RUN > $RUN/runner.log 2>&1"
            rc=$?
            log_f=$RUN/train_$pr.log
        fi
        dry || printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$tag" "$m" "$arm" "$var" "$r" \
            "$rc" "$LOAD0" "$log_f" "$rt" "$pr" >> "$O/runs.tsv"
        fixown "$RUN"
        dry || sleep "$POST_RUN_SLEEP_S"
    done
done
plan_report a2a
