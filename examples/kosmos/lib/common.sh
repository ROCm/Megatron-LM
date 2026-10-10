#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# lib/common.sh: shared host-side helpers (logging, DRYRUN, container exec, GPU guards, shuffle, ports, plan). Source it.

KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$KIT/config.sh"
DRYRUN=${DRYRUN:-0}

log()  { echo "[$(date '+%F %T')] $*"; }
warn() { echo "[$(date '+%F %T')] WARNING: $*" >&2; }
die()  { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }
dry()  { [ "$DRYRUN" = 1 ]; }

# run CMD...: execute, or print under DRYRUN=1.
run() {
    if dry; then
        printf '[dryrun]'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    "$@"
}

# manual CMD: a step left to the user (network, images, containers, apt).
manual() { echo "[manual] $*"; }

# wfile PATH: write stdin to PATH, or print it under DRYRUN=1.
wfile() {
    if dry; then
        echo "[dryrun] write $1:"
        sed 's/^/[dryrun]   | /'
        return 0
    fi
    mkdir -p "$(dirname "$1")" && cat > "$1"
}

# cexec CT 'cmd': run in container CT as root (or on the host when CT is empty), CPU work.
cexec() {
    local ct=$1; shift
    if dry; then
        echo "[dryrun] ${ct:+$DOCKER exec -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES $ct }bash -lc \"$(echo "$*" | tr -s ' ')\""
        return 0
    fi
    if [ -n "$ct" ]; then
        "$DOCKER" exec -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" "$ct" bash -lc "$*"
    else
        bash -lc "$*"
    fi
}

need_ct() {
    dry && return 0
    [ -z "$1" ] && return 0
    "$DOCKER" inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true ||
        die "container $1 is not running (run_all.sh manual prints how to create it)"
}

load1() { cut -d' ' -f1 /proc/loadavg; }

# wait_load TAG: hold until the host 1-min loadavg is below LOAD_MAX (sets LOAD0 to the value at release).
wait_load() {
    LOAD0=$(load1)
    if dry; then
        [ "$LOAD_MAX" != 0 ] && echo "[dryrun] wait until the 1-min loadavg < $LOAD_MAX (poll ${LOAD_POLL_S}s)"
        return 0
    fi
    [ "$LOAD_MAX" = 0 ] && return 0
    local t0=$SECONDS
    while awk -v l="$(load1)" -v m="$LOAD_MAX" 'BEGIN { exit !(l >= m) }'; do
        if [ "$LOAD_WAIT_MAX_S" != 0 ] && [ $((SECONDS - t0)) -ge "$LOAD_WAIT_MAX_S" ]; then
            warn "loadavg $(load1) still >= $LOAD_MAX after ${LOAD_WAIT_MAX_S}s: starting anyway"
            break
        fi
        log "host loadavg $(load1) >= $LOAD_MAX, waiting ${LOAD_POLL_S}s before $1"
        sleep "$LOAD_POLL_S"
    done
    LOAD0=$(load1)
}

# wait_idle TAG: hold until no process on the node has a KFD (GPU) context.
wait_idle() {
    [ "$GPU_IDLE_CHECK" = 1 ] || return 0
    if dry; then
        echo "[dryrun] wait until /sys/class/kfd/kfd/proc is empty (no GPU context on the node)"
        return 0
    fi
    local t0=$SECONDS n
    while n=$(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l) && [ "$n" -gt 0 ]; do
        [ $((SECONDS - t0)) -ge "$GPU_IDLE_WAIT_S" ] && die "$n processes still hold GPU contexts after ${GPU_IDLE_WAIT_S}s"
        log "$n processes hold GPU contexts ($(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')), waiting before $1"
        sleep 30
    done
}

# gpu CT TAG 'cmd': one GPU invocation: idle + load guards, optional flock, then the command in CT.
gpu() {
    local ct=$1 tag=$2; shift 2
    wait_idle "$tag"
    wait_load "$tag"
    if dry; then
        echo "[dryrun] ${LOCK_FILE:+flock $LOCK_FILE }${ct:+$DOCKER exec -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES $ct }bash -lc \"$(echo "$*" | tr -s ' ')\"   # GPU"
        return 0
    fi
    log "GPU $tag (loadavg $LOAD0)"
    local cmd=()
    [ -n "$ct" ] && cmd=("$DOCKER" exec -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" "$ct")
    cmd+=(bash -lc "$*")
    if [ -n "$LOCK_FILE" ]; then
        flock "$LOCK_FILE" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# sampler_start FILE / sampler_stop: "epoch load1" every LOAD_SAMPLE_S seconds while a step runs.
sampler_start() {
    dry && return 0
    mkdir -p "$(dirname "$1")"
    ( trap 'exit 0' TERM; while :; do echo "$(date +%s) $(load1)"; sleep "$LOAD_SAMPLE_S"; done ) >> "$1" &
    SAMPLER=$!
    trap 'sampler_stop' EXIT
}
sampler_stop() { [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null; SAMPLER=; }

# fixown PATH...: hand files a container created as root back to the caller.
fixown() {
    [ "$FIXOWN" = 1 ] || return 0
    dry && return 0
    [ -n "$KOSMOS_CT" ] && cexec "$KOSMOS_CT" "chown -R $(id -u):$(id -g) $* 2>/dev/null; true"
}

# shuffle SEEDSTR ITEMS...: deterministic shuffle, the LCG of KOSMOS bench/common/env.sh kosmos_shuffle.
shuffle() {
    local s a i j t
    s=$(printf '%s' "$1" | cksum | cut -d' ' -f1)
    shift
    a=("$@")
    for ((i = ${#a[@]} - 1; i > 0; i--)); do
        s=$(( (s * 6364136223846793005 + 1442695040888963407) & 0x7fffffffffffffff ))
        j=$(( (s >> 31) % (i + 1) ))
        t=${a[i]}; a[i]=${a[j]}; a[j]=$t
    done
    echo "${a[*]}"
}

# next_port: advance PORT to the next free TCP port (call directly, not in $(...)).
PORT=${PORT:-$PORT0}
next_port() {
    PORT=$((PORT + 1))
    while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$((PORT + 1)); done
}

# plan_add MIN / plan_report STEP: run count and GPU-minute estimate of a step.
PLAN_N=0; PLAN_MIN=0
plan_add()    { PLAN_N=$((PLAN_N + 1)); PLAN_MIN=$(awk -v a="$PLAN_MIN" -v b="$1" 'BEGIN { print a + b }'); }
plan_report() { log "PLAN $1: $PLAN_N GPU runs, estimated $(awk -v m="$PLAN_MIN" 'BEGIN { printf "%.0f min (%.1f h)", m, m / 60 }') of GPU time"; }

# done_log LOG PATTERN: true if LOG exists and contains PATTERN (the run completed earlier; skipped on a rerun).
done_log() { [ -f "$1" ] && grep -q "$2" "$1"; }
