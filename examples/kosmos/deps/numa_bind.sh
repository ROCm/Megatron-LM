#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Per-rank NUMA binding, Primus convention (Primus runner/helpers/numa_bind.sh): used as
#   torchrun ... --no-python numa_bind.sh <script.py | cmd> ARGS...
# GPU bus id of LOCAL_RANK from `amd-smi list --csv`, NODE = /sys/bus/pci/devices/<bus>/numa_node, then
# numactl --cpunodebind=$NODE --membind=$NODE. Appended to $NUMA_BIND_LOG per rank, read in the process that then execs
# the trainer (same pid): Cpus_allowed_list / Mems_allowed_list from /proc/self/status, the memory policy of the first
# /proc/self/numa_maps entry (--membind is a mempolicy; Mems_allowed_list is the cpuset and stays 0-1), and
# numactl --show's cpubind / membind.
mapfile -t BUS_ID < <(amd-smi list --csv | awk -F, 'NR>1 && $2!="" {print $2}')
NODE=$(cat /sys/bus/pci/devices/"${BUS_ID[$LOCAL_RANK]}"/numa_node)
if [[ $# -gt 0 && ( "${1}" == *.py || "${1}" == "-m" ) ]]; then
    set -- python3 "$@"
fi
LOG=${NUMA_BIND_LOG:-/dev/stderr}
exec numactl --cpunodebind="$NODE" --membind="$NODE" bash -c '
    st=$(grep -E "^(Cpus_allowed_list|Mems_allowed_list)" /proc/self/status | tr "\t\n" "  ")
    pol=$(grep -m1 -oE "(bind|prefer|interleave|default|local)[:0-9,-]*" /proc/self/numa_maps)
    sh=$(numactl --show | grep -E "^(cpubind|membind)" | tr "\n" " ")
    echo "[numa_bind] rank=$RANK local_rank=$LOCAL_RANK bus=$1 node=$2 pid=$$ $st mempolicy=$pol $sh" >> "$3"
    shift 3; exec "$@"' _ "${BUS_ID[$LOCAL_RANK]}" "$NODE" "$LOG" "$@"
