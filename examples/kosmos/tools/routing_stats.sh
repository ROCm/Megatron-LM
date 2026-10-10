#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# tools/routing_stats.sh DIR: the routing a run actually had, from its MOE_ROUTING_STATS files DIR/rank<R>.txt (one line
# per router call: layer number, then this rank's tokens per expert). Calls are matched by line across the ranks and
# summed. Prints "calls rank_max_over_mean hottest_expert_over_mean kosmos_cf_needed", each the maximum over calls:
# the busiest rank's routed rows over the mean per rank; the hottest expert's rows over the mean per expert; the rows
# the busiest rank receives with each expert padded to 256 (KOSMOS's arena rows) over T*k. "- - - -" without files.
set -uo pipefail
D=$1
n=$(ls "$D"/rank*.txt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "- - - -"; exit 0; }
awk -v ep="$n" '
    FNR == 1 { nf++ }
    { if (nf == 1) { L[FNR] = $1; E = NF - 1 } else if ($1 != L[FNR]) bad = 1
      for (e = 2; e <= NF; e++) C[FNR, e - 1] += $e
      if (FNR > nc) nc = FNR }
    END {
        if (bad || nf != ep || E % ep) { print "- - - -"; exit }
        el = E / ep
        for (c = 1; c <= nc; c++) {
            tot = 0; hot = 0; rmax = 0; pmax = 0
            for (d = 0; d < ep; d++) {
                r = 0; p = 0
                for (e = 1; e <= el; e++) { v = C[c, d * el + e]; r += v; p += int((v + 255) / 256) * 256; if (v > hot) hot = v }
                tot += r; if (r > rmax) rmax = r; if (p > pmax) pmax = p
            }
            if (!tot) continue
            a = rmax / (tot / ep); b = hot / (tot / E); f = pmax / (tot / ep)
            if (a > A) A = a; if (b > B) B = b; if (f > F) F = f
        }
        printf "%d %.3f %.3f %.3f\n", nc, A, B, F
    }' "$D"/rank*.txt
