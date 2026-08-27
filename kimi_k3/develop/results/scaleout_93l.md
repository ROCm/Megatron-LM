# G46 — 93 L configurations, validated analytically

> `kimi_k3/config/scaleout.py`; tests in `test_k3_p12_scaleout.py`.
> Raw table: `results/raw/scaleout_93l.txt`. No cluster job is launched from here (R10.1).

Every number below is derived from a **measurement**, not an estimate (R9.1):
bytes per parameter from `results/opt_mem.md` (`dist_muon` at DP=8: **7.87**),
parameter counts from `tools/mem_budget.py`'s analytic breakdown, and the 82 GiB
non-parameter headroom from the 4 L run in `results/official_smoke.md`, where
16.31 B params/rank at 7.87 B/param accounted for the rest of a 202 GiB peak.

## The smallest configuration that fits

| config | GPUs | nodes | params/GPU | state GiB | + headroom | fits |
|---|---|---|---|---|---|---|
| pp8 · ep28 | 224 | 28 | 19.25 B | 141.1 | 223.1 | **yes** |
| pp8 · ep32 | 256 | 32 | 17.73 B | 129.9 | 211.9 | **yes** |
| pp4 · ep56 | 224 | 28 | 26.34 B | 193.1 | 275.1 | yes, with 13 GiB spare |
| pp8 · ep16 | 128 | 16 | 28.36 B | 207.9 | 289.9 | no |
| pp8 · ep8 | 64 | 8 | 49.64 B | 363.8 | 445.8 | no |

**28 nodes** is the floor with aligned pipeline boundaries. `pp4 · ep56` reaches
it too but leaves 13 GiB of headroom against 65 GiB for `pp8 · ep28`, so it is
the fragile one.

## PP cannot exceed 8 without splitting an AttnRes block

A layer appends a residual slot when its 0-indexed position is a multiple of 12,
and **the prefix sum resets at that moment** — `prefix_sum = None`, the stream
restarts. A boundary there hands the next stage a payload whose prefix half is
fresh rather than half-accumulated, and whose slot count is exactly `layer / 12`.

93 layers give **seven whole blocks**, so there are only eight places a stage can
begin on a block boundary. Every `pp16` row in the table is therefore flagged. It
is not a correctness failure — P5 proved bitwise transport of a partial prefix
across a stage boundary (G21) — but a `pp16` layout cuts a block in half, and any
repartition or recompute then has to reason about a prefix that began on another
stage. The planner reports the constraint instead of quietly emitting the layout.

| pp | layout | boundaries | payload slots crossing |
|---|---|---|---|
| 4 | 24, 24, 24, 21 | 24, 48, 72 | 2, 4, 6 |
| 8 | 12 x 7, then 9 | 12, 24, …, 84 | 1, 2, 3, 4, 5, 6, 7 |

The last stage is short in both, and it is the one to watch: it carries the tail
block **and** the extra layer, including the final MLA pair (layers 92 and 93,
1-indexed) — the one place the 3:1 KDA/MLA stride breaks. A boundary that lands
between those two is called out by name.

Payload cost grows with depth: the last boundary in a `pp8` layout carries 8x the
hidden state, against 2x at the first. Pipeline stages are not interchangeable
here, and a balanced *layer* split is not a balanced *bandwidth* split.

## EP ladder

8, 16, 28, 32, 56 — all divide 896, and a test asserts it, because an EP that
does not divide the expert count fails deep inside the dispatcher rather than at
configuration time.

## Owed

G47: the dispatcher A/B matrix (stock all-to-all vs DeepEP vs MoRI vs MoonEP) and
the QB-load report template, both as plans rather than implementations.
