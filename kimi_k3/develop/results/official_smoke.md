# G28 — the 4 L official config on one node

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.official_smoke --preset 4L --ep 8 --seq 512`
> Raw per-rank rows in `official_smoke_raw.jsonl`. 8 × MI355X (288 GB),
> bf16 weights, `dist_muon`, full activation recompute, EP = 8, TP = PP = 1.

## It fits

| measurement | value |
|---|---|
| parameters per rank | **16.31 B** (of 94.0 B total) |
| after model construction | 30.38 GiB |
| after optimizer construction | 133.40 – 137.96 GiB (8.78 – 9.08 B/param) |
| **peak during training** | **188.70 – 202.22 GiB** (12.43 – 13.32 B/param) |
| headroom on a 288 GB card | ≈ 71 GB at the worst rank |
| three training steps | losses 13.35 / 13.55 / 13.37, all finite, all ranks `ok` |

None of the plan's documented fallbacks (shorter sequence, higher gradient
accumulation, precision-aware Adam) were needed at `seq = 512`.

## What it confirms

**The analytic per-rank parameter model was exact.** `06-capacity-and-parallelism.md`
predicted 16.3 B parameters per rank for this configuration from
`P_nonexpert / (TP·PP) + P_expert / (TP·PP·EP)`. The measurement is 16.31 B.

**The optimizer figure sits above G5's 7.87 B/param, and it should.** Under
EP = 8 with a world size of 8, the expert-parallel DP group has size 1, so
`dist_muon` shards only the non-expert parameters — 98 % of the model is expert
weight whose master and momentum are not shared with anyone. G5's 7.87 was
measured at DP = 8 with EP = 1, the opposite corner. The two numbers agree with
the same model of what `LayerWiseDistributedOptimizer` shards; they are not in
tension.

**Per-rank spread is ±4 %** (188.7 – 202.2 GiB), the same whole-tensor sharding
imbalance G5 saw at DP = 8.

## What this does not establish

`seq = 512`, three steps, mock data. It says the configuration *fits and steps*,
not that it trains well or that it fits at 8 k. The sequence-length scaling is
activation-dominated and the AttnRes payload grows with it
(`results/attn_res.md`), so 8 k needs its own measurement — that is the nightly
job's, along with the 8 L config and the EP ladder.
