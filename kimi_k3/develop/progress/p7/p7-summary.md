# P7 — Trainer (COMPLETE)

## 1. Objective

Turn the completed decoder into something that trains, and measure what it costs.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/training/pretrain_kimi_k3.py` | `model_provider` / `forward_step` in Megatron's shapes, plus a self-contained `train_smoke` loop |
| `kimi_k3/tools/tokenizer.py` | tiktoken ranks + the released special-token ids |
| `kimi_k3/tests/test_k3_p7_trainer.py` | 7 tests |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G27** — it trains | **GREEN at tiny** | 30 steps on a fixed batch drive the loss from 8.36 to ~0.0 under **both** `dist_muon` and `adam`; on fresh random tokens it sits at `ln(4096) = 8.32`, i.e. chance |
| **G28** — memory report | **GREEN** | the 4 L official config **fits on one node**: 16.31 B params/rank, peak 188.7–202.2 GiB of 288 GB, three steps with finite losses on all 8 ranks, no fallbacks needed (`results/official_smoke.md`) |
| **G29** — checkpoint round-trip | **GREEN** | save, rebuild from a *different* init, load, and the forward matches bitwise |

`pytest kimi_k3/tests/ -q` → **168 passed, 1 skipped**.

## 4. Two assertions, not one

A single "the loss went down" check would pass on a broken model. So the trainer
is gated twice:

* on **fresh random tokens** the loss must sit near `ln(vocab_size)`. Far above
  means something is broken; far *below* would mean the model can see its own
  labels — a leak that a decreasing-loss check would happily accept.
* on a **fixed batch** the loss must fall by more than 2 nats. That is the
  difference between "the optimizer ran" and "the model learns", and it
  exercises KDA, gated MLA, the AttnRes layer, LatentMoE and the router
  together.

Both pass under `dist_muon` and `adam`, which also confirms the Muon parameter
split does not strand any tensor: every parameter is in exactly one group and
each group's optimizer moves it.

## 5. The official-width run

It fits, and the analytic model was exact: `06-capacity-and-parallelism.md`
predicted **16.3 B parameters per rank** for 4 L at EP = 8, and the measurement is
16.31 B. Peak memory is 188.7–202.2 GiB of 288 GB at `seq = 512` with full
recompute, so none of the documented fallbacks were needed.

The optimizer figure (8.78–9.08 B/param) sits *above* G5's 7.87 and should:
under EP = 8 with a world size of 8, the expert-parallel DP group has size 1, so
`dist_muon` shards only the non-expert 2 % of the model. G5 measured the opposite
corner (DP = 8, EP = 1). Both follow from the same model of what
`LayerWiseDistributedOptimizer` shards.

What it does not establish: `seq = 512`, three steps, mock data. Sequence-length
scaling is activation-dominated and the AttnRes payload grows with it, so 8 k
needs its own measurement.

## 6. Still owed elsewhere

Unchanged from earlier phases: production-geometry KDA
parity (G15 nightly), the 8-rank EP exercise (G26), and the AITER kernel path
(P10) which needs the workspace checkout bumped and rebuilt.

## 7. Commit chain

| commit | scope |
| --- | --- |
| **110c84c69** | P7 T7.1/T7.2/T7.5: trainer, tokenizer, checkpoint round-trip |

**P7 is partial**: T7.3 and T7.4 (the 4 L official run and its memory report) are
owed to the nightly job.
