# P7 — Trainer (PARTIAL: tiny end-to-end green, official-width owed)

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
| **G28** — memory report | **owed** | the 4 L official run needs 8 GPUs and belongs to the nightly job; the analytic model and the G5 measurements stand in until then |
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

## 5. What is honestly not done

The 4 L official config (94 B parameters, 8 GPUs, EP=8) has not been run. That is
G28's substance, and with it the plan's documented fallbacks (seq 4 k, higher
grad-accum, precision-aware Adam). It needs a multi-GPU launch and a real time
budget, so it belongs to the nightly job rather than to an interactive session.

Also unfinished from earlier phases, and unchanged: production-geometry KDA
parity (G15 nightly), the 8-rank EP exercise (G26), and the AITER kernel path
(P10) which needs the workspace checkout bumped and rebuilt.

## 6. Commit chain

| commit | scope |
| --- | --- |
| **110c84c69** | P7 T7.1/T7.2/T7.5: trainer, tokenizer, checkpoint round-trip |

**P7 is partial**: T7.3 and T7.4 (the 4 L official run and its memory report) are
owed to the nightly job.
