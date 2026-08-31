# P12 — Scale-out preparation (COMPLETE; the rest needs more than one node)

## 1. Objective

Work out what running 93 L would take, and hand a human something they can launch.
No cluster job is launched from here (R10.1).

## 2. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G46** — 93 L configs validated analytically | **GREEN** | floor **28 nodes**; every constant measured (`results/scaleout_93l.md`) |
| **G47** — dispatcher A/B + templates | **GREEN** | plan asserted against the pin; arms A and B **run** (`results/dispatcher_ab.md`) |
| **T12.5** — flatness scripts | **GREEN** | `tools/flatness_probe.py`, calibrated (`results/flatness.md`) |

## 3. The answer: 28 nodes

`pp8 x ep28`, 224 GPUs, seq 8192, `fla`, full recompute, mbs 1 — **59 GiB spare**.
Two constraints bind, and neither is the node count: **pp must be 8** (pp4 fails at
every EP; pp16 is cheaper but breaks AttnRes block alignment) and **ep must be at
least 28** (per-layer cost is ~5.7 GiB at 16/28/32 experts per rank and ~57–65 at
56/112 — a tenfold jump between 32 and 56).

Same headline as the incoming plan, reached differently: that number came from two
errors that cancelled — a flat 82 GiB headroom measured at 4 layers/seq 512 applied
to a 12-layer stage at seq 8192, against a flat 7.87 B/param applied to expert
weights that shard differently. Compensating mistakes are worse than a wrong
answer, because nothing looks wrong.

### The 28-node configuration is micro-batch-1 only

| micro-batch | mid-stage | |
|---|---:|---|
| 1 | 228.9 GiB | fits |
| **2** | **297.5 GiB** | **does not fit** |

State (160.3 GiB) does not scale with micro-batch; activations (68.6 GiB) do. So
the 59 GiB margin is **not** room for a bigger micro-batch — it is safety margin.
Throughput at 28 nodes has to come from data parallelism, i.e. more nodes, not
from larger micro-batches. Anyone planning a token budget needs this before they
plan it.

## 4. What the single-node runs actually say about performance

Every timing figure this project has is from a **correctness harness**, not a
throughput benchmark, and it is worth being blunt about the difference.

| configuration | steady iteration | peak/rank | kernel launches |
|---|---:|---:|---:|
| 4 L official, EP=8, seq 512 | **2649.6 ms** | 193.6 GiB | 321,163 |
| 4 L official, EP=4, 2 layers, seq 512 | 1747.6 ms | 139.8 GiB | 213,113 |

At mbs 1 and seq 512 that is **512 tokens per 2.65 s = 193 tokens/s** across eight
GPUs. Roughly 23 TFLOP per iteration (8 x 5.62 B active x 512 tokens, recompute
included) is **~1.1 TFLOP/s per GPU** — a fraction of a percent of the part's bf16
peak.

**That is expected and not a finding about K3.** The configuration is chosen for
memory and correctness: micro-batch 1, short sequence, full recompute, DP=1. The
run is latency-bound — 321 k kernel launches to process 512 tokens — and the
per-iteration fixed costs dominate. The Muon step alone is 21.1 % of device time
and runs **once per iteration regardless of batch size**, so at mbs 1 it is
amortised over almost nothing.

What the single-node runs *do* establish, and what they cannot:

* **established** — where the time goes (Muon 21.1 %, `addmm` 12.3 %, collectives
  6.6 %, AttnRes 0.09 %), that the ranking is stable across a 2x depth change, that
  dispatcher choice moves ~1 %, and that memory scales as `results/scaleout_93l.md`
  says;
* **not established** — anything about tokens/s at a real training shape. That
  needs larger micro-batches, which the memory model says 28 nodes does not have
  room for, so it is a multi-node question too.

## 5. Remaining, and all of it needs more than one node

| item | why it is blocked | tool |
|---|---|---|
| **≥13-layer AttnRes proxy** | **structurally impossible on one node.** Two AttnRes slots need 13 layers; 13 layers at official width is **484 GiB/GPU at EP=8**, 1.7x a card. EP=7 at 4 L was tried and OOMed, pinning the model's edge. Every measurement so far has `K ≤ 1` slot against 8 at 93 L, and the mixer is `O(K+1)` — so its share **cannot be ranked** from any single-node trace | `proxy_ep8.py --layers` |
| **Production-width twins** | tiny/40 steps can only say "not resolvable here" — that is what priced per-head Muon's 1.75 % cost against an unmeasurable benefit. The **band must be re-measured** at production geometry: it moved when the KDA default flipped, which established that a band belongs to a configuration | `twin_run.py` |
| **G41 at scale** | QAT trains and its offset is stable at 4 L (drift −5.0e-04). A *convergence* claim needs real data. The sharper question is deployment: serving without the activation quantisation the model was trained under changes **one token in ten** | `qat_twin.py` |
| **Dispatcher arms C–E** | needs `deep_ep`/`mori`, neither installed — and **lowest value**: A and B are 1 % apart and collectives are ~11 % of device time, so no dispatcher can win more than that. Worth running only where communication dominates, which is the same multi-node condition the AttnRes proxy needs | `proxy_ep8.py --dispatcher` |

## 6. Two assumptions a reader must not inherit silently

* **The expert-locality cliff is bracketed, not explained.** Per-layer cost jumps
  ~10x between 32 and 56 experts per rank. The 28-node answer depends on `ep28`
  sitting on the good side at 32. `headroom_gib()` **raises** rather than
  extrapolate across it, so anyone wanting `ep16` must measure that gap.
* **`expert-DP > 1` bytes/param is estimated at 0.72x, never measured.** One node
  cannot produce that configuration; it is flagged `expert_bpp_measured: False`.
