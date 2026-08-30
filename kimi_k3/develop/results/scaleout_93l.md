# G46 — what training 93 L actually takes

> `kimi_k3/config/scaleout.py`; tests in `test_k3_p12_scaleout.py`.
> Measurements: `results/raw/headroom_fla.jsonl`, `headroom_final.jsonl`.
> No cluster job is launched from here (R10.1).

93 L, **seq 8192**, `fla` KDA backend (the default since 2026-08-30), full
recompute, micro-batch 1, MI355X at 288 GiB.

| shape | GPUs | nodes | experts/rank | state | mid-stage | last stage | AttnRes-aligned | fits |
|---|---|---|---|---|---|---|---|---|
| pp4 × ep28 | 112 | 14 | 32 | 320.6 | 457.9 | 457.4 | yes | no |
| pp4 × ep32 | 128 | 16 | 28 | 290.5 | 427.8 | 427.3 | yes | no |
| pp4 × ep56 | 224 | 28 | 16 | 199.9 | 337.2 | 336.7 | yes | no |
| pp8 × ep28 | 224 | 28 | 32 | 160.3 | 228.9 | 228.4 | yes | **yes** |
| pp8 × ep32 | 256 | 32 | 28 | 145.2 | 213.8 | 213.3 | yes | **yes** |
| pp8 × ep56 | 448 | 56 | 16 | 100.0 | 168.6 | 168.1 | yes | **yes** |
| pp16 × ep28 | 448 | 56 | 32 | 80.2 | 114.5 | 125.5 | **no** | **yes** |
| pp16 × ep32 | 512 | 64 | 28 | 72.6 | 106.9 | 117.9 | **no** | **yes** |
| pp16 × ep56 | 896 | 112 | 16 | 50.0 | 84.3 | 95.3 | **no** | **yes** |

## **28 nodes** — `pp8 × ep28`, 224 GPUs, 59 GiB of margin

Not a squeeze: a fifth of each card is free on both the mid and last stages.

## Two constraints bind, and neither is the node count

**`pp` must be 8.** `pp4` fails at every EP — 24 layers per stage exceeds the card
regardless of sharding. `pp16` is cheaper still but breaks AttnRes block
alignment: 93 layers give only eight block-aligned cut points, so any larger PP
leaves a residual block straddling two stages. 28 nodes is the *aligned* floor.

**`ep` must be at least 28.** Per-layer cost has a cliff in expert locality at
seq 8192:

| experts per rank | per KDA+MoE layer |
|---|---|
| 16, 28, 32 | **~5.7 GiB** |
| 56 | ~57 GiB |
| 112 | ~65 GiB |

Ten-fold, between 32 and 56. `ep28` puts 32 experts on a rank and sits just
inside. `headroom_gib()` **raises** above 32 rather than extrapolating — the
mechanism is bracketed, not explained, and a fit through a discontinuity would be
invented rather than measured.

## Where every number comes from

| quantity | value | measured how |
|---|---|---|
| non-expert | **6.00 B/param** | L=1 dense-only at DP=8 |
| expert, expert-DP=1 | **10.66 B/param** | four consistent deltas, L=1..4 |
| per KDA+MoE layer, seq 8192 | **5.72 GiB** | marginals 6.26 / 4.05 / 6.84 at 32 experts/rank |
| fixed: embedding, output, loss | **16.66 GiB** | L=1 at seq 8192 — last stage only |
| params per GPU | analytic | confirmed to the digit by G26 (16,306,993,312) |

`tools/proxy_ep8.py --no-trace` records resident memory on both sides of
optimizer construction, so headroom is peak-minus-**measured**, never
peak-minus-estimate.

## What this replaces, and why the first answer was wrong twice over

The first version of this document said 28 nodes too — by two errors that
cancelled. It applied a **flat 82 GiB headroom** measured at 4 layers and seq 512
to a 12-layer stage at seq 8192 (far too low), against a **flat 7.87 B/param**
applied to expert weights that shard differently (also too low). Getting the same
answer from compensating mistakes is worse than getting a different one, because
nothing looks wrong.

Three things were retracted on the way here, all from computing rather than
measuring:

1. **"~24 GiB of headroom per MoE layer"** — an artefact of the wrong
   bytes-per-param. Measured across an 8x range of expert locality it is flat.
2. **"expert-DP=1 implies 15.17 B/param"** — a correct reading of
   `parallel_state.py` about the *group size*, and wrong about the *cost*.
   Measurement gives 10.66.
3. **"a 108 GiB fixed term"** — 87 % of it was the **eager KDA oracle**, which
   keeps the recurrent state at every timestep for autograd: 109 GiB per layer at
   seq 8192 against `fla`'s 10.2. Bisection found it; it was neither the
   vocab-163840 output layer (removing it changed nothing) nor the naive fp32
   loss (15 GiB in isolation).

That last one is why this document is dated after the R5.3 backend flip. On the
eager default, **28 nodes does not fit at all**.

## Sensitivity

Sequence length is the largest lever — the per-layer term is ~5.7 GiB at seq 8192
and ~1.5 at seq 512. Micro-batch scales the activation term directly; mbs 2 adds
roughly 69 GiB to a mid-stage and would consume the margin.

## Not modelled

* **Pipeline in-flight micro-batches.** Every measurement is pp=1. Under 1F1B an
  early stage holds several micro-batches at once; with full recompute that should
  be layer inputs only (~117 MB/layer at seq 8192), but "should be" is not a
  measurement.
* **expert-DP > 1.** Estimated at 0.72x and flagged `expert_bpp_measured: False`.
  One node cannot produce that configuration.
* **Context parallelism**, which would divide the sequence term and is the obvious
  lever if micro-batch or sequence has to grow.
