# G34 — twin runs and the noise band

> `python -m kimi_k3.tools.twin_run --preset tiny --steps 40 --out results/raw/twin_run_tiny.json`
> Raw: `results/raw/twin_run_tiny.json`. Tiny preset, 40 steps, fixed batch, `dist_muon`, lr 1e-4, fp32.

## The band first

Two runs that should be equivalent will not produce identical losses, so "did
this change the model" cannot be answered against zero. It is answered against
the movement the configuration produces for no reason at all: the same run under
three seeds.

| pair | max Δ | mean Δ | final-quarter Δ |
|---|---|---|---|
| seed 0 vs 1 | 0.2382 | 0.0837 | 0.0488 |
| seed 0 vs 2 | 0.1135 | 0.0449 | 0.0334 |
| seed 1 vs 2 | 0.1511 | 0.0609 | 0.0175 |
| **band** (worst of each) | **0.2382** | **0.0837** | **0.0488** |

Three statistics rather than one: `max` catches a single spike a mean would
absorb, `mean` catches a small constant offset, and the final quarter catches
slow divergence — which is the failure that matters and the one the other two
miss.

## The twins

| axis | max Δ | mean Δ | final Δ | inside the band |
|---|---|---|---|---|
| KDA backend, eager vs `fla` | 0.1371 | 0.0223 | 0.0184 | **yes** |
| recompute off vs full | 0.0000 | 0.0000 | 0.0000 | **yes** |

Final loss on the fixed batch: 0.18603 eager, 0.19012 fla — the two backends
train the same model to the same place, and the difference is a third of the
seed-to-seed spread.

## The recompute twin is bitwise zero, and that is the right answer

An activation checkpoint replayed correctly reproduces the same numbers, so 0.0
is what a *working* recompute path produces. The problem is that 0.0 and "the
flag did nothing" look identical from the outside. So the harness proves the path
fired: it counts calls into `tensor_parallel.checkpoint` on each side and records
them — **0 with recompute off, 4 with it on**, one per layer of the tiny preset.
Without that count this twin would be a gate that cannot fail.

## What this does not establish

The band is measured at tiny width, 40 steps, on a fixed batch. It is a check
that a refactor did not move the model, not a convergence result. Production-width
twins over ~1 k steps are scheduled work, not CI, and the band has to be
re-measured at that geometry — a band measured here says nothing about one there.
The comparison is also same-seed by construction: the twin shares its
initialisation with its partner, while the band's runs do not, which makes the
band the more permissive of the two comparisons.
