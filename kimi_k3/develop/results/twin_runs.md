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

**This band is for the `eager` KDA backend**, which was the default when it was
measured. Re-measured on 2026-08-30 with identical seeds, steps and batch but the
`fla` default (R5.3 flip), it is **0.2679 / 0.1165 / 0.1230**. A noise band is a
property of a configuration, not of a project: comparing a twin against a band
measured under a different default is comparing against the wrong yardstick.

| backend | max Δ | mean Δ | final-quarter Δ |
|---|---|---|---|
| `eager` (2026-08-27) | 0.2382 | 0.0837 | 0.0488 |
| **`fla` (2026-08-31, current default)** | **0.2679** | **0.1165** | **0.1230** |

Each band is reproducible to the digit on its own default, across separate runs.

## Both twins re-validated against the current band (2026-08-31)

Certifying a twin against a band measured under a different default is comparing
against the wrong yardstick, so both were re-run rather than assumed to carry over:

| axis | max Δ | mean Δ | final-quarter Δ | inside |
|---|---|---|---|---|
| KDA backend, eager vs `fla` | **0.0765** | **0.0192** | **0.0151** | yes |
| recompute off vs full | **0.0000** | **0.0000** | **0.0000** | yes |

The KDA twin came in **tighter** than on the eager-era band — 0.0765 against 0.1371
on max, and 0.0151 against 0.0184 on the final quarter — while the band it is
measured against got wider. Both twins now sit further inside than before.

That is worth stating carefully, because it is easy to over-read. The twin
compares the same two backends either way; what changed is which one the *rest of
the model* runs, and the band it is judged against. It is not evidence that `fla`
improved, only that the comparison did not degrade when the default moved.

The recompute twin is still bitwise 0.0, with the checkpoint path still proven to
fire — 0 calls off, 4 on. A correct recompute replays a forward, so the backend
should not matter; that is a prediction about a different kernel's determinism,
and it is now measured rather than reasoned.

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

---

# G55 — re-run under SiTU, and the three-seed band was too narrow

> `python -m kimi_k3.tools.twin_run --preset tiny --steps 40 --seeds 0 1 2 3 4 5`
> Raw: `results/raw/twin_run_tiny_situ.json` (3 seeds),
> `results/raw/twin_run_tiny_situ_6seed.json` (6 seeds).

Every twin verdict before this ran **GeGLU** experts (G52). The band is a property
of a configuration, so it was re-measured rather than reused -- the same rule this
document already applied across the eager->fla flip.

## The band

| seeds | pairs | max Δ | mean Δ | final Δ |
|---|---|---|---|---|
| 0,1,2 | 3 | 0.2911 | 0.0900 | 0.0593 |
| **0..5** | **15** | **0.3114** | **0.1271** | **0.0794** |

The three-seed estimate is low on every statistic, worst on `final_delta` (+34%).
That is structural, not luck: the band is a **max over pairs**, and three seeds
samples that maximum three times. Across the 15 pairs the median `final_delta` is
0.0465 against a max of 0.0794, so the tail the band is trying to capture is well
above typical.

## The axes, against the six-seed band

| axis | max Δ | mean Δ | final Δ | inside band |
|---|---|---|---|---|
| `recompute` | **0.0** | **0.0** | **0.0** | yes -- bitwise, 4 checkpoint calls vs 0 |
| `kda_backend` | 0.1702 | 0.0266 | 0.0754 | **yes** |
| `per_head_muon` | 0.3598 | 0.1721 | 0.1044 | no (max and mean both outside) |
| `situ_activation` | 0.1776 | 0.0709 | 0.0892 | no (`final` only) |

`recompute` is bitwise identical, which is the correct answer and unchanged by
SiTU -- recompute cannot alter arithmetic. `kda_backend` is inside: eager vs fla is
still not shown to move the model.

## The near-miss

Against the three-seed band `kda_backend` read **outside** it, on `final_delta`
alone (0.0754 vs 0.0593), and it would have been written up as a finding: "SiTU
tightened run-to-run variance and exposed a pre-existing eager/fla difference". It
is not true. With a properly sampled band the twin is inside.

Quantified, because one near-miss does not establish how bad the default was:
**16 of the 20 possible three-seed subsets produce a band that fails
`kda_backend`.** The default was wrong 80% of the time on this axis. It is now six
seeds, and the docstring carries the measurement.

## The control, and an honest caveat about it

`situ_activation` (the G52 fix itself) was added as a control: if wiring the
correct activation moved the loss no further than reseeding does, the tool would
not be measuring anything. It lands outside the band -- but **only on
`final_delta`** (0.0892 vs 0.0794), and inside on max and mean. That is a weaker
control than a single-statistic reading suggests, and by the standard applied to
`kda_backend` above it is close enough to the boundary to deserve the same
scepticism. What can be said is that the fix is *not* clearly inside the band; a
stronger statement needs more seeds or more steps.

## Scope

Tiny preset, 40 steps, fixed batch, fp32. This re-validates the twin axes under
the corrected activation. It does **not** re-validate the QAT convergence study or
the flatness probes, which also predate G52.
