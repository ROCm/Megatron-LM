# Per-head Muon — twin run, KDA only

> `python -m kimi_k3.tools.twin_run --preset tiny --steps 40 --axes per_head_muon`
> Raw: `results/raw/twin_per_head_muon.json`. Tiny, 40 steps, fixed batch,
> `dist_muon`, fp32, **fla** KDA backend (the default since 2026-08-30).

## The question

Per-head Muon costs **1.75x** on the Muon step, which is 21.1 % of device time
(`results/per_head_muon_cost.md`). The split itself is exact — G35 shows one split
step is bitwise-equal to N independent core-Muon steps — and the argument for it
is about optimisation behaviour: orthogonalising the stacked matrix normalises
functionally independent heads against one another and takes a spectral scale for
a matrix the model never uses as one.

So the question is not "does it change the update" (G35's control asserts it
does) but **does it change the update by more than reseeding does**. If not, there
is no evidence here for paying 1.75x.

## Result

| | max Δ | mean Δ | final-quarter Δ |
|---|---|---|---|
| noise band (3 seeds) | 0.2679 | 0.1165 | 0.1230 |
| **per-head vs whole-matrix** | **0.2415** | **0.1104** | **0.0239** |
| | inside | inside | inside, by 5x |

**Inside the band on all three statistics.** Final losses 0.16548 baseline against
0.15067 per-head — and the final-quarter delta (0.0239) is *five times smaller*
than the band's (0.1230), meaning the two runs converge closer to each other than
two seeds of the same configuration do.

## What that does and does not license

It does **not** say per-head Muon is worthless. Forty steps at tiny width on a
fixed batch cannot resolve an optimisation-quality difference; the band being
wider than the effect is exactly the regime where a twin says "not measurable
here", not "not real".

It does say there is **no evidence in hand** for paying 1.75x on the largest row
in the model. The default stays **off** (`k3_per_head_muon: bool = False`), and it
should stay off until a production-width run over many more steps says otherwise.
That is scheduled work, not a benchmark.

## The band moved, and that is worth recording

`results/twin_runs.md` recorded the band as **0.2382 / 0.0837 / 0.0488** on
2026-08-27. Re-measured here with the same seeds, same steps, same fixed batch, it
is **0.2679 / 0.1165 / 0.1230**.

Nothing about the protocol changed — the **KDA backend default did**, from `eager`
to `fla` (R5.3 flip, 2026-08-30). Different kernel, different rounding, different
seed-to-seed spread. The old band was reproducible to the digit across two runs on
the old default, and this one is reproducible on the new one.

The consequence is procedural: **a noise band is a property of a configuration,
not of a project.** Any twin compared against a band measured under a different
default is comparing against the wrong yardstick. `twin_runs.md` now carries both
bands with the backend that produced each.
