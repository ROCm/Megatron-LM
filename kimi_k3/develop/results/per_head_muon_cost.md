# What per-head Muon costs

> `/tmp/perhead_bench.py`-style measurement, gfx950, `float32_matmul_precision="medium"`
> (what `TensorParallelMuon` runs in), 5 Newton-Schulz steps, quintic coefficients.
> P11's budget table listed this as the first thing to check. This is that check.

## The result

| matrix | whole | per-head | ratio |
|---|---:|---:|---:|
| KDA q/k/v `[12288, 7168]` → 96 × `[128, 7168]` | 15.69 ms | 27.29 ms | **0.57x** |
| KDA o_proj `[7168, 12288]` → 96 × `[7168, 128]` | 15.50 ms | 27.73 ms | **0.56x** |
| MLA q_b `[18432, 1536]` → 96 × `[192, 1536]` | 1.42 ms | 24.93 ms | **0.06x** |

Per-head is **~1.75x slower** on the KDA projections and **17x slower** on MLA's
`q_b`. Extrapolated across the attention matrices of a 93 L model at TP=1:
**8.00 s → 13.97 s** per optimizer step. (The absolute figure is a single-GPU
extrapolation and not comparable to the P11 trace, where matrices are sharded;
the **ratio** is the finding.)

## Why MLA is so much worse

Newton-Schulz is five matmuls per matrix. Splitting `[18432, 1536]` into 96 slices
of `[192, 1536]` turns 5 large GEMMs into 480 small ones — the work barely
changes, the launch count does. KDA's slices are `[128, 7168]`, still small but
wide enough to amortise; MLA's are `[192, 1536]`, and at that size the step is
launch-bound rather than compute-bound.

So the cost is not "per-head is expensive" but "per-head is expensive **when the
slice is small**", and MLA's `q_b` at 1536 wide is where it falls off.

## What this does and does not say

It does **not** say per-head Muon is wrong. P9's argument stands and is about
optimisation behaviour, not speed: Newton-Schulz drives singular values toward 1
and `get_muon_scale_factor` then rescales by the matrix's shape, so orthogonalising
the stacked matrix normalises functionally independent heads against one another
and computes a spectral scale for a matrix the model never uses as one. G35 shows
the split is exact — one split step is bitwise-equal to N independent core-Muon
steps.

What it says is that the correctness argument has a **measured price**, on the
largest single row in the model (the Muon step, 21.1 % of device time at EP=8),
and that the price is not uniform across K3's attention matrices.

## Recommendation

`PerHeadMuon` is opt-in — `tag_k3_heads` has to be called, and nothing in
`build_optimizer` calls it. Leave it that way, and if it is enabled, consider
tagging **KDA only**: the 0.57x on KDA buys the per-head spectral scale where the
slices are wide, while MLA's 0.06x pays 17x for the same property on matrices
small enough that the stacked scale is a far weaker objection.

That selective policy is not implemented. It would be a one-line change to
`head_split` (drop the MLA table, or gate on slice width), and it should be made
on a convergence result rather than on this timing alone — a 1.75x optimizer step
may well be worth it if per-head Muon trains better, which is a twin-run question
(P9's protocol), not a benchmark question.

## Owed

Whether per-head Muon actually trains better. Nothing measured here bears on it.
