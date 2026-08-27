# P9 — Training equivalence, per-head Muon, resume hardening (COMPLETE)

## 1. Objective

Show that changes which should not move the model do not move it, give Muon a
head split K3 actually needs, and make a resume survive the pipeline.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/optim/per_head_muon.py` | head splits for KDA and MLA projections, `PerHeadMuon`, the parameter-group policy |
| `kimi_k3/optim/resume.py` | the `dist_muon` resume workaround and its detector |
| `kimi_k3/attention/gated_mla.py` | `_clip_qk` implemented; per-head max-logit recording |
| `kimi_k3/tools/twin_run.py` | noise band, twin axes, axis-engagement evidence |
| `kimi_k3/tools/pp_resume_probe.py` | PP=2 save/resume with two negative controls |
| `kimi_k3/tests/test_k3_p9_{per_head_muon,qk_clip,twin_run,resume}.py` | 25 tests |

`pytest kimi_k3/tests/ -q` -> **208 passed, 3 skipped**.

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G34** — noise band, twins inside it | **GREEN at tiny** | band max 0.2382 / mean 0.0837 / final 0.0488; eager-vs-`fla` at 0.1371 / 0.0223 / 0.0184 (`results/twin_runs.md`) |
| **G35** — per-head Muon parity | **GREEN** | one split step **bitwise-equal** to N core-Muon steps on the head slices, on both head axes; policy asserted against a real model |
| **G36** — `clip_qk` active and logged | **GREEN at tiny** | core's own `clip_qk(model)` drives our MLA unmodified; max logit falls to the threshold, heads already under it untouched |
| **G37** — resume stability under PP | **GREEN, bitwise** | 10 post-resume steps identical with and without dropout; both controls fail (`results/pp_resume.md`) |

## 4. The finding: `dist_muon` could not be resumed at all

`ChainedOptimizer.state_dict()` returns a **list** when more than one optimizer is
chained, and `dist_muon` always chains two. `LayerWiseDistributedOptimizer.load_state_dict`
calls `.values()` on it. The layer-wise optimizer cannot read what its own
`state_dict()` writes, so every `dist_muon` resume fails — on the optimizer P0
measured at 7.87 B/param and this project recommends for K3. Register: **A15**.
The workaround is fork-local (R2.1) with a tripwire that fails once core fixes it.

## 5. Two things that would have passed for the wrong reason

* The **recompute twin is bitwise 0.0**, which is what a correct recompute path
  produces — and also what a flag that does nothing produces. The harness now
  counts calls into `tensor_parallel.checkpoint`: 0 off, 4 on.
* The **RNG negative control was inert** at first, because dropout is 0 at tiny
  and nothing else drew from the generator. With dropout at 0.1 it diverges by
  0.129 while the real resume stays bitwise.

Both were caught by asking what the control would look like if it were broken.

## 6. Why the head split is ours

Core keys `is_qkv` on the literal name `linear_qkv.weight` and carries an explicit
`TODO(deyuf): support MLA`. K3 has no fused QKV, so nothing would ever split, and
each `[12288, 7168]` attention matrix would go through Newton-Schulz whole —
normalising each head's gradient against the others and taking a spectral scale
for a matrix the model never uses as one.

A detail that falls out and is worth stating: TP splits attention **by head**, so
every head slice is complete on one rank and is orthogonalized with
`partition_dim=None` and no `tp_group`. That is the opposite of core's fused-QKV
split, where a slice still spans the sharded query-group axis.

## 7. Owed

The 1 k-step production-width twins remain scheduled work, not CI, and the band
has to be re-measured at that geometry — this one says nothing about it. G36's
"no divergence over the window" is shown at tiny only. Resume does not yet cover
`torch_dist` sharded checkpoints or a DP size that changes across the restart.
