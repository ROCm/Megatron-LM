# G66/G67 — three attempts at making Muon's Newton-Schulz cheaper

Muon is ~81 % of the proxy iteration and is third-party code
(`emerging_optimizers` 0.3.0 + `megatron/core/optimizer/muon.py`), so every lever
here is an injection at the boundary rather than an edit. G65 established what it
is spending that time on: **719 matrices per rank, 93 % of them expert weights**,
each orthogonalised alone in a 5-step Newton-Schulz of three GEMMs.

Three levers were tried. Two are dead; the third is wired behind
`--muon-batch-ns`.

## What Newton-Schulz actually does per step

```
A = X @ X.mT          # symmetric
B = b*A + c*(A @ A)   # symmetric (A is)
X = a*X + B @ X       # not symmetric
```

Two of the three GEMMs are symmetric, which is what makes a SYRK-shaped kernel
interesting at all. The third is the largest of the three and cannot be helped.

Two facts that the levers depend on, both measured rather than assumed:

- **Operands are bf16, not fp32.** `OrthogonalizedOptimizer.step` enters a
  `fp32_matmul_precision("medium")` context for the whole step, and
  `newton_schulz` responds by casting X to bf16 explicitly
  (`muon_utils.py:209`). Sampling `torch.get_float32_matmul_precision()` outside
  that context reports `highest` and is misleading; the trace shows Tensile
  `_BBS_BH_` (bf16) kernels, not `_S_B_`.
- **`partition_dim` is None for every parameter.** K3 runs
  `--muon-tp-mode blockwise`, and `TensorParallelMuon.orthogonalize` maps that to
  `partition_dim=None`, so `newton_schulz_tp` short-circuits to a purely local
  `newton_schulz` with no collective inside the iteration.

---

## Lever 1 — the in-tree Triton SYRK kernel (G66): **1.63–2.19× slower**

`emerging_optimizers/triton_kernels/syrk.py` ships a `tsyrk_ex` and
`muon_utils.newton_schulz_step_tsyrk` uses it for the two symmetric GEMMs. It
never runs, for exactly one reason: Megatron builds `ns_kwargs` from `steps`,
`tp_group`, `partition_dim`, `tp_mode`, `coefficient_type` (`muon.py:106`) and
never passes `use_syrk`, so it takes the library default `False`. The precision
gate it sits behind (`== "medium"`) is already open.

`install_muon_syrk()` injects the flag. Wrapping `newton_schulz` rather than
`newton_schulz_tp` matters — only the former has the parameter, and the TP
wrapper delegates to it by module-level name, so one patch covers both paths.

It works, and it is slower: **1.63–2.19×** against the ordinary GEMM at K3's
shapes, plus a 486-config autotune sweep costing **18.6 min** on first call. The
triangular schedule halves the MFMA work but a single expert's Gram is too small
to fill the GPU in the first place, so halving the tiles just idles CUs.

Kept in tree behind `--muon-syrk`, off by default. It is the honest record of why
the kernel that is already installed does not help.

## Lever 2 — batched Newton-Schulz via `torch.baddbmm` alone: **1.02–1.07×**

`newton_schulz` already dispatches 3-D input to `batched_newton_schulz_step`
(`muon_utils.py:201`). Stacking 16 same-shaped matrices and calling it once cuts
the launch count 16× but barely moves the clock: the matrices are already at
~78 % of peak individually, so there is little idle to recover, and `baddbmm` is
*slower per element* than the `addmm` the serial path uses.

Measured alone this is not worth the stacking copies. It is a prerequisite for
lever 3, not a lever itself.

## Lever 3 — quack-flydsl `batched_tsyrk_ex` (G67): **1.055× end to end**

From `github.com/wenchenvincent/quack-flydsl` @ `wip/amd-flydsl-port`. Needs
FlyDSL 0.2.x on the path: 0.1.1.dev409 has no `flydsl.expr.math`, 0.3.2 renamed
`expr.vector` → `expr.Vector`. The global install is pinned by `amd-aiter` (MoRI
imports it), so 0.2.4 lives in a separate tree reached by `PYTHONPATH`.

Both K3 Gram shapes clear `can_use_batched_tsyrk` (N % 256 == 0, K % 128 == 0).

### Kernel alone, against `torch.baddbmm`

| Gram | B | baddbmm | quack | speedup | cost for all 336 |
|---|---:|---:|---:|---:|---:|
| (3584, 3584) from (B, 3584, 6144) | 1 | 0.143 ms | 0.170 | 0.84× | 57.2 ms |
| | 4 | 0.536 | 0.381 | 1.41× | 32.0 |
| | 8 | 0.985 | 0.788 | 1.25× | 33.1 |
| | 16 | 2.004 | 1.395 | **1.44×** | **29.3** |
| (3072, 3072) from (B, 3072, 3584) | 1 | 0.070 | 0.112 | 0.63× | 37.6 |
| | 4 | 0.265 | 0.243 | 1.09× | 20.4 |
| | 8 | 0.476 | 0.398 | 1.20× | 16.7 |
| | 16 | 0.816 | 0.682 | **1.20×** | **14.3** |

**At B=1 the kernel loses** (0.84× / 0.63×) for the same reason lever 1 lost, and
its own docstring says so. Batching is what turns the triangular schedule from a
liability into a win. This is why lever 1 and lever 3 disagree despite being the
same idea.

### Whole optimizer step, isolated

> `python -m kimi_k3.tools.muon_batch_ns {serial,batch,syrk} --n 48 --batch 16`,
> one rank, 48 matrices of each expert shape plus one unbatchable (896, 7168).

| arm | steady step | vs serial |
|---|---:|---:|
| serial | 199.24 ms | 1.000× |
| batched, `baddbmm` | 191.17 | 1.042× |
| batched, quack `batched_tsyrk_ex` | 180.88 | **1.102×** |

The 1.44× on the Gram becomes 1.10× on the step, because only two of the three
GEMMs are symmetric, the unaccelerated third is the largest, and normalize /
stack / dtype conversions are unchanged.

### Whole iteration, 8 ranks

> `torchrun --standalone --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L
> --ep 8 --seq 512 --iterations 12 --warmup 4 --no-trace [--muon-batch-ns 16
> [--muon-batch-syrk]]`. Raw: `results/raw/muon_batch_ns_e2e.jsonl`.

Each arm was run twice, the second pass in reversed order, because G50 retracted a
1.44× that was an artefact of non-adjacent runs.

| arm | pass a | pass b | mean | vs base |
|---|---:|---:|---:|---:|
| base | 1838.6 ms | 1837.6 | 1838.1 | 1.000× |
| `--muon-batch-ns 16` | 1819.1 | 1819.8 | 1819.5 | 1.010× |
| `+ --muon-batch-syrk` | 1742.1 | 1741.7 | **1741.9** | **1.055×** |

**96.2 ms per iteration.** The order-reversed spread is 1.0 / 0.7 / 0.4 ms against
a 96 ms separation, so no significance test is needed to separate the syrk arm;
batching alone (18.6 ms) also clears its own 0.7 ms spread.

For scale, this single lever is **4.5× the combined win of all four fused model
kernels** (AttnRes + SiTU + gated RMSNorm + causal conv, together 1.2 %).

**Peak memory is unchanged** -- byte-identical per-rank peaks across all three
arms (185.93 / 185.95 / 186.28 / 186.30 / 190.99 / 199.42 ×2 GiB). Peak occurs in
forward/backward, not in the optimizer step, so the `(B, M, N)` staging buffer is
free. This was worth checking: peak memory is what sizes the 93 L cluster
(`results/scaleout_93l.md`), and a lever that traded 1 % of time for GiB would not
be worth taking.

### Numerics

After 4 steps, worst rel-L2 of the parameter against the serial path:

| arm | vs serial | vs the other batched arm |
|---|---:|---:|
| batched, `baddbmm` | 2.79e-04 | — |
| batched, quack | 2.80e-04 | 2.27e-04 |

Both batched arms differ from serial by the same amount, and quack adds nothing
on top — the drift is bf16 kernel selection (`baddbmm` vs `addmm`), not the
triangular kernel. The Gram itself carries ~1.5e-03 rel-err against an
fp32-accumulate reference, but Newton-Schulz is self-correcting and it does not
survive to the parameter.

The unbatchable (896, 7168) parameter comes out **bit-identical**, which is the
check that the serial fallback inside the batched step is untouched.

## How the batching works

`install_muon_batched_ns` replaces `TensorParallelMuon.step` with the same loop,
split in two: the per-parameter prologue (weight decay, momentum EMA, Nesterov)
writes directly into a slice of a `(B, M, N)` buffer, then one
`scaled_orthogonalize_fn` call covers the whole bucket, then each parameter takes
its own slice back. `scaled_orthogonalize_fn` needed no change — it reads its
scale from `size(-2)/size(-1)`, which is already correct for 3-D.

Buckets key on (shape, dtype, tp_group). A parameter is batchable only if it is
2-D, has `partition_dim is None`, and is not QKV-split; everything else runs the
original serial path verbatim. That restriction is load-bearing: the TP branches
of `newton_schulz_tp` all-gather along `partition_dim` and would mis-handle the
extra leading dimension.

`_contract_muon_batched_ns` asserts that the four things the copied loop depends
on still exist in `OrthogonalizedOptimizer.step`, so an IFU that changes the loop
fails loudly instead of silently diverging.

### One hazard this created, and the guard for it

Calling `scaled_orthogonalize_fn` directly bypasses any `orthogonalize` override.
`PerHeadMuon` is exactly such an override (`optim/per_head_muon.py:189`) and keys
on a `k3_head_split` attribute the tagging pass puts on KDA attention matrices --
which are 2-D with `partition_dim=None`, so they would have landed in a bucket and
had their head split **silently dropped** under `--k3-per-head-muon`. Parameters
carrying that attribute are now excluded from batching, and the pin contract
asserts `PerHeadMuon.orthogonalize` still keys on it.

## What is left on the table

The third GEMM of each step (`B @ X`, the largest of the three) is not symmetric
and stays on the general path. `gemm_symmetric_pingpong`, quack's 2-D fast path,
is still unmeasured in-model -- but lever 1 predicts it will lose for the same
B=1 reason, so the batched kernel is the one that matters.
