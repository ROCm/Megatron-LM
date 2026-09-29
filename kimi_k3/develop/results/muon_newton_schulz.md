# G66/G67 — three attempts at making Muon's Newton-Schulz cheaper

Muon dominates the proxy iteration and is third-party code
(`emerging_optimizers` 0.3.0 + `megatron/core/optimizer/muon.py`), so every lever
here is an injection at the boundary rather than an edit. **Its share is quoted
from kernel attribution, not from the region annotation**: the profiler reports
`Optimizer.step#TensorParallelMuon.step` as 3006.0 ms against a 1843.3 ms
iteration, which is impossible -- the ROCTracer duplicate-flow artifact that
`summarise()` was patched for is only partly neutralised, and any "Muon is N % of
the iteration" figure taken from that row (including the ~81 % quoted in earlier
gate notes) is unsupported. What is measurable is that **GEMM is 949.2 ms of the
1843.3 ms baseline iteration (51 %), and effectively all of it is Newton-Schulz**:
`aten::mm` + `aten::addmm` is 10,293 calls against the 10,785 that G65 predicts
from 719 matrices x 5 steps x 3 GEMMs. G65 established what it
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

### Batch size: the kernel is flat, the iteration is not

B=16 was carried over from the microbenchmark grid, not chosen. Sweeping it
(each point measured 2-3x, reproducing to within 3 ms):

| B | steady | vs base | worst-rank peak | peak cost |
|---|---:|---:|---:|---:|
| 2 | 2800.0 ms | **1.52x slower** | 199.42 GiB | -- |
| 4 | 2310.2 | 1.26x slower | 199.42 | -- |
| 8 | 1951.1 | 1.06x slower | 199.42 | -- |
| 16 | 1741.9 | 1.055x | 199.42 | free |
| 24 | 1681.9 | 1.093x | 199.56 | +0.14 GiB |
| 32 | 1700.0 | 1.081x | 202.84 | +3.4 |
| 48 | 1640.2 | 1.121x | 209.41 | +10.0 |
| 56 | 1706.3 | 1.078x | 212.69 | +13.3 |
| 112 | 1696.2 | 1.084x | 235.66 | +36.2 |
| 168 | 1567.2 | **1.173x** | 258.62 | +59.2 |
| 336 | **OOM** | -- | 263.97 | -- |

Two things here are worth more than the numbers.

**Small B is a trap.** Below 16 batching is *worse than not batching*: B=2 costs
1.52x. The control says this is not the staging copies (their volume is constant
in B, ~19 ms of bandwidth total) -- B=2 with plain `baddbmm` and no quack is
**2924.5 ms**, worse still. It is `bmm` falling off a kernel-selection cliff at
small batch, which `emerging_optimizers` flags in its own docstring
(`muon_utils.py:332`). Anyone picking a conservative small batch size would land
1.5x slower than leaving the feature off.

**The curve is reproducibly non-monotone and it is not the kernel.** B=48 beats
B=56 and B=112 while using less memory than either, in three independent runs.
Timed alone (`scratchpad/bsyrk_curve.py`), `batched_tsyrk_ex` is *flat*: the total
Gram cost for all 336 matrices is 28.3-30.2 ms (fc1) and 14.0-15.5 ms (fc2) at
every B from 16 to 168, with no inversion. The kernel saturates by B=16 and then
scales linearly. So every bit of the 1742 -> 1567 ms gain from B=16 to B=168
comes from *outside* the symmetric GEMM, and batching the full bucket of 336
would buy nothing from the kernel even if it fit -- it OOMs on a 13.78 GiB
allocation with 6.83 GiB free.

**Recommended default: B=24** -- 1.093x for +0.14 GiB, the largest gain that costs
no peak memory at all. Peak is set by forward/backward up to ~B=24 and by the
optimizer above it, so that is the crossover, not a round number. B=48 is
defensible at +10 GiB; B=168 is not at +59, since peak memory is what sizes the
93 L cluster (`results/scaleout_93l.md`).

### Exactly which GEMM the batched syrk replaced

Three arms traced at the same geometry, differing only in the lever. Rank 0, one
traced iteration. `b48` -> `b48syrk` changes *only* the kernel behind the two
symmetric matmuls, so the diff is the attribution:

| kernel | b48 | b48syrk | delta |
|---|---:|---:|---:|
| `Cijk_Alik_Bljk_...MT256x256x64` | 280.5 ms / 64 | **0.0 / 0** | -280.5 |
| `Cijk_Ailk_Bljk_...MT256x256x64` | 421.1 / 125 | 257.6 / 69 | -163.5 |
| `Cijk_Ailk_Bljk_...MT256x240x64` | 85.3 / 28 | 86.7 / 28 | +1.4 |
| `Cijk_Ailk_Bjlk_...MT256x256x64` | 69.0 / 21 | 68.8 / 21 | -0.2 |
| `nt_pp_bf16_256x256x64` (quack) | 0.0 / 0 | **372.2 / 156** | +372.2 |
| **GEMM total** | **855.9** | **785.3** | **-70.6** |

**`batched_tsyrk_ex` took over 442.8 ms of Tensile GEMM and does it in 372.2 ms --
1.19x on the work it actually replaced.** The two replaced GEMMs separate by
operand layout: `Alik_Bljk` (transposed A) is the Gram `X @ X.mT` and vanishes
entirely; `Ailk_Bljk` serves both `A @ A` and `B @ X`, so syrk takes the `A @ A`
half and the 257.6 ms / 69 calls left behind is the non-symmetric third GEMM,
which no symmetric kernel can help. 64 + 56 = 120 replaced calls against 156
quack calls.

### Batching and the kernel are different mechanisms

`base` -> `b48` (batching, no quack) is not mainly a GEMM-efficiency win:

| | base | b48 | b48syrk |
|---|---:|---:|---:|
| iteration | 1843.3 ms | 1741.4 | 1641.3 |
| GEMM device time | 949.2 | 855.9 | 785.3 |
| **kernel launches** | **59,050** | **19,142** | 19,179 |
| Memcpy DtoD | 61.8 ms / 6,227 | -- | 61.1 / 896 |

Batching cuts launches **3.1x** and improves Tensile's tile choice: the serial
path scatters across four suboptimal tiles (`MT192x192`, `192x224`, `256x192`,
`256x224`) over 6,461 calls, while the batched shapes get `MT256x256` uniformly.
Quack then wins on the symmetric math. The two are additive, which is what
1839.6 -> 1740.7 -> 1640.4 shows.

### What the B curve is actually selecting

Tracing B=48 against B=56 (the reproducible 66 ms inversion) locates it, and it is
not a GEMM effect -- consistent with the kernel being flat in B:

| delta ms | B=48 | B=56 | kernel |
|---:|---:|---:|---|
| **+60.0** | 106.0 / 26 | 166.0 / 41 | `reduce_kernel` / `aten::linalg_vector_norm` |
| **+55.3** | 0.0 / 0 | 55.3 / 51 | `elementwise_kernel_manual_unroll<128,8>` |
| +23.1 | 86.5 | 109.6 | Tensile `MT256x240x64` |
| -16.8 | 256.6 | 239.8 | Tensile `MT256x256x64` |
| -68.8 | 68.8 / 21 | 0.0 / 0 | Tensile `Bjlk_...MT256x256x64` |

The cause is `F.normalize(x, p=2, dim=(-2,-1))` at the top of `newton_schulz` --
the spectral-norm rescale, not any matmul. At B=56 its reduction splits across 41
kernels instead of 26 and costs 60 ms more, and a second elementwise kernel
appears that does not exist at B=48. The GEMM changes are a near-wash. **B selects
reduction kernels, not GEMM kernels**, which is why the curve is jagged and why the
isolated GEMM benchmark showed nothing.

### What is left, counted rather than estimated

From the `b48syrk` trace, the largest non-GEMM device time in the iteration:

| item | ms | calls |
|---|---:|---|
| NCCL (`record_param_comms` + `allreduce_coalesced`) | 220.6 | 73 |
| `linalg_vector_norm` / `reduce_kernel` (the `normalize`) | 106.1 | 26 |
| `elementwise_kernel_manual_unroll<128,4>` | 88.2 | 676 |
| Memcpy DtoD | 61.1 | 896 |

Two levers are visible and neither needs a third-party kernel: the `normalize` at
106 ms is a fusable reduce-then-divide *and* the thing making the B curve jagged,
and the momentum/staging traffic across ~1,600 launches is what
`torch._foreach_*` exists for. An earlier version of this section estimated the
non-GEMM block at 700-850 ms from FLOP ratios; that was too high, and these
counted figures replace it.

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
