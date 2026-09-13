# G6 — measured AttnRes payload and mixer cost

> Produced by `kimi_k3/tools/attn_res_probe.py` on one MI355X; raw rows in
> `attn_res_raw.jsonl`. Production width: `H = 7168`, `S = 8192`, `B = 1`,
> 93 layers, `attn_res_block_size = 12`, bf16 activations, PP = 8.
> Regenerate: `python -m kimi_k3.tools.attn_res_probe --width production`.

## 1. Pipeline payload

Layout `[12, 12, 12, 12, 12, 11, 11, 11]`. Multipliers are
`1 + ceil((last + 1) / 12)`, so both this split and the `12×7 + 9` variant give
the same sends.

| stage | layers (0-idx) | recv × | send × | send MB | in-flight MB (1F1B warm-up) |
|---:|---|---:|---:|---:|---:|
| 0 | 0–11 | 1 | 2 | 224 | 1792 |
| 1 | 12–23 | 2 | 3 | 336 | 2352 |
| 2 | 24–35 | 3 | 4 | 448 | 2688 |
| 3 | 36–47 | 4 | 5 | 560 | **2800** |
| 4 | 48–59 | 5 | 6 | 672 | 2688 |
| 5 | 60–70 | 6 | 7 | 784 | 2352 |
| 6 | 71–81 | 7 | 8 | 896 | 1792 |
| 7 | 82–92 | 8 | — | — | — |

**Worst case is the middle of the pipeline, not the end**: the payload grows with
depth while the in-flight microbatch count falls, and the product peaks at stage 3
at **2.8 GB** — roughly **5.6 GB** counting saved input *and* output tensors.
Uncomfortable but not disqualifying, and context parallelism divides it directly.

`pack` / `unpack` round-trip exactly, and the analytic `payload_bytes` matches a
real tensor's footprint (asserted in the probe and in
`tests/test_k3_p0_attn_res.py`).

## 2. Mixer cost — one mix at production width

| K+1 | fwd peak MB | fwd+bwd peak MB | fwd ms | fwd+bwd ms | stack MB (bf16 / fp32) |
|---:|---:|---:|---:|---:|---|
| 2 | 1 568 | 2 876 | 1.81 | 5.02 | 224 / 448 |
| 3 | 2 352 | 4 144 | 2.28 | 6.55 | 336 / 672 |
| 5 | 3 920 | 6 832 | 3.24 | 9.58 | 560 / 1 120 |
| 7 | 5 489 | 9 520 | 4.17 | 12.57 | 784 / 1 568 |
| **9** | **7 057** | **12 208** | **5.15** | **15.65** | 1 008 / 2 016 |

Cost is linear in `K+1`, as the shapes predict.

## 3. What the whole model pays, per microbatch

186 mixes (93 layers × 2, plus one at the model output), mean `K+1` = **5.39**:

| metric | eager fp32 mix (the release's semantics) | same mix kept in bf16 |
|---|---:|---:|
| bytes read per forward | 109.6 GiB | 109.6 GiB |
| eager mixer **forward** | **≈ 635 ms** | ≈ 427 ms |
| peak for one mix at `K+1 = 9`, fwd | 7 057 MB | 3 024 MB |
| peak for one mix at `K+1 = 9`, fwd+bwd | 12 208 MB | 6 160 MB |

The fp32 upcast the release performs costs **2.3× memory** and **≈1.5× time**.
We keep it — it is the released semantics and the oracle's contract — but a fused
kernel can hold the fp32 accumulation in registers and pay neither.

### Why this is the headline perf item

* For scale: the routed-expert GEMMs are ≈ 797 TFLOP per microbatch at this
  shape, i.e. **~800 ms at 40 % MFU**. The *eager AttnRes forward alone is the
  same order as all of the MoE compute*, and its backward is ~3× the forward.
* **Recompute is mandatory, not optional.** Saving each mix's fp32 stack for
  backward would cost ≈ 1.27 GB × 186 ≈ **236 GB per microbatch**. With recompute
  the peak is bounded by one live pair (≈ 4–12 GB depending on `K`), and the
  traffic is paid twice.
* Therefore the P11 fused mixer has two jobs: never materialise the `[T, K+1, H]`
  fp32 stack, and make non-recomputed AttnRes affordable. **Budget: ≤ 10 % of the
  eager forward (≤ 64 ms per microbatch) and no fp32 stack in HBM.**

## 4. Caveats

* Single-GPU microbenchmark: no TP/CP sharding of `S` or `H`, no overlap with
  other kernels. The in-flight payload figures are analytic from measured
  per-boundary bytes, not from a running PP job — G7 exercises the real
  transport, and P11's trace gives the in-situ share.
* `torch.compile` was not applied; the P11 comparison should include it as the
  cheap baseline before any hand-written kernel.

---

# G56 — "fused" was never fused, and the rename found a live bug

Raised in review: *"attn_res_mix_fused does not really fuse it."* Correct.

## It is chunking

```python
return torch.cat([attn_res_mix(prefix_sum[start:start+chunk], ...)
                  for start in range(0, prefix_sum.shape[0], chunk)], dim=0)
```

A Python loop calling the eager mixer on row slices. **No kernel, no fusion.** It
issues the eager op sequence once *per chunk*, so it launches strictly more work
than the eager path -- which is exactly what G44/G45 measured and nobody read as a
contradiction at the time:

| | baseline | "fused" |
|---|---|---|
| kernel launches | 321,116 | **321,163 (+47)** |
| peak HBM | 193.63 GiB | 193.63 GiB (0.00) |
| steady iteration | 2653.5 ms | 2649.6 ms (noise) |

*More* launches and *zero* memory saved. A genuinely fused kernel would show fewer
launches; that number was in the results table for two weeks arguing against the
name.

The name also propagated a promise. `attn_res.py`'s header said "the fused kernel
in P11 must reproduce it" and described a fold that was "free for the P11 fused
kernel". **No such kernel was ever written** -- P11 rescoped once the trace put
AttnRes at 0.09% of device time. The docstrings were describing planned work as
though it existed.

## Renamed

`attn_res_mix_chunked`, `k3_attn_res_chunked`, `--k3-attn-res-chunked`,
`AttnResMixer.chunked`, `tools/proxy_ep8 --chunked-attn-res`, and
`tests/test_k3_p11_chunked_attn_res.py`. `attn_res_mix_fused`,
`--k3-attn-res-fused` and `--fused-attn-res` remain as aliases; the config field
`k3_attn_res_fused` is kept, deprecated, and warns.

## The rename exposed a real bug

`k3_transformer_block.py:38` read `k3_attn_res_fused` to build the **model-output**
mixer, separately from `k3_transformer_layer.py` which builds the per-layer ones.
Renaming the field left the block reading the now-`None` deprecated alias, and
`decoder.output_attn_res` came back with `chunked=None` -- so with the flag on, the
output mix silently ran the **eager** path while all eight layer mixes ran chunked.

The pre-existing test `test_the_flag_selects_the_path_and_the_model_agrees` is what
caught it, and its docstring says exactly why it exists: *"a flag that reaches only
some of them would still pass a unit test."* That test was written for this failure
mode and duly found it.

Whether the bug predates the rename is worth being precise about: **it does not.**
Before the rename both sites read the same field name, so both got the flag. The
rename created it and the test caught it within minutes. What the episode shows is
that the flag has **two independent construction sites**, which is a standing
hazard the test now guards.

## A process note on how I nearly missed it

My first sweep for stale references was `grep ... | head -20`, which returned
exactly 20 lines -- and I treated that as the complete list. `k3_transformer_block.py`
was line 21. Truncating an enumeration and then reasoning from it as if it were
exhaustive is how the block site got left behind.

## Status unchanged

The optimisation is still correct (G43, bit-identical forward at any chunk size)
and still pointless at any geometry that fits on one node: the temporary it removes
is 29 MB at 4 layers / seq 512, against the 109.6 GiB it was built for at 93 layers
/ seq 8192. Default stays off. What changed is only that the name no longer claims
something that does not exist.

---

# G58 — the real fused kernel

> `kimi_k3/block/attn_res_triton.py`, gated by `tests/test_k3_p11_triton_attn_res.py`.
> `--k3-attn-res-triton`. Measured on one MI355X, bf16 in/out, fp32 math.

The kernel P11 promised and never wrote (G56). Per token, two passes straight out
of the original tensors -- no `cat`, no fp32 `[T, K+1, H]` materialisation:

    pass 1   s2[j] = sum_h v[j,h]^2 ; d[j] = sum_h v[j,h]*w[h]
             score = d * rsqrt(s2/H + eps) ; p = softmax(score)
    pass 2   out[h] = sum_j p[j] * v[j,h]

The fold that makes it cheap is the one the release's own code makes obvious: the
RMSNorm gain and the `[1, H]` projection only ever appear multiplied, so they
collapse to one `[H]` vector **and the normalisation becomes a per-candidate
scalar applied after the dot product** instead of to `H` elements before it.

## Forward

| T | K | eager | chunked | **triton** | speed-up | rel-L2 | GB/s | % of copy bw |
|---|---|---|---|---|---|---|---|---|
| 512 | 2 | 0.171 | 0.164 | **0.041** | 4.17x | 1.3e-05 | 1250 | 23.8% |
| 512 | 8 | 0.356 | 0.356 | **0.042** | 8.40x | 2.2e-05 | 3295 | 62.8% |
| 4096 | 8 | 2.569 | 2.570 | **0.257** | 10.01x | 2.9e-05 | 4349 | 82.9% |
| **8192** | **8** | 5.125 | 5.225 | **0.486** | **10.55x** | 2.7e-05 | 4592 | **87.5%** |

Peak temporary at `T=8192, K=8, H=7168`:

| | peak temp |
|---|---|
| eager | 6.891 GiB |
| chunked (4096) | 3.500 GiB |
| **triton fused** | **0.109 GiB** |

**63x less transient memory than eager, 32x less than the chunked mixer**, and
10.6x faster. For contrast, the chunked mixer it replaces was +47 kernel launches
and 0.00 GiB at the geometry that fits on one node.

## Tuning: the intuitive knob is backwards

First working version ran 1.951 ms (21.7% of bandwidth) at production shape. The
whole gap was `num_warps`:

| num_warps | ms | GB/s |
|---|---|---|
| **4** | **0.522** | **4276** |
| 8 | 0.959 | 2327 |
| 16 | 1.655 | 1348 |

More warps is 1.83x then 3.17x *worse*. Each program already holds a `[BJ, BH]`
tile; splitting it further pushes the loads below the width at which they
coalesce. `BH=2048` beats 512 by 1.5x for the same reason. Tuned defaults are in
the module with this measurement beside them.

## What it is not

**Not bit-identical, and cannot be.** It reduces over `H` in tiles rather than in
torch's order. Every other AttnRes gate asserts bit-equality; this one is gated on
a measured tolerance (fp32 rel-L2 ~1e-06, bound set at 1e-4).

**The backward is not accelerated.** A raw Triton kernel returns a tensor with no
`grad_fn` -- calling it directly in a training step would train nothing through
this path while the loss still fell, because every other path still trains. The
`_FusedAttnResMix` wrapper recomputes the mix through the eager oracle and
differentiates that, so gradients are **bit-identical** to eager (0.0 on all four
inputs, asserted) and every existing backward gate still applies. But the backward
re-materialises the temporaries the forward avoids, so **the memory win above is
forward-only**.

A real backward kernel is the remaining work, and it is tractable -- the gradient
of a softmax over a scalar-scaled dot product is closed form -- but it is a second
kernel, not a tweak to this one.

## Scope

Single-GPU microbenchmark at production tensor shape. Not yet measured inside a
model step, where AttnRes was 0.09% of device time at the proxy geometry (seq 512,
K<=1) -- the shape where this kernel is worth least. Its case rests on production
geometry, which no single node can run.


---

# G59 — the backward kernel

> `kimi_k3/block/attn_res_triton.py`, `tests/test_k3_p11_triton_attn_res.py`.
> Measured on one MI355X, bf16 tensors, fp32 math, T 8192 / K 8 / H 7168.

G58 shipped a fused forward with an **eager recompute** for the backward:
gradients were bit-identical, but the backward re-materialised the
`[T, K+1, H]` fp32 temporaries the forward avoids, so the memory win was
forward-only. This is the backward as its own kernel.

## The gradients collapse to three scalars

With `p = softmax(d*r)`, `r = rsqrt(s2/H + eps)`, `d = <v,w>`, `s2 = <v,v>`:

    gp[j]   = <g, v[j]>
    ds[j]   = p[j] * (gp[j] - sum_i p[i] gp[i])       # softmax
    dd[j]   = ds[j] * r[j]                            # through d
    ds2[j]  = ds[j] * d[j] * (-0.5 * r[j]^3 / H)      # through the rsqrt
    dv[j,h] = p[j]*g[h] + dd[j]*w[h] + 2*ds2[j]*v[j,h]
    dw[h]   = sum_t sum_j dd[j] * v[j,h]

So the backward is the same two passes as the forward, not a larger computation.

## Full forward + backward

| | eager | chunked | **triton** |
|---|---|---|---|
| T 4096 | 8.376 ms / 5.414 GiB | 8.371 / 5.414 | **1.363 ms / 0.057 GiB** |
| **T 8192** | 16.665 ms / 10.828 GiB | 17.779 / 6.891 | **2.456 ms / 0.113 GiB** |

**6.8x faster, 96x less peak memory.** Gradient accuracy against the eager
oracle: fp32 ~9e-06, bf16 ~3e-04 on all four inputs.

## Three versions of one reduction

`dw` is the only cross-token reduction and it took three attempts. Recorded
because each failed for a different reason and the first two looked fine:

| approach | fwd+bwd | peak | norm/proj grad error |
|---|---|---|---|
| `atomic_add` from the main kernel | 6.63 ms | 0.11 GiB | ok |
| bf16 GEMV | **1.82 ms** | **0.11 GiB** | **3.05e-03** |
| fp32 einsum over token chunks | 3.61 ms | 0.55 GiB | 2.3e-04 |
| **separate fp32 reduction kernel** | **2.46 ms** | **0.11 GiB** | **2.3e-04** |

* **Atomics**: 8192 programs x 7 chunks x 1024 lanes is ~58M atomic adds onto
  7168 addresses. The kernel ran at ~640 GB/s, an eighth of what its traffic
  justifies -- it was not bandwidth-bound, it was contending.
* **bf16 GEMV**: fastest, and wrong in a way only one of four gradients showed.
  It rounds the fp32 `dd` *before* summing, giving 3.05e-03 on the norm/proj
  gradients against 1.9e-04 on the others. Those two are **parameter** gradients
  and the AttnRes path is specified fp32 (R7.3), so this was not acceptable even
  though the error sits at bf16 epsilon.
* **fp32 einsum over chunks**: correct, but upcasts `v` a chunk at a time --
  5x the peak memory and 1.5x the time.
* **Separate kernel**: `dd` stays fp32, `v` is read in its native dtype, grid is
  (H blocks, T blocks) writing partials that the caller sums. Both properties at
  once.

A bug worth recording from that sequence: the chunk size was written
`1 << 20 // max(1, K*H//256)`, which parses as `1 << (20 // ...)` = `1 << 0` = 1.
It ran 8192 single-token einsums and took **235 ms**, 14x *slower* than eager, and
the only symptom was the timing.

## Status

Not bit-identical and cannot be -- reductions over `H` run in tiles. Gated on
measured tolerances at both precisions, plus a test that the backward allocates
under an eighth of eager's peak, so the memory property cannot silently regress
the way it did between G58 and this.

Still a single-GPU microbenchmark at production tensor shape. At the proxy
geometry (seq 512, K <= 1) AttnRes was 0.09% of device time, so the in-model win
there will be negligible; the case rests on production geometry, which no single
node can run.
