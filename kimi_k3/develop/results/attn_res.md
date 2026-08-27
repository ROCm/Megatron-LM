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
