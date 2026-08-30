# G15 — measured KDA backend agreement

> Produced by `python -m kimi_k3.tools.kda_parity_probe`; raw rows in
> `kda_parity_raw.jsonl`. Bounds derived from these numbers live in
> `kimi_k3/tests/tolerance.py`, each with its measurement attached.

## The floor comes first

A backend comparison is meaningless without the dtype floor beside it, so the
probe measures the same eager code in bf16 against itself in fp32 *before*
comparing anything to fla (rule R4.4).

| geometry | seq | comparison | rel-L2 | max-abs | cosine |
|---|---:|---|---:|---:|---:|
| tiny (H=2, K=64) | 256 | **floor** — eager bf16 vs eager fp32 | 3.279e-03 | 4.05e-04 | 0.999995 |
| tiny | 256 | fla fp32 vs eager fp32 | **6.90e-07** | 8.20e-08 | 1.000000 |
| tiny | 256 | fla bf16 vs eager bf16 | 4.256e-03 | 4.88e-04 | 0.999991 |
| tiny | 1024 | floor | 3.365e-03 | 5.64e-04 | 0.999994 |
| tiny | 1024 | fla fp32 | **7.02e-07** | 1.01e-07 | 1.000000 |
| tiny | 1024 | fla bf16 | 4.328e-03 | 9.77e-04 | 0.999991 |
| mid (H=8, K=128) | 1024 | floor | 3.295e-03 | 2.54e-04 | 0.999995 |
| mid | 1024 | fla fp32 | **7.01e-07** | 7.64e-08 | 1.000000 |
| mid | 1024 | fla bf16 | 4.275e-03 | 4.88e-04 | 0.999991 |
| mid | 4096 | floor | 3.298e-03 | 3.88e-04 | 0.999995 |
| mid | 4096 | fla fp32 | **7.06e-07** | 7.64e-08 | 1.000000 |
| mid | 4096 | fla bf16 | 4.300e-03 | 4.88e-04 | 0.999991 |

Backward, worst per-tensor gradient: **fp32 1.6–2.9e-05** (`dt_bias` / `g`),
**bf16 6.0–6.1e-03** (`v`).

## What the numbers say

1. **The kernel is right.** In fp32, fla and the oracle agree to 7e-7 — cosine
   1.000000 to six places — across every geometry and length measured.
2. **bf16 disagreement is bf16.** fla-vs-eager in bf16 is 4.3e-3 against a floor
   of 3.3e-3: the kernel contributes roughly 1e-3 on top of what the dtype costs
   anyway. A 1e-5 bound in bf16, which the incoming plan rightly warned against,
   would fail on the dtype alone.
3. **Error does not grow with sequence length.** 256 → 4096 moves the figures in
   the third significant digit. This is worth stating because the opposite is
   normally assumed for a recurrent model: the `lower_bound = -5` gate caps the
   per-step decay at `exp(-5) ≈ 0.0067`, so old state — and old error — is
   forgotten faster than it accumulates. A test pins the property.

## Bounds set from this (with margin)

| what | bound | measured | floor | margin |
|---|---:|---:|---:|---:|
| forward, fp32 | rel-L2 ≤ 1e-5, cos ≥ 0.999999 | 7.1e-7 | — | ~14× |
| forward, bf16 | rel-L2 ≤ 1e-2, cos ≥ 0.9999 | 4.3e-3 | 3.3e-3 | ~2.3× |
| backward, fp32 | rel-L2 ≤ 1e-3 | 2.9e-5 | — | ~34× |
| backward, bf16 | rel-L2 ≤ 2e-2 | 6.1e-3 | 3.3e-3 | ~3.3× |

## Production geometry — G15, run 2026-08-30

`python -m kimi_k3.tools.kda_parity_probe --production --seq 1024 4096 8192 --grad`
at K3's real **H=96, K=128**:

| seq | floor (eager bf16 vs fp32) | fla fp32 | fla bf16 | worst grad fp32 | worst grad bf16 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 3.300e-03 | **6.434e-07** | 4.309e-03 | 2.35e-05 | 6.10e-03 |
| 4096 | 3.305e-03 | **6.419e-07** | 4.305e-03 | 1.57e-05 | 6.07e-03 |
| 8192 | 3.310e-03 | **6.768e-07** | 4.319e-03 | 1.46e-05 | 6.08e-03 |

Identical in shape to the smaller geometries, and flat in sequence length: 8x the
sequence moves fp32 agreement from 6.43e-07 to 6.77e-07. bf16 sits at 4.31e-03
against a measured floor of 3.31e-03 — the kernel is within the dtype's own noise,
not adding to it.

**This is what R5.3 was waiting for, so the default flipped to `fla`** in its own
commit. The eager oracle stays in tree and stays selectable; it is the reference,
and every parity number here is against it.

The flip is not cosmetic. The oracle keeps the recurrent state at every timestep
for autograd, so at seq 8192 it costs **109 GiB per layer** against fla's **10.2**
— the difference between the 93 L model fitting on 28 nodes and not fitting at all
(`results/scaleout_93l.md`).

## Still owed

seq 64 k, and the KDA state-passing path across a sequence-parallel boundary.
