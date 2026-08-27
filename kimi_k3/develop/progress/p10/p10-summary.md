# P10 — MXFP4 / MXFP8 QAT (COMPLETE except the scheduled tier)

## 1. Objective

Make the quantisation numerics match the release, and run the real kernel.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/moe/k3_qat.py` | `f32_to_e8m0`, and `compute_e8m0_scale` rewritten to the measured rule |
| `kimi_k3/moe/k3_a8w4.py` | the AITER Triton a8w4 forward with an STE backward |
| `kimi_k3/tests/test_k3_p10_{quantizer,a8w4}.py` | 17 tests |
| `kimi_k3/tests/test_k3_p0_qat.py` | two tests corrected — one asserted a property the format does not have |

`pytest kimi_k3/tests/ -q` -> **226 passed, 4 skipped**.

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G38** — quantiser exactness | **GREEN** | **byte-identical** to `aiter.per_1x32_f4_quant`; MXFP8 bit-identical to AITER with `scale_type=fp8_e8m0` (`results/mxfp4_scale_rule.md`) |
| **G39** — a8w4 forward | **GREEN** | rel-L2 **1.66e-3** vs the dequantised matmul, at four shapes including 3584→3072 (`results/a8w4.md`) |
| **G40** — STE gradients | **GREEN, exact** | rtol 0, atol 0 against the fake-quant reference, for both input and weight gradients |
| **G41** — 4 L QAT run, serve parity | **owed** | 8-GPU scheduled tier |

## 4. The scale rule was wrong, and the release could settle it

P0 implemented the OCP formula `X = 2**(floor(log2(amax)) - 2)`, targeting E2M1's
largest *normal* (6.0) by flooring. That rule clips — a group peaking at 3.99
comes back as 3.0, a 25 % error on its largest element.

Rather than argue from the spec, the question was put to the released weights.
Histogramming a real expert's 344,064 groups by peak element magnitude gives
25.75 % at 3.0, 55.10 % at 4.0, 19.15 % at 6.0. **The floor-to-6.0 rule can only
ever leave a peak at 4.0 or 6.0**, so a quarter of the released groups are
impossible under it. The release targets the largest *power of two* (4.0) and
rounds the exponent to nearest — which is AITER's `f32_to_e8m0`, reproduced here
bit trick and all, because its midpoint is a mantissa of 1.5 rather than the
geometric mean 1.414.

Import is unaffected: dequantisation is unambiguous, so P8's converter and the
anchored-parity result stand. Register: **A16**.

## 5. Two deferrals that turned out not to be needed

* **The kernel.** P0's finding D1 said the K3 a8w4 assets were absent and deferred
  G8 to an AITER checkout bump. The *opus* path is indeed absent — but
  `aiter.ops.triton.moe.moe_op_gemm_a8w4` is there, is Triton, and needs no
  rebuild. G39 and G40 are green at the pinned checkout.
* **TE's `MXFP4Quantizer`**, the plan's cross-check target, does not exist at the
  pin at all. AITER is the better target anyway — it is what the release serves
  through — and both facts are asserted rather than assumed.

## 6. Three things that were quietly wrong

* **wgrad against the wrong activations.** The STE backward multiplied the
  original activations rather than the quantised ones that actually reached the
  kernel. It still trains. G40's *exact* criterion caught it; a tolerance would
  have passed.
* **Unswizzled weight scales.** The kernel does not raise — it returns a
  plausible tensor wrong by rel-L2 0.53. Now a test with a 320x margin.
* **A P0 test asserted a property the format does not have**: a full-range group
  round-tripping exactly. It cannot, because a peak of `6 * 2**k` lands exactly on
  the tie, which rounds up and doubles the grid. Replaced with the true property
  and an explicit test of the exception.

## 7. Owed

G41. Also: this wraps a *single-expert* GEMM — routing many experts through one
fused call is the `moe_stage{1,2}` shape the opus path would provide, and it is
the thing worth revisiting if that checkout is ever bumped.
