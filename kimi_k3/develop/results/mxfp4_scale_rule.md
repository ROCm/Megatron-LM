# G38 — which MXFP4 scale rule the release actually uses

> Measured off a real released expert: `layers.2...experts.0.w1`, 5.2 MiB of
> packed data plus 0.3 MiB of scales, fetched by range request.
> Cross-check: `aiter.per_1x32_f4_quant`. Tests: `test_k3_p10_quantizer.py`.

## The question

An MX scale is a shared power of two per group of 32. Two rules are in
circulation, and they disagree by one exponent on a large fraction of groups:

* **target the largest normal (6.0), by flooring** — the OCP formula
  `X = 2**(floor(log2(amax)) - 2)`. This is what P0 implemented.
* **target the largest power of two (4.0), rounding to nearest** —
  `X = round_to_pow2(amax / 4)`.

The difference is not cosmetic. Under the first rule the scaled peak lands in
`[4, 8)`, so any group whose peak has a mantissa above 1.5 is **clamped** to
E2M1's 6.0. A group peaking at 3.99 gets `X = 2**-1`, scales to 7.98, and comes
back as 3.0 — a **25 % error on the group's largest element**. Under the second
the scaled peak lands in `[2.83, 5.66]` and nothing ever clamps.

## The measurement

Dequantise a released expert and histogram each group's peak *element magnitude*
before the scale is applied. Both rules constrain this differently, so the
histogram identifies the rule.

| peak magnitude | released (344,064 groups) | predicted by target-4.0 | possible under floor-to-6.0 |
|---|---|---|---|
| 3.0 | **25.75 %** | 30.7 % | **impossible** |
| 4.0 | 55.10 % | 51.5 % | possible |
| 6.0 | 19.15 % | 17.8 % | possible |

The floor-to-6.0 rule can only ever leave a peak at 4.0 or 6.0. **A quarter of
the released groups are impossible under it.** The target-4.0 rule's prediction
for a log-uniform mantissa is what is there.

## Confirmation, and the fix

`compute_e8m0_scale` now implements the measured rule, reproducing AITER's
`f32_to_e8m0` bit trick exactly rather than approximating it — the midpoint is a
mantissa of 1.5, not the geometric mean 1.414, because that is what the hardware
does. Against `aiter.per_1x32_f4_quant` the result is now **byte-identical**:
same packed nibbles, same e8m0 scales, at input scales 0.02, 3.0 and 100.0.

MXFP8 agrees too, once AITER is asked for the right thing: `per_1x32_f8_scale_f8_quant`
defaults to **fp32** scales, which is not the MX format; with
`scale_type=fp8_e8m0` it is bit-identical to ours.

## What this does not change

Import. Dequantisation is unambiguous — packed value times scale — so P8's
converter and the anchored-parity result are unaffected, and its round-trip claim
covers packed data it never re-quantises.

## One property worth stating rather than discovering later

Exact round-trip fails for a group whose peak is exactly `6 * 2**k`: `amax / 4`
is then exactly `1.5 * 2**(k-1)`, the tie, which rounds **up**; the grid doubles
and the group's smallest level falls off it. Re-quantising an already-quantised
tensor therefore moves 19.15 % of scales by one — precisely the groups that
peaked at 6.0. It does not affect QAT, which re-quantises fp32 masters where the
tie has measure zero, but it does mean quantisation is not idempotent.

## The plan's cross-check target does not exist

G38 named TE's `MXFP4Quantizer`. At the pin TE ships `MXFP8Quantizer` and
`NVFP4Quantizer` only, and NVFP4 is a different format (block 16, E4M3 scales,
plus a global scale) that cannot stand in. AITER is the better target regardless:
it is the library whose kernels the release's serving path runs. Both substitutions
are asserted in the tests so the absence stays recorded.
