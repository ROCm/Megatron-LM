# G62 — fused KDA elementwise: gated RMSNorm and the causal short conv

> `kimi_k3/attention/kda_triton.py`, gated by `tests/test_k3_p3_kda_fused.py`.
> `--k3-kda-fused-elementwise` (default on). Production shape: B 1, T 8192,
> D 12288, H 96, head_dim 128, W 4.

Both are transcriptions of things the release ships **fused** and this fork wrote
out eagerly. `gated_rms_norm`'s own docstring names its counterpart
`FusedRMSNormGated`; we had eight aten ops under that name. `causal_short_conv`
was `F.pad` + two transposes + a miopen depthwise conv + SiLU, with a
materialised padded copy.

They run in **69 of 93 layers**, so this is not a rounding error at full depth.

| op | eager fwd+bwd | fused | speed-up |
|---|---|---|---|
| `gated_rms_norm` | 3.917 ms | **1.547 ms** | **2.53x** |
| `causal_short_conv` | 3.287 ms | **1.743 ms** | **1.89x** |

Accuracy against the oracles, fp32 / bf16:

| | forward | dx | dw | dgate |
|---|---|---|---|---|
| `gated_rms_norm` | 6.6e-08 / 1.4e-05 | 6.6e-08 / 1.4e-05 | 1.9e-07 / 7.4e-06 | 8.5e-08 / 1.6e-05 |
| `causal_short_conv` | 6.5e-08 / 2.9e-03 | 7.4e-08 | 1.2e-07 | — |

The short conv's bf16 forward difference (2.9e-03) is the kernel accumulating its
four taps in fp32 where miopen accumulates in bf16 -- the kernel is the *more*
accurate of the two, and the eager path remains the declared oracle only because
every other gate is written against it.

## The left pad is arithmetic, not a copy

`F.conv1d` needs `[B, D, T]`, so the eager path transposes in, pads a new tensor,
convolves, transposes back. The kernel reads `[B, T, D]` in place and masks taps
whose source index is negative. That removes the pad temporary and both
transposes, which is most of the traffic at D = 12288.

Because the pad became a mask, an off-by-one in the tap offset would leak future
tokens into earlier outputs and still look numerically plausible, so the gate
asserts causality directly: perturb the tail of the sequence, assert the head of
the output is unchanged.

## A bug worth recording

The short-conv backward indexed its `dw` partials by time-tile only, so **every
batch wrote the same slot and the last one won**. The forward and `dx` were both
correct to 7e-08; only `dw` was wrong, and only for `B > 1` -- at `B = 1` it is
invisible. The gate now runs B in {1, 2, 3} for that reason.

That is the second kernel this session whose only defect was in indexing rather
than arithmetic (G58: int32 overflow; G59/G60: Triton specializing an integer arg
equal to 1). The maths has been right first time in every case; the addressing has
not.

## Scope

Standalone microbenchmarks at production tensor shape, not yet re-traced in-model.
At the proxy geometry these two sites totalled 1.86 ms of a 1842 ms iteration, so
the in-model effect will be small; the case is full depth and full sequence.
