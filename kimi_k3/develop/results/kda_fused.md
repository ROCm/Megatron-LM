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

---

## In-model, all four fused kernels enabled

> `torchrun --standalone --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L \`
> `  --ep 8 --seq 512 --triton-attn-res --record-shapes --with-stack`
> Raw: `results/raw/allfused_run.jsonl`. Trace:
> `traces/proxy_ep8_4L_rank0_alltoall_tritonar_shapes_stack.{json,stacks}`.

SiTU (G61), AttnRes forward and backward (G58/G59), KDA gated RMSNorm and causal
short conv (G62), all active.

**The binary check passes: `miopen_depthwise_convolution` and
`aten::convolution_backward` are absent from the trace.** A named kernel either
runs or it does not, so this needs no tolerance judgement -- the eager conv path
is gone.

All nine Triton kernels dispatch:

| kernel | ms | calls |
|---|---|---|
| `_dw_reduce_kernel` | 1.423 | 8 |
| `_conv_bwd` | 0.862 | 9 |
| `_situ_fwd` | 0.656 | 14 |
| `_situ_bwd` | 0.345 | 7 |
| `_conv_fwd` | 0.329 | 18 |
| `_grn_fwd` / `_grn_bwd` | 0.356 | 9 |
| `_attn_res_kernel` / `_bwd` | 0.351 | 23 |

| region | baseline | all fused |
|---|---|---|
| steady iteration | 1858.2 ms | **1836.4 ms** |
| `k3.layer` | 87.24 ms | **72.27 ms** (-17%) |
| `k3.kda` | 14.65 ms | **10.38 ms** (-29%) |
| `k3.attn_res` | 7.65 ms | **0.85 ms** (-9x) |
| kernel launches | 59,943 | **58,354** |

### What this does and does not establish

The **total iteration moved 1.2%**. The iteration is 81% Muon, and all of this
work is in the other 19%. Each kernel is a genuine 1.9x-6.8x at production shape;
at seq 512 with K <= 1 there is an order of magnitude less to collect, which was
the stated expectation going in and is what happened.

So this trace establishes that the kernels are **correct and dispatching in a real
step** -- not that they pay off yet. Their case is full depth and full sequence,
which no single node can run.

One thing the trace surfaces: **`_dw_reduce_kernel` is now the largest fused
kernel at 1.423 ms**, and at `K <= 1` it is doing almost no useful work, so that is
nearly all fixed overhead. At production `K = 8` it amortises; if AttnRes ever
matters at small `K`, that reduction is the thing to revisit.

### A process note

Two attempts at this run failed on `DistNetworkError ... code: -98` before I
looked properly. The cause was mine: picking a "free" port by `bind()`-then-
`close()` leaves it in `TIME_WAIT`, so **the probe poisoned the port it had just
reported as free**. `torchrun --standalone` picks and holds its own port and
removes the race.
