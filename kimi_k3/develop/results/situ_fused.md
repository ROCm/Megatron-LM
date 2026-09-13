# G61 — the fused SiTU kernel, and where the elementwise time actually is

> `kimi_k3/moe/situ_triton.py`, gated by `tests/test_k3_p6_situ_fused.py`.
> `--k3-situ-fused` (default on). Raw: `results/raw/situ_fused_run.jsonl`.
> Trace: `traces/proxy_ep8_4L_rank0_alltoall_tritonar_shapes_stack.json`.

## What the trace showed

`situ_glu` was plain PyTorch, and G52 made it the activation on every routed
expert. Measured in-model: **10 distinct aten ops, 350 launches, 5.05 ms** against
a `k3.moe` region of 8.97 ms -- more than half the MoE region, for two
transcendentals and a few multiplies, because each step was its own pass over an
`[~8000, 3072]` tensor:

| op | ms | calls |
|---|---|---|
| `mul` | 1.792 | 56 |
| `copy_` (the `.float()`/`.to()`) | 1.126 | 42 |
| `div` | 0.890 | 28 |
| `tanh` | 0.815 | 28 |
| `sigmoid` | 0.429 | 14 |

Core would normally fuse an activation, but `bias_activation_fusion` only knows
SwiGLU and GeGLU, and the seam SiTU must use -- `use_te_activation_func`, because
core's GLU cannot express the tanh-limited *up* branch -- explicitly disables
fusion. G52 bought the right math at the cost of an unfused activation.

## The kernel

One read of `[gate | up]`, compute in registers, one write. Backward is closed
form, so it is one more read plus one write rather than a recompute:

    d(situ_a)/dgate = (1 - t_g^2) * s + beta * t_g * s * (1 - s)
    dL/dgate = dy * up_t * d(situ_a)/dgate
    dL/dup   = dy * situ_a * (1 - t_u^2)

| | forward | fwd+bwd | peak |
|---|---|---|---|
| eager | 0.391 ms | 1.099 ms | 0.739 GiB |
| **fused** | **0.070 ms** | **0.190 ms** | **0.047 GiB** |
| | 5.6x | **5.8x** | **15.7x** |

Accuracy against the oracle: fp32 **2.9e-07**, bf16 **2.9e-05**, both directions.

`tl.math.tanh` does not exist in this Triton build, so the kernel uses
`2*sigmoid(2x) - 1`. That is only safe because sigmoid saturates, so there is a
test asserting finiteness and agreement at input 1e4 rather than trusting the
identity.

## In-model, from the trace

| | before | after |
|---|---|---|
| SiTU device time | 5.05 ms | **0.65 ms** (7.8x) |
| distinct aten ops under SiTU | 10 | **1** (`_FusedSituGLU`) |
| `k3.attn_res` (G58/G59 kernel) | 7.65 ms | **0.89 ms** (8.6x) |
| steady iteration | 1858.2 ms | **1841.9 ms** |
| kernel launches | 59,943 | 58,964 |

What remains attributed to SiTU is `reshape`/`view`/`empty` at **zero** device
time. The end-to-end check matters for the same reason it did in G60: the unit
tests show the kernel computes the right thing, only a trace shows core dispatches
it inside a real step.

## The survey: there is nothing else worth fusing in our code

Elementwise and reduction device time by call site, whole iteration:

| ms | calls | site |
|---|---|---|
| **110.11** | 1360 | `optimizer.py:560 step_with_ready_grads` |
| **68.95** | 3400 | `muon.py:90 scaled_orthogonalize_fn` |
| 33.34 | 769 | `distributed_data_parallel.py:431 hook` |
| 13.93 | 685 | `_multi_tensor_copy_this_to_that` |
| 5.05 | 168 | `situ.py:19 situ_glu` **(now 0.65)** |
| 0.98 | 72 | `kda.py:53 gated_rms_norm` |
| 0.88 | 36 | `kda.py:32 causal_short_conv` |
| 0.55 | 48 | `k3_moe_layer.py:49 _latent_norm` |
| 0.42 | 69 | `attn_res.py:31 score_vector` |

**226 of 244 ms sits in the optimizer.** `optimizer.py:560` is literally
`self.optimizer.step()` -- that 110 ms is elementwise work inside
`emerging_optimizers`, attributed to the deepest *Megatron* frame because the real
frames are in a third-party package. `muon.py:90` is Newton-Schulz's scaling, same
story.

Everything our fork owns now totals **~2.8 ms**. Four more kernels would be needed
to chase it, each saving well under a millisecond. Stopping here is the right call:
the mass is in code this repo does not own, which is the same conclusion every
other performance thread in this phase reached -- **the optimizer is the
bottleneck, and it is not ours.**

If the optimizer path is to be attacked, the honest options are: reduce
`muon_num_ns_steps` (5 today), batch the 680 same-shaped expert matrices into one
grouped Newton-Schulz instead of 3400 individual calls, or switch to Adam (already
measured 4.7x faster, G47). The middle one is the real fix and it is a change to
`emerging_optimizers`.
