# G51 — achieved TFLOP/s inside the model, and why the roofline overstated it

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 8 \`
> `  --seq 512 --record-shapes --with-stack --trace-dir develop/profile/traces`
> `torchrun --nproc_per_node=8 -m kimi_k3.tools.mm_attribution --preset 4L --ep 8 --record-shapes`
> Raw: `results/raw/gemm_efficiency_s512.json`, `results/raw/mm_attribution.json`,
> `results/raw/trace_full.jsonl`. Traces are gitignored run artefacts.

## What was wrong before

`tools/roofline.py` (added 2026-08-31, never gated, no results doc) times **isolated**
`a @ b` calls at **seq 8192**. Its output — "GEMMs at 92-99% of peak" — was quoted
repeatedly as though it described this model running. It does not. Two independent
problems:

1. **Standalone, not in-model.** Nothing in it observes the model.
2. **Wrong `m`.** It uses `m = 8192`; the traces it was compared against run at
   **seq 512**, so the GEMMs are 16x shorter and far less efficient.

A third error was found while checking it: the attention benchmark used **128
heads**; the model has **96** (`num_attention_heads: 96`). TFLOP/s survived
(69.9% vs the 69.7% reported, since numerator and denominator both scale) but the
per-call times were 33% too high -- 3.043 ms became 2.281 ms at seq 8192.

The dimensions in the roofline table are otherwise correct: every row traces to a
real config value (`q_lora_rank 1536 -> 96x192`, `kv_lora_rank 512 -> 96x256`,
`k3_routed_expert_hidden_size 3584`, `moe_ffn_hidden_size 3072`).

## The model's own GEMMs, measured in situ

Peak on this device: **1,322.9 TFLOP/s**.

| shape | role | calls | TFLOP/s | % peak |
|---|---|---|---|---|
| `[512,7168]x[7168,163840]` | output layer | 1 | 1,228.8 | **92.9%** |
| `[512,163840]x[163840,7168]` | output dgrad | 1 | 1,188.5 | 89.8% |
| `[512,7168]x[7168,12288]` | KDA qkv | 30 | 835.4 | 63.2% |
| `[512,12288]x[12288,7168]` | KDA o_proj | 21 | 747.7 | 56.5% |
| `[12288,512]x[512,7168]` | wgrad | 13 | 654.4 | 49.5% |
| `[7168,512]x[512,12288]` | wgrad | 4 | 587.7 | 44.4% |

**44-93%, not 92-99%.** Only the output layer reaches roofline-like numbers,
because only it has a large enough `n` to absorb `m = 512`.

## The bottleneck, precisely located

Splitting kernel time by phase (`tools/mm_attribution.py`):

| kernel class | forward+backward | optimizer step |
|---|---|---|
| **GEMM** | **83.5 ms / 316 calls** | **1,177.5 ms / 10,210 calls** |
| attention | 0.2 ms / 5 | 0 |
| collectives | 198.4 ms / 68 | 352.7 ms / 4 |

**93.4% of GEMM time is the optimizer.** The 10,210 calls match Newton-Schulz's
expected 10,200 (680 matrices x 5 NS steps x (1 `mm` + 2 `addmm`)). So the
`Cijk_*` rows that dominate the P11 trace at 67.5% of kernel time are **Muon's
orthogonalisation, not the model**. The model's entire forward+backward GEMM work
is 83.5 ms.

`--with-stack` pins it to the line:

```
orthogonalized_optimizer.py(155):_step
  muon.py(137):_orthogonalize
    muon.py(90):_scaled_orthogonalize_fn
      muon_utils.py(228):_newton_schulz_tp
        muon_utils.py(127):_newton_schulz
          muon_utils.py(293):_newton_schulz_step
            <built-in method addmm>        1,502,158 us
```

### And the shapes say *why* it is slow

| NS shape | calls | TFLOP/s | % peak |
|---|---|---|---|
| `mm [7168,67584]x[67584,7168]` | 5 | 1,311.3 | **99.1%** |
| `mm [3584,6144]x[6144,3584]` | 1,680 | 1,012.0 | 76.5% |
| `mm [3072,3584]x[3584,3072]` | 1,680 | 918.8 | 69.5% |
| `addmm [3584,6144]` | 1,680 | 747.4 | 56.5% |
| `addmm [3072,3584]` | 1,680 | 691.3 | 52.3% |
| `addmm [3072,3072]` | 1,680 | 653.9 | 49.4% |

Muon is orthogonalising **expert** matrices — `3584x6144`, `3072x3584`, `3072x3072`
— 1,680 calls each at 49-76% of peak. The handful of large matrices reach 99%, but
they are 5 calls against 1,680.

**This is a shape problem, not a kernel-quality problem.** Newton-Schulz runs 5
iterations over thousands of medium matrices, none individually large enough to
saturate the GPU. Nothing about the pin, the grouped-GEMM backend, or the
attention path touches it — which is consistent with G50 finding the whole TE
2.12 -> 2.18 + CK move worth only 1.038x.

## Two incidental findings

**Gradient clipping forces a device-to-host sync every step.**

```
layer_wise_optimizer.py(279):_step -> optimizer.py(1316):_step
  -> layer_wise_optimizer.py(259):_get_grad_norm
    -> clip_grads.py(51):_get_grad_norm_fp32 -> <built-in method item>
```

`clip_grad=1.0` reads the grad norm back to the host. This is **not**
proxy-specific — a real trainer clips too, so the sync is in the production path.
An earlier note in this session wrongly attributed the step's largest `.item()` to
the proxy's own `float(loss.detach())` (`proxy_ep8.py:138`); the stacks show it is
the grad norm.

**The "slow second layer" is a drain point, not extra work.** Forward host time by
layer: 3.25, **20.81**, 7.50, 6.66 ms, against near-uniform GPU time of
2.99/7.92/7.54/6.56. Layer 1 issues exactly the same ops as layer 2 (38 `copy_`,
40 `to`, 23 `_to_copy`) but spends 14.85 ms in a blocking `hipMemcpyWithStream`
against layer 2's 2.57 ms. The GPU is busy for 19.74 of those 20.81 ms, and the
11.94 ms elementwise burst inside the window belongs to **no** `k3.*` region — it
is earlier queued work draining. The host raced through the dense layer 0 and
layer 1 is simply where it first had to wait. The blocking call comes from
`aten::copy_ < aten::_to_copy < aten::to < CheckpointFunction`, i.e. activation
recompute (`recompute_granularity="full"`).

## Limits of this measurement

- **TE-dispatched GEMMs carry no shapes.** TE's linears go through its own C++
  extension and surface as raw Tensile kernels with no aten parent, so
  `record_shapes` cannot reach them. The model rows above are Megatron's own
  paths; the routed experts are absent by construction. Their efficiency was
  measured separately by standalone `te.GroupedLinear` benchmark: **24-29% of
  peak**, against the 48-55% `roofline.py` implied with a dense stand-in.
- **`with_flops` does not count elementwise ops**, so rows like `aten::mul`
  (155.6 ms, `[512,2,7168]`) show 0.0 TFLOP/s. Unmeasured, not free.
- **seq 512 only.** The model side is small here; the seq-8192 regime the roofline
  describes is still unvalidated against a real run.
- `with_stack` did **not** materially inflate wall clock: steady 1,844.8 ms
  against G50's warm 1,842.3.

## What should change

`tools/roofline.py` should derive its shapes from `preset()` rather than typed
constants, take `m` from the sequence length actually being compared against, and
measure the grouped path for experts. Until then its output is a machine-capability
reference, not a statement about K3.
