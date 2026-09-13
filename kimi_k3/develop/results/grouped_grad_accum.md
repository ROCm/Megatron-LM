# G63 — the elementwise kernels around the grouped GEMM, identified; and a dead end

> Raised in review: *"what is
> `vectorized_templated_elementwise_kernel<4, CUDAFunctor_add<float>, ...,
> LoadWithCast<2>, StoreWithCast<1>, float, float, c10::BFloat16>` for?"*

## What it is

**DDP's gradient accumulation**, `main_grad (fp32) += grad (bf16)`. The template
signature says so on its own: `CUDAFunctor_add<float>` with `LoadWithCast<2>` and
`StoreWithCast<1>` over `<float, float, BFloat16>` is a mixed-dtype add, casting
on both load and store. Attributed from the trace:

    aten::add_  <-  distributed_data_parallel.py(431): hook
    775 launches, 33.20 ms

and 672 of those 775 are the routed experts:

| calls | ms | shape | |
|---|---|---|---|
| 336 | 14.57 | `[6144, 3584]` | expert fc1 weight |
| 336 | 7.81 | `[3584, 3072]` | expert fc2 weight |
| 103 | 10.82 | assorted | everything else in the model |

336 = **112 experts x 3 MoE layers**. TE's `GroupedLinear` keeps `weight0 ..
weight111` as separate parameters, so DDP registers a backward hook per expert and
issues 336 small adds per matrix where one contiguous add would do.

## The fix TE offers, and why it is not usable here

`GroupedLinear` takes `single_grouped_weight`, which stores the grouped weights as
one parameter. Three things had to be discovered to even test it:

1. **Core never passes it**, and `NVTE_GROUPED_LINEAR_SINGLE_PARAM` only *gates* an
   explicitly requested `True` -- setting the env var alone does nothing.
2. Injecting it through core's `_get_extra_te_kwargs` **breaks every other TE
   module**: that helper is shared, so `RMSNorm.__init__() got an unexpected
   keyword argument`. It has to be patched onto `GroupedLinear.__init__` itself.
3. With that, the layout does change as advertised: **8 parameters per grouped
   module become 1** (`weight0..weight7` -> `weight`).

Then it fails, twice over:

* **QAT breaks.** `RuntimeError: GroupedTensor only supports view(-1) for
  distributed optimizer flatten`. TE's fused mode stores the weight as a custom
  `GroupedTensor`; `k3_qat_wiring`'s `parametrize` hook needs real views. The
  *finder* is fine -- `expert_weight_names` already handles both `weight` and
  `weight{i}` -- the clash is the tensor type. K3 ships MXFP4-quantised routed
  experts, so QAT is not optional for matching the release.
* **It faults the GPU.** At EP=8, with QAT off entirely,
  `Memory access fault by GPU node-N` on every rank. The default run in the same
  script succeeded, so it is this path.

TE marks the feature EXPERIMENTAL. On this stack it is not merely
QAT-incompatible; it does not run.

## Status

`k3_grouped_linear_single_param` stays **default off**, with the patch and a pin
contract kept so the next TE bump can re-test cheaply rather than rediscovering
all three obstacles. **22.4 ms remains on the table** -- it is the largest single
elementwise item left anywhere in the step, model or optimizer, and it is bounded
above by what a working fused-parameter layout would save.

If it is wanted before TE fixes this, the other route is to make QAT quantise the
fused buffer as one tensor instead of parametrizing per expert -- which may be
*simpler* than the current wiring, not harder. That does nothing about the GPU
fault, so TE has to work first.
