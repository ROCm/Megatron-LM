# T12.5 — continued-pretrain flatness

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.flatness_probe --preset 4L --ep 8 \`
> `    --seq 256 --steps 40 --lr 1e-6 --quantisation-baseline`
> Raw: `results/raw/flatness_raw.jsonl`. Evaluation against the released
> checkpoint is human-run (R10.1); this is the harness plus its calibration.

## What it catches

A conversion defect **looks like success on a loss curve**. It spikes at step 0,
the model re-converges over a few hundred steps, and by the time anyone looks the
curve is flat and the evidence is gone. Reading the last few steps says "healthy",
and they genuinely are.

## Flatness is a shape, not a slope

The first version tested `drift < -tolerance` and **failed a healthy run**:
continued pretraining is supposed to descend, and 0.002 per step over 40 steps is
0.08 of drift, entirely normal. Recovery is distinguished by *deceleration* —
a model climbing back from broken weights drops hard then flattens, while genuine
training descends at a steady rate.

| window | drift | deceleration | verdict |
|---|---:|---:|---|
| flat | ~0 | 0.00 | flat |
| steady descent, 0.002/step x 40 | −0.078 | 0.00 | **flat** |
| recovery `3.40 → 2.50` | −0.87 | **0.53** | not flat |
| one spike in a quiet window | ~0 | 0.00 | not flat (`spike`) |

## The threshold has to know the data's noise

The first end-to-end run returned **"not flat"** — and was wrong. A fresh model on
random tokens has a typical step-to-step move of **0.137**; its "spike" of 0.2423
is 1.7x that, and it tripped a fixed 0.05 tolerance. Ordinary sampling noise,
reported as a defect.

The threshold is now the larger of what the caller asked for and what the window
can **resolve** (3 x the step noise), and the verdict carries `conclusive` saying
which regime the answer came from. Re-scored, the same run reads:

    arrival 13.385   (chance ln(163840) = 12.007)
    drift -0.0690    deceleration -0.1859    spike 0.2423
    step_noise 0.1371 -> resolution 0.4114
    flat = True, conclusive = False
    "a 'flat' verdict here means 'nothing larger than the noise', not 'nothing'"

That is the honest reading of a fresh model on synthetic tokens: nothing to see,
and the window could not have seen much anyway. It is the same lesson the twin-run
band taught — **a threshold is a property of a configuration, not of a project.**

### The noise estimator is the median, and that matters

A spike creates two large consecutive deltas. Using the *mean* would let the event
inflate the very estimate meant to judge it, pushing the threshold past its own
detection. With the mean, the single-spike case above stops convicting. The median
is unmoved by the outliers being detected.

## C6 is measured, not assumed

Finding **C6**: the released routed experts are MXFP4 and the original BF16 weights
were never published, so `dequantize_on_import` starts a continued-pretrain run
from **quantised-then-dequantised** weights. A small arrival bump is *expected*.

`--quantisation-baseline` measures it: the same batch with expert weights as
loaded, then fake-quantised through the same MXFP4 path. **Measured 0.0749** at
4 L official. Any arrival offset within that is attributable to C6; beyond it is
not. A test asserts the offset is nonzero — zero would mean the baseline silently
measures nothing and any real bump would look excusable.

Measuring it also exposed a bug in the probe: it left the weight parametrizations
attached, keeping both the fp32 master and the quantised copy resident, and the
training loop that followed **OOMed at official width**. `disable_qat_experts` now
undoes hooks *and* parametrizations, restoring the original parameter rather than
the quantised value.

## To run it for real

    --checkpoint <converted release checkpoint>   the actual subject
    --expect-loss <released model's own figure>   turns arrival into a comparison

On real held-out data the step noise falls and the window becomes conclusive.
Without both of those this measures the harness, which is what it was used for here.
