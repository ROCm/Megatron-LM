# G41 — does QAT train, and how far does it sit from BF16?

> `for mode in bf16 qat; do torchrun --nproc_per_node=8 -m kimi_k3.tools.qat_twin \`
> `    --preset 4L --ep 8 --seq 256 --steps 30 --batch-pool 8 --mode $mode --json out.jsonl; done`
> `python -m kimi_k3.tools.qat_twin --mode report --json out.jsonl`
> Raw: `results/raw/qat_twin_raw.jsonl`. 4 L official, EP=8, `dist_muon`, lr 1e-5, bf16.

## The question is the trend, not the size

Every other twin in this project compares things that *should* be identical, so
the test is whether they sit inside the seed noise. QAT is not one of those:
quantising the routed experts to MXFP4 is **expected** to move the loss. What
would be fatal is a gap that **grows** — quantisation noise the model cannot train
through. So the statistic is the offset's drift, first half against second.

## Result

| | |
|---|---|
| BF16 | 13.229 → 9e-05 over 30 steps |
| QAT | 13.225 → 8e-05 |
| both trained | **yes** |
| mean offset | **−7.7e-04** |
| drift (second half − first half) | **−5.0e-04** |
| worst single step | 0.0219, at step 13 — **0.37 % of the loss at that step** |

The offset is **negative**: QAT sits marginally *below* BF16, which at this
magnitude means the two are indistinguishable, not that quantisation helps. The
gap does not grow. The curves have the same shape throughout — 13.2, a plateau
near 5.9, a drop to 0.14, then the floor.

**Read over steps 0–23, not all 30.** Both curves reach ~1e-04 by step 24 and sit
there, so the last six steps contribute a delta of ~1e-05 that would flatter the
drift statistic for the wrong reason. The window is the region where the loss is
still above 0.01 — 24 of the 30 steps.

## Serving parity: activation quantisation is not free to drop

The trained QAT model's forward, against the same weights served the way a server
would: same quantised weights, **no** activation fake-quant.

| | |
|---|---|
| rel-L2 | **0.0414** |
| argmax agreement | **90.6 %** |
| max abs logit change | 0.875 (logit std 1.69) |

So serving these weights *without* the activation quantisation they were trained
under changes 1 token in 10. That is a real decision, not a rounding error: either
serve with matching MXFP8 activations, or train without activation quantisation
and accept the weight-only regime. It is not something to discover after a
conversion.

## Two ways this measurement was wrong first, and both looked fine

Recorded because both produced *plausible numbers*, which is the dangerous kind of
wrong.

* **A single fixed batch.** At 4 L official the model memorises one batch
  completely and both curves reach **exactly 0.0** by step 20. The offset then
  reported `second_half = 2e-22` and "stable" — true, and meaningless, because
  both sides were sitting on zero and a divergence could not have appeared if it
  existed. The trend test built to catch divergence was blind. A pool of eight
  distinct batches fixes it.
* **Serving parity on a freshly initialised model.** It reported rel-L2 **1.046**
  and **1.2 %** argmax agreement — which reads like catastrophic failure and is
  simply a 2.6 % perturbation measured against a network with no signal in it.
  Taken on the trained model the same measurement is 0.041 and 90.6 %.

The quantiser was checked directly in both cases (2.6 % rel-L2, dtype preserved)
before the experiment was blamed — the tooling was never at fault, the experiment
was.

## Scope

4 layers, synthetic tokens, 30 steps. It establishes that QAT trains and does not
diverge, and it puts a number on the serving gap. It is **not** a convergence
result: that needs real data over many more steps, and it is scheduled work.
