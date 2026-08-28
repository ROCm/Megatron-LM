# G26 — expert parallelism across 8 ranks, and where non-determinism starts

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.ep_smoke --preset 4L --ep 8 --seq 256 --steps 3`
> Raw: `results/raw/ep_smoke_raw.jsonl`. 4 L official, EP=8, bf16, `dist_muon`.

## The sharding is exactly right

| | measured | expected |
|---|---|---|
| local experts per rank | **112** | 896 / 8 = 112 |
| parameters per rank | **16,306,993,312** | 16.31 B (`tools/mem_budget.py`) |
| spread across ranks | **zero** — all eight identical | — |
| starved experts | **0 of 896** | — |
| load, max / mean | **1.77** | — |
| load, min / mean | **0.44** | — |

The parameter count matches the analytic model to the digit, on every rank. No
expert goes unused: a router that had collapsed would still train and still
descend, and the only symptom would be ranks doing no work — which is the reason
this gate is worth eight GPUs at all.

## Two identical eval forwards are not bitwise, and this is where it starts

Repeating the same forward under `eval()` and `no_grad` gives logits differing by
**max-abs 0.53–0.79**, against a logit std of 1.69. Far too large for accumulated
float noise, so it was traced rather than explained away.

**4.0 % of tokens are routed to different experts** — 31 of 768 token-layer pairs.
That accounts for the whole difference: a token sent to a different expert
produces a materially different output. But it raises the real question, because
identical inputs must produce identical router logits — a near-tie has nothing to
flip it. Something upstream had to move first.

Recording *which* MoE layer differs answers it. Across all eight ranks:

| MoE layer | tokens re-routed (of 256) |
|---|---|
| **1st** (layer 1; layer 0 is dense) | **0 on every rank** |
| 2nd | 4 – 8 |
| 3rd | 15 – 29 |

**The first MoE router is bitwise identical on all eight ranks.** Everything
before it — embedding, the dense layer, KDA, gated MLA, the AttnRes mixes — is
deterministic. The non-determinism enters *inside* the MoE expert computation
(grouped GEMM and the all-to-all combine, both of which accumulate with atomics),
and then compounds: its output perturbs the next router, 4–8 tokens flip, and the
layer after that sees both the float noise and the changed expert outputs, so
15–29 flip.

So this is not a defect in K3's attention, its residual stream, or its router. It
is the known MoE accumulation path, amplified by top-16-of-896 routing.

Ruled out on the way, each by reading the code rather than assuming:

* **dropout** — at tiny in fp32 the same test is bitwise zero with dropout on
  *and* off, so `eval()` propagates correctly;
* **core's expert-bias accumulator**, the trap P0 documented in
  `results/pp_payload.md` — core guards the update with `torch.is_grad_enabled()`,
  and this check runs under `no_grad`;
* **our own quantile estimator** — guarded by `self.training and
  torch.is_grad_enabled()`, so it does not fire either.

## What this means for every other gate

**A bitwise gate cannot be run at bf16 with expert parallelism.** The gates in
this project that demand exact equality — G21's PP parity, G37's resume, G43's
fused mixer — all run in fp32, at tiny, or with expert bias off, which is why they
can. That was previously a convention inherited from P0's noise-floor work; it is
now a measured requirement with a located cause.

Reproducible routing at production settings is what core's `router_replay` exists
for. Exercising it is owed.

## Owed

`router_replay` record-and-replay, which would pin routing across forwards and let
a determinism check run at bf16 with EP. This run does not exercise it.
