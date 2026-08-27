# P6 — LatentMoE, quantile-balancing router, SiTU (COMPLETE)

## 1. Objective

The last core piece: routed experts at the 3584-wide latent with the norm core
lacks, shared experts at the hidden width, the released routing rule, and a
balancing scheme whose evidence is honestly labelled.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/moe/k3_moe_layer.py` | `K3MoELayer`: `postprocess` inserts `routed_expert_norm` before the up-projection |
| `kimi_k3/moe/k3_router.py` | `QuantileBalancingRouter`, the histogram estimator, the bias rule, and a transcription of `KimiMoEGate.forward` |
| `kimi_k3/specs/layer_specs.py` | MoE layers build `K3MoELayer` with the K3 router |
| `kimi_k3/tests/test_k3_p6_moe.py` | 10 tests |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G23** — SiTU | **GREEN** | matches the release formula; both branches provably tanh-capped |
| **G24** — LatentMoE | **GREEN** | latent norm demonstrably in the path (changing its gain changes the model output); routed experts at 3584, shared at 7168; only layer 1 dense |
| **G25** — routing + balancing | **GREEN** | routing matches the transcription: selection uses the bias, weights come from the **unbiased** scores; the bias rule agrees exactly with an independently written reference; estimator error 2e-2 at 64 bins and 2e-3 at 1024 |
| **G26** — EP | **partial** | single-rank paths green; the 8-rank exercise belongs to the nightly job |

`pytest kimi_k3/tests/ -q` → **161 passed, 1 skipped**.

## 4. Two halves, two standards of evidence

**Routing is published.** `KimiMoEGate.forward` is in the release, so the test is
a transcription comparison, and it pins the detail that would otherwise be got
backwards: the correction bias steers *selection* but never reaches the
*weights*, which are gathered from the unbiased scores. Reversing that changes
what the model optimises, and nothing would crash.

**Balancing is not.** The K3 report names Quantile Balancing but gives no
algorithm, and the release ships no reference. So the implementation here is
explicitly *our formulation* — each expert should win `topk / num_experts` of the
tokens, so read the score at that quantile from a running per-expert histogram
and bias every expert to the pooled threshold — and it is gated on internal
consistency only: exact agreement with a plainly-written reference, measured
estimator error, and the behaviour it exists to produce (a skewed router's load
ratio falls from 2.0 to under 1.6). No test claims release parity, and the module
docstring says why.

This is the same distinction the review drew when it demoted the plan's
load-ratio band from a gate to an observation.

## 5. Hand-off to P7

The decoder is now complete and every piece is gated: KDA, gated NoPE MLA,
AttnRes placement and transport, LatentMoE with the K3 router. P7 is the trainer
— `pretrain_kimi_k3.py`, the tokenizer, the 4 L official config on the measured
optimizer recipe, and the memory report.

Two things P7 inherits that are worth knowing: `dist_muon` is the sharded Muon
path (7.87 B/param at DP=8, measured in G5), and `DistributedDataParallelConfig`
must carry `use_distributed_optimizer` too or the first step dies in
`_copy_main_params_to_model_params`.

## 6. Commit chain

_pending._
