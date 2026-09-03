# Response to the Phase 1 review

> Branch `dev/wen/kimi-k3`, checked 2026-09-02 against the working tree.
> Six items, verified individually. Summary: the list is accurate, one stated
> blocker does not reproduce, and item 1.4 sat on top of a second defect that
> the review did not reach. Both 1.4 defects are now fixed (G48).

| # | Item | Status |
|---|---|---|
| 1.1 | Wire the real Megatron trainer | **Accurate** — further along than stated |
| 1.2 | Small-config real BF16 run | **Accurate on scope; blocker does not reproduce** |
| 1.3 | Depth-1 MTP | **Accurate** — not implemented, spec is right |
| 1.4 | Quantile-balancing sign-off | **Fixed** — plus a second defect found |
| 1.5 | TP+EP(+PP) proxy | **Accurate** |
| 1.6 | AdamW scaffold | **Done** — premise partly outdated |

## 1.1 — Wire the real Megatron trainer

Accurate, and two sub-items already exist: `add_kimi_k3_args`
(`config/arguments.py:16`) and `k3_config_from_args`
(`config/k3_config_builder.py:29`). `model_provider` / `forward_step` /
`loss_func` are written to `megatron.training.pretrain` shapes.

Genuinely missing: the call to `pretrain` itself (only `train_smoke`, a
hand-rolled loop), and checkpoint save/load, which appears nowhere under
`kimi_k3/`. Note `optim/resume.py` exists — the A15 workaround for
`dist_muon`'s broken `load_state_dict` — so resume is not from zero, but it is
Muon-specific and will need revisiting if the optimizer changes (see 1.6).

**Do 1.4's dispatch fix before this lands.** See below.

## 1.2 — Small-config real BF16 run

Accurate on scope: mock data only, no real LR schedule.

**"EP8 blocked by the 1.4 router-EP fault" does not reproduce here.** G26
`ep_smoke` passes at EP=8 (112 experts/rank, zero spread across ranks, 0 of 896
starved). Three separate 8-iteration EP=8 proxy jobs completed on 2026-09-01.
`HSA 0x1016` appears nowhere in our records.

The likely reconciliation is in 1.4: **the bias-update path is not reachable
from the proxy at all.** `proxy_ep8.one_step` does forward / backward /
`finish_grad_sync` / `opt.step` and never calls `finalize_model_grads`. If the
fault is in the bias or histogram collectives, our EP=8 runs could not have hit
it — it is a real fault in a path we had never executed, which is consistent
with both observations rather than contradicting either.

## 1.3 — Depth-1 MTP

Not implemented, deliberately: the release ships `num_nextn_predict_layers = 0`,
so there are no MTP weights to convert and nothing to anchor parity against.
Recorded as v1 out-of-scope in three places.

The review's spec checks out. Core has the machinery
(`transformer/multi_token_prediction.py`, `mtp_num_layers`), so this is
configuration rather than new modelling. **Layer 93 is MLA** (verified against
`kda_layers_1idx`; 24 MLA / 69 KDA), so "MTP layer == Gated-MLA(NoPE+gate) +
LatentMoE, not GPT/KDA" is correct and `decoder_layer_specs[-1]` is the right
source.

Two constraints for whoever picks it up:

- `transformer_config.py:2161` caps `mtp_num_layers` at 1 under
  `overlap_moe_expert_parallel_comm`, then requires `pp > 1`. We are pp8, so the
  PP requirement is satisfied; the MoE comm-overlap interaction is untested.
- **AttnRes is the open risk.** Whether the MTP head sits inside or outside the
  final fp32 softmax mix is not answerable from config. This is the same class
  of problem that put VPP out of scope (core rejects our custom tensor shapes
  under interleaving, `schedules.py:949`).

With no released weights this is convergence-testable only — every other
component here was signed off by anchored parity, and that method is not
available. Scope it accordingly.

## 1.4 — Quantile-balancing sign-off

Already built: `ScoreQuantileEstimator`, `quantile_balancing_bias`,
`QuantileBalancingRouter`, `released_gate_reference`.

**The review's all-reduce concern was correct.** The histogram was accumulated
from local tokens only, so per-rank biases diverged by up to 3.3e-02 at 8 ranks.
Fixed via `pooled_histogram()`, reduced over the same TPxCPxDP group core uses.
Now 0.000e+00 disagreement.

**A second defect was found while confirming it, and it is the more serious
one.** `finalize_model_grads._update_router_expert_bias` never calls
`module.update_expert_bias()` — it uses the free function
`get_updated_expert_bias` and copies core's own `sign()` step in. Quantile
balancing would have been silently discarded the moment 1.1 landed, with no
visible symptom: the bias still moves, balancing still looks healthy, and the
algorithm running is core's. Fixed by `install_router_bias_dispatch()`.

Full detail, measurements and the remaining gaps: [`../results/router_bias_sync.md`](../results/router_bias_sync.md).

Remaining sub-items from the review, not yet done:

- **EP8 iter-2 fault** — cannot reproduce; please share the launch command and
  commit. The dispatch fix changes which code runs on that path, so it is worth
  re-running against this tree before hunting it.
- **Checkpoint/restore under EP resharding** — buffers are `persistent=True`,
  but resharding is unverified.
- **Estimator error vs bin width** — already covered by
  `test_k3_p6_moe.py::test_estimator_quantile_error_is_bounded_by_the_bin_width`.
- **Selection-only assertion** (bias steers top-k, never weights) — covered by
  the `released_gate_reference` equivalence test.

## 1.5 — TP+EP(+PP) proxy

Accurate. `tools/proxy_ep8.py:65-66` hardcodes `tensor_model_parallel_size=1`
and `pipeline_model_parallel_size=1`. `config/scaleout.py` holds the measured
constants to calibrate against.

Worth noting the reduce group added in 1.4 is
`get_tensor_and_data_parallel_group(with_context_parallel=True)` and has only
been exercised at TP=1/PP=1 — this item is what would validate it.

## 1.6 — AdamW scaffold

Done on 2026-09-01, including CPU offload. Measured at EP=8, 4L, seq 512:

| arm | steady | peak HBM |
|---|---|---|
| `dist_muon` | 1,845 ms | 191.0 GiB |
| `adam_dist` | **390 ms** | 228.2 GiB |
| `adam_dist` + CPU offload | 10,260 ms | **140.7 GiB** |

"Don't carry AdamW optimizer-state mem into 1.5 scaleout" is right, and now
quantified: **+37.2 GiB, 5.76 B/param** of resident optimizer state.

"Keep Muon as the recipe" is worth revisiting **on cost**: Adam is 4.7x faster
here because Newton-Schulz disappears — Muon's step was 1,501 ms of its 1,845,
and 58% of that was NS `addmm`. That is a proxy-scale figure and will compress
at 93 layers / seq 8192, and it says nothing about convergence, which remains
the real argument for Muon. Recorded so the trade is explicit rather than
assumed.

CPU offload works (100% of state tensors on host, verified per-tensor) but costs
26x at this geometry; the overhead is param-bound and fixed per step, so it
should compress at production sequence length.

The AdamW/Adam decay mode was A/B'd after this table was taken (G49): reversed
order gives 330.4 vs 330.5 ms and identical peak memory, so the mode is free and
these numbers stand as AdamW results. That A/B also found that a **cold proxy
run reads ~6x slow** on this node — relevant to anyone comparing arms measured
on different days. See [`../results/weight_decay_ab.md`](../results/weight_decay_ab.md).

Two notes: the flag is `--optimizer adam_dist`, not `adamw`. And
`--optimizer dist_muon --cpu-offload` now **raises** rather than silently
no-opping — under Muon only the 0.085% scalar group is eligible (finding A20),
so it would report a saving that is not real.
