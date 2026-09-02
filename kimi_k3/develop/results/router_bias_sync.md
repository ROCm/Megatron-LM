# G48 — the router's expert bias, across ranks and through core

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.router_sync_probe`
> `BYPASS_REDUCE=1` reproduces the pre-fix behaviour.
> Raised by a branch review (Phase 1 item 1.4). Two defects, one found by the
> review and one found while confirming it.

## Defect 1 — the histogram was never reduced across ranks

`ScoreQuantileEstimator.update()` accumulated an EMA histogram of routing
scores from **local tokens only**. Every rank therefore read a different
quantile off a different distribution and derived a different
`expert_bias` — so the same token could be routed to different experts
depending on which rank happened to hold it, and nothing anywhere asserted
otherwise. Core reduces its own `tokens_per_expert` over the TPxCPxDP group for
exactly this reason (`moe_utils.py:1201`); our override replaced that path and
dropped the collective with it.

Measured at 8 ranks, each fed a deliberately different score distribution:

| | histogram mass | max bias disagreement |
|---|---|---|
| local only (pre-fix) | 32,768 | **3.324e-02** |
| pooled (`pooled_histogram()`) | 262,144 = 8 x | **0.000e+00** |

The fix reduces into a **copy**, never the persistent buffer: `histogram` is an
EMA re-reduced on every call, so an in-place sum would multiply it by the world
size once per step.

`is_populated()` is now reduced too. It gates the call to `quantile()`, which
now contains a collective — a rank whose own histogram was empty would have
skipped it and hung every other rank.

## Defect 2 — core never called our update at all

Found while confirming defect 1, and the more serious of the two.

`finalize_model_grads._update_router_expert_bias` (`:293-319`) collects every
module exposing an `expert_bias`, computes a new value with the free function
`get_updated_expert_bias` — core's fixed `sign()` step — and copies it in. **It
never consults the module.** `QuantileBalancingRouter.update_expert_bias` was
therefore dead code on any path that runs through `megatron.training.pretrain`:
core would recompute the bias from scratch every iteration and overwrite ours.

The failure is silent. The bias still moves every step, load balancing still
looks like it is working, and the only symptom is that the algorithm being run
is core's and not K3's.

This had not fired yet because nothing reached it. `proxy_ep8.one_step` does not
call `finalize_model_grads`; only `train_smoke` drives the update, through its
own per-module loop (`pretrain_kimi_k3.py:175-178`). It would have fired the
moment Phase 1 item 1.1 landed.

`core_patch.install_router_bias_dispatch()` rebinds that module-scope symbol so
routers implementing `update_expert_bias` handle themselves and every other
module still takes core's original path. Installed from `build_k3_model` when
`moe_router_enable_expert_bias` is set, idempotent, guarded by pin contract
`router expert-bias update dispatches to the module` (rule R2.2).

## What was wrong with the tests

All router tests were single-rank, so defect 1 was untestable by construction.

`test_expert_bias_is_updated_during_training` was docstringed *"the
quantile-balancing update has to be reached by the training loop"* but called
`update_expert_bias()` by hand — it asserted the function mutates the bias, not
that anything reaches it. Renamed to
`test_expert_bias_moves_when_the_update_is_called`, which is what it checks.
Reachability is now covered by
`test_k3_p6_moe.py::test_core_bias_update_dispatches_to_the_router`.

## Still open

- The estimator and `expert_bias` buffers are `persistent=True`, but
  checkpoint/restore under **EP resharding** is unverified (review item 1.4).
- The reduce group is `get_tensor_and_data_parallel_group(with_context_parallel=True)`,
  matching core. Not yet exercised under TP>1 or PP>1 — blocked on review
  item 1.5.
