# Plan-0 — Kimi K3 Training in ROCm/Megatron-LM

This directory is the **development plan** for adding Kimi-K3 training support to
`ROCm/Megatron-LM`. After any interruption, pick up at the first phase in
[`01-roadmap.md`](01-roadmap.md) whose status is not `done`.

## Document index

| File | Description |
|---|---|
| [`00-review-findings.md`](00-review-findings.md) | **Review of the incoming "rev 2" plan**, verified against the pinned repo and the released model. Read this to understand why plan-0 differs. |
| [`01-roadmap.md`](01-roadmap.md) | **Roadmap.** Principles, 13 phases, dependency graph, milestones, top risks, scope. Start here. |
| [`02-target-architecture.md`](02-target-architecture.md) | **Target design.** K3 modules → Megatron parents, config mapping, the AttnRes pipeline protocol, state-dict mapping. |
| [`03-code-layout.md`](03-code-layout.md) | **Landing list.** Every file, its owner phase, and the interface contracts. |
| [`04-phase-details.md`](04-phase-details.md) | **Per-phase tasks**, exit criteria and risks. |
| [`05-test-strategy.md`](05-test-strategy.md) | **Gates G1–G47**, tiers, tolerance harness, banned warnings, CI ladder, test tree. |
| [`06-capacity-and-parallelism.md`](06-capacity-and-parallelism.md) | **Memory and parallelism math** — corrected optimizer model, AttnRes payload cost, PP/EP/CP layout rules. |

Ground truth about the model itself lives one level up in
[`../architecture/01-kimi-k3-architecture-deep-dive.md`](../architecture/01-kimi-k3-architecture-deep-dive.md);
the working rules are in [`../rules/rule.md`](../rules/rule.md).

## One-line summary

> Land Kimi-K3 as a **fork-local `kimi_k3/` package** that rides existing core
> mechanisms (`adjust_tensor_shapes_fn`, `MoESubmodules.router`, `moe_latent_size`,
> `clip_qk`, `dist_muon`) instead of forking them: gate feasibility first (P0),
> then build the four K3-specific pieces — KDA, gated NoPE MLA, AttnRes, LatentMoE
> with quantile balancing — each against an FP32 oracle (P3–P6), then trainer,
> converter, training equivalence, QAT and performance (P7–P11).

## Phase map

```
        ┌─ gate ─┐   ┌──── scaffolding ────┐
        P0  ──────►  P1  ──►  P2
                                │
              ┌─────────────────┼─────────────────┬─────────────────┐
              ▼                 ▼                 ▼                 ▼
        P3 KDA            P4 gated MLA      P5 AttnRes + PP    P6 LatentMoE
              └─────────────────┴────────┬────────┴─────────────────┘
                                         ▼
                                  P7 single-node trainer
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
             P8 converter          P10 MXFP4 QAT         P11 perf + fused AttnRes
                   │                     │                     │
                   ▼                     │                     │
             P9 twin runs + Muon ────────┴─────────────────────┘
                                         ▼
                                 P12 scale-out prep [CLUSTER]
```

## What plan-0 changes relative to the incoming "rev 2" plan

1. **AttnRes pipeline transport is redesigned.** Core back-props only
   `output_tensor[0]`, so the payload is **one packed tensor**, not two; and core
   already provides `adjust_tensor_shapes_fn`, so no schedule is re-implemented.
   A dedicated gate (G20) fails if a payload gradient is ever dropped.
2. **The Muon capacity model is corrected.** `dist_muon` shards master weights
   and momentum across DP; the per-GPU cost is `6 + 8/DP`, and 98 % of the
   parameters are sharded by EP — neither appeared in the old table.
3. **The ground truth is completed** from the released model:
   `attn_res_block_size = 12` (8 slots), the exact AttnRes math, KDA's two
   distinct gates, the in-kernel q/k L2 norm, shared experts on the 7168-wide
   hidden, and the MLA `192**-0.5` scale.
4. **`GPTModel.__init__` is not duplicated** — a scoped symbol rebinding gives
   the same "no transient core block" guarantee with a fraction of the IFU
   surface.
5. **Performance is planned** (P11): the AttnRes fp32 mixer is a predictable
   ~120 GB-of-reads-per-forward bottleneck, and it gets a trace, a budget and a
   fused kernel.
6. **Process is added**: rules, gates G1–G47, a status tracker, per-phase
   summaries, an IFU tripwire test, and a banned-warning ratchet.

## What we are **not** doing in v1

1. **MTP** — the release ships `num_nextn_predict_layers = 0`.
2. **The vision tower** — text tower only; vision keys are explicitly skipped by the converter.
3. **KCP / 1 M context** — v1 trains ≤ 64 k.
4. **VPP with AttnRes** — core rejects custom tensor shapes under interleaving.
5. **a8w4 backward** and **dense a8w4 on gfx950** — the kernels do not exist.
6. **MoonEP / DeepEP integration** — evaluated on paper in P12.

## How to start working

1. Read [`../rules/rule.md`](../rules/rule.md), then [`01-roadmap.md`](01-roadmap.md).
2. Open [`04-phase-details.md`](04-phase-details.md) at the phase you are picking up.
3. Work the task list; tick rows in [`../progress/status.md`](../progress/status.md)
   with the commit SHA (R1.3).
4. Close the phase with `../progress/p<id>/p<id>-summary.md` (R3.5).
5. Investigations go to [`../notes/`](../notes/) as `YYYY-MM-DD-<topic>.md`.
