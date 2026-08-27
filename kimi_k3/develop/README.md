# Kimi K3 in ROCm/Megatron-LM — Development Workspace

`kimi_k3/develop/` is the development knowledge base for **adding Kimi-K3
training support to ROCm/Megatron-LM**. All plans, verification records,
technical analysis and per-phase records live here.

**Production code never lands here.** Code lands under `kimi_k3/` (the
sibling directories: `config/`, `attention/`, `block/`, `moe/`, `model/`,
`optim/`, `pipeline/`, `tools/`, `training/`, `tests/`). `megatron/**` is
read-only and guard-enforced (see [`rules/rule.md` §2](rules/rule.md)).

## Directory Layout

```
kimi_k3/develop/
├── README.md                     ← this file (quick index)
├── rules/
│   └── rule.md                   ← project-wide working rules (guard, gates, commits, tolerances)
├── architecture/
│   └── 01-kimi-k3-architecture-deep-dive.md   ← ground truth, verbatim-verified against the release
│
├── plan-<n>/                     ← per-plan documents (plan-0 today)
│   ├── README.md                 high-level pitch + links
│   ├── 00-review-findings.md     review of the incoming plan / current tree (severity-tagged)
│   ├── 01-roadmap.md             phase overview, dependency graph, milestones, top risks
│   ├── 02-target-architecture.md K3 reference → Megatron target module map + contracts
│   ├── 03-code-layout.md         file landing list + phase × file matrix
│   ├── 04-phase-details.md       per-phase tasks, exit criteria, risks
│   ├── 05-test-strategy.md       gate matrix, tiers, tolerance harness, CI ladder
│   └── 06-capacity-and-parallelism.md  memory model, optimizer recipes, EP/PP/CP layout math
│
├── notes/                        ← ad-hoc investigations, `YYYY-MM-DD-<topic>.md`
├── profile/                      ← perf-trace analysis reports (created in P11)
├── perf/                         ← operator + end-to-end perf tables (created in P11)
└── progress/
    ├── status.md                 ← real-time per-task status (5-column rows)
    └── p<id>/                    ← per-phase scratch + `p<id>-summary.md` close-out
```

## Current Status

- [x] **Step 1** — architecture ground truth (`architecture/`), verified against the
      released `moonshotai/Kimi-K3` and against `ROCm/Megatron-LM @ a1b00d4`.
- [x] **Step 2** — development plan (`plan-0/`), incorporating the review of the
      incoming "rev 2" plan (`plan-0/00-review-findings.md`).
- [-] **Step 3** — code development, in progress:

| phase | state | gates |
|---|---|---|
| **P0** feasibility | complete | G1–G8 green |
| **P1** scaffold, guard, CI | complete | G9–G10 green |
| **P2** config, presets, construction | complete | G11–G13 green |
| **P3** KDA | complete | G14–G16 green |
| **P4** gated MLA (NoPE) | complete | G17–G18 green |
| **P5** AttnRes layer + transport | complete | G19–G22 green |
| **P6** LatentMoE, QB router, SiTU | complete | G23–G25 green, G26 partial |
| **P7** trainer | complete | G27–G29 green |
| **P8** | complete | converter both directions, 497,220 tensors accounted for, anchored parity on real layer-0 weights (cosine 0.999973) |
| **P9** | complete | twin-run noise band, per-head Muon, QK-clip, bitwise resume under PP=2 (found A15: dist_muon could not resume at all) |
| **P10** | complete (G41 owed) | MXFP4 scale rule corrected against the released weights (A16); a8w4 forward at rel-L2 1.66e-3; STE gradients exact |
| **P11** | partial | chunked AttnRes mixer bit-identical (G43); baseline trace says the Muon step is 21 % and AttnRes 0.1 % — phase rescoped. EP=8 blocked on a shared node |
| **P12** | partial | 93 L configs validated: floor 28 nodes, PP capped at 8 by AttnRes block alignment (G46). G47 owed |

The decoder is complete and trains end to end: 30 steps on a fixed batch drive the
loss from 8.36 to ~0.0 under both `dist_muon` and `adam`, and the 4 L official
config (94 B parameters) trains on a single node at EP = 8, peaking at 189–202 GiB
of 288. `pytest kimi_k3/tests/ -q` → **168 passed, 1 skipped**.

Owed to a nightly job rather than an interactive session: production-geometry KDA
parity (G15), the 8-rank EP exercise (G26), sequence-length memory scaling beyond
512, and the AITER a8w4 kernel path (P10 — needs the workspace checkout bumped
and rebuilt).

Task-level tracking: [`progress/status.md`](progress/status.md).

## Suggested Reading Order

| Audience | Path |
|---|---|
| **First-time reader / want the working rules first** | [`rules/rule.md`](rules/rule.md) |
| **Reviewer / 5-min overview** | [`plan-0/README.md`](plan-0/README.md) → [`plan-0/01-roadmap.md`](plan-0/01-roadmap.md) |
| **Reviewer / "is the incoming plan right?"** | [`plan-0/00-review-findings.md`](plan-0/00-review-findings.md) |
| **Developer / about to pick up a phase** | [`plan-0/04-phase-details.md`](plan-0/04-phase-details.md), jump to that phase |
| **Developer / where does my file go?** | [`plan-0/03-code-layout.md`](plan-0/03-code-layout.md) |
| **Anyone asking "what IS Kimi K3?"** | [`architecture/01-kimi-k3-architecture-deep-dive.md`](architecture/01-kimi-k3-architecture-deep-dive.md) |
| **Anyone sizing a run** | [`plan-0/06-capacity-and-parallelism.md`](plan-0/06-capacity-and-parallelism.md) |

## Naming Conventions

- **`kimi_k3/...`** — landing path for all production changes.
  `megatron/**` is read-only; the only allowlisted root files are
  `.github/workflows/kimi_k3.yml` and `.pre-commit-config.yaml`.
- **`kimi_k3/develop/`** — design docs / notes / progress only, **no production code**.
- When citing core, use `megatron/<path>.py:<line>` at the pinned SHA; when citing
  the K3 release, use `HF moonshotai/Kimi-K3:<file>` plus the symbol name.
- Phases are `P<id>`; gates are `G<n>` (monotonic, never reused); rules are `R<n.m>`.
