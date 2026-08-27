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

- [x] **Step 1** — architecture ground truth (see `architecture/`), verified against
      the released `moonshotai/Kimi-K3` `config.json` + modeling sources and
      against `ROCm/Megatron-LM @ a1b00d4259e92dc4a07a0be2c24088fe827f4b6e`.
- [x] **Step 2** — development plan (see `plan-0/`), incorporating the review of
      the incoming "rev 2" plan (`plan-0/00-review-findings.md`).
- [ ] **Step 3** — code development. Next action: **P0 feasibility gates**
      (`plan-0/04-phase-details.md` § P0). Nothing may be ported before P0 is green.

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
