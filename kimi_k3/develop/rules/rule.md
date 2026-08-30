# Kimi K3 Development Rules

> Working rules for the Kimi-K3 integration into ROCm/Megatron-LM.
> New rules are appended to the smallest matching section. Retired rules are
> kept with `~~strike-through~~` plus a retirement date — they are the audit
> trail. Rules are atomic and self-contained: pick the section, then read one rule.
>
> Last updated: 2026-08-27 (R3.6 execute-don't-infer).

---

## §1. Review / commit workflow

### R1.1 — Stop before commit for review
After finishing a phase or task, **do not commit automatically**. Present a
one-page summary of what changed, which gates passed, and what was de-scoped.
Commit only after the user says "commit".

### R1.2 — Push only on explicit prompt
Never `git push` until the user explicitly asks. A local commit is the natural
pause point.

### R1.3 — Status-pin commit pattern
Every feature commit (`feat(kimi-k3)[P<id>]: ...`) is followed **immediately**
by a docs-only commit that pins the matching `progress/status.md` rows to the
feature SHA:

```
docs(kimi-k3)[P<id>]: pin status.md P<id> cells to the P<id> SHA (<feature-sha>)
```

The pin commit only touches `develop/progress/status.md` and
`develop/progress/p<id>/p<id>-summary.md`. Never fold it into the feature commit.

### R1.4 — Commit message format
`feat|fix|refactor|docs|test(kimi-k3)[P<id>]: <summary>`, subject ≤ 100 chars.
Add the plan tag when a commit opens or closes a plan:
`docs(kimi-k3)[plan-0]: open plan-0`.

---

## §2. Core isolation (the load-bearing rule)

### R2.1 — `megatron/**` is read-only
`kimi_k3/ci/no_core_diff_guard.sh` fails any diff outside `kimi_k3/` except the
allowlist: `.github/workflows/kimi_k3.yml` and `.pre-commit-config.yaml`.
The guard runs on every commit and in CI stage 0.

### R2.2 — Namespace patching is allowed; file edits are not
Rebinding a *module attribute* at runtime from our own entry point (e.g.
`megatron.core.models.gpt.gpt_model.TransformerBlock = K3TransformerBlock`
inside a scoped context manager) produces **no diff** and is therefore allowed.
Every such patch must:
1. live in exactly one file (`kimi_k3/model/core_patch.py`),
2. be scoped (context manager or explicit install/uninstall — never import-time
   global mutation),
3. carry an assertion test that fails loudly when the patched symbol moves or
   changes signature after an IFU (see R4.5).

### R2.3 — Escalate, don't work around silently
Anything genuinely impossible without editing `megatron/**` stops the phase:
write the minimal upstream-shaped diff into `kimi_k3/upstream_proposals/NNN-*.patch`
plus an issue draft, get human sign-off, and continue with a **documented**
workaround. Never quietly fork a core file into `kimi_k3/`.

### R2.4 — Prefer core mechanisms over re-implementation
Before writing a mechanism, grep core for it at the pin. The recurring examples:
`adjust_tensor_shapes_fn` (PP payload shapes), `MoESubmodules.router` (router
injection), `qk_clip.clip_qk` (QK-clip), `router_replay` (routing determinism),
`moe_latent_size` (LatentMoE), `LayerWiseDistributedOptimizer` (Muon sharding).
A phase that re-implements one of these without recording *why* fails review.

---

## §3. Documentation conventions

### R3.1 — English in dev docs
All of `develop/` is written in English.

### R3.2 — Plan layout
Every plan ships under `develop/plan-<n>/` with:

| file | purpose |
|---|---|
| `00-review-findings.md` | severity-tagged review that motivated the plan (optional after plan-0) |
| `01-roadmap.md` | phase overview table, dependency graph, milestones, top risks |
| `02-target-architecture.md` | reference → Megatron module map and interface contracts |
| `03-code-layout.md` | file landing list + phase × file matrix |
| `04-phase-details.md` | per-phase tasks, exit criteria, risks |
| `05-test-strategy.md` | gate matrix, tiers, tolerance harness, CI ladder |
| `06-capacity-and-parallelism.md` | memory / parallelism math (optional after plan-0) |
| `README.md` | pitch + links |

### R3.3 — Gate naming
Gates are `G<n>`, monotonically increasing across the whole project lifetime.
**Never reuse a gate number.** Sub-letters (`G12a` / `G12b`) are allowed when one
parent gate has several sub-criteria owned by distinct scripts.

### R3.4 — Every claim about core or the release carries a citation
A factual statement about Megatron core cites `megatron/<file>.py:<line>` at the
pinned SHA. A statement about Kimi K3 cites the released artefact
(`config.json` field, `modeling_*.py` symbol, or a safetensors header shape).
Statements taken from the K3 report are labelled **[report]** and are treated as
weaker evidence than the release artefacts.

### R3.6 — Claims about a dependency's behaviour are executed, not inferred
**Added 2026-08-27 after a retraction.** A statement about what a third-party
function *does* — ignores an argument, mis-handles a dtype, is slow — is written
down only after running it. Reading a signature is not running a function:
`chunk_kda` was declared to "silently ignore `A_log`" on the strength of its
signature, and the claim reached four documents before an actual call disproved
it (`../notes/2026-08-27-fla-signature-check.md` §1). Signature-shaped evidence
is admissible for *structure* (a symbol exists, a parameter is accepted); only
execution is admissible for *behaviour*.

### R3.5 — Per-phase summary file
Every phase that ships work closes with `develop/progress/p<id>/p<id>-summary.md`
containing, in order: 1. Objective · 2. What changed (file → what) · 3. Gates
(table with numbers) · 4. Measurements · 5. De-scope decisions · 6. Hand-off to
the next phase · 7. Artefacts shipped · 8. Known follow-ups · 9. Commit chain.
It lands in the same commit as the status pin (R1.3), not the feature commit.

---

## §4. Test conventions

### R4.1 — Test file naming
`kimi_k3/tests/test_k3_p<id>_<short_name>.py`. The phase id in the filename is
the phase that introduced the test; later phases extending it keep the name.

### R4.2 — Tiny by default, production shapes behind `--run-slow`
Unit and parity tests use the `tiny` preset by default. Production-width
parametrisations are marked `pytest.mark.slow` and run only in the nightly
8-GPU job. **Never instantiate an official preset (4L/8L/93L) in a default-tier
test** — 8L official is ≈215 B params and 93L is ≈2.78 T.

### R4.3 — Official presets are validated analytically, and cannot be constructed
**Amended 2026-08-27 (P2).** Meta device does *not* make construction free:
Megatron and TE modules place parameters on `torch.cuda.current_device()`
regardless of an ambient device context, so "build it on meta and count" would
really allocate 2.78 T parameters at 93 L. `build_k3_model` refuses non-tiny
presets unless the caller passes `allow_official=True`. Official presets are
checked by analytic parameter counting alone, against the per-component table in
[`../architecture/01-kimi-k3-architecture-deep-dive.md`](../architecture/01-kimi-k3-architecture-deep-dive.md),
never by allocating weights.

### R4.4 — Component-specific tolerances, floor-measured first
There is no project-wide tolerance. Every kernel/backend is compared against an
**FP32 oracle** on three statistics — rel-L2, max-abs, cosine. The bound is set
in the component's first parity task by (a) measuring the eager-fp32 vs
eager-bf16 spread as the floor, (b) setting the bound above it with recorded
margin, (c) writing the rationale into the test docstring. A bound is never
relaxed silently; every change is called out in the phase status row.
fp64 `torch.autograd.gradcheck` applies **only to true-autograd modules**
(KDA, AttnRes mixer, MLA gate) and **never to STE paths** — an STE backward is
deliberately not the derivative of its forward, so gradcheck is invalid there by
construction. STE is validated against an explicit fake-quant reference module.

### R4.5 — Pin-drift assertions
Every core mechanism we depend on (R2.4) and every namespace patch (R2.2) has an
assertion test in `kimi_k3/tests/test_k3_p1_pin_contracts.py` that fails when the
symbol, signature, or behaviour drifts. These run in CI stage 0 (CPU) and are the
first thing re-run after an IFU rebase.

### R4.6 — Correctness ratchet
Every phase keeps all previously green gates green. Each phase opens with a
ratchet check and records the pass count in its status row. Any drop blocks the
phase commit. Pre-existing unrelated failures are documented in the phase summary
after being reproduced with the phase's changes stashed.

### R4.7 — Determinism + `router_replay` for anything touching routing
Tests that touch routing pin seeds and use core's `router_replay` so a rerun
reproduces expert assignment exactly.

---

## §5. Run scripts and env-var conventions

### R5.1 — `${VAR:-DEFAULT}` guards
Every override-able variable in `kimi_k3/training/*.sh` and the per-phase
`progress/p<id>/run_*.sh` scripts uses `${VAR:-DEFAULT}` so any knob can be
flipped from the command line without editing the script.

### R5.2 — `K3_*` namespace for overrides
Shape / config overrides use the `K3_*` prefix (`K3_SEQ_LENGTH`,
`K3_NUM_LAYERS`, `K3_EP`). Feature flags use the matching un-prefixed
`UPPER_SNAKE_CASE` name that mirrors the config field
(`USE_K3_FLA_KDA` ↔ `use_k3_fla_kda`).

### R5.3 — Backend switches are explicit and default to the safe path
Every kernel has an eager FP32 oracle that stays in tree permanently and is
selectable at runtime. Fast backends (fla `chunk_kda`, AITER a8w4) default
**off** until their parity gate is green at production shapes, then the default
flip is its own commit with the gate numbers in the message.

**Flipped so far:** `k3_kda_backend` -> `fla` on 2026-08-30, after G15 ran at
H=96 / K=128 and seq 1024/4096/8192 (fp32 6.4-6.8e-07, bf16 4.31e-03 against a
3.31e-03 floor). The gate was nearly flipped a day early on the strength of "the
slow suite is green" -- which was true and irrelevant, because no slow test
covered production geometry. *Green somewhere* is not the condition this rule
states, and wanting the flip (it decides whether 93 L fits on 28 nodes) is
exactly when to re-read the rule rather than the summary.

---

## §6. Git hygiene

### R6.1 — Never commit logs, traces, or heavy artefacts
`*.log`, chrome traces (`*.pt.trace.json`), `*.tgz`, `*.tfevents*`, forensic
outputs, and checkpoint shards are gitignored per `progress/p<id>/.gitignore`.
Committed artefacts are: scripts, rendered reports (`*.md` / `*.html`), and
curated tables.

### R6.2 — Never commit vendored clones
Reference clones (fla, AITER, TransformerEngine, HF Kimi-K3) stay outside the
tree. Ported reference code is a **file with a provenance header**, never a clone.

### R6.3 — No interactive git
No `git rebase -i`, `git add -i`, `git commit --amend`, no force-push to
`rocm_dev` or `dev/wen/kimi-k3`.

---

## §7. Code style

### R7.1 — No narrative comments
Comments explain non-obvious intent, trade-offs, or constraints the code cannot
convey — never "increment the counter". "Why this over the obvious alternative"
comments are encouraged, especially around core-version workarounds.

### R7.2 — Provenance + license header on every ported file
A file adapted from another repository carries `Source: <repo>@<sha>:<path>`,
the adaptation summary, and a pointer to the `LICENSES` section of
`kimi_k3/PINS.md`. **A header alone is not sufficient** — the license
compatibility conclusion must be recorded in `PINS.md` before the file lands.

### R7.3 — Dtype contract
K3 numerics that the release performs in fp32 are performed in fp32 by us and
are not "optimised" to bf16 without a gate: the AttnRes mixer (`_apply_attn_res`
upcasts to fp32), the MoE router logits, the KDA `beta`/`A_log`/`dt_bias` path,
the SiTU activation, and the MLA attention output **[report]**. Everything else
follows the input dtype.

### R7.4 — Imports at the top of the file
No mid-function imports. A genuinely deferred import gets a named helper near the
top with a docstring explaining the deferral (typically optional backends).

---

## §8. Dispatch precedence

### R8.1 — KDA backend precedence
`kimi_k3/attention/kda_backends.py` dispatches in this order:
1. `fla` `chunk_kda` (when `use_k3_fla_kda=True` **and** the pinned signature
   check passed at import),
2. eager FP32 oracle (`kda_eager_fp32.py`) — always available, always correct,
   slow.
A `flydsl` slot is reserved and unimplemented in v1.

### R8.2 — Expert backend precedence
1. AITER a8w4 fused MoE (`use_k3_qat_a8w4=True`, gfx950 only),
2. BF16 fake-quant reference (QAT semantics, no serving parity),
3. plain BF16 grouped GEMM.
Falling back logs one warning line; the banned-warning ratchet (R9.2) decides
whether that line is allowed in a given smoke.

---

## §9. Measurement discipline

### R9.1 — Capacity claims are measured, never assumed
No configuration ships with a node count, an optimizer recipe, or a
bytes-per-parameter figure that is not backed by a row in
`develop/results/opt_mem.md` produced by `kimi_k3/tools/mem_budget.py`.
Analytic estimates are allowed in planning documents **only when labelled
"estimate — to be measured in <gate>"**.

### R9.2 — Banned-warning ratchet
Smoke runs must grep clean against the running banned-warning set. The current
set is pinned in `plan-<n>/05-test-strategy.md` under "Banned warnings". Each
plan may extend it; entries are never removed silently.

### R9.3 — Forensic attribution before a perf fix
Before designing a perf fix, attribute the bottleneck to a single source line
via the trace's correlation table. The forensic helper script is committed under
`progress/p<id>/_forensics*.py`; its output is not.

### R9.4 — 10 % de-scope rule
A ranked-bottleneck row worth < 10 % of steady iteration wall time de-scopes the
corresponding optimisation task. De-scoped tasks stay in tree as plan follow-ups.

---

## §10. Environment

### R10.1 — Hardware tiers
**[CPU-OK]** anywhere · **[GPU]** 1×MI355X (gfx950) · **[8-GPU]** one node ·
**[CLUSTER]** multi-node, human-executed (the agent prepares configs + analysis
only and never launches a cluster job).

### R10.2 — Pins
All five dependency SHAs (Megatron fork, TransformerEngine, AITER, fla, HF
Kimi-K3 revision) live in `kimi_k3/PINS.md` together with the `LICENSES` section.
A pin bump is its own commit and re-runs the P0 gates listed in R10.3.

### R10.3 — After any IFU on `rocm_dev`
Rebase `dev/wen/kimi-k3`, then re-run, in order: the guard, `test_k3_p1_pin_contracts.py`
(R4.5), G1–G3, and only then other work. An IFU is never bundled with feature work.

### R10.4 — Branch naming
Active development branch: **`dev/wen/kimi-k3`**, opened off `rocm_dev` at the
pinned SHA (first commit `2a66e4ce5`, docs only). PRs land into `rocm_dev` from
this branch. Any document that needs to name the branch cites this rule rather
than repeating the name, so a rename is a one-line change here.

---

## §11. House-keeping

### R11.1 — Adding a rule
Smallest matching `§N`; `R<n.m> — <one-line title>` heading; first sentence says
when the rule started and why; no cross-rule dependencies inside a rule body.

### R11.2 — Retiring a rule
Strike through heading and body, append
`**Retired YYYY-MM-DD: <one-line reason>**`. Do not delete.
