# P0 — Feasibility gates (IN PROGRESS)

> Written at the first checkpoint of P0, not at its close. Four of eight tasks
> are done, one is blocked on an external dependency, three are not started.
> Format per rule R3.5.

## 1. Objective

Prove the four risky mechanics and pin the dependencies **before any porting
starts**: the config path, block injection, the AttnRes pipeline payload, and
the optimizer memory model — plus close the open ground-truth questions.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/config/k3_transformer_config.py` | `KimiK3TransformerConfig(MLATransformerConfig)` with the K3 fields, layer-kind helpers (`is_kda_layer`, `appends_attn_res_slot`, `attn_res_slots_before`) and the 1e-6 LoRA-norm epsilon |
| `kimi_k3/config/k3_config_builder.py` | `k3_config_from_args` / `config_from_preset`; core's builder is never called |
| `kimi_k3/config/presets.py` | tiny / 4L / 8L / 93L; official layer list verbatim from `config.json` |
| `kimi_k3/model/core_patch.py` | scoped `k3_block_class()` rebinding + `assert_pin_contracts()` (7 core mechanisms) |
| `kimi_k3/model/k3_gpt_model.py` | `K3GPTModel`; `set_input_tensor` enforces the single packed payload |
| `kimi_k3/block/k3_transformer_block.py` | `K3TransformerBlock` skeleton (P5 fills in the state handling) |
| `kimi_k3/tools/mem_budget.py` | analytic parameter + optimizer-memory model; the oracle for G13 |
| `kimi_k3/tests/` | 22 tests across config / construction / parameter count, plus `conftest.py` |
| `kimi_k3/tests/fixtures/release_shapes.json` | 50 real tensor shapes + dtypes from the released headers (6 KB, no weight data) |
| `kimi_k3/PINS.md` | four of five pins resolved; licenses recorded |
| `develop/notes/2026-08-27-release-audit.md` | G2 evidence |
| `develop/notes/2026-08-27-fla-signature-check.md` | why G1 is red |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G1** — pins + licenses + fla signature | **RED (blocked)** | 4/5 pins resolved; licenses recorded for all five. `fla` unresolved: `chunk_kda` on `main` takes neither `A_log` nor `dt_bias`, spells the layout flag `state_v_first`, and ends in `**kwargs` so the released call is silently accepted and wrong |
| **G2** — release artefact audit | **GREEN** | 4/4 open items closed with evidence; fixture committed |
| **G3** — config on the real inheritance path | **GREEN** | 9 tests |
| **G4** — model construction, no transient core block | **GREEN** | 4 tests; constructor spy sees 0 core blocks; meta and cuda both build |
| **G5** — optimizer memory | **GREEN** | 16 runs / 40 rank-rows; `dist_muon` 7.87 B/param at DP=8 vs `muon` 15.17 flat; analytic `6 + 8/DP + c` fits with c ≈ 0.9–1.2 |
| **G6** — AttnRes payload sizing | not started | needs 1 GPU |
| **G7** — PP=2 packed payload + gradient assertion | not started | needs 2 GPUs |
| **G8** — external assets + one-expert QAT | not started | AITER pin known stale (finding D1) |

`pytest kimi_k3/tests/ -q` → **22 passed**.

## 4. Measurements

Analytic parameter model, checked tensor-by-tensor against the released headers:

| preset | total | active/token |
|---|---:|---:|
| tiny | 10.73 M | 6.86 M |
| 4L official | 93.99 B | 5.62 B |
| 8L official | 214.70 B | 10.06 B |
| **93L official** | **2.779 T** | **104.19 B** |

Every per-component formula matches the checkpoint exactly, with one documented
delta: the checkpoint's `A_log` is `[128]` and ours is `[96]` (+32 per KDA layer).
Routed experts are **98.0 %** of all parameters.

### Optimizer memory (G5)

| recipe | DP=1 | DP=2 | DP=4 | DP=8 |
|---|---:|---:|---:|---:|
| `adam` | 18.02 | 18.02 | 18.02 | 18.02 |
| `adam` + dist-opt | 18.02 | 12.02 | 9.03 | 7.52 |
| the same + precision-aware | — | 11.02 | — | 7.27 |
| `muon` | 15.17 | — | — | 15.17 |
| `dist_muon` | 15.17 | 11.00 | 8.91 | **7.87** |

`dist_muon` at DP = 8 is **1.93×** cheaper than plain `muon`, and identical to it
at DP = 1. The capacity tables in `plan-0/06-capacity-and-parallelism.md` §2 are
now generated from these rows rather than from an assumed 14 B/param.

## 5. Findings added to the register

| id | severity | what |
|---|---|---|
| **A13** | HIGH | The local (non-TE) MLA path cannot be constructed at the pin: `MLASelfAttention` passes `k_channels`/`v_channels` to a `DotProductAttention` that accepts neither. Construction gates therefore run on 1 GPU with the TE spec. |
| **D5** | CRIT | `fla`'s `chunk_kda` silently ignores `A_log`, `dt_bias` and `transpose_state_layout`. The signature check must compare parameter names, not "did it raise". |
| **B5** | resolved | `A_log` is `[128]` with the last 32 exactly zero — measured, no longer assumed. |
| **B7** | confirmed | The MLA LoRA norms really do use 1e-6. |

## 6. Hand-off

- **P0 continues** with G5–G8 (all GPU work) and the `fla` bisection for G1.
- **P2 is unblocked**: the config path, presets, layer pattern and construction
  mechanics are proven, and `mem_budget.py` is the analytic oracle it needs.
- **P3 is unblocked despite G1 being red**: the FP32 oracle is defined by the
  released call's *semantics*, not by fla's current signature. The backend stays
  `eager` (rule R5.3) until the pin resolves and G15 runs.
- **P8 gets a corrected mapping table** (`plan-0/02-target-architecture.md` §9)
  built from real key names rather than guesses.

## 7. Artefacts

`kimi_k3/tests/fixtures/release_shapes.json` · `kimi_k3/PINS.md` ·
`develop/notes/2026-08-27-{release-audit,fla-signature-check}.md` · this file.
Reference downloads (config, modeling sources, 60 MB index) stay outside the tree
(rule R6.2).

## 8. Known follow-ups

- `fla` revision bisection (blocks G1, then G15).
- AITER pin bump or a written descope of the a8w4 fast path (blocks G8, then P10).
- Construction gates need 1 GPU because of A13 — the CI ladder in
  `plan-0/05-test-strategy.md` moves G4/G12 from stage 0 to stage 1. (done)
- `DistributedDataParallelConfig` must carry `use_distributed_optimizer` too, or
  the first optimizer step dies in `_copy_main_params_to_model_params`. Nothing
  validates the pair at construction time — remember this at P7 bring-up.
- The AttnRes projections are `[1, hidden]`, i.e. 2-D, so core's Muon path sends
  them through Newton–Schulz as rank-1 matrices. Decide in P9 (T9.3) whether to
  exclude them.
- `dist_muon` shard balance is ±4 % at DP=8 on the probe's shape mix; re-measure
  at 4 L official shapes with grouped-GEMM experts (G28).

## 9. Commit chain

| commit | scope |
| --- | --- |
| **94a40436c** | P0 T0.2–T0.4: config path, block injection, parameter oracle, 22 tests, release audit; G2/G3/G4 green, G1 red and blocking |

T0.5–T0.8 (all GPU) continue in this phase and will extend this summary.
