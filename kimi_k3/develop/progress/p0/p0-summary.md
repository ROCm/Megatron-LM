# P0 — Feasibility gates (COMPLETE)

> **All eight gates are green.** Nothing is porting-blocked. Format per rule R3.5.

## 1. Objective

Prove the four risky mechanics and pin the dependencies **before any porting
starts**: the config path, block injection, the AttnRes pipeline payload, and
the optimizer memory model — plus close the open ground-truth questions.

**Outcome:** all four are proven, three of them by measurement rather than
argument. Two of the plan's own claims turned out to be stale rather than wrong
(AITER), one of mine was wrong and is retracted (fla), and the one genuine
blocker (fla's KDA backward not compiling on gfx950) turned out to be a Triton
version, fixed by upgrading Triton alone — no torch change, which matters because
TE, apex and AITER are all built against this torch.

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
| `develop/notes/2026-08-27-fla-signature-check.md` | the fla retraction and what actually blocked G1 |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G1** — pins + licenses + fla | **GREEN** | All 5 pins + licenses. The released `chunk_kda` call works as written (D5 retracted); the KDA backward was blocked by **Triton 3.6.0**, fixed by upgrading Triton alone to 3.7.1 (D6 resolved) — no torch change |
| **G2** — release artefact audit | **GREEN** | 4/4 open items closed with evidence; fixture committed |
| **G3** — config on the real inheritance path | **GREEN** | 9 tests |
| **G4** — model construction, no transient core block | **GREEN** | 4 tests; constructor spy sees 0 core blocks; meta and cuda both build |
| **G5** — optimizer memory | **GREEN** | 16 runs / 40 rank-rows; `dist_muon` 7.87 B/param at DP=8 vs `muon` 15.17 flat; analytic `6 + 8/DP + c` fits with c ≈ 0.9–1.2 |
| **G6** — AttnRes payload sizing | **GREEN** | payload peaks at 2.8 GB in flight mid-pipeline; eager mixer 109.6 GiB read and ≈635 ms forward per microbatch; P11 budget set at ≤64 ms |
| **G7** — PP=2 packed payload + gradient assertion | **GREEN** | bitwise match against PP=1 (loss 0.0, grads 0.0 / 131 params); negative control gives identical loss with gradients wrong by 8.1e-4 |
| **G8** — external assets + one-expert QAT | **GREEN** | AITER assets present on `main` (D1 resolved); TE MXFP4 is a stub (D2 sharpened); STE grads exactly match the fake-quant reference; cache refresh and checkpoint round-trip exact |

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

### AttnRes cost (G6)

| metric | measured |
|---|---:|
| payload per boundary | 224 MB → 896 MB |
| worst in-flight payload | **2.8 GB** (stage 3 of 8; ×2 with saved input+output) |
| one mix at K+1=9 | 7.1 GB fwd / 12.2 GB fwd+bwd; 5.15 / 15.65 ms |
| whole model, per microbatch | 186 mixes, mean K+1 5.39, **109.6 GiB read**, ≈635 ms eager forward |
| fp32 tax vs a bf16 mix | 2.3× memory, ≈1.5× time |

For scale, all of the routed-expert GEMMs are ≈800 ms per microbatch at 40 % MFU:
the eager AttnRes forward alone is the same order. Recompute is mandatory
(saving the fp32 stacks would cost ≈236 GB per microbatch), and P11's fused mixer
now has a budget: ≤10 % of the eager forward and no fp32 stack in HBM.

### Pipeline transport (G7)

PP=2 with the packed payload is **bitwise identical** to the single-stage run —
loss and all 131 parameter gradients. The negative control (block-residual slots
detached at creation, emulating a two-tensor payload) returns an **identical
loss** with gradients wrong by 8.1e-4: a loss-only check would have passed it.
That is finding A1 reproduced on demand, and the reason G20 asserts gradient flow.

Core's own `adjust_tensor_shapes_fn` supplied the per-stage shapes (recv 1→2,
send 2→3); no schedule was re-implemented, and the hook is bound only when PP > 1.

### QAT prototype (G8)

| check | result |
|---|---|
| AITER K3 a8w4 assets at `main` `e9e1278b1` | **present** — tuned CSV (gfx950 / 896 experts / top-16 / 3584 / `Situv2`), both opus stages, the enum entry |
| TE `MXFP4Quantizer` at 2.12 | **stub** — `quantize_impl` and dequantise-from-packed both raise, so no cross-check exists |
| STE grads vs the fake-quant reference | **exactly equal** on all three weights |
| packed cache after an optimizer step | stale until refreshed, **exact** afterwards |
| checkpoint round-trip (masters + packed + scales) | bit-exact, outputs identical |

The prototype's first run **failed**, and usefully: activation quantisation had
no STE, so the cast into `float8_e4m3fn` killed the gradient to `w1`/`w3` — the
whole gate/up half of the expert silently received zeros while `w2` looked fine.
Both the module and its reference now use an activation STE.

## 5. Findings added to the register

| id | severity | what |
|---|---|---|
| **A13** | HIGH | The local (non-TE) MLA path cannot be constructed at the pin: `MLASelfAttention` passes `k_channels`/`v_channels` to a `DotProductAttention` that accepts neither. Construction gates therefore run on 1 GPU with the TE spec. |
| ~~**D5**~~ | retracted | The "silently ignored kwargs" claim was wrong — drawn from the signature without running the call. fla handles all three deliberately. |
| **D6** | HIGH | fla's KDA **backward** fails to compile on gfx950 (`chunk_intra.py:395`, TritonAMDGPUPipeline, triton 3.6.0). Forward works. |
| **D7** | MED | The published fla wheel ships no `fla/ops`; pin git, not PyPI. |
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
- Determinism traps for any future parity harness: `hidden_dropout` /
  `attention_dropout` default to 0.1, and the router's expert-bias accumulator
  mutates during forward. Both make two "identical" runs differ by ~1e-4, which
  would inflate a measured floor and hide a real defect.
- The P0 block is transport-faithful but layer-approximate (it wraps a stock
  `TransformerLayer` rather than splitting the mixes around attention and MLP).
  P5 makes it faithful; no parity gate may cite it until then.

## 9. Commit chain

| commit | scope |
| --- | --- |
| **94a40436c** | P0 T0.2–T0.4: config path, block injection, parameter oracle, 22 tests, release audit; G2/G3/G4 green, G1 red and blocking |

| **73114c73a** | P0 T0.5: optimizer-memory measurement (G5 green); capacity section regenerated from data |

| **187ac2c01** | P0 T0.6: AttnRes mixer oracle, payload protocol, sizing (G6 green); P11 budget set |

| **d6c51b7f1** | P0 T0.7: PP=2 packed-payload transport, bitwise exact (G7 green) |

| **c07ef0507** | P0 T0.8: MXFP4 quantiser, SiTU, one-expert QAT prototype, AITER/TE asset verification (G8 green) |

| **396ba6423** | P0 T0.1: triton 3.7.1 unblocks the KDA backward (G1 green) |

**P0 closes here with all eight gates green.** Next: P1 (scaffold, guard, CI
wiring), then P2.
