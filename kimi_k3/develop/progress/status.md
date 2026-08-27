# Kimi K3 Integration Progress Tracker

> Tick `[x]` and fill in the commit SHA + date when a task is done.
> Task granularity matches [`../plan-0/04-phase-details.md`](../plan-0/04-phase-details.md).
> Row format (R2.2 of the DS-V4 lineage, R1.3 here): `[ ] | task | commit | date | note`.
> `[-] ~~task~~` marks a de-scoped task — keep the original text struck through.
> `TBD-p<id>` is the placeholder before the pin commit lands.
>
> Project-wide rules live in [`../rules/rule.md`](../rules/rule.md).

## P0 — Feasibility gates (G1–G8)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T0.1 `PINS.md` (5 SHAs + LICENSES) + fla contract check | `396ba6423` | 2026-08-27 | **G1 GREEN.** All 5 pins + licenses recorded. The released `chunk_kda` call is accepted as written (the "silently ignored kwargs" reading was wrong — retracted, D5). The real blocker was **Triton, not torch**: upgrading triton alone 3.6.0 → **3.7.1** makes the KDA backward compile on gfx950 (D6 resolved) — no torch change, which matters because TE/apex/AITER are built against it. Check is functional, not signature-based: `tests/test_k3_p0_fla_contract.py` |
| [x] | T0.2 Release artefact audit (`A_log` shape, `KimiRMSNorm` eps, key prefixes + vision count, tokenizer ids) | `94a40436c` | 2026-08-27 | **G2 GREEN.** `A_log` F32 `[128]`, `[96:]` exactly 0; `KimiRMSNorm` default eps **1e-6** (LoRA norms confirmed); 497,220 tensors / 96 shards, 168 vision+projector keys to skip; KDA and MLA both under `self_attn`, MoE under `block_sparse_moe`. Fixture: `tests/fixtures/release_shapes.json` |
| [x] | T0.3 Config on the real inheritance path | `94a40436c` | 2026-08-27 | **G3 GREEN**, 9 tests. Type survives, MLA fields populated, core builder sentinel never fires, layer pattern + slot schedule pinned |
| [x] | T0.4 Meta-device model construction + analytic param count | `94a40436c` | 2026-08-27 | **G4 GREEN**, 4 tests. Constructor spy sees 0 core blocks; rebinding does not leak; 93L analytic total 2.779 T, every per-component formula matches the released headers. Needs 1 GPU (TE spec) because of finding A13 |
| [x] | T0.5 Optimizer memory measurement incl. `dist_muon` | `73114c73a` | 2026-08-27 | **G5 GREEN.** 16 runs / 40 rank-rows. `dist_muon` 15.17 / 11.00 / 8.91 / **7.87** B/param at DP 1/2/4/8 vs plain `muon` flat at 15.17 → the plan's 14 B/param holds only at DP=1. `adam` 18.02 flat; +dist-opt 7.52 at DP=8; +precision-aware 7.27. Per-rank spread ±4 % at DP=8 (whole-tensor sharding). Report: `results/opt_mem.md`; method + the DDP-config trap: `notes/2026-08-27-optimizer-memory-method.md` |
| [x] | T0.6 AttnRes payload + mixer-temporary sizing | `187ac2c01` | 2026-08-27 | **G6 GREEN.** Payload peaks mid-pipeline at **2.8 GB** in flight (stage 3 of 8), 224→896 MB per boundary. One mix at K+1=9: 7.1 GB fwd / 12.2 GB fwd+bwd, 5.15/15.65 ms. Whole model 186 mixes, mean K+1 5.39, **109.6 GiB read per forward**, eager mixer fwd **≈635 ms/microbatch** vs ≈800 ms for all routed-expert GEMMs → recompute mandatory, P11 budget set at ≤64 ms. fp32 tax measured at 2.3× memory / 1.5× time. Report: `results/attn_res.md` |
| [x] | T0.7 PP=2 packed-payload prototype + payload-gradient assertion | `d6c51b7f1` | 2026-08-27 | **G7 GREEN.** PP=2 is **bitwise identical** to PP=1: loss diff 0.0, worst gradient diff 0.0 across all 131 params. Negative control (slots detached) gives **identical loss** but gradients wrong by 8.1e-4 — finding A1 reproduced on demand. Shapes supplied through core's own `adjust_tensor_shapes_fn`; no schedule re-implemented. Report: `results/pp_payload.md`; rerun: `progress/p0/run_g7.sh` |
| [x] | T0.8 External asset verification + one-expert QAT prototype | `c07ef0507` | 2026-08-27 | **G8 GREEN.** AITER K3 assets **confirmed present** on `main` `e9e1278b1` (D1 resolved — our checkout was 5 months stale); TE's MXFP4 quantiser is a **stub** at 2.12 (`quantize_impl` and dequantise-from-packed both raise), so `moe/k3_qat.py` is the reference outright (D2 sharpened). One-expert prototype: STE grads **exactly** match the fake-quant reference, packed cache refresh is exact after a step, checkpoint round-trips. 15 tests. Kernels not yet exercised — needs an aiter rebuild, P10 |

## P1 — Scaffold and environment (G9–G10)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T1.1 Branch + skeleton | `c96fad823` | 2026-08-27 | branch `dev/wen/kimi-k3`; package skeleton + `kimi_k3/README.md` |
| [x] | T1.2 `no_core_diff_guard.sh` + allowlist | `c96fad823` | 2026-08-27 | **G9 GREEN.** Passes on the branch (59 files, all in `kimi_k3/`), rejects a staged core edit with exit 1, honours exactly the two allowlisted root files, rejects any other root file, and compares against the **merge-base** so an IFU rebase does not flag upstream's own core commits. 6 hermetic tests in a throwaway repo |
| [x] | T1.3 `test_k3_p1_pin_contracts.py` (IFU tripwire) | `c96fad823` | 2026-08-27 | **8 contracts**, each its own test with a message saying what to re-check: block injection, `adjust_tensor_shapes_fn` (accepted by 1F1B, rejected by VPP/no-PP), single-tensor `backward_step`, router injection + `postprocess`, `clip_qk`, `dist_muon` sharding + dist-opt rejection, config substitution, MLA `k_channels` (A13) |
| [x] | T1.4 `test_env.py` + `tools/capture_env.py` | `c96fad823` | 2026-08-27 | **G10 GREEN.** torch 2.10 / triton **3.7.1** (asserted >=3.7 with the reason) / TE 2.12 / fla 0.6.0 / 8x MI355X. AITER K3 kernels skip with a pointer until the checkout is bumped (P10) |
| [x] | T1.5 Pre-commit hook + CI workflow | `c96fad823` | 2026-08-27 | `.pre-commit-config.yaml` gains a local `kimi-k3-no-core-diff` hook; `.github/workflows/kimi_k3.yml` wires CI stages 0 and 1. Both YAML files validated. **Human action needed:** `pre-commit install`, and set the runner labels + container image (marked `TODO(maintainer)`) |

## P2 — Config, presets, model construction (G11–G13)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T2.1 Final config + builder + args | `c1d4db3d4` | 2026-08-27 | `config/arguments.py` with `add_kimi_k3_args`; every `k3_*` config field settable from the CLI (asserted); a preset **fills gaps without overruling explicit flags** (two-pass parse). Defaults are the released values, asserted field by field |
| [x] | T2.2 Presets tiny / 4L / 8L / 93L | `c1d4db3d4` | 2026-08-27 | officials are **analytic only** — meta device does not prevent allocation (finding A14), so `build_k3_model` refuses them without `allow_official=True` |
| [x] | T2.3 Layer specs (KDA vs MLA, dense vs MoE) | `c1d4db3d4` | 2026-08-27 | **G11 GREEN.** `specs/layer_specs.py` splits the *plan* (pure config, CPU-testable) from the spec; core takes heterogeneous `layer_specs` natively. 93L: 69 KDA / 24 MLA matching the literal list, layer 0 dense at 33792, tail `M M`, slots at 0/12/…/84. Local plan slices by PP stage |
| [x] | T2.4 `core_patch.py` + `k3_gpt_model.py` + `model/build.py` | `c1d4db3d4` | 2026-08-27 | **G12 GREEN.** `build_k3_model(preset)` is the single assembly path; layers are heterogeneous as the plan says; config type survives; AttnRes mixers present per layer + at the output |
| [x] | T2.5 `tools/mem_budget.py` analytic table | `c1d4db3d4` | 2026-08-27 | **G13 GREEN.** 4L 94.0 B / 8L 214.7 B / 93L 2.779 T, each within 2 % of target, and the per-component subtotals cross-checked against the layer plan |

## P3 — KDA (G14–G16)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T3.1 `kda_eager_fp32.py` FP32 oracle | `157d86a18` | 2026-08-27 | **G14 GREEN.** **Bit-identical** to fla's own `naive_recurrent_kda`; 7e-7 vs `chunk_kda` and `fused_recurrent_kda`; fp64 gradcheck passes. Gate and recurrence both read from fla source, not from the paper |
| [x] | T3.2 `kda.py` + `kda_backends.py` (two gates, released kwargs) | `157d86a18` | 2026-08-27 | module parameter count equals `mem_budget.kda_layer_params` exactly, which was itself checked against the released checkpoint. `K3KDASelfAttention` adapts to Megatron's sequence-first interface; KDA layers in a real model now use real KDA |
| [x] | T3.3 Tolerance floors + bounds recorded | `157d86a18` | 2026-08-27 | **G15 GREEN** at tiny/mid. fp32 7.0e-7 · bf16 4.3e-3 against a **3.3e-3 dtype floor** · bwd 2.9e-5 fp32 / 6.1e-3 bf16. Error does **not** grow with sequence length — the `-5` gate bounds accumulation. Report: `results/kda_parity.md`. Production geometry owed to the nightly |
| [x] | T3.4 `sharded_state_dict` + state layout round-trip | `157d86a18` | 2026-08-27 | **G16 GREEN.** State carries across a split sequence; `transpose_state_layout` inverts; state dict round-trips |
| [x] | T3.5 Behavioural tests | `157d86a18` | 2026-08-27 | causality, gate bounds incl. saturation at both ends, softplus branch, the epsilon-inside-the-sqrt L2 norm, batch independence under the seq-first transpose |

## P4 — Gated MLA with NoPE (G17–G18)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T4.1 `K3GatedMLA`: rotary bypassed, `192**-0.5` scale | `231544065` | 2026-08-27 | **G17 GREEN.** Own module, not a subclass: once rotary is gone, the scale changes and a gate is inserted, nothing of core's MLA forward survives (and core has no NoPE mode — A9). Shape contract preserved so the converter stays a rename |
| [x] | T4.2 Full-rank sigmoid output gate + fp32 attention output | `231544065` | 2026-08-27 | gate placement pinned by construction: zeroing `g_proj` makes sigmoid exactly 0.5, so a gate *before* `o_proj` must scale its output by exactly 0.5 |
| [x] | T4.3 `clip_qk` hook exposed | `231544065` | 2026-08-27 | present on both the module and the Megatron adapter; core's walker skips layers without it |
| [x] | T4.4 Fused backend at 192/128 | `231544065` | 2026-08-27 | **G18 GREEN.** SDPA fwd+bwd in bf16 at the real head-dim asymmetry, V padded to `q_head_dim` and sliced back exactly as the release does for FlashAttention |
| [x] | T4.5 LoRA-norm epsilon | `231544065` | 2026-08-27 | 1e-6, asserted, and shown to matter: changing it changes the output |

## P5 — AttnRes block and pipeline transport (G19–G22)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T5.1 `attn_res.py` fp32 mixer oracle | `187ac2c01` | 2026-08-27 | **G19 GREEN** (P0): bit-for-bit against the released `_apply_attn_res`; fp64 gradcheck |
| [x] | T5.2 `attn_res_pp.py` protocol | `187ac2c01` | 2026-08-27 | single packed tensor; module docstring is the spec |
| [x] | T5.3 `k3_transformer_layer.py` (two mixes, slot append, prefix reset) | `f405615d2` | 2026-08-27 | **G22 GREEN.** Every layer matches a **verbatim transcription** of `_forward_attn_residual`, driven with the same weights, after perturbing the mixer projections off their zero init so a uniform mix cannot hide a misplacement |
| [x] | T5.4 `k3_transformer_block.py` state + recompute | `f405615d2` | 2026-08-27 | recompute written, not inherited: core's `_checkpointed_forward` assumes one hidden tensor and would drop the block residual. Output within 1e-6 and every gradient within 1e-4 of the non-recomputed run |
| [x] | T5.5 `pipeline/k3_schedule.py` | `d6c51b7f1` | 2026-08-27 | **G20/G21 GREEN** (G7): bitwise PP=2 parity; the negative control gives an identical loss with gradients wrong by 8.1e-4 |
| [x] | T5.6 Behavioural tests | `f405615d2` | 2026-08-27 | slot appended only at boundaries; prefix restarts there; the slot is the prefix *before* attention; determinism (tiny preset now sets dropout to 0 — the trap from the G7 report came back and cost a test) |

## P6 — LatentMoE, QB router, SiTU (G23–G26)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T6.1 `situ.py` (fp32, β=4, β₂=25) | `c07ef0507` | 2026-08-27 | **G23 GREEN** (P0/G8): matches the release formula; both branches tanh-capped |
| [x] | T6.2 `k3_moe_layer.py` postprocess override | TBD-p6 | 2026-08-27 | **G24 GREEN.** The latent norm is demonstrably in the path — changing its gain changes the model output. Routed experts at 3584, shared at 7168 (finding B6), only layer 1 dense |
| [x] | T6.3 `k3_router.py` QB router + estimator | TBD-p6 | 2026-08-27 | **G25 GREEN.** Routing matches the transcription (bias steers selection, weights come from the **unbiased** scores). Balancing is **our formulation** — the report names QB but publishes no algorithm — gated on exact agreement with an independent reference, estimator error (2e-2 @ 64 bins, 2e-3 @ 1024) and behaviour (skewed load ratio 2.0 → <1.6). No test claims release parity |
| [-] | T6.4 EP path + `router_replay` determinism | TBD-p6 | 2026-08-27 | **G26 partial**: single-rank paths green; the 8-rank exercise belongs to the nightly job |
| [x] | T6.5 Grouped-GEMM smoke | TBD-p6 | 2026-08-27 | tiny preset runs the MoE stack end to end with gradients |

## P7 — Single-node trainer (G27–G29)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T7.1 `pretrain_kimi_k3.py` end to end | | | |
| [ ] | T7.2 `tools/tokenizer.py` | | | |
| [ ] | T7.3 4 L official on the measured recipe | | | documented fallbacks if it does not fit |
| [ ] | T7.4 Memory report | | | G28; analytic model within 10 % |
| [ ] | T7.5 Dist-checkpoint save/load | | | G29 |

## P8 — Converter and anchored parity (G30–G33)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T8.1 `tools/mapping.py` + explicit invariants | | | G30; vision keys skipped **and counted** |
| [ ] | T8.2 `tools/convert.py` both directions | | | `dequantize_on_import` |
| [ ] | T8.3 Synthetic + named-real-shard round-trip | | | ask before downloading |
| [ ] | T8.4 Truncated-HF parity on real layers 1–4 | | | G32 |

## P9 — Training equivalence, per-head Muon (G34–G37)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T9.1 `tools/twin_run.py` + measured noise band | | | G34 |
| [ ] | T9.2 Twin axes (eager vs fla, recompute) | | | |
| [ ] | T9.3 `optim/per_head_muon.py` + group policy | | | G35; core's `is_qkv` does not cover MLA |
| [ ] | T9.4 Core `clip_qk` wiring | | | G36 |
| [ ] | T9.5 Save→resume stability under PP=2 | | | G37 |

## P10 — MXFP4 / MXFP8 QAT (G38–G41)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T10.1 `k3_qat.py` quantiser + our OCP-MX reference | | | G38; TE ships no NumPy reference at the pin |
| [ ] | T10.2 `k3_qat_experts.py` a8w4 fwd + STE bwd | | | G39, G40 |
| [ ] | T10.3 Checkpointing of packed data + scales + masters | | | |
| [ ] | T10.4 QAT-vs-BF16 twin + serve parity | | | G41 |

## P11 — Perf baseline and fused AttnRes (G42–G45)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T11.1 EP=8 proxy + trace | | | G42 |
| [ ] | T11.2 Baseline report + per-task budgets | | | G42; forensic attribution before any fix |
| [ ] | T11.3 `attn_res_fused.py` | | | G43, G44 |
| [ ] | T11.4 Next-ranked target (10 % rule) | | | |
| [ ] | T11.5 Post-phase trace + perf tables | | | G45 |

## P12 — Scale-out preparation (G46–G47) **[CLUSTER]**

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T12.1 93 L configs on measured recipes | | | G46 |
| [ ] | T12.2 PP layouts aligned to AttnRes blocks | | | boundaries at layer ≡ 11 (mod 12) |
| [ ] | T12.3 EP ladder configs + QB load template | | | G47 |
| [ ] | T12.4 Dispatcher A/B matrix | | | plan only |
| [ ] | T12.5 Continued-pretrain flatness scripts | | | |
