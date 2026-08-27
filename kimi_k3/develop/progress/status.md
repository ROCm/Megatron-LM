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
| [ ] | T0.1 `PINS.md` (5 SHAs + LICENSES) + `check_fla_signature.py` | | | G1; `fla` is **not installed** in the current container — this is the first blocking task |
| [ ] | T0.2 Release artefact audit (`A_log` shape, `KimiRMSNorm` eps, key prefixes + vision count, tokenizer ids) | | | G2; CPU-only, needs `config.json` + index + one shard header |
| [ ] | T0.3 Config on the real inheritance path | | | G3; core substitutes `MLATransformerConfig` at `arguments.py:1230-1232` |
| [ ] | T0.4 Meta-device model construction + analytic param count | | | G4; constructor spy must see zero core `TransformerBlock` |
| [ ] | T0.5 Optimizer memory measurement incl. `dist_muon` | | | G5; per parameter group, as a function of DP |
| [ ] | T0.6 AttnRes payload + mixer-temporary sizing | | | G6; feeds `06-capacity-and-parallelism.md` §3 |
| [ ] | T0.7 PP=2 packed-payload prototype + payload-gradient assertion | | | G7; the finding-A1 guard |
| [ ] | T0.8 External asset verification + one-expert QAT prototype | | | G8; AITER K3 assets absent from the 2026-04-03 checkout, TE path differs from the plan |

## P1 — Scaffold and environment (G9–G10)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T1.1 Branch + skeleton | | | |
| [ ] | T1.2 `no_core_diff_guard.sh` + allowlist, proven against a deliberate core edit | | | G9 |
| [ ] | T1.3 `test_k3_p1_pin_contracts.py` (IFU tripwire) | | | G10; nine core mechanisms asserted |
| [ ] | T1.4 `test_env.py` + version capture into `PINS.md` | | | G10 |
| [ ] | T1.5 Pre-commit hook script + CI workflow file | | | human registers the hook |

## P2 — Config, presets, model construction (G11–G13)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T2.1 Final config + builder + args, full release→config mapping | | | |
| [ ] | T2.2 Presets tiny / 4L / 8L / 93L | | | officials are meta-device only |
| [ ] | T2.3 Layer specs (KDA vs MLA, dense vs MoE) | | | `+1` indexing from the literal list |
| [ ] | T2.4 `core_patch.py` + `k3_gpt_model.py` | | | scoped rebinding, not a full `__init__` override |
| [ ] | T2.5 `tools/mem_budget.py` analytic table | | | oracle for G13 |

## P3 — KDA (G14–G16)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T3.1 `kda_eager_fp32.py` FP32 oracle | | | must place the q/k L2 norm where the kernel does |
| [ ] | T3.2 `kda.py` + `kda_backends.py` (two gates, released kwargs) | | | eager default until G15 green at production shapes |
| [ ] | T3.3 Tolerance floors + bounds recorded | | | seq {1k, 8k, 64k} |
| [ ] | T3.4 `sharded_state_dict` + state layout round-trip | | | G16 |
| [ ] | T3.5 Behavioural tests (causality, lower-bound clamp, safe_gate) | | | |

## P4 — Gated MLA with NoPE (G17–G18)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T4.1 `K3GatedMLA`: rotary bypass + `192**-0.5` scale | | | core MLA has no NoPE mode |
| [ ] | T4.2 Full-rank sigmoid output gate + fp32 attention output | | | gate applies before `o_proj` |
| [ ] | T4.3 Expose `clip_qk` for core's walker | | | block must keep `.layers` |
| [ ] | T4.4 Fused-attention bring-up at 192/128 | | | record the fallback if rejected |
| [ ] | T4.5 LoRA-norm epsilon per the G2 answer | | | |

## P5 — AttnRes block and pipeline transport (G19–G22)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T5.1 `attn_res.py` fp32 mixer oracle | | | exact release math |
| [ ] | T5.2 `attn_res_pp.py` pack/unpack/slots_before + protocol docstring | | | single packed tensor |
| [ ] | T5.3 `k3_transformer_layer.py` (two mixes, slot append, prefix reset) | | | |
| [ ] | T5.4 `k3_transformer_block.py` (state ownership, output mix) | | | |
| [ ] | T5.5 `pipeline/k3_schedule.py` (bind only when PP>1) | | | G21 |
| [ ] | T5.6 Behavioural tests | | | G19, G22 |

## P6 — LatentMoE, QB router, SiTU (G23–G26)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [ ] | T6.1 `situ.py` (fp32, β=4, β₂=25) | | | G23 |
| [ ] | T6.2 `k3_moe_layer.py` postprocess override (`routed_expert_norm`) | | | G24; shared experts on hidden 7168 |
| [ ] | T6.3 `k3_router.py` QuantileBalancingRouter + estimator | | | G25 exact-update parity |
| [ ] | T6.4 EP path + `router_replay` determinism | | | G26; load ratio reported, not gated |
| [ ] | T6.5 Grouped-GEMM smoke at tiny then at an 896 divisor | | | |

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
