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
| [x] | T6.2 `k3_moe_layer.py` postprocess override | `06216dc08` | 2026-08-27 | **G24 GREEN.** The latent norm is demonstrably in the path — changing its gain changes the model output. Routed experts at 3584, shared at 7168 (finding B6), only layer 1 dense |
| [x] | T6.3 `k3_router.py` QB router + estimator | `06216dc08` | 2026-08-27 | **G25 GREEN.** Routing matches the transcription (bias steers selection, weights come from the **unbiased** scores). Balancing is **our formulation** — the report names QB but publishes no algorithm — gated on exact agreement with an independent reference, estimator error (2e-2 @ 64 bins, 2e-3 @ 1024) and behaviour (skewed load ratio 2.0 → <1.6). No test claims release parity |
| [-] | T6.4 EP path + `router_replay` determinism | TBD-g26 | 2026-08-28 | **G26 GREEN for the EP path**: 8 ranks, **112 local experts each** (896/8 exact), params per rank identical at **16,306,993,312**, **0 of 896 starved**, load max/mean 1.77. Determinism is **not** bitwise at bf16+EP and the cause is located — finding **A18**: the first MoE router is identical on every rank, so the source is inside the MoE accumulation, not our attention. `router_replay` still owed (`results/ep_smoke.md`) |
| [x] | T6.5 Grouped-GEMM smoke | `06216dc08` | 2026-08-27 | tiny preset runs the MoE stack end to end with gradients |

## P7 — Single-node trainer (G27–G29)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T7.1 `pretrain_kimi_k3.py` end to end | `110c84c69` | 2026-08-27 | **G27 GREEN at tiny.** 30 steps on a fixed batch drive the loss 8.36 → ~0.0 under **both** `dist_muon` and `adam`; on fresh random tokens it sits at `ln(4096) = 8.32` (chance), which catches both a broken model and a label leak |
| [x] | T7.2 `tools/tokenizer.py` | `110c84c69` | 2026-08-27 | tiktoken ranks + special ids asserted against the released `config.json` |
| [x] | T7.3 4 L official config on the measured recipe | `0cc8f376b` | 2026-08-27 | **It fits.** 8× MI355X, EP=8, seq 512, `dist_muon`, full recompute: **16.31 B params/rank** (the analytic model predicted 16.3 B), peak **188.7–202.2 GiB** of 288 GB, three steps with finite losses on all ranks. None of the documented fallbacks were needed |
| [x] | T7.4 Memory report | `0cc8f376b` | 2026-08-27 | **G28 GREEN.** `results/official_smoke.md`. 12.43–13.32 B/param at peak; 8.78–9.08 after optimizer construction — above G5's 7.87 because at EP=8 with world 8 the expert-DP group is size 1, so `dist_muon` shards only the non-expert 2 %. Seq 8 k and the EP ladder remain the nightly's |
| [x] | T7.5 Checkpoint save/load | `110c84c69` | 2026-08-27 | **G29 GREEN.** Save, rebuild from a *different* init, load, forward matches bitwise |

## P8 — Converter and anchored parity (G30–G33)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T8.1 `tools/mapping.py` + explicit invariants | `93b2c57ad` | 2026-08-27 | **G30 GREEN.** All **497,220** released tensors accounted for: 497,052 mapped, **168 skipped by name** (the vision tower and projector), **zero unmapped**, 247,296 MXFP4 pairs, 69 KDA / 24 MLA layers inferred from which tensors exist. A 5 KB pattern fixture stands in for the 60 MB index |
| [x] | T8.2 `tools/convert.py` both directions | `93b2c57ad` | 2026-08-27 | **G31 GREEN.** bf16 round-trips exactly; `A_log` trims only when the padding is really zero; MXFP4 experts dequantise on import; exporting experts **refuses** rather than writing bf16 under a `weight_packed` name |
| [x] | T8.3 Synthetic round-trip in CI | `93b2c57ad` | 2026-08-27 | 13 tests, no network, no weights |
| [-] | T8.4 Checkpoint-anchored parity | `93b2c57ad` | 2026-08-27 | **G32 GREEN for one KDA layer on real weights**: 847 MiB fetched by range request, our module and the **release's own** module both load it with 0 missing / 0 unexpected, forward agrees to **rel-L2 7.4e-3, cosine 0.999973** (inside the measured bf16 bound). The full four-layer truncated-model slice is still owed. Report: `results/anchored_parity.md` |
| [x] | T8.5 Tokenizer round-trip | `110c84c69` | 2026-08-27 | **G33 GREEN** (P7): special ids asserted against the released `config.json` |

## P9 — Training equivalence, per-head Muon, resume hardening (G34–G37)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T9.1 `tools/twin_run.py` + noise band | `76070d0b7` | 2026-08-27 | **G34 GREEN.** Band from 3 seeds at tiny/40 steps: max 0.2382, mean 0.0837, final-quarter 0.0488 (`results/twin_runs.md`) |
| [x] | T9.2 Twin axes | `76070d0b7` | 2026-08-27 | eager-vs-`fla` KDA **inside the band** on all three statistics (0.1371 / 0.0223 / 0.0184); recompute on-vs-off **bitwise 0.0**, with the checkpoint path proven to fire (0 calls off, 4 on) so the zero is not vacuous |
| [x] | T9.3 `optim/per_head_muon.py` + group policy | `23ade88df` | 2026-08-27 | **G35 GREEN.** One split step is **bitwise-equal** to N core-Muon steps on the head slices, both axes; head slices are TP-local so `partition_dim=None`. Policy asserted against a real model. **Cost measured 2026-08-30**: per-head is **0.57x** on KDA and **0.06x** on MLA `q_b` (launch-bound at 1536 wide) — `results/per_head_muon_cost.md`. Stays opt-in, now behind `--k3-per-head-muon` and **KDA only** (MLA excluded: 17x). Twin at tiny/40 steps is **inside the noise band on all three statistics** — no evidence for paying 1.75x (`results/per_head_muon_twin.md`) |
| [x] | T9.4 Wire `clip_qk` | `23ade88df` | 2026-08-27 | **G36 GREEN** at tiny: core's own `clip_qk(model)` drives our MLA unmodified; max logit recomputed per head and clipped to the threshold. The NoPE `k_rot` slice is shared across heads, so its whole correction lands on the query side |
| [x] | T9.5 Resume under PP=2 | `544733536` | 2026-08-27 | **G37 GREEN, bitwise** over 10 post-resume steps, with and without dropout. Both negative controls fail. Found finding **A15**: `dist_muon` could not be resumed at all as shipped (`results/pp_resume.md`) |

## P10 — MXFP4 / MXFP8 QAT (G38–G41)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T10.1 quantiser + oracle | `9b1d2e8c7` | 2026-08-27 | **G38 GREEN.** The scale rule was wrong and the released weights said so — 25.75 % of a real expert's groups are impossible under the OCP floor-to-6.0 formula. Corrected to the measured rule; now **byte-identical** to `aiter.per_1x32_f4_quant`, and MXFP8 bit-identical with `scale_type=fp8_e8m0`. Finding **A16** |
| [x] | T10.2 a8w4 forward + STE backward | `07d103d34` | 2026-08-27 | **G39 GREEN** at rel-L2 **1.66e-3** including K3's own 3584→3072 geometry, via `aiter.ops.triton.moe.moe_op_gemm_a8w4` — no checkout bump needed, contrary to P0's D1 deferral. **G40 GREEN, exact** (rtol 0, atol 0) |
| [x] | T10.3 Checkpointing of packed data + scales + masters | `de2d10b1e` | 2026-08-27 | landed in P0 with the expert prototype; packed caches are buffers and ride the checkpoint |
| [x] | T10.4 QAT-vs-BF16 twin, serve parity | TBD-g41 | 2026-08-28 | **G41 GREEN.** 4 L official, EP=8: both train, offset **−7.7e-04**, drift **−5.0e-04**, worst step **0.37 % of the loss**. Serving without the activation quantisation it was trained under moves **1 token in 10** (rel-L2 0.041, argmax 90.6 %) — `results/qat_twin.md` |

## P11 — Performance baseline and AttnRes optimisation (G42–G45)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T11.1 EP=8 proxy + trace | `ef9bae4c1` | 2026-08-27 | **G42 GREEN.** 4 L official, **EP=8**, seq 512: steady **2653.5 ms** (spread 2643–2659), peak **193.6 GiB**/rank, 321 k launches. The earlier 2-layer run is kept as a depth cross-check — the ranking is unchanged |
| [x] | T11.2 Baseline report + budgets | `72daec696` | 2026-08-27 | `profile/profile-baseline-2026-08-27.md`. **The prediction was wrong**: AttnRes is 0.1 % of device time, the Muon step is 21 % |
| [x] | T11.3 Fused (chunked) AttnRes mixer | TBD-g43 | 2026-08-28 | **G43 GREEN at both tiers, with the guarantee scoped to what was measured.** Forward **bit-identical** at any chunk size; per-token gradients bit-identical **at the default chunk** (below ~1024 rows at release width rocBLAS picks a different kernel and ~1 element in 1e4 moves by 4e-04 of scale); the shared `[H]` gain gradient grows **sub-linearly** in the chunk count — 64x the chunks for 4x the error, 7.7e-07 of magnitude at 256 chunks. Behind `--k3-attn-res-fused`, default off |
| [x] | T11.4 Whatever the ranking says next | `ef9bae4c1` | 2026-08-27 | **rescoped and settled.** Muon step **21.1 %**, AttnRes **0.09 %** — the ranking held across a 2x depth change, retiring the caveat that 21 % was overstated. Under R9.4 no AttnRes fix is authorised |
| [x] | T11.5 Post-phase trace | `ef9bae4c1` | 2026-08-27 | **G45 GREEN**, **G44 has nothing to meet.** Fused vs baseline at EP=8: **0.00 GiB** memory delta, −0.15 % time (noise). The temporary the chunking removes is **29 MB** at this geometry, not 109 GiB — that figure is production shape, which no single node can run |

## P12 — Scale-out preparation (G46–G47)

|     | Task | commit | date | note |
| --- | --- | --- | --- | --- |
| [x] | T12.1 93 L configs from the measured recipe | `a5c3a9f62` | 2026-08-27 | **G46 GREEN.** Floor is **28 nodes** (pp8 x ep28: 19.25 B params/GPU, 141 GiB state + 82 GiB measured headroom). `results/scaleout_93l.md` |
| [x] | T12.2 PP layouts + legality test | `a5c3a9f62` | 2026-08-27 | **PP cannot exceed 8** with aligned boundaries — 93 layers give seven whole AttnRes blocks, so eight cut points. Every `pp16` candidate is flagged, not silently emitted |
| [x] | T12.3 EP ladder | `a5c3a9f62` | 2026-08-27 | 8/16/28/32/56, each asserted to divide 896 |
| [x] | T12.4 Dispatcher A/B matrix | `1cdda9dd8` | 2026-08-27 | **G47 GREEN** as a plan (`plan-0/07-dispatcher-ab.md`). The premise is overtaken: core already ships DeepEP **and** MoRI as flex backends — nothing to port, only runtimes to install. Found **A17**: core's "cannot enable both" guard is unreachable, so asking for MoRI + DeepEP silently gives DeepEP |
| [ ] | T12.5 Continued-pretrain flatness scripts | — | — | owed |

