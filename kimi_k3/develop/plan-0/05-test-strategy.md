# 05 — Test Strategy

> How each phase proves it is correct. Gates are `G<n>`, monotonic and never
> reused (R3.3). Every phase keeps all earlier gates green (R4.6).

## Test pyramid

```
                  ▲ convergence / twin runs        (P9, P10 — scheduled)
                ▲ end-to-end distributed smoke     (P7, P11 — nightly 8 GPU)
              ▲ checkpoint-anchored parity         (P8 — nightly)
            ▲ module parity vs FP32 oracle         (P3–P6 — tiny in CI, prod nightly)
          ▲ behavioural + gradcheck + protocol     (P3–P6 — CI)
        ▲ guard, pin contracts, config, meta-device (P1, P2 — CPU, per commit)
```

## Tiers

| Tier | Where it runs | Shapes |
|---|---|---|
| **fast** (default) | per-commit CI, CPU or 1 GPU | `tiny` preset only (R4.2) |
| **release** (`pytest --run-slow`) | nightly, 8 GPU | official widths, seq {1 k, 8 k, 64 k} |
| **integration** | nightly, 8 GPU | 4 L official, 50 iterations |
| **scheduled/manual** | on request | twin runs, 8 L multi-node, 1 k-step runs |

## Tolerance harness (R4.4)

`kimi_k3/tests/tolerance.py` provides one comparison function used by every
parity test:

```python
compare(actual, oracle) -> {"rel_l2": ..., "max_abs": ..., "cos": ...}
```

Procedure for a new component, in this order:

1. Run **eager-fp32 vs eager-bf16** on the same inputs → this spread is the floor.
2. Set the component's bound above the floor with an explicit margin.
3. Record floor, bound and margin in the test docstring, together with the shape
   and seq length they were measured at.
4. Any later change to a bound is called out in the phase status row; bounds are
   never relaxed silently.

`gradcheck` (fp64) is used **only** for true-autograd modules: the KDA eager
oracle, the AttnRes mixer, the MLA gate. It is **invalid for STE paths** and
must not appear in any QAT test (an STE backward is deliberately not the
derivative of its forward). STE is validated against an explicit BF16 fake-quant
module with identical quantisation and ordinary autograd.

## Gate matrix

| Gate | Phase | Type | What it checks |
|---|---|---|---|
| **G1** | P0 | CPU | `PINS.md` complete (5 SHAs + LICENSES); `check_fla_signature.py` accepts the full released `chunk_kda` kwarg set incl. `use_qk_l2norm_in_kernel` |
| **G2** | P0 | CPU | Release audit closed: `A_log` checkpoint shape, `KimiRMSNorm` default eps, exact key prefixes + vision-key count, tokenizer ids — each recorded with its evidence |
| **G3** | P0 | CPU | `type(config) is KimiK3TransformerConfig`, MLA fields populated, `core_transformer_config_from_args` never on the call stack |
| **G4** | P0 | CPU | Meta-device tiny `K3GPTModel`: decoder is `K3TransformerBlock`; **no core `TransformerBlock` ever constructed**; rebinding uninstalled; 93 L analytic count within 1 % |
| **G5** | P0 | GPU | Measured bytes/param, per parameter group, for adam / precision-aware adam+dist-opt / muon / **dist_muon**, as a function of DP → `develop/results/opt_mem.md`, produced by `tools/opt_mem_probe.py` + `tools/opt_mem_report.py`. Measured as the CUDA allocator delta from before construction to after two optimizer steps, so masters, moments, gradient buffers and all-gather scratch are all counted |
| **G6** | P0 | GPU | AttnRes payload bytes per stage boundary vs slot count, and mixer fp32 temporaries, at tiny and official width |
| **G7** | P0 | GPU | PP=2 packed payload: loss + gradient parity vs single-stage, **and** a payload-gradient assertion (a perturbed slot on stage 1 produces non-zero gradient on stage 0) |
| **G8** | P0 | GPU | External assets present at the pins (AITER K3 a8w4 + `Situv2`; TE MXFP4 quantizer entry point) **and** one-expert QAT prototype: a8w4 forward, STE vs fake-quant, packed-cache refresh, checkpoint round-trip |
| **G9** | P1 | CPU | Guard rejects a core edit, accepts a `kimi_k3/` edit, honours the two-file allowlist, computes the diff against the merge-base |
| **G10** | P1 | GPU | `test_env.py` imports resolve; **pin contracts** (R4.5) all assert: `gpt_model.TransformerBlock`, `adjust_tensor_shapes_fn` accepted by 1F1B and asserted `None` by VPP/no-pipelining, `MoESubmodules.router`, `moe_layer.postprocess` shape, `clip_qk` walker, muon/dist-opt rejection, `dist_muon` → `LayerWiseDistributedOptimizer` |
| **G11** | P2 | CPU | 93 L layer pattern: 69 KDA / 24 MLA from the literal `kda_layers` list with `+1` indexing; layers 92 **and** 93 MLA; layer 1 dense with intermediate 33792; AttnRes slots at 0-indexed 0/12/…/84 (8 slots) |
| **G12** | P2 | CPU | Final config + construction path (promoted from G3/G4, run against shipped code) |
| **G13** | P2 | CPU | Meta-device parameter count for tiny/4L/8L/93L vs `tools/mem_budget.py`, per component and in total, within 1 % |
| **G14** | P3 | GPU | fp64 `gradcheck` on the tiny eager KDA module |
| **G15** | P3 | GPU | `fla` vs FP32 oracle, fwd+bwd, seq {1 k, 8 k, 64 k}; three statistics; thresholds + measured floors recorded (fast tier tiny, release tier official) |
| **G16** | P3 | GPU | Recurrent-state round-trip under `transpose_state_layout=True`; `sharded_state_dict` round-trip for `A_log`/`dt_bias`/conv |
| **G17** | P4 | GPU | K3 MLA vs eager release-math reference: NoPE (no rotation of the 64 shared dims), `scaling = 192**-0.5`, gate before `o_proj`, LoRA-norm eps |
| **G18** | P4 | GPU | Fused attention fwd+bwd at 192/128 head dims with the CK/AITER env set; `clip_qk` reached by core's walker |
| **G19** | P5 | GPU | AttnRes mixer vs oracle + fp64 gradcheck; behavioural tests (zero-init ⇒ uniform weights; early-block perturbation propagates) |
| **G20** | P5 | GPU | **Payload-gradient flow** — fails if any packed component's gradient is dropped (the finding-A1 guard; verified by temporarily reverting to a two-tensor payload) |
| **G21** | P5 | GPU | PP=2 vs single-stage loss and gradient parity with per-stage slot counts |
| **G22** | P5 | GPU | Recompute on/off loss-identical; `block_size = num_layers` degenerate case |
| **G23** | P6 | CPU/GPU | SiTU-GLU parity vs the release formula, fp32 math contract, β = 4 / β₂ = 25 |
| **G24** | P6 | GPU | LatentMoE forward parity on fixed weights incl. `routed_expert_norm` placement and shared experts at hidden width |
| **G25** | P6 | CPU | **Quantile-Balancing exact update parity** vs the ported reference on fixed synthetic margins (uniform / heavy-tail / ties); estimator error-vs-bin-width documented |
| **G26** | P6 | 8 GPU | EP smoke on 8 ranks + `router_replay` determinism; load ratio reported (not gated) |
| **G27** | P7 | 8 GPU | 4 L official, 50 iterations, no NaN/Inf, loss decreasing, no banned warnings |
| **G28** | P7 | 8 GPU | Memory report committed; analytic model reproduces measurement within 10 % |
| **G29** | P7 | 8 GPU | Full-model dist-checkpoint save/load; resumed loss matches |
| **G30** | P8 | CPU | Mapping dry run over the full released index: zero unmapped tensors, invariants asserted, vision keys skipped **and counted** |
| **G31** | P8 | GPU | Converter round-trip: bf16 exact; byte-exact packed round-trip only for an unchanged checkpoint with preserved scales; otherwise dequantised-value equivalence |
| **G32** | P8 | GPU | Checkpoint-anchored parity on real layers 1–4 vs a truncated HF reference; per-layer statistics within component tolerances; truncated-logit match reported |
| **G33** | P8 | CPU | Tokenizer round-trip + special-token ids on a fixed corpus |
| **G34** | P9 | 8 GPU | Twin-run noise band measured from 3 seeds and committed; eager-vs-fla twin inside the band at tiny and 4 L |
| **G35** | P9 | GPU | Per-head Muon single-step parity vs a torch reference on fixed gradients; K3 parameter-group policy asserted (which tensors go to Muon vs the scalar optimizer) |
| **G36** | P9 | 8 GPU | `clip_qk` active, max-attention-logit logged, no divergence over the window |
| **G37** | P9 | 8 GPU | Save → resume bitwise-stable loss for 20 steps under `PP=2` |
| **G38** | P10 | GPU | MXFP4/MXFP8 quantiser exact vs our OCP-MX reference; round-trip cross-check against TE's `MXFP4Quantizer` |
| **G39** | P10 | GPU | a8w4 forward within MXFP8-round-trip tolerance of the dequantised matmul |
| **G40** | P10 | GPU | **STE gradients exactly match the BF16 fake-quant reference** (no gradcheck anywhere in this gate) |
| **G41** | P10 | 8 GPU | 4 L QAT trains; QAT-vs-BF16 offset stable; serving-logit parity statistics reported |
| **G42** | P11 | 8 GPU | EP=8 proxy smoke + chrome trace + ranked-bottleneck report **with per-task budgets** |
| **G43** | P11 | GPU | Fused AttnRes mixer fwd+bwd parity vs the eager mixer at fast and release tiers |
| **G44** | P11 | 8 GPU | Fused mixer meets the G42 budget; actual delta recorded in the status row |
| **G45** | P11 | 8 GPU | Post-phase trace + report; no banned warnings; no HBM regression |
| **G46** | P12 | CPU | 93 L configs: meta-device construction, parameter count, memory budget, PP-layout legality (boundaries aligned to AttnRes blocks) |
| **G47** | P12 | CPU | EP ladder configs + dispatcher A/B matrix + analysis notebook committed |

## Banned warnings (R9.2)

A smoke run must grep clean against this set; each plan may extend it.

1. `fallback to eager KDA backend` — allowed only in runs that deliberately select eager.
2. `fallback to BF16 fake-quant experts` — allowed only before G39 is green.
3. `adjust_tensor_shapes_fn is not supported` — always fatal (means the binding leaked into VPP or PP=1).
4. `Shared expert overlap not supported` raised as a *warning* rather than an assertion — always fatal; the assertion must be surfaced, not masked.
5. `NaN`/`Inf` in loss, grad-norm, or router scores.
6. `recompiling` / Dynamo cache-limit messages from the fused mixer (P11 onward).
7. `unmapped tensor` from the converter.

## CI ladder

| Stage | Trigger | Content | Budget |
|---|---|---|---|
| **0** | every commit | guard (G9) · pin contracts (G10) · config + layer pattern + parameter count (G3, G11, G13) · mapping dry run (G30) · QB exact-parity math (G25) · tokenizer (G33) | < 5 min, CPU |
| **1** | every commit | **model construction (G4, G12)** — needs a GPU because the local non-TE MLA spec cannot be constructed at the pin (finding A13) · tiny-tier parity: KDA (G14, G15-fast), MLA (G17-fast), AttnRes (G19, G20, G22), SiTU/MoE (G23, G24-fast) · synthetic converter round-trip (G31-synthetic) | < 20 min, 1 GPU |
| **2** | nightly | release-tier parity (G15, G17, G18, G43) · 4 L integration (G27–G29) · EP smoke (G26) · anchored parity (G32) · QAT smoke (G38–G40) | 8 GPU |
| **3** | scheduled / manual | twin runs (G34), QAT convergence (G41), perf traces (G42, G44, G45), multi-node 8 L | 8 GPU / cluster |

A red parity gate blocks merges that touch its component (R4.6).

## Test tree

```
kimi_k3/tests/
├── tolerance.py                          shared 3-statistic comparison + floor helper
├── conftest.py                           tiny-preset fixtures, empty_cache autouse, --run-slow
├── test_env.py                           P1  imports + versions
├── test_k3_p1_pin_contracts.py           P1  IFU tripwire (R4.5)
├── test_k3_p2_config.py                  P2  G3/G12 config path
├── test_k3_p2_layer_pattern.py           P2  G11
├── test_k3_p2_param_count.py             P2  G13
├── test_k3_p3_kda_gradcheck.py           P3  G14
├── test_k3_p3_kda_parity.py              P3  G15 (fast + slow)
├── test_k3_p3_kda_state.py               P3  G16
├── test_k3_p4_mla_parity.py              P4  G17
├── test_k3_p4_mla_backend.py             P4  G18
├── test_k3_p5_attn_res_mixer.py          P5  G19
├── test_k3_p5_payload_grad.py            P5  G20  ← the design-error guard
├── test_k3_p5_pp_parity.py               P5  G21
├── test_k3_p5_recompute.py               P5  G22
├── test_k3_p6_situ.py                    P6  G23
├── test_k3_p6_latent_moe.py              P6  G24
├── test_k3_p6_qb_router.py               P6  G25
├── test_k3_p6_ep_smoke.py                P6  G26
├── test_k3_p8_mapping.py                 P8  G30
├── test_k3_p8_convert_roundtrip.py       P8  G31
├── test_k3_p8_anchored_parity.py         P8  G32
├── test_k3_p8_tokenizer.py               P8  G33
├── test_k3_p9_per_head_muon.py           P9  G35
├── test_k3_p10_quantizer.py              P10 G38
├── test_k3_p10_a8w4_forward.py           P10 G39
├── test_k3_p10_ste_backward.py           P10 G40
└── test_k3_p11_fused_attn_res.py         P11 G43
```
