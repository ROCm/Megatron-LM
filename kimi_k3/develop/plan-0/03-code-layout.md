# 03 — Code Landing List

> Paths and responsibilities only; the design behind them is in
> [`02-target-architecture.md`](02-target-architecture.md), the task lists in
> [`04-phase-details.md`](04-phase-details.md).
> Prefix legend: `+` new file · `~` modified later in the project · `!` new package.

## 0. Top-level layout

```
kimi_k3/                                    ! everything we own
├── README.md                               + entry doc (points at develop/)
├── PINS.md                                 + 5 SHAs + LICENSES section + env versions (P1, re-pinned on bump)
├── config/
│   ├── k3_transformer_config.py            + KimiK3TransformerConfig(MLATransformerConfig)
│   ├── k3_config_builder.py                + k3_config_from_args()  (never core_transformer_config_from_args)
│   ├── presets.py                          + tiny / 4L / 8L / 93L
│   └── arguments.py                        + add_kimi_k3_args(parser)
├── model/
│   ├── core_patch.py                       + scoped `k3_block_class()` rebinding (R2.2), pin-contract helpers
│   └── k3_gpt_model.py                     + K3GPTModel(GPTModel): block injection + packed set_input_tensor
├── attention/
│   ├── kda.py                              + KimiDeltaAttention module + submodules dataclass
│   ├── kda_backends.py                     + dispatch: fla chunk_kda | eager  (+ reserved flydsl slot)
│   ├── kda_eager_fp32.py                   + FP32 oracle: same math as the released kernel call
│   └── gated_mla.py                        + K3GatedMLA(MLASelfAttention): NoPE, 192**-0.5 scale, output gate, clip_qk hook
├── block/
│   ├── attn_res.py                         + AttnResMixer (fp32 eager oracle, exact release math)
│   ├── attn_res_pp.py                      + pack/unpack + slots_before() + shape helpers (the protocol)
│   ├── k3_transformer_layer.py             + K3TransformerLayer(TransformerLayer): two mixes + slot append
│   └── k3_transformer_block.py             + K3TransformerBlock(TransformerBlock): state ownership, output mix
├── pipeline/
│   └── k3_schedule.py                      + binds adjust_tensor_shapes_fn when PP>1; no schedule re-implementation
├── moe/
│   ├── k3_moe_layer.py                     + K3MoELayer(MoELayer): postprocess override for routed_expert_norm
│   ├── k3_router.py                        + QuantileBalancingRouter(TopKRouter) + histogram estimator
│   ├── situ.py                             + situ_glu() activation (fp32 math)
│   ├── k3_qat.py                           + MXFP4/MXFP8 pack/unpack, e8m0 scales, RNE/SR   (P10)
│   └── k3_qat_experts.py                   + KimiK3QATGroupedMLP (a8w4 fwd + STE bwd)       (P10)
├── specs/
│   ├── layer_specs.py                      + per-layer spec factory (KDA vs MLA, dense vs MoE)
│   └── block_spec.py                       + block-level spec incl. AttnRes submodules
├── optim/
│   └── per_head_muon.py                    + per-head momentum partition + core clip_qk wiring   (P9)
├── tools/
│   ├── mapping.py                          + release-key ↔ Megatron-key table + invariants
│   ├── convert.py                          + shard-wise converter, both directions
│   ├── tokenizer.py                        + tiktoken o200k-style, vocab 163840, bos/eos/pad ids
│   ├── mem_budget.py                       + analytic param/optimizer/activation model (regenerated from measurements)
│   ├── twin_run.py                         + twin-run driver + noise-band statistic
│   └── attn_res_probe.py                   + payload/temporary sizing instrument (P0-F7, reused in P11)
├── training/
│   ├── pretrain_kimi_k3.py                 + entry point (model_provider, forward_step, dataset provider)
│   └── configs/                            + per-config yaml/sh (tiny, 4L, 8L, 93L, QAT, EP ladder)
├── tests/                                  + see 05-test-strategy.md for the full tree
├── ci/no_core_diff_guard.sh                + guard with the two-file allowlist
├── upstream_proposals/                     + NNN-*.patch + issue drafts (never applied without sign-off)
├── repro/                                  + minimal reproducers for upstream bugs
├── results/                                + opt_mem.md, parity/, payload sizing tables
└── develop/                                ← docs only (this tree)

.github/workflows/kimi_k3.yml               ~ ALLOWLISTED root file (CI wiring)
.pre-commit-config.yaml                     ~ ALLOWLISTED root file (hook registration; human registers)
```

## 1. Phase × file matrix

| Phase | Files | Depends on |
|---|---|---|
| **P0** | `PINS.md`, `config/k3_transformer_config.py` (skeleton), `config/k3_config_builder.py` (skeleton), `model/{core_patch,k3_gpt_model}.py` (skeleton), `block/attn_res_pp.py` (prototype), `pipeline/k3_schedule.py` (prototype), `tools/{mem_budget,attn_res_probe}.py`, `moe/k3_qat*.py` (one-expert prototype), `results/opt_mem.md` | pinned repos only |
| **P1** | `ci/no_core_diff_guard.sh`, `tests/test_env.py`, `tests/test_k3_p1_pin_contracts.py`, `.github/workflows/kimi_k3.yml`, `.pre-commit-config.yaml`, `README.md` | P0 |
| **P2** | `config/*` (final), `specs/layer_specs.py`, `model/*` (final), `tests/test_k3_p2_*.py` | P1 |
| **P3** | `attention/{kda,kda_backends,kda_eager_fp32}.py`, `tests/test_k3_p3_kda_*.py` | P2 |
| **P4** | `attention/gated_mla.py`, `tests/test_k3_p4_mla_*.py` | P2 |
| **P5** | `block/*`, `pipeline/k3_schedule.py`, `specs/block_spec.py`, `tests/test_k3_p5_attn_res_*.py` | P2 |
| **P6** | `moe/{k3_moe_layer,k3_router,situ}.py`, `tests/test_k3_p6_moe_*.py` | P2 |
| **P7** | `training/pretrain_kimi_k3.py`, `training/configs/4l_official.*`, `tools/{tokenizer,mem_budget}.py`, `results/opt_mem.md` (refresh) | P3+P4+P5+P6 |
| **P8** | `tools/{mapping,convert}.py`, `tests/test_k3_p8_convert_*.py`, `results/parity/` | P3+P4+P6 |
| **P9** | `optim/per_head_muon.py`, `tools/twin_run.py`, `training/configs/twin_*.sh`, `tests/test_k3_p9_*.py` | P7+P8 |
| **P10** | `moe/{k3_qat,k3_qat_experts}.py`, `training/configs/qat_4l.*`, `tests/test_k3_p10_qat_*.py` | P6+P7 |
| **P11** | `develop/progress/p11/run_*.sh`, `develop/profile/*.md|.html`, `develop/perf/{attn_res,proxy_ep8}.md`, `block/attn_res_fused.py` | P7 (+P5) |
| **P12** | `training/configs/{93l_*,ep_ladder_*}`, `develop/notes/*.md`, analysis notebooks | P9+P10+P11 |

## 2. Interface contracts

### 2.1 Entry point

`kimi_k3/training/pretrain_kimi_k3.py` is the only place that:

1. builds the config via `k3_config_from_args()` (never core's builder),
2. constructs `K3GPTModel` inside `k3_block_class(K3TransformerBlock)`,
3. installs the AttnRes shape binding via `k3_schedule.install(pp_size)` — a
   no-op when `PP == 1` (`schedules.py:631` asserts the hook is `None` there),
4. registers K3 arguments through `add_kimi_k3_args`.

Nothing else in the tree performs a namespace patch (R2.2).

### 2.2 Decoder state contract

Between layers, the decoder carries `(prefix_sum [S,B,H], block_residual [S,B,K,H])`.
Across a pipeline boundary it carries **one packed tensor** of shape
`[(1+K)·S, B, H]` (`attn_res_pp.pack`). Stage-0 input and last-stage output are
plain `[S, B, H]`. This contract is written in `attn_res_pp.py`'s module
docstring and asserted by G19/G20/G21.

### 2.3 Backend contract

Every accelerated backend implements the same signature as its FP32 eager oracle
and is selected by a config field, never by import-time availability. Availability
failures downgrade with exactly one warning line (subject to the banned-warning
ratchet, R9.2).

### 2.4 Checkpoint contract

Every new module implements `sharded_state_dict`. Invariants that the converter
asserts (never "if present"): `A_log` padding/trim rule, `dt_bias == [12288]`,
gate-slot ordering inside `linear_fc1`, expert MXFP4 pack + e8m0 scale pairing,
and an explicit skip-list for vision keys with a reported count.

## 3. Reference porting

| Source | Lands as | Requirement |
|---|---|---|
| Released `modeling_kimi_linear.py` math | `kda_eager_fp32.py`, `attn_res.py`, `situ.py`, `k3_router.py` | re-implemented against Megatron conventions, with the release's exact math; provenance header (R7.2) |
| Primus eager references (if used) | same files | provenance header **and** a recorded license-compatibility conclusion in `PINS.md` before the file lands |
| `fla` | runtime dependency only | pinned SHA whose `chunk_kda` signature matches the released call (G1) |
| AITER | runtime dependency only | pinned SHA that actually contains the K3 a8w4 assets (G8) |
