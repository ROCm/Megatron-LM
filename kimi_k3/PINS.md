# Kimi K3 — dependency pins

> Gate **G1**. Every SHA the project builds against, plus the license
> compatibility conclusion for each (rule R2.2 / R7.2). A pin bump is its own
> commit and re-runs G1–G3 (rule R10.3).
>
> Status: **G1 is NOT green** — `fla` is pinned and its forward verified, but its
> KDA backward does not compile on gfx950; see §2 and
> `develop/notes/2026-08-27-fla-signature-check.md`.
> Last updated: 2026-08-27.

## 1. Pins

| Repo | Pin | How it was resolved | Verified here |
|---|---|---|---|
| **ROCm/Megatron-LM** | `a1b00d4259e92dc4a07a0be2c24088fe827f4b6e` (`rocm_dev`) | local checkout; the branch `dev/wen/kimi-k3` is cut from it | yes — all pin contracts assert against it (`kimi_k3/model/core_patch.py`) |
| **ROCm/TransformerEngine** | `2.12.0.dev0+40434cf6` (installed wheel) | version string carries the source SHA `40434cf6` | yes — `MXFP4BlockScaling`, `MXFP8BlockScaling`, `MXFP4Quantizer` located |
| **AITER** | `e9e1278b1` (origin/main, 2026-08-27) | git; the workspace checkout was 5 months stale | **yes** — `kimik3_a8w4_tuned_fmoe.csv`, `ops/opus/moe_stage{1,2}_a8w4.py` and `ActivationType.Situv2` all present |
| **fla** (flash-linear-attention) | git `5e02dd3` (0.6.0) — **forward OK, backward blocked** | git clone; the PyPI wheel ships no `fla/ops` | forward verified by running the released call; backward fails to compile on gfx950 (see §2) |
| **HF moonshotai/Kimi-K3** | `a590ce090cb049c93a33dfe8c208ec652aa20503` (lastModified 2026-08-20) | HF model API | yes — config, modeling sources and shard headers read at this revision |

## 2. `fla` — usable for forward, blocked for backward (G1 red)

**Correction:** an earlier version of this section claimed the released
`chunk_kda` call was silently mis-handled by `fla`. That was wrong; see
`develop/notes/2026-08-27-fla-signature-check.md` §1 for the retraction.

`fla` git `main` **accepts the released Kimi-K3 call exactly as written**:
`A_log` and `dt_bias` are read from `**kwargs` when `use_gate_in_kernel=True`
(and the function raises if `A_log` is missing without a `lower_bound`), and
`transpose_state_layout` is an accepted alias for `state_v_first`. Verified by
running it: forward produces correct-shaped output and perturbing `A_log`
changes the result.

What is actually blocking:

1. **The backward does not compile on gfx950.** `chunk_kda_bwd_intra` fails in
   Triton's AMD backend (`fla/ops/kda/chunk_intra.py:395`, `TritonAMDGPUPipeline`,
   `RuntimeError: PassManager::run failed`) with triton 3.6.0. Forward-only is
   not enough to train.
2. **The PyPI wheel is partial.** `flash-linear-attention==0.5.2` ships no
   `fla/ops`, so install from git.

Pin: **git `5e02dd3`** (version string 0.6.0), forward-verified, backward blocked.
`k3_kda_backend` stays `eager` until the backward compiles and G15 is green
(rule R5.3) — which does not block P2–P6, since the FP32 oracle is the contract.

G1's check is **functional**: run the released call and compare against the
oracle. A signature-based check would have failed on a working library, because
`A_log` legitimately arrives through `**kwargs`.

## 3. AITER — resolved

The K3 assets the plan named **do exist** on `main` (`e9e1278b1`, 2026-08-27); the
workspace checkout was simply five months old (2026-04-03), which is what review
finding D1 actually caught. Verified at this SHA:

| asset | status |
|---|---|
| `configs/model_configs/kimik3_a8w4_tuned_fmoe.csv` | present — 35 tuned rows: `gfx950`, `model_dim=3584`, `expert=896`, `topk=16`, `ActivationType.Situv2`, act `float8_e4m3fn` x weight `float4_e2m1fn_x2`, `QuantType.per_1x32`, `flydsl_moe{1,2}_afp8_wfp4_*` kernels |
| `ops/opus/moe_stage1_a8w4.py` | present — `opus_moe_stage1_a8w4_fwd(..., situ_beta=4.0, situ_linear_beta=25.0)`, fp8 output with an e8m0 `out_scale` |
| `ops/opus/moe_stage2_a8w4.py` | present |
| `ActivationType.Situv2` | present in the enum and in the fused-MoE dispatch |

**Not yet exercised.** Running the kernels needs the workspace checkout bumped
and its JIT modules rebuilt; P10 does that. P0 verified presence, pinned the SHA,
and built the numerics contract the kernels must match
(`kimi_k3/moe/k3_qat.py`).

## 4. LICENSES

| Repo | License | Compatibility conclusion |
|---|---|---|
| ROCm/Megatron-LM | NVIDIA Megatron-LM license (BSD-3-style, per-file NVIDIA copyright) | We add files under `kimi_k3/` and modify nothing in `megatron/**`; our additions carry their own headers. Compatible. |
| ROCm/TransformerEngine | Apache-2.0 (upstream) / MIT for the AMD fork's added files | Runtime dependency only; no code vendored. Compatible. |
| AITER | MIT (© Advanced Micro Devices) | Runtime dependency only; no code vendored. Compatible. |
| fla (flash-linear-attention) | MIT (© 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li) | Runtime dependency. If any eager reference is *ported*, the file carries a provenance header naming this license (rule R7.2). Compatible. |
| HF moonshotai/Kimi-K3 | "Kimi K3 License" — MIT-style grant covering weights, config, inference **and training** code, with attribution conditions | We read config/modeling sources to derive an independent implementation and we convert weights. Permitted. Any file transcribing released code carries a provenance header naming this license. |

**Nothing is vendored into this tree.** Reference clones stay outside the repo
(rule R6.2); the only committed artefact derived from the release is
`kimi_k3/tests/fixtures/release_shapes.json` (tensor shapes and dtypes, no weights).

## 5. Environment observed at pin time

| Component | Version |
|---|---|
| torch | 2.10.0+git94c6e04 |
| triton | 3.6.0 |
| transformer_engine | 2.12.0.dev0+40434cf6 |
| GPUs | 8 × AMD Instinct MI355X (gfx950) |
| fla | 0.6.0 (git `5e02dd3`), installed from source |
