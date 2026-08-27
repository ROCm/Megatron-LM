# Kimi K3 — dependency pins

> Gate **G1**. Every SHA the project builds against, plus the license
> compatibility conclusion for each (rule R2.2 / R7.2). A pin bump is its own
> commit and re-runs G1–G3 (rule R10.3).
>
> Status: **G1 is NOT green** — the `fla` pin is unresolved and blocking; see
> §2 and `develop/notes/2026-08-27-fla-signature-check.md`.
> Last updated: 2026-08-27.

## 1. Pins

| Repo | Pin | How it was resolved | Verified here |
|---|---|---|---|
| **ROCm/Megatron-LM** | `a1b00d4259e92dc4a07a0be2c24088fe827f4b6e` (`rocm_dev`) | local checkout; the branch `dev/wen/kimi-k3` is cut from it | yes — all pin contracts assert against it (`kimi_k3/model/core_patch.py`) |
| **ROCm/TransformerEngine** | `2.12.0.dev0+40434cf6` (installed wheel) | version string carries the source SHA `40434cf6` | yes — `MXFP4BlockScaling`, `MXFP8BlockScaling`, `MXFP4Quantizer` located |
| **AITER** | `f299f579a` (2026-04-03) — **stale, must be bumped** | local checkout at `/workspace/aiter` | **no** — the K3 a8w4 assets are absent at this SHA (see §3) |
| **fla** (flash-linear-attention) | **UNRESOLVED — blocking** | PyPI 0.5.2 and GitHub `main` both mismatch the released call | **no** — see §2 |
| **HF moonshotai/Kimi-K3** | `a590ce090cb049c93a33dfe8c208ec652aa20503` (lastModified 2026-08-20) | HF model API | yes — config, modeling sources and shard headers read at this revision |

## 2. `fla` — why the pin is unresolved (G1 red)

The released `modeling_kimi_linear.py` calls:

```python
chunk_kda(q, k, v, g, beta,
          A_log=self.A_log, dt_bias=self.dt_bias,
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True, safe_gate=..., lower_bound=-5.0,
          transpose_state_layout=True, cu_seqlens=cu_seqlens)
```

`chunk_kda` on fla `main` accepts **neither `A_log` nor `dt_bias`**, and names the
state-layout flag **`state_v_first`**, not `transpose_state_layout`. Its signature
ends in `**kwargs`, so those three arguments are **silently swallowed** instead of
raising — a call that looks correct would run with the decay parameters ignored.

Consequences, in order:

1. The pin must be the revision (or fork) the release was built against — "latest"
   is wrong and, worse, wrong *quietly*.
2. `tools/check_fla_signature.py` must assert **by parameter name** against
   `inspect.signature`, never by "the call did not raise".
3. Until the pin is resolved, `k3_kda_backend` stays `eager` (rule R5.3) and the
   FP32 oracle is the only KDA path. This does not block P2–P6.

## 3. AITER — assets not present at the local SHA

At `f299f579a` there is no `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv`,
no `aiter/ops/opus/` directory and no `ActivationType.Situv2`. P10 (QAT) needs
them; P0-T0.8 must bump the pin to a SHA that has them or record the fast path as
descoped (review finding D1).

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
| fla | **not installed** |
