# Kimi K3 — dependency pins

> Gate **G1**. Every SHA the project builds against, plus the license
> compatibility conclusion for each (rule R2.2 / R7.2). A pin bump is its own
> commit and re-runs G1–G3 (rule R10.3).
>
> Status: **G1 GREEN** — all five pins resolved, licenses recorded, and the
> released `chunk_kda` call verified functionally (forward *and* backward) after
> the triton 3.6.0 → 3.7.1 upgrade. See §2.
> Last updated: 2026-09-01.

## 1. Pins

| Repo | Pin | How it was resolved | Verified here |
|---|---|---|---|
| **ROCm/Megatron-LM** | `a1b00d4259e92dc4a07a0be2c24088fe827f4b6e` (`rocm_dev`) | local checkout; the branch `dev/wen/kimi-k3` is cut from it | yes — all pin contracts assert against it (`kimi_k3/model/core_patch.py`) |
| **ROCm/TransformerEngine** | `2.18.0.dev0+8f377e4` — **moved 2026-09-01**, was `2.12.0.dev0+40434cf6` | built from source at fork HEAD (`8f377e4`, 2026-08-29); version string carries the SHA | yes — see §6 for what the move fixed and which results predate it |
| **AITER** | `e9e1278b1` (origin/main, 2026-08-27) | git; the workspace checkout was 5 months stale | **yes** — `kimik3_a8w4_tuned_fmoe.csv`, `ops/opus/moe_stage{1,2}_a8w4.py` and `ActivationType.Situv2` all present |
| **fla** (flash-linear-attention) | git `5e02dd3` (0.6.0) | git clone; the PyPI wheel ships no `fla/ops` | **forward and backward both verified** on triton 3.7.1 by `tests/test_k3_p0_fla_contract.py` |
| **HF moonshotai/Kimi-K3** | `a590ce090cb049c93a33dfe8c208ec652aa20503` (lastModified 2026-08-20) | HF model API | yes — config, modeling sources and shard headers read at this revision |

## 2. `fla` — resolved (G1 green)

Two corrections led here, both recorded in
`develop/notes/2026-08-27-fla-signature-check.md`:

1. The released Kimi-K3 `chunk_kda` call **is accepted as written**. `A_log` and
   `dt_bias` are read from `**kwargs` when `use_gate_in_kernel=True`, the function
   raises if `A_log` is missing without a `lower_bound`, and
   `transpose_state_layout` is an accepted alias for `state_v_first`. The earlier
   "silently ignored" reading was inferred from the signature and is retracted.
2. The real blocker was **Triton**, not fla and not torch: `chunk_kda_bwd_intra`
   failed to compile on gfx950 under triton 3.6.0. **Upgrading Triton alone to
   3.7.1 fixes it** — forward and backward both run, with finite non-zero
   gradients at `(T,H,K) = (128,4,64)` and `(2048,8,128)`. No torch change was
   needed, which matters: TE, apex and AITER are all built against this torch.

Pin: **git `5e02dd3`** (0.6.0). Install from git — the PyPI wheel ships only
`fla/layers` and `fla/models`, no `fla/ops`.

`k3_kda_backend` nevertheless stays `eager` by default (rule R5.3): compiling is
not the same as being *correct*, and the fla KDA-backward bug history (#807,
#785) is why the FP32 oracle is permanent. G15 flips the default, at production
shapes, against the oracle.

G1's check is **functional, never signature-based**
(`tests/test_k3_p0_fla_contract.py`): a name-based check would have failed on a
working library.

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
| triton | **3.7.1** (upstream PyPI; 3.6.0 could not compile fla's KDA backward). `torch.compile` re-verified on it — including inductor's Triton MM templates under `max-autotune` and Megatron's `jit_fuser` fusion path, which is `torch.compile` by default. Guarded by `tests/test_k3_p0_torch_compile_contract.py` |
| transformer_engine | **2.18.0.dev0+8f377e4** (built from source; see §6) |
| GPUs | 8 × AMD Instinct MI355X (gfx950) |
| fla | 0.6.0 (git `5e02dd3`), installed from source |


## 6. TransformerEngine pin moved, 2026-09-01

`2.12.0.dev0+40434cf6` → `2.18.0.dev0+8f377e4` (ROCm fork HEAD), **built from
source**. PyPI's `transformer_engine` is a metapackage over
`transformer_engine_cu12`/`cu13` and is CUDA-only — there is no ROCm wheel, and
installing it would shadow the working build.

    NVTE_FRAMEWORK=pytorch PYTORCH_ROCM_ARCH=gfx950 \
        pip install --no-build-isolation --no-deps .

Two build notes. `git clone --depth 1 --recursive` leaves **`3rdparty/ck_jit`
empty**, and the only symptom is `[AITER-BUILD] CK-JIT build failed` — pip hides
the underlying `ck_jit_build.py: No such file or directory`, so run CMake by hand
to see it. And the default builds `gfx942;gfx950`; pinning `PYTORCH_ROCM_ARCH`
halves the work. The previous install is archived at
`/tmp/te_backup/te_2.12.0.dev0+40434cf6.tgz`.

### Why it moved

The **CK fused-attention backward returned NaN** at `hd192_hd128` — K3's gated
MLA shape. On the old pin `q_a_layernorm` and `q_b_proj` came back NaN while
`kv_a_layernorm` stayed finite, deterministic 6/6, with AOTRITON clean. At HEAD
all three fused-attention paths agree at 7.9297e-01, so `k3_mla_backend` defaults
to `te` with no `NVTE_FUSED_ATTN_*` pin. That is worth **2.97x** on the MLA
attention operator (6.459 → 2.169 ms, 29.1 % → 72.2 % of peak) and 1.67x on the
whole MLA layer at seq 8192 (11.51 → 6.89 ms).

### What was re-verified on the new build

| check | result |
|---|---|
| full test suite | **288 passed, 11 skipped** |
| anchored MLA parity vs the release | **rel-L2 5.8240e-03, cosine 0.999983** — identical to the old build |
| CK grouped GEMM vs hipBLASLt | 1.536 ms vs 2.212 ms — **1.44x**, unchanged |
| MLA attention operator | 2.169 ms / 72.2 % of peak — unchanged |
| Newton-Schulz on ROCm (**A20**) | still absent: no `newton_schulz`, `CusolverMpCtx` or `cusolvermp_ctx_create`. The finding rested on reading `cuda_only_cpp_sources` and the `_IS_HIP_EXTENSION` guard; a HEAD build now confirms it empirically |

### Results measured on the OLD pin and not re-run

These were measured against `2.12.0.dev0+40434cf6`. Anchored parity coming back
identical is good evidence none of them shifted, but none has been re-measured,
and in this project a number belongs to the configuration it was taken on:

* `results/kda_parity.md` — G15, including production geometry. KDA runs through
  `fla`, not TE, so a TE bump should not touch it.
* `results/anchored_parity_4l.md` — the four-layer slice. Only the **MLA layer**
  was re-anchored on the new build (identical); KDA layers 1–2, the AttnRes sites
  and the routing check were not, and re-running them needs the 49 GiB slice again.
* `results/scaleout_93l.md`, `profile/profile-baseline-2026-08-27.md` — the memory
  model, the 28-node floor, and the whole device-time ranking. Attention is a
  ~155 ms line of which the TE swap removes ~60 ms, so the AttnRes mixer's
  dominance is unaffected, but the absolute timings predate the bump.
* `results/opt_mem.md`, `results/pp_resume.md`, `results/qat_twin.md`,
  `results/twin_runs.md`, `results/router_replay.md`, `results/ep_smoke.md`,
  `results/a8w4.md`, `results/mxfp4_scale_rule.md` — optimizer, transport, QAT and
  routing paths, none of which route through TE's attention.

The one that would most repay re-running is the **P11 device-time trace**: MLA
attention was measured before the swap, so its share of the ranking is now stale
by roughly 60 ms per forward.
