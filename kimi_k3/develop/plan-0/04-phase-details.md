# 04 — Per-Phase Detail

> Each phase is **Tasks → Exit criteria → Risks / notes**. Gate definitions live
> in [`05-test-strategy.md`](05-test-strategy.md); file paths in
> [`03-code-layout.md`](03-code-layout.md). When a task is done, tick it in
> [`../progress/status.md`](../progress/status.md) with the commit SHA (R1.3),
> and close the phase with `progress/p<id>/p<id>-summary.md` (R3.5).
> Tier tags: **[CPU-OK]** · **[GPU]** 1×MI355X · **[8-GPU]** one node ·
> **[CLUSTER]** human-run.

---

## P0 — Feasibility gates (nothing is ported before this phase is green)

### Tasks

1. **T0.1 Pins + licenses. [CPU-OK]** Exact SHAs for the five repos (this
   Megatron fork incl. the `a1b00d4` lineage check, TransformerEngine, AITER,
   `fla`, and the HF `moonshotai/Kimi-K3` revision) into `kimi_k3/PINS.md`,
   with a `LICENSES` section naming each repo's license and the compatibility
   conclusion. Ship `tools/check_fla_signature.py`: import `chunk_kda` and
   assert the released kwarg set is accepted — `q, k, v, g, beta, A_log,
   dt_bias, initial_state, output_final_state, use_qk_l2norm_in_kernel,
   use_gate_in_kernel, use_beta_sigmoid_in_kernel, safe_gate, lower_bound,
   transpose_state_layout, cu_seqlens` (finding B4).
2. **T0.2 Release artefact audit. [CPU-OK]** Download only `config.json`,
   `model.safetensors.index.json`, the modeling/configuration `.py` files and
   **one** shard header (ask before any bulk download; R6.1). Close the four
   open ❓ items and record the answers in `develop/notes/2026-XX-release-audit.md`:
   (a) `A_log` shape in the checkpoint — `[96]` or `[128]`-padded (finding B5);
   (b) `KimiRMSNorm`'s default `eps` and therefore whether the MLA LoRA norms
   use 1e-5 (finding B7); (c) exact key prefixes and the vision-key count
   (finding B8); (d) tokenizer ids and special tokens.
3. **T0.3 Config inheritance path. [CPU-OK]** Skeleton
   `KimiK3TransformerConfig(MLATransformerConfig)` + `k3_config_from_args()`.
   Prove the constructed object is our class with MLA fields populated and that
   `core_transformer_config_from_args` was never called (sentinel patch during
   the test only).
4. **T0.4 Model construction. [CPU-OK]** Tiny `K3GPTModel` on **meta device**
   via the scoped rebinding (§3 of `02-target-architecture.md`). Assert the
   decoder type, assert **no core `TransformerBlock` was ever constructed**
   (constructor spy), assert the rebinding is uninstalled afterwards. Add the
   analytic 93 L parameter count against the architecture table.
5. **T0.5 Optimizer memory measurement. [GPU]** A tiny-but-real model measured
   under: `adam`, precision-aware `adam` + `--use-distributed-optimizer`,
   `muon`, and **`dist_muon`** (which shards master weights and momentum via
   `LayerWiseDistributedOptimizer`). Report **per parameter group** (the 2-D
   Muon group vs the scalar/Adam group — `muon.py:283-302`) and as a function of
   DP size, producing `results/opt_mem.md` with a `B/param(DP)` curve. This
   replaces the incoming plan's flat 14 B/param figure (finding A6).
6. **T0.6 AttnRes payload sizing. [GPU]** `tools/attn_res_probe.py` measures, at
   tiny and at production width: the packed payload bytes per stage boundary as
   a function of slot count, the fp32 temporaries inside `_apply_attn_res`, and
   the wall-clock of the eager mixer. Output feeds
   `06-capacity-and-parallelism.md` and sets the P11 budget.
7. **T0.7 PP=2 packed-payload prototype. [GPU]** Tiny model, `PP=2`, packed
   single-tensor payload, `adjust_tensor_shapes_fn` bound only when `PP>1`.
   Loss and gradient parity against the single-stage run, **plus** an explicit
   assertion that a perturbation of a `block_residual` slot on stage 1 produces
   a non-zero gradient on stage 0 (the finding-A1 guard). No VPP claim.
8. **T0.8 External assets + one-expert QAT. [GPU]** Verify at the pinned SHAs:
   AITER's K3 a8w4 assets (`configs/model_configs/kimik3_a8w4_tuned_fmoe.csv`,
   `aiter/ops/opus/moe_stage{1,2}_a8w4.py`, `ActivationType.Situv2`) and TE's
   MXFP4 quantizer entry point (in this workspace it is
   `transformer_engine/pytorch/tensor/mxfp4_tensor.py:MXFP4Quantizer`, **not**
   `custom_recipes/quantization_mxfp4.py`, and there is no NumPy reference —
   findings D1, D2). Then a single routed expert: packed a8w4 forward, STE
   backward vs an explicit BF16 fake-quant module, fp32 masters, packed-cache
   refresh on step, checkpoint round-trip of packed data + scales.

### Exit criteria

**G1–G8 green** (see `05-test-strategy.md`). A red gate is *reported*, not
worked around: if T0.8 fails on the AITER assets, P10's fast path is descoped
in writing and the phase still closes; if T0.7 fails, the AttnRes transport
design is re-opened before any block code is written.

### Risks / notes

- The tolerance floor for later phases starts here: T0.6 and T0.7 record
  eager-fp32 vs eager-bf16 spreads on the mixer, which is the first entry in the
  tolerance table (R4.4).
- T0.5 must run long enough to include the **step**, not just allocation —
  `LayerWiseDistributedOptimizer` all-gathers parameters after the step, so peak
  is not at construction.
- Do not bulk-download shards. A single shard header answers T0.2(a).

---

## P1 — Scaffold and environment

### Tasks

1. **T1.1 [CPU-OK]** Branch `dev/wen/kimi-k3` off `rocm_dev` at the pinned SHA (open since `2a66e4ce5`);
   create the `kimi_k3/` skeleton per `03-code-layout.md`.
2. **T1.2 [CPU-OK]** `ci/no_core_diff_guard.sh`: fail any diff outside
   `kimi_k3/` except `.github/workflows/kimi_k3.yml` and
   `.pre-commit-config.yaml`. Prove it by attempting a deliberate core edit in
   a scratch commit and confirming the guard rejects it.
3. **T1.3 [CPU-OK]** `tests/test_k3_p1_pin_contracts.py` — the IFU tripwire
   (R4.5). One assertion per core mechanism we ride:
   `gpt_model.TransformerBlock` exists at module scope and `GPTModel.__init__`
   still constructs it; `forward_backward_pipelining_without_interleaving`
   accepts `adjust_tensor_shapes_fn`; the interleaved and no-pipelining
   schedules still assert it is `None`; `MoESubmodules.router` is a builder
   field; `moe_layer.postprocess` still calls `fc2_latent_proj` with no norm;
   `optimizer/qk_clip.py:clip_qk` walks `.decoder.layers` and skips modules
   without `clip_qk`; `'muon' in optimizer` still rejects
   `--use-distributed-optimizer`; `dist_muon` still routes to
   `LayerWiseDistributedOptimizer`.
4. **T1.4 [GPU]** `tests/test_env.py`: import `fla.chunk_kda` (pinned
   signature), `transformer_engine.pytorch`, the AITER a8w4 entry points; write
   the resolved versions into `PINS.md`.
5. **T1.5 [CPU-OK]** Pre-commit hook script (guard + formatting) committed; a
   human registers it. CI workflow file wired for stage 0/1.

### Exit criteria

G9 (guard rejects a core edit, accepts a `kimi_k3/` edit, allowlist honoured),
G10 (env imports + pin-contract assertions green).

### Risks / notes

- The guard must run on the **diff against the merge-base**, not the working
  tree, or an IFU rebase trips it on every core commit it pulls in.

---

## P2 — Config, presets, model construction

### Tasks

1. **T2.1 [CPU-OK]** Final `KimiK3TransformerConfig` + `k3_config_from_args()`
   + `add_kimi_k3_args()`, with the full release→config mapping table from
   `02-target-architecture.md` §2.1 implemented and unit-tested field by field.
2. **T2.2 [CPU-OK]** `presets.py`: `tiny` (hidden 512, KDA head_dim 64 × 2
   heads, latent 256, 8 experts / top-2, 4 layers in a true `[K,K,K,M]`
   pattern, `attn_res_block_size = 2`, seq 128), `4L` / `8L` / `93L` official.
   Officials are meta-device only (R4.3).
3. **T2.3 [CPU-OK]** `specs/layer_specs.py`: per-layer kind resolution
   (`is_kda = (layer_idx + 1) in kda_layers`), dense-vs-MoE selection
   (`first_k_dense_replace = 1`, `ffn_hidden_size = 33792` on layer 1),
   MoE spec with `K3MoELayer` + `QuantileBalancingRouter` placeholders.
4. **T2.4 [CPU-OK]** `model/core_patch.py` + `model/k3_gpt_model.py` final,
   with the constructor-spy test promoted from P0.
5. **T2.5 [CPU-OK]** `tools/mem_budget.py` implements the architecture §8 analytic
   table and is the oracle for the parameter-count gate.

### Exit criteria

G11 (layer-pattern conformance for 93 L: 69 KDA / 24 MLA, `+1` indexing, layers
92 **and** 93 both MLA, layer 1 dense with intermediate 33792, AttnRes slots at
0-indexed 0/12/…/84 → 8 slots), G12 (config type survives + K3 block constructed
+ no transient core block), G13 (meta-device 93 L parameter count within 1 % of
the analytic table; per-component subtotals also checked).

### Risks / notes

- **The `+1` indexing convention is the most likely off-by-one in the project.**
  G11 pins it with the literal list from `config.json`, not a stride formula —
  the tail `…, 88, 92, 93` breaks the 3:1 stride once.
- Officials must never be instantiated with real weights in a test (R4.2).

---

## P3 — KDA mixer **[core]**

### Tasks

1. **T3.1 [GPU]** `kda_eager_fp32.py`: the FP32 oracle. Implements q/k/v
   projections → per-branch SiLU short conv (k=4) → **in-kernel-equivalent**
   q/k L2 normalisation → `sigmoid(beta)` → decay from `A_log`/`dt_bias`/`g`
   with the `lower_bound = -5` clamp → chunked delta-rule recurrence →
   `o_norm(o, sigmoid(g_out))` → `o_proj`.
2. **T3.2 [GPU]** `kda.py` + `kda_backends.py`: the module, the two-gate
   plumbing (`f_a/f_b` decay gate vs full-rank `g_proj` output gate — finding
   B3), and the `fla` backend mirroring the released call kwarg-for-kwarg.
   Eager is the default until G15 is green at production shapes (R5.3).
3. **T3.3 [GPU]** Tolerance work: measure eager-fp32 vs eager-bf16 spread as the
   floor at seq {1 k, 8 k, 64 k}, then set the fla-vs-oracle bounds on rel-L2,
   max-abs and cosine with recorded margin (R4.4).
4. **T3.4 [GPU]** `sharded_state_dict` for `A_log`, `dt_bias`, conv weights and
   the recurrent-state layout (`transpose_state_layout=True` round-trip).
5. **T3.5 [CPU-OK]** Behavioural tests independent of the backend: causality
   (token *t* cannot affect outputs < *t*), `lower_bound` clamp behaviour,
   `safe_gate` on/off equivalence when the bound is inactive.

### Exit criteria

G14 (fp64 `gradcheck` on the tiny eager module — true autograd, so gradcheck is
valid here), G15 (fla vs FP32 oracle: tiny in CI, production shapes nightly,
fwd+bwd, three statistics, thresholds and their floors recorded in the test
docstring), G16 (state + checkpoint round-trip).

### Risks / notes

- Upstream `fla` KDA **backward** bug history (#807, #785) is why the eager
  oracle is permanent, not a bring-up crutch. A production-shape backward
  failure goes to `repro/` and the backend default stays eager.
- A 64 k-token bf16 recurrent backward will not meet a 1e-5 bound. That is
  expected; the bound is whatever the measured floor plus margin says (R4.4).
- `chunk_kda` performs the q/k L2 norm; the oracle must do it in the same place
  or the two disagree by more than any tolerance.

---

## P4 — Gated MLA with NoPE **[core]**

### Tasks

1. **T4.1 [GPU]** `K3GatedMLA(MLASelfAttention)`: bypass rotary entirely (core
   supports only `rope`/`yarn` — finding A9), produce `k_rot` MQA-style and
   expand it un-rotated, set `softmax_scale = (128 + 64) ** -0.5`.
2. **T4.2 [GPU]** Full-rank sigmoid output gate applied to the flattened
   attention output **before** `linear_proj`; optional fp32 attention output
   (`k3_mla_fp32_attn_output`, default on — [report]).
3. **T4.3 [GPU]** Expose `clip_qk` so core's `optimizer/qk_clip.py:clip_qk`
   picks the layer up (finding A8); confirm KDA layers are skipped by the
   `hasattr` guard and that the block keeps a `.layers` attribute.
4. **T4.4 [GPU]** Fused-attention bring-up at the 192/128 head-dim combination
   (`NVTE_FUSED_ATTN_CK=1`, `NVTE_CK_USES_{FWD,BWD}_V3=1`,
   `NVTE_CK_IS_V3_ATOMIC_FP32=1`); if the backend rejects the combination,
   record the fallback (unfused / padded-V) with its cost.
5. **T4.5 [GPU]** LoRA-norm epsilon handling per the P0-T0.2(b) answer.

### Exit criteria

G17 (parity vs an eager MLA reference implementing the release math — NoPE,
scale, gate, LoRA norms — within component tolerances), G18 (fwd+bwd on the
fused backend at production head dims; `clip_qk` reached by core's walker in a
1-layer harness).

### Risks / notes

- The 64 un-rotated "rope" dims are *content*, not position. A reviewer who
  "fixes" this by enabling rotary silently destroys parity — the test docstring
  says so explicitly.
- `scaling = 192**-0.5`, not `128**-0.5`. This is the second most likely silent
  parity bug after the `+1` layer indexing.

---

## P5 — AttnRes block and pipeline transport **[core]**

### Tasks

1. **T5.1 [CPU-OK]** `attn_res.py`: the fp32 mixer oracle, transcribed from the
   release (architecture §6) — `cat(slots, prefix)` → RMS-normalise → score with
   `norm.weight * proj.weight` → softmax over `K+1` → weighted sum.
2. **T5.2 [CPU-OK]** `attn_res_pp.py`: `pack` / `unpack` / `slots_before` plus
   the module docstring that *is* the protocol specification.
3. **T5.3 [GPU]** `k3_transformer_layer.py`: two mixes per layer, slot append at
   `layer_idx % block_size == 0` with the `prefix_sum` reset, accumulation after
   attention and after the MLP.
4. **T5.4 [GPU]** `k3_transformer_block.py`: state ownership, stage-0 seeding,
   last-stage output mix + final norm, recompute wrapping that carries **both**
   state tensors.
5. **T5.5 [GPU]** `pipeline/k3_schedule.py`: bind `adjust_tensor_shapes_fn`
   when `PP > 1` (finding A2/A3); compute per-stage slot counts from the local
   layer range.
6. **T5.6 [GPU]** Behavioural tests: zero-initialised projections ⇒ uniform
   softmax over slots; perturbing an early block moves the last-layer output;
   `block_size = num_layers` degenerates to a single-block model.

### Exit criteria

G19 (mixer parity vs oracle + fp64 gradcheck at tiny), **G20 (payload-gradient
flow — the finding-A1 guard)**, G21 (PP=2 vs single-stage loss and gradient
parity), G22 (recompute on/off loss-identical; behavioural tests green).

### Risks / notes

- G20 is the gate that would have caught the incoming plan's design error. It
  must fail if either payload component's gradient is dropped — verify by
  temporarily reverting to a two-tensor payload and confirming the test goes red.
- Payload memory is a first-order cost (up to 9× the normal PP tensor). P5 keeps
  the eager mixer; the fused version is P11's job. Record the measured payload
  bytes per boundary in the phase summary.
- VPP stays descoped; if it becomes required it is an `upstream_proposals/` item,
  not a workaround.

---

## P6 — LatentMoE, Quantile-Balancing router, SiTU **[core]**

### Tasks

1. **T6.1 [CPU-OK]** `situ.py`: `situ_glu` in fp32 with `β = 4.0`,
   `β₂ = 25.0`, cast back at the end; wired as `MLPSubmodules.activation_func`
   for both routed and shared experts.
2. **T6.2 [GPU]** `k3_moe_layer.py`: subclass `MoELayer`, override
   `postprocess()` to insert `routed_expert_norm = RMSNorm(3584)` before
   `fc2_latent_proj` (finding A10). Shared experts at hidden width 7168 with
   intermediate 6144 (finding B6). `moe_shared_expert_overlap` must stay off —
   surface core's assertion rather than masking it.
3. **T6.3 [GPU]** `k3_router.py`: `QuantileBalancingRouter(TopKRouter)` — fp32
   sigmoid scores, bias added **for selection only**, weights gathered from the
   unbiased scores, renormalised, scaled by 1.0; plus the QB histogram estimator
   updating the bias.
4. **T6.4 [8-GPU]** EP path: 8-rank all-reduce/all-to-all exercise with
   `router_replay` determinism; report the load ratio (max/mean) as an
   observation, never as a gate.
5. **T6.5 [GPU]** Expert-parallel + grouped-GEMM smoke at tiny width with 8
   experts, then with `num_experts` chosen from the divisors of 896.

### Exit criteria

G23 (SiTU parity + fp32 contract), G24 (LatentMoE forward parity vs the release
math on fixed weights, including norm placement), G25 (**QB exact-update parity**
against the ported reference algorithm on fixed synthetic margins:
uniform / heavy-tail / ties), G26 (EP smoke + routing determinism).

### Risks / notes

- QB has no checkpoint anchor — it is a training-time behaviour from the report.
  Exact update parity against a reference implementation is the only real gate;
  the healthy-load-ratio band (1.5–2.5) is an observation to report.
- The estimator's bin width trades bias against responsiveness; document the
  error-vs-bin-width curve in the phase summary.

---

## P7 — Single-node trainer

### Tasks

1. **T7.1 [8-GPU]** `pretrain_kimi_k3.py` end to end: config build, model
   construction, schedule binding, forward step, loss.
2. **T7.2 [CPU-OK]** `tools/tokenizer.py` (tiktoken o200k-style, vocab 163840,
   `bos 163584 / eos 163586 / pad 163839`, XTML specials, media placeholder id).
3. **T7.3 [8-GPU]** 4 L official config on the **measured** optimizer recipe
   from G5, full recompute, seq ≤ 8 k. Documented fallbacks if it does not fit:
   seq 4 k → higher grad-accum → precision-aware Adam (with the reason written
   into the config header).
4. **T7.4 [8-GPU]** Memory report: measured peak and persistent bytes/param,
   activation memory with and without recompute, AttnRes payload contribution —
   into `results/opt_mem.md` (refresh) and `06-capacity-and-parallelism.md`.
5. **T7.5 [8-GPU]** Checkpoint save/load of the whole model (dist-checkpointing).

### Exit criteria

G27 (50 iterations, no NaN/Inf, loss decreasing), G28 (memory report committed;
the analytic model in `mem_budget.py` reproduces the measurement within 10 %),
G29 (save/load round-trip, resumed loss matches).

### Risks / notes

- This job is the **8-GPU nightly integration job**, not per-commit CI.
- If the 4 L config does not fit under any documented fallback, the phase closes
  with the measurement and a re-plan, not with a shrunken definition of success.

---

## P8 — Converter and checkpoint-anchored parity

### Tasks

1. **T8.1 [CPU-OK]** `tools/mapping.py` from the released index + modeling file.
   Rows carry expected shape and dtype. Invariants are explicit, never
   "if present": `A_log` per the P0 answer, `dt_bias == [12288]`, gate-slot
   order in `linear_fc1`, MXFP4 pack + e8m0 scale pairing, explicit vision skip
   rule with a reported count.
2. **T8.2 [GPU]** `tools/convert.py`: shard-wise, both directions,
   `dequantize_on_import` for MXFP4 experts.
3. **T8.3 [CPU-OK]** Synthetic tiny checkpoint round-trip in CI; 2–3 **named**
   real shards for the real-data path (ask before downloading).
4. **T8.4 [GPU]** Checkpoint-anchored parity on real layers 1–4
   (① KDA+dense ② KDA+MoE ③ KDA+MoE ④ MLA+MoE) against a **deliberately
   truncated** HF reference (same four layers + embedding + final norm + head).
   Router replay pinned, 16 fixed prompts, per-layer hidden-state statistics.

### Exit criteria

G30 (dry run validates the full index, zero unmapped tensors, invariants
asserted, vision keys counted), G31 (round-trip: bf16 exact; byte-exact packed
round-trip claimed **only** for an unchanged checkpoint with preserved scales —
after training or requantisation the criterion is dequantised-value
equivalence), G32 (per-layer parity within component tolerances; truncated-logit
match reported as a number, not a pass/fail), G33 (tokenizer round-trip on a
fixed corpus).

### Risks / notes

- A 4-layer slice cannot reproduce full-93 L logits and is not asked to.
- After `dequantize_on_import`, "matches the released model" means "matches the
  dequantised release" — say it in the report (finding C6).

---

## P9 — Training equivalence, per-head Muon, resume hardening

### Tasks

1. **T9.1 [8-GPU]** `tools/twin_run.py`: fixed tokenized corpus (hash
   committed), fixed GBS/MBS/accum, LR schedule, seeds; statistic = per-step
   loss delta + max |Δloss| over the window + final-window mean delta, compared
   against a **3-seed noise band measured first**.
2. **T9.2 [8-GPU]** Twin axes: eager-vs-`fla` KDA backend; recompute on/off;
   (optional) an NVIDIA-twin config export.
3. **T9.3 [GPU]** `optim/per_head_muon.py`: per-head momentum partition before
   Newton–Schulz for MLA/KDA projections (core's `is_qkv` path keys on
   `linear_qkv.weight` and carries an explicit `TODO: support MLA`, so the head
   split is ours — finding A7), plus the explicit K3 parameter-group policy
   (which tensors go to Muon vs the scalar optimizer).
4. **T9.4 [GPU]** Wire core `clip_qk` (`--qk-clip`, alpha, threshold) and record
   the max-attention-logit trace.
5. **T9.5 [8-GPU]** Save → resume bitwise-stable loss for 20 steps under `PP=2`
   with the packed payload.

### Exit criteria

G34 (noise band measured and committed; eager-vs-fla twin inside the band at
tiny and 4 L), G35 (single-step per-head Muon parity against a torch reference
on fixed gradients), G36 (`clip_qk` active and logged; no divergence), G37
(resume stability under PP).

### Risks / notes

- The Muon-vs-AdamW loss comparison is a **reported experiment with
  independently tuned LRs**, never an acceptance gate.
- 1 k-step official-width twins are scheduled/manual, not CI.

---

## P10 — MXFP4 / MXFP8 QAT **[core]**

### Tasks

1. **T10.1 [GPU]** `k3_qat.py`: MXFP4 pack/unpack with e8m0 scales (group 32)
   and MXFP8 activation quantisation matching AITER's `per_1x32`. Because TE
   ships no NumPy reference at the pin (finding D2), the oracle is **our own
   OCP-MX reference**, cross-checked for round-trip agreement against
   `MXFP4Quantizer`. RNE default; stochastic rounding behind an off-by-default
   flag.
2. **T10.2 [GPU]** `k3_qat_experts.py`: `KimiK3QATGroupedMLP` — AITER a8w4
   forward with the SiTU betas passed through; STE backward routing dgrad/wgrad
   to blockwise-FP8 / BF16 grouped GEMM against dequantised weights; fp32
   masters; packed-cache refresh on optimizer step (semantics proven in G8).
3. **T10.3 [GPU]** Checkpointing of packed data + scales + masters.
4. **T10.4 [8-GPU]** QAT-vs-BF16 twin per the P9 protocol (stable offset, no
   divergence trend) and QAT-forward-vs-serving logit comparison on a converted
   slice.

### Exit criteria

G38 (quantizer exactness vs our reference + TE round-trip cross-check),
G39 (a8w4 forward within MXFP8-round-trip tolerance of the dequantised matmul),
G40 (**STE gradients match the fake-quant reference exactly** — gradcheck is
invalid here by construction and must not appear), G41 (4 L QAT trains; offset
report + serve-parity statistics committed).

### Risks / notes

- If the AITER assets were absent at G8, this phase ships the BF16 fake-quant
  path only, the serve-parity criterion is deferred **in writing**, and the
  reason is recorded in the phase summary.
- QAT does not reduce training memory; MXFP4 shrinks serving only.

---

## P11 — Performance baseline and AttnRes/KDA optimisation

### Tasks

1. **T11.1 [8-GPU]** EP=8 proxy script at official widths with a reduced layer
   count and a calibrated `seq_length`; 10-iteration smoke; chrome trace over
   one steady iteration.
2. **T11.2 [8-GPU]** Baseline report (`develop/profile/profile-baseline-ep8-<date>.{md,html}`):
   cold/steady iteration time, GPU vs CPU active, top-N kernels, launch counts,
   attention vs MoE vs comm vs AttnRes split, ranked bottlenecks, **and the
   per-task improvement budgets** that the following tasks are measured against
   (R9.3 forensic attribution before any fix).
3. **T11.3 [GPU]** `block/attn_res_fused.py`: fuse the mixer so the
   `(T, K+1, H)` fp32 concat is never materialised — compute the RMS statistic
   and the score in one pass, then the weighted sum in a second, with an
   autograd-correct backward. Land behind `k3_attn_res_fused` (default off until
   its gate is green).
4. **T11.4 [8-GPU]** Whatever the ranked list says next (KDA chunk kernel,
   dispatcher, router) — subject to the 10 % de-scope rule (R9.4).
5. **T11.5 [8-GPU]** Post-phase trace + report + `perf/attn_res.md` and
   `perf/proxy_ep8.md` table updates.

### Exit criteria

G42 (proxy smoke + baseline trace + ranked report with budgets), G43 (fused
mixer fwd+bwd parity vs the eager mixer at fast and release tiers), G44 (measured
delta meets the budget from G42, actual numbers recorded), G45 (post-phase trace
+ report; no banned warnings; no memory regression).

### Risks / notes

- The AttnRes mixer is the *predictable* bottleneck (≈120 GB of reads per
  forward at production shape, architecture §6.1) — but the trace decides, not the
  prediction. If the trace disagrees, the phase rescopes and says so (the
  DeepSeek-V4 P29 precedent).
- Do not tune before attributing (R9.3), and do not chase a row worth < 10 %
  of steady iteration time (R9.4).

---

## P12 — Scale-out preparation **[CLUSTER]**

### Tasks

1. **T12.1** 93 L configurations parameterised by the **measured** optimizer
   recipe and the measured payload cost — node counts come from
   `results/opt_mem.md` + `tools/mem_budget.py`, never from an estimate.
2. **T12.2** PP layouts via explicit `pipeline_model_parallel_layout`, annotated
   with AttnRes block boundaries (multiples of 12) and the final double-MLA;
   a config test asserts every boundary is legal.
3. **T12.3** EP ladder configs (EP ∈ {8, 16, 28, 32, 56}) + a QB-load report
   template.
4. **T12.4** Dispatcher A/B matrix (stock all-to-all vs DeepEP port vs MoRI vs
   a MoonEP evaluation) as a plan, not an implementation.
5. **T12.5** Continued-pretrain-flatness evaluation scripts for the released
   checkpoint.

### Exit criteria

G46 (93 L configs validated analytically: meta-device construction, parameter
count, memory budget, layout legality), G47 (EP ladder + dispatcher A/B
templates + analysis notebook committed).

### Risks / notes

- The agent prepares; a human runs. No cluster job is launched from here (R10.1).
- KCP / 1 M context stays deferred.
