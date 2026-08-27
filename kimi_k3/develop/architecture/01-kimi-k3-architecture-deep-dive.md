# Kimi K3 — Architecture Ground Truth

> **Purpose:** one place that says exactly what the model *is*, so no phase
> re-derives it and no two documents disagree.
> **Evidence tiers (R3.4):** ✅ = read from the released artefacts
> (`HF moonshotai/Kimi-K3` `config.json` / `modeling_kimi_linear.py` /
> safetensors headers) · 📄 **[report]** = from arXiv 2607.24653, weaker ·
> ❓ = open, owned by a P0/P2 gate.
> Verified 2026-08-26 against the released `config.json` and modeling sources.

---

## 1. One paragraph

Kimi K3 is a 2.78 T-parameter (≈104 B active/token) hybrid-attention MoE
language model. Its backbone is 93 decoder layers in a repeating
`KDA → KDA → KDA → MLA` pattern (69 Kimi-Delta-Attention linear-attention
layers, 24 gated Multi-head-Latent-Attention layers), every layer wired into an
**Attention-Residuals (AttnRes)** stream instead of a plain residual, every
layer but the first using a **latent MoE** (896 routed experts, top-16, run at a
3584-wide latent) with SiTU-GLU experts. There is no positional encoding
anywhere (NoPE) — position information comes from KDA's recurrence and its short
convolutions. The released checkpoint is MXFP4-packed on routed experts only,
matching the disclosed QAT recipe. The release is multimodal
(`KimiK3ForConditionalGeneration` with a 27-layer vision tower); **v1 of this
integration trains the text tower only.**

---

## 2. Top-level config (✅ verbatim from `config.json`)

| Field | Value | Consequence for us |
|---|---|---|
| `model_type` (top) / `text_config.model_type` | `kimi_k3` / `kimi_linear` | Text config is **nested under `text_config`** — the converter must strip that level. |
| `architectures` | `KimiK3ForConditionalGeneration` | Checkpoint also holds `vision_tower.*` / `mm_projector.*`; converter must skip them **explicitly**, not "if present". |
| `hidden_size` | 7168 | |
| `num_hidden_layers` | 93 | 93 = 3 × 31; PP splits are never even (see `06-capacity-and-parallelism.md`). |
| `vocab_size` | 163840 | |
| `tie_word_embeddings` | **false** | Separate `lm_head`; 2.35 B embedding params. |
| `rms_norm_eps` | 1e-5 | But see §5 ❓ (MLA LoRA norms may not receive it). |
| `max_position_embeddings` | 1048576 | v1 trains ≤ 64 k; 1 M is descoped. |
| `num_nextn_predict_layers` | 0 | **No MTP.** |
| `attn_res_block_size` | **12** | The single most consequential number for pipeline design (§6). |
| `hidden_act` | `situ` | `activation_situ_beta=4.0`, `activation_situ_linear_beta=25.0` (§7). |
| `first_k_dense_replace` | 1 | Layer 1 is dense FFN with `intermediate_size=33792`. |
| `moe_layer_freq` | 1 | Every other layer is MoE. |
| `quantization_config` | `mxfp4-pack-quantized`, group 32, `scale_dtype uint8` (e8m0), symmetric, `ignore` = `self_attn`, `shared_experts`, `mlp.(gate\|up\|gate_up\|down)_proj`, `lm_head`, `vision_tower`, `mm_projector` | Routed experts only; everything else is bf16 in the release. |

### 2.1 Layer pattern (✅)

`text_config.linear_attn_config.full_attn_layers` =
`[4, 8, 12, …, 88, 92, 93]` — **24 entries, 1-indexed**, and `kda_layers` lists
the complementary 69. Note the tail: **92 and 93 are both MLA** (the `…, 88, 92, 93`
ending breaks the strict 3:1 stride once, at the very end).

```
1-indexed layer:  1   2   3   4   5   6   7   8  ...  88  89  90  91  92  93
kind:            KDA KDA KDA MLA KDA KDA KDA MLA ... MLA KDA KDA KDA MLA MLA
FFN:            dense MoE MoE MoE MoE MoE MoE MoE ... MoE MoE MoE MoE MoE MoE
```

Internally HF uses 0-indexed `layer_idx` and `config.is_kda_layer(i)` compares
`i + 1` against the lists. **Every off-by-one in this project traces back to
that convention** — the conformance test (G11) pins it.

---

## 3. KDA — Kimi Delta Attention (✅ verbatim)

```python
projection_size = head_dim * num_heads         # 128 * 96 = 12288
q_proj, k_proj, v_proj : Linear(7168, 12288, bias=False)
q_conv1d, k_conv1d, v_conv1d : ShortConvolution(12288, kernel_size=4, activation='silu')
A_log   : Parameter(log(uniform(1, 16)))       # declared [num_heads] = [96]  (see ❓ below)
f_a_proj: Linear(7168, 128)   f_b_proj: Linear(128, 12288)     # low-rank DECAY gate
dt_bias : Parameter(empty(12288), dtype=fp32)
b_proj  : Linear(7168, 96)                                      # beta, cast to fp32
g_proj  : Linear(7168, 12288)                                   # FULL-RANK OUTPUT gate (use_full_rank_gate=true)
o_norm  : FusedRMSNormGated(128, eps=rms_norm_eps, activation='sigmoid')
o_proj  : Linear(12288, 7168)
```

forward, in order:
1. `q,k,v = {q,k,v}_proj(x)` → each through its own SiLU short conv (k=4).
2. `g = f_b_proj(f_a_proj(x))` reshaped to `[..., 96, 128]` — the **decay** gate.
3. `beta = b_proj(x).float()` — `[..., 96]`.
4. the kernel call (**this exact kwarg set is the contract**):

```python
o, recurrent_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=self.A_log, dt_bias=self.dt_bias,
    initial_state=recurrent_state, output_final_state=True,
    use_qk_l2norm_in_kernel=True,      # <-- q/k L2 norm happens INSIDE the kernel
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,   # <-- sigmoid(beta) inside
    safe_gate=self.gate_lower_bound is not None,
    lower_bound=self.gate_lower_bound, # -5.0
    transpose_state_layout=True,
    cu_seqlens=cu_seqlens,
)
```

5. output gate: `g_out = g_proj(x)` → `[..., 96, 128]`; `o = o_norm(o, g_out)`
   (RMSNorm over `head_dim`, multiplied by `sigmoid(g_out)`).
6. `o = o_proj(o.flatten(-2))`.

**Two distinct gates.** `f_a/f_b` (rank 128) feeds the *in-kernel decay*;
`g_proj` (full rank) is the *output* gate consumed by `o_norm`. Any document or
test that says "the KDA gate" without saying which one is ambiguous and wrong.

**❓ open item — `A_log` shape.** The modeling code declares `[num_heads] = [96]`;
the incoming plan reports the released first-shard header as `[128]` with the
last 32 values zero. Both can hold if the checkpoint pads for kernel alignment.
**G2 resolves this from the actual shard header** and the converter asserts the
outcome (trim `[128]→[96]` on import with `A_log[96:] == 0` asserted, zero-pad on
export) instead of guessing. `dt_bias` is `[12288]` in both readings.

`ShortConvolution` (from `fla`) applies `silu` after a causal depthwise conv of
width 4 over the *projected* q/k/v — not over the hidden state.

---

## 4. Gated MLA with NoPE (✅ verbatim)

```python
q_a_proj  : Linear(7168, 1536)        q_a_layernorm : KimiRMSNorm(1536)   # eps NOT passed ❓
q_b_proj  : Linear(1536, 96*192)      # q_head_dim = qk_nope(128) + qk_rope(64) = 192
kv_a_proj_with_mqa : Linear(7168, 512 + 64)
kv_a_layernorm     : KimiRMSNorm(512)                                     # eps NOT passed ❓
kv_b_proj : Linear(512, 96*(128+128))
o_proj    : Linear(96*128, 7168)
g_proj    : Linear(7168, 96*128)      # mla_use_output_gate = true
scaling   = q_head_dim ** -0.5        # 192**-0.5, NOT 128**-0.5
assert self.use_nope ; self.rotary_emb = None
```

- The 64 "rope" dims exist but **are never rotated**. `k_rot` is produced once
  (MQA-style, one head) and `expand`ed across all 96 heads, then concatenated to
  the nope part. It is extra shared content, not position.
- Output gate: `attn_output = attn_output * g_proj(x).sigmoid()` applied
  **before** `o_proj`, on the flattened `[B, S, 96*128]` layout — i.e. full-rank
  and elementwise, not per-head-scalar.
- The FA2 path pads V from 128 → 192 and slices the output back — the same
  head-dim asymmetry DeepSeek MLA has, so TE/AITER fused attention should accept
  it, but the combination is a P4 gate (G17), not an assumption.
- 📄 **[report]** attention output is kept in FP32 during training.

---

## 5. Norms

Every layer: `input_layernorm`, `post_attention_layernorm`, plus the two AttnRes
norms (§6). All `KimiRMSNorm` with `eps = rms_norm_eps = 1e-5` **except**
`q_a_layernorm` / `kv_a_layernorm`, which are constructed **without** an `eps`
argument and therefore take `KimiRMSNorm`'s class default.
**❓ G12 reads that default out of the released modeling file and, if it differs
from 1e-5, the K3 config carries a separate `mla_lora_norm_eps` field.** This is
a classic silent-parity-drift trap: a 1e-6 vs 1e-5 epsilon on a 1536-wide norm
moves logits far more than any kernel tolerance we set.

---

## 6. AttnRes — Attention Residuals (✅ verbatim; the design driver)

State carried through the whole decoder is **two** tensors:

- `prefix_sum` — `[B, S, H]`, the running sum inside the current block;
- `block_residual` — `[B*S, num_blocks, H]`, one frozen slot per completed block,
  starting empty (`new_zeros(B*S, 0, H)`).

The mixer:

```python
def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    # prefix_sum: (T, H)   block_residual: (T, K, H)
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)   # (T, K+1, H)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()  # (H,)
    scores = (k * score_weight).sum(-1)          # (T, K+1)
    probs = scores.softmax(-1).unsqueeze(1)      # (T, 1, K+1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)
```

Per decoder layer (`_forward_attn_residual`):

```
prefix_sum = hidden_states
if block_residual has slots:  hidden = _apply_attn_res(prefix_sum, block_residual,
                                                       self_attention_res_proj, self_attention_res_norm)
if layer_idx % 12 == 0:       block_residual = cat([block_residual, prefix_sum.unsqueeze(1)]);  prefix_sum = None
hidden = input_layernorm(hidden);  hidden = self_attn(hidden)
prefix_sum = (prefix_sum + hidden) if prefix_sum is not None else hidden
hidden = _apply_attn_res(prefix_sum, block_residual, mlp_res_proj, mlp_res_norm)
hidden = post_attention_layernorm(hidden);  hidden = moe_or_mlp(hidden)
prefix_sum = prefix_sum + hidden
return prefix_sum, block_residual
```

and after the last layer, once more at model level with
`output_attn_res_norm` / `output_attn_res_proj`.

Per-layer parameters: 2 × `RMSNorm(7168)` + 2 × `Linear(7168, 1)` = 28 672
(negligible in count, decisive in behaviour).

### 6.1 Facts that fall out of this code

1. **Slot creation is at 0-indexed `layer_idx % 12 == 0`** → slots after layers
   0, 12, 24, 36, 48, 60, 72, 84 → **8 slots total**; the trailing block is 9
   layers (93 = 7×12 + 9).
2. **`prefix_sum` resets at each block boundary** (set to `None`, then to the
   attention output) — the residual stream restarts per block; the old stream is
   frozen into a slot.
3. `block_residual` **grows monotonically**, so the tensor crossing a pipeline
   stage boundary has a *stage-dependent* shape.
4. The mixer runs in **fp32** and materialises `(T, K+1, H)` fp32 twice per
   layer. At `T = 8192`, `K+1 = 9`, `H = 7168` that is **2.1 GB of fp32
   temporaries per mix**, ~120 GB of reads per forward across the whole model at
   the average `K+1 ≈ 5.5`. This is the single largest non-GEMM cost in the model
   and the reason plan-0 has a dedicated fused-mixer phase (P11).
5. Backward needs every slot → slots are activations, not constants.

---

## 7. LatentMoE + router + SiTU (✅ verbatim)

```python
gate(hidden)                            # router runs on the 7168-wide hidden, BEFORE the down-proj
h = routed_expert_down_proj(hidden)     # 7168 -> 3584
y = experts(h)                          # 896 experts, top-16, each 3584 -> 3072 -> 3584
y = routed_expert_norm(y)               # RMSNorm(3584, eps=rms_norm_eps)   <-- AFTER combine, BEFORE up-proj
y = routed_expert_up_proj(y)            # 3584 -> 7168
y = y + shared_experts(hidden)          # shared experts run on 7168 with intermediate 2*3072 = 6144
```

Router (`KimiMoEGate`):

```python
logits = F.linear(hidden.float(), weight.float())     # weight: [896, 7168], fp32
scores = logits.sigmoid()
scores_for_choice = scores + e_score_correction_bias  # noaux_tc bias, choice only
topk_idx = topk(scores_for_choice, 16)                # num_expert_group = topk_group = 1 -> no grouping
topk_weight = scores.gather(1, topk_idx)              # weights come from the UNBIASED scores
topk_weight /= topk_weight.sum(-1, keepdim=True) + 1e-20      # moe_renormalize = true
topk_weight *= routed_scaling_factor                  # 1.0
```

Expert / MLP body — SiTU-GLU:

```python
gate, up = w1(x), w3(x)                       # separate w1 (gate) / w3 (up), fp32 activation math
situ_a = beta * tanh(gate / beta) * sigmoid(gate)      # beta = 4.0
up     = linear_beta * tanh(up / linear_beta)         # linear_beta = 25.0
y      = w2((situ_a * up).to(x.dtype))
```

SiTU is a **soft-capped** SwiGLU: both branches are `tanh`-limited, which is why
the release ships no Hadamard rotation for QAT — SiTU is the outlier control.

📄 **[report]** training additionally uses **Quantile Balancing** on the router
(a balancing update on the `e_score_correction_bias`) and **MoonEP** for
perfectly balanced expert parallelism. Neither is visible in the inference
modeling file; QB is implemented from the report and validated against a ported
reference implementation (G25), not against the checkpoint.

---

## 8. Parameter budget (derived from §2–§7; the analytic oracle for G13)

Per-layer, `H=7168`, KDA/MLA projection width `P=12288`, latent `L=3584`,
expert intermediate `I=3072`, `E=896`:

| Component | Formula | Params |
|---|---|---:|
| KDA layer | 5·H·P + H·128 + 128·P + H·96 + 3·P·4 + P + 96 | **443,740,768** |
| MLA layer | H·1536 + 1536·(96·192) + H·576 + 512·(96·256) + P·H + H·P + 2048 | **232,196,096** |
| MoE layer | E·3·L·I + 3·H·6144 + 2·H·L + E·H + L | **29,784,935,936** |
| Dense FFN (layer 1) | 3·H·33792 | **726,663,168** |
| AttnRes (per layer) | 2·H + 2·H | 28,672 |
| Embedding + lm_head (untied) | 2·163840·H | **2,348,810,240** |

Totals:

| Group | Count | Params |
|---|---:|---:|
| KDA layers | 69 | 30.618 B |
| MLA layers | 24 | 5.573 B |
| MoE layers | 92 | 2 740.214 B |
| Dense FFN | 1 | 0.727 B |
| Embeddings + head | — | 2.349 B |
| Norms + AttnRes | — | ≈0.004 B |
| **Total** | | **≈ 2.779 T** ✅ matches the 2.78 T headline |
| **Active / token** (16/896 experts) | | **≈ 104 B** (derived, not from the report) |

`kimi_k3/tools/mem_budget.py` implements this table; G13 asserts the meta-device
parameter count of the 93 L preset against it within 1 %.

---

## 9. Tokenizer + special tokens (✅)

tiktoken o200k-style, `vocab_size = 163840`, `bos = 163584`, `eos = 163586`,
`pad = 163839`, `media_placeholder_token_id = 163605`, XTML specials
`<|open|> <|close|> <|sep|> <|end_of_msg|>`, `transformers` pinned at 4.56.2.

---

## 10. Quantization in the release (✅) vs QAT in training (📄 [report])

The shipped checkpoint is `compressed-tensors` `mxfp4-pack-quantized`: group 32,
uint8 (e8m0) scales, symmetric, **routed experts only**. 96 shards ≈ 1.56 TB.
The training recipe is **W-MXFP4 / A-MXFP8 QAT with an STE backward in high
precision**; no Hadamard; stochastic rounding unstated (we default to RNE and
keep SR behind an off-by-default flag). MXFP4 shrinks *serving*, not training —
QAT does not reduce training memory.

---

## 11. What v1 deliberately does not implement

MTP (`num_nextn_predict_layers = 0` anyway) · the vision tower / `mm_projector`
· KCP and 1 M context · a8w4 backward (dgrad has no transposed-scale FP4 weight
copy; wgrad has no weight operand) · dense-layer a8w4 on gfx950 (AITER's dense
a8w4 GEMM is gfx1250-only) · interleaved pipelining (VPP) with AttnRes
(core rejects custom tensor shapes under VPP — see `01-roadmap.md` risk table).
