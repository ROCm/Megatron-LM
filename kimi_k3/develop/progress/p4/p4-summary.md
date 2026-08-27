# P4 — Gated MLA with NoPE (COMPLETE)

## 1. Objective

The other half of K3's attention: multi-head latent attention with no positional
encoding, a `192 ** -0.5` scale, a full-rank sigmoid output gate, and LoRA norms
at eps 1e-6.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/attention/gated_mla_eager_fp32.py` | the FP32 oracle, transcribed from `KimiMLAAttention.forward` |
| `kimi_k3/attention/gated_mla.py` | `K3GatedMLA` (fused or eager) + `K3GatedMLASelfAttention` |
| `kimi_k3/specs/layer_specs.py` | MLA layers now build real gated MLA |
| `kimi_k3/tests/test_k3_p4_gated_mla.py` | 11 tests |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G17** — released math | **GREEN** | fused vs eager oracle 1.5e-7; parameter count equals the analytic model exactly; scale, gate placement, NoPE sharing and LoRA eps each pinned separately |
| **G18** — fused backend | **GREEN** | `scaled_dot_product_attention` runs fwd+bwd at the real 192/128 head-dim asymmetry in bf16, with finite non-zero gradients; `clip_qk` hook present for core's walker |

## 4. Why this is a module, not a subclass

The plan proposed subclassing core's `MLASelfAttention` and overriding the rotary
path. Once rotary is removed, the softmax scale changes and a gate is inserted
before the output projection, almost nothing of core's forward survives — and
finding A9 established that core supports no NoPE mode at all (`rope_type` takes
only `"rope"` and `"yarn"`). What is reused is the **shape contract**, so the
converter stays a rename: `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `g_proj`, `o_proj`.

The 192/128 asymmetry is handled the way the release handles it for
FlashAttention: pad V to `q_head_dim`, slice the output back.

## 5. Four silent-failure modes, four tests

None of these would crash. Each would train.

| what | how it is pinned |
|---|---|
| scale is `192**-0.5`, not `128**-0.5` | asserted against both values, at the official preset |
| the output gate is applied, full-rank, **before** `o_proj` | zeroing `g_proj` makes sigmoid exactly 0.5, so a gate before a linear map must scale its output by exactly 0.5 — asserted |
| NoPE: `k_rot` is shared across heads and never rotated | every head's 64 rope dims are asserted equal, and no rotary module exists |
| LoRA norms use 1e-6 | changing the epsilon must change the output, so the norms are demonstrably applied, and the config default is asserted |

## 6. Hand-off to P5

Both attention kinds are real now, so `specs/layer_specs.py` is final apart from
the MoE half. P5 makes the AttnRes layer placement faithful — the P0 block wraps
a stock `TransformerLayer` rather than splitting the two mixes around the
attention and MLP halves.

One thing P5 should decide: `AttnResMixer` initialises its projection to zero, so
the mix starts uniform. That is a reasonable init, but it also means the norm
gain receives exactly zero gradient at step 0 (the score is identically zero
regardless of it). Harmless — the gain starts learning as soon as the projection
moves — but it looks alarming in a "which parameters have no gradient" check, so
it should be stated rather than rediscovered.

## 7. Commit chain

_pending._
