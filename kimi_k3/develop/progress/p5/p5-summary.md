# P5 — AttnRes layer and block (COMPLETE)

## 1. Objective

Put the two attention-residual mixes where the release puts them — around the
attention and MLP halves of each layer — and make activation recompute carry both
state tensors.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/block/k3_transformer_layer.py` | `K3TransformerLayer`: the release's exact sequence, replacing both residual adds |
| `kimi_k3/block/k3_transformer_block.py` | owns the state and the model-level mix; recompute over `(prefix_sum, block_residual)` |
| `kimi_k3/specs/layer_specs.py` | layers are `K3TransformerLayer` |
| `kimi_k3/config/presets.py` | tiny preset sets dropout to 0 |
| `kimi_k3/tests/test_k3_p5_attn_res_layer.py` | 10 tests |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G19** — mixer | **GREEN** (P0) | bit-for-bit against the released `_apply_attn_res`, fp64 gradcheck |
| **G20** — payload gradients | **GREEN** (G7) | bitwise PP=2 parity; the negative control gives an identical loss with gradients wrong by 8.1e-4 |
| **G21** — PP parity | **GREEN** (G7) | loss 0.0 and gradients 0.0 across all 131 parameters |
| **G22** — placement + recompute | **GREEN** | every layer matches a verbatim transcription of `_forward_attn_residual`; recompute is numerically invisible forward and backward |

## 4. The test that matters

`released_forward_attn_residual` is a **verbatim transcription** of
`KimiDecoderLayer._forward_attn_residual`, written in the release's batch-first
`[B, S, H]` terms with its own flat `(T, K, H)` reshapes, driven with our layer's
own weights. Our sequence-first layer has to *agree* with it, not merely resemble
it. All four tiny-preset layers match.

Weaker checks would have passed with the mixes in the wrong place, which is the
entire failure this phase exists to exclude — so the mixer projections are
perturbed away from their zero init first, because a uniform mix hides placement
errors by construction.

Three further properties are pinned separately, because each is load-bearing and
none is visible in a loss curve: a slot is appended **only** at block boundaries;
the prefix sum **restarts** there (a boundary layer starts its stream from its
attention output, not its input); and the appended slot is the prefix **before**
this layer's attention.

## 5. Recompute had to be written, not inherited

Core's `_checkpointed_forward` assumes a single hidden-state tensor. Ours carries
two, and dropping the block residual from the saved inputs would silently
recompute it from the wrong thing. Since AttnRes is recompute-*mandatory* at
production width — keeping the mixer's fp32 stacks would cost ≈236 GB per
microbatch (`develop/results/attn_res.md`) — this path is not optional, so it is
implemented and gated: output within 1e-6 and every parameter gradient within
1e-4 of the non-recomputed run.

## 6. A trap that came back

The G7 report recorded that Megatron defaults `hidden_dropout` and
`attention_dropout` to 0.1, so two "identical" forwards differ by ~1e-4. It cost
a test here before the note was remembered. The tiny preset now sets both to 0,
which is what a test geometry should do — so the trap is disarmed for every
future determinism check rather than re-diagnosed each time.

## 7. Hand-off to P6

The decoder is now structurally complete: real KDA, real gated MLA, real AttnRes
placement, recompute. P6 is the last core piece — LatentMoE with the norm core
lacks, the quantile-balancing router, and SiTU experts.

## 8. Commit chain

| commit | scope |
| --- | --- |
| **f405615d2** | P5 T5.3/T5.4/T5.6: faithful layer placement, recompute, behavioural gates |

**P5 closes here.** Next: P6 (LatentMoE, quantile-balancing router, SiTU experts).
