# G32 (full) — four-layer anchored parity against the release

> Real released weights, 49.25 GiB across shards 1–4, fetched with the user's
> explicit approval (the plan caps unattended downloads at 10 GB) and **deleted
> after validation**. Scripts kept; weights not.

## Why a four-layer slice

Every other parity test in this tree compares our code against **our own** eager
oracle — and those oracles were written by reading the release. A misreading is
invisible to them, because the oracle encodes it too. Only the release's own module
on the release's own weights can catch "we built a coherent model that is not
Kimi K3".

Four layers contain every kind exactly once:

| layer | attention | FFN | AttnRes |
|---|---|---|---|
| 0 | KDA | **dense** (`first_k_dense_replace = 1`) | **creates the slot** |
| 1, 2 | KDA | MoE | mixes |
| 3 | **gated MLA** (1-indexed 4, the first full-attention layer) | MoE | mixes |

Before this, only **one KDA layer** had ever been anchored. Gated MLA, AttnRes and
the routing path had none.

## Results

| component | weights | parameters | rel-L2 | max-abs | cosine |
|---|---|---|---|---|---|
| **gated MLA** (`te` backend) | layer 3 | 232,196,096 = ours | **5.82e-03** | 2.20e-03 | **0.999983** |
| **gated MLA** (`sdpa`, release workaround) | layer 3 | 232,196,096 = ours | 5.82e-03 | 2.20e-03 | 0.999983 |
| **KDA** | layer 1 | 443,740,384 = ours | **7.84e-03** | 1.53e-05 | 0.999969 |
| **KDA** | layer 2 | 443,740,384 = ours | **7.41e-03** | 3.05e-05 | 0.999973 |
| **AttnRes**, attention site | layer 0 | — | **0.00e+00** | 0.00e+00 | **1.000000** |
| **AttnRes**, MLP site | layer 0 | — | **0.00e+00** | 0.00e+00 | **1.000000** |
| **routing** | layer 1 gate + bias | — | top-16 sets **identical for 100 % of tokens**; gathered weights 7.04e-08 | | |

Every module loaded with **zero missing and zero unexpected** keys in both
implementations, and parameter counts match the release exactly. The bf16 figures
sit at the measured dtype floor (~4e-3, `results/kda_parity.md`); AttnRes is
**bit-identical** because that path runs in fp32.

The routing check used the **real** `e_score_correction_bias` (nonzero, max 0.0844),
so the bias-for-selection-only rule — bias steers the top-k, weights come from the
*unbiased* scores — is confirmed against real values rather than a transcription.

Shapes confirmed on real tensors: `routed_expert_down_proj [3584, 7168]`,
`routed_expert_up_proj [7168, 3584]`, `routed_expert_norm [3584]` (the norm core's
`MoELayer.postprocess` lacks), `shared_experts.gate_proj [6144, 7168]`.

## The one thing this caught, and it was the harness

The first MLA run gave **rel-L2 1.52, cosine 0.615** — a structural disagreement.
It was not the model. The release passes `attention_mask` straight to
`eager_attention_forward`, so `attention_mask=None` means **bidirectional**
attention there, while our `_sdpa` forces `is_causal=True`. Supplying a causal
mask took it to 5.82e-03.

Worth recording because the failure was loud and the cause was mundane. A 2x
difference in output std was the tell — a wrong scale or a missing gate would look
like that too, and the gate path was verified identical by reading
(`g_proj(hidden).sigmoid()` applied **before** `o_proj`) before the mask was
suspected.

## What is not covered

The routed expert FFN forward. Dequantised experts are ~59 GB per layer, so the
release block and ours cannot both be resident on one GPU, and top-16-of-896
routing means a subset of experts gives a wrong answer rather than a partial one.
The evidence for that path is independent and strong: P8 accounted for all 497,220
released tensors with zero unmapped, and P10 showed our MXFP4 quantiser is
**byte-identical** to `aiter.per_1x32_f4_quant`. It is anchored transitively, not
directly.

## Re-run 2026-08-31 after the TE attention swap

The MLA attention operator moved from `scaled_dot_product_attention` with V padded
to 192 (the release's own workaround) to TransformerEngine's `DotProductAttention`
with `kv_channels=(192, 128)`, which handles the asymmetric head dims natively.

Re-anchored against the release's own module on real layer-3 weights, the two
backends are **indistinguishable**: rel-L2 5.8240e-03 (`te`) against 5.8230e-03
(`sdpa`), differing from each other by 1e-6. The swap is numerically free.

    attention operator, seq 8192   6.459 ms -> 2.207 ms   2.93x, 29.1% -> 70.9% of peak
    full MLA layer,     seq 8192  11.62  ms -> 7.00  ms   1.66x

Only layer 3's MLA tensors were re-fetched for this (0.43 GiB), not the 49 GiB slice.
