# G65 — which shapes Muon applies to, and one it should not have

> `torchrun --standalone --nproc_per_node=8 -m kimi_k3.tools.muon_shapes --preset 4L --ep 8`
> Raw: `results/raw/muon_shapes_4L.json`. Per rank, 4 L official, EP=8.

Muon's rule is `megatron/core/optimizer/muon.py:295-300`: a parameter is
orthogonalised iff it is **2-D** and not flagged
`is_embedding_or_output_parameter`. Everything else goes to the nonlinear (Adam)
group.

## The groups

**Muon: 719 tensors, 13.957 B params.** 672 of them -- 93% -- are expert weights:

| shape | count | role |
|---|---|---|
| (6144, 3584) | 336 | expert `linear_fc1` (gate+up) |
| (3584, 3072) | 336 | expert `linear_fc2` (down) |
| (67584, 7168), (7168, 33792) | 1 each | dense layer 1 MLP |
| (12288, 7168) x5, (7168, 12288) | 3 each | KDA `q/k/v/g/o_proj` |
| (12288, 7168), (7168, 6144) | 3 each | shared-expert fc1/fc2 |
| (7168, 12288), (12288, 7168) | 1 each | MLA `o_proj`, `g_proj` |
| (3584, 7168), (7168, 3584) | 3 each | MoE `fc_latent_proj` |
| (18432, 1536), (24576, 512), (1536, 7168), (576, 7168) | 1 each | MLA `q_b`, `kv_b`, `q_a`, `kv_a` |
| (896, 7168) | 3 | router |
| (12288, 128), (128, 7168), (96, 7168) | 3 each | KDA `f_b`, `f_a`, `b_proj` |

336 = 112 experts/rank x 3 MoE layers. That is the whole Newton-Schulz cost:
719 matrices x 5 NS steps x 3 GEMMs = **10,785 GEMM calls per step**, matching the
10,210 counted in the trace.

**Nonlinear (Adam): 52 tensors, 2.349 B.** Embedding and output layer (both
`(163840, 7168)`, flagged), the KDA convs `(12288, 1, 4)` (3-D), `dt_bias`, all
`(7168,)` norms -- and now the AttnRes pseudo-queries.

## The AttnRes pseudo-query was in the wrong group

`AttnResMixer.proj` was stored `[1, H]`, mirroring the checkpoint's tensor layout.
That made it 2-D, so `len(param.shape) == 2` put it in Muon: 9 tensors per rank at
4 L, **187 at 93 layers**.

Newton-Schulz on a rank-1 tensor is row normalisation, measured directly:

| input row norm | output row norm | cosine(in, out) |
|---|---|---|
| 0.85 | **0.9783** | 1.000000 |
| 84.58 | **0.9783** | 1.000000 |
| 8430.50 | **0.9783** | 1.000000 |

Magnitude discarded, direction preserved, at any input scale. That is Muon
behaving *consistently* -- it replaces an update with its orthogonal polar factor,
and a rank-1 tensor has one singular value -- but it is not what the parameter
wants.

**The report is explicit.** S2.2: *"a layer-specific learnable pseudo-query
`q_l = w_l in R^d`"* -- a vector. S2.5: *"Kimi K3 adopts Muon as the optimizer for
its **matrix** parameters"*. A vector pseudo-query is therefore not a Muon
parameter. Our `proj` **is** that pseudo-query: `score_vector = norm.weight *
proj`, matching `phi(q,k) = exp(q^T RMSNorm(k))`.

So this is a **spec-compliance fix, not a tuning choice**, which matters because
the effect is unmeasurable here: 9 rank-1 vectors against a twin-harness noise band
of ~17% on the late loss would return "not significant" and mean nothing. The
report decides it; no convergence run is needed or would help.

Fixed by storing the pseudo-query as `[H]`, the shape the report gives it. The
converter squeezes the released `[1, H]` on the way in and restores it on the way
out -- symmetric, exactly as `pad_a_log` mirrors `trim_a_log`. The round-trip gate
caught the missing reverse transform immediately: every value equal, shape lost.

Result: Muon 728 -> **719** tensors, nonlinear 43 -> **52**. `attn_res_*.proj` now
sits at `(7168,)` beside its gain partner `attn_res_*.weight`, which is what the
report describes.

## Also confirmed by the report

S2.5 applies the per-head refinement to **attention projections (Q, K, V)**:
momentum partitioned along the head dimension, orthogonalised per head. That
matches `optim/per_head_muon.py`, whose `SPLIT_KINDS` was restricted to KDA in A20
on cost grounds -- the report's rationale ("heads with larger gradient scales
dominate the shared update direction") is about Q/K/V generally, so whether MLA
should also be split is a live question A20 closed on measurement, not on the
report.

## Still open

The report says nothing about `k3_situ_*`-style vector parameters elsewhere, and
nothing about the router `(896, 7168)`, which is a genuine matrix and stays in
Muon. No other 2-D-but-not-a-matrix parameter appears in the enumeration above.
