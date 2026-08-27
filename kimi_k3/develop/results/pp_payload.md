# G7 — AttnRes payload across a pipeline boundary

> Reproduce: `kimi_k3/develop/progress/p0/run_g7.sh` (2 GPUs).
> Tooling: `kimi_k3/tools/pp_payload_probe.py`. Raw rows: `pp_payload_raw.jsonl`.
> Model: `tiny` preset (4 layers, `[K,K,K,M]`, `attn_res_block_size = 2`),
> fp32, TP=1, PP=2, one microbatch.

## Result

| run | loss vs reference | worst gradient diff | verdict |
|---|---:|---:|---|
| reference, PP=1, repeated once | 0.0 | 0.0 | deterministic |
| **PP=2, packed payload** | **0.0** | **0.0** across all 131 parameters | **MATCH** |
| PP=2, slots detached (negative control) | **0.0** | **8.1e-4** | **MISMATCH** |

The pipelined run is **bitwise identical** to the single-stage run — loss and
every parameter gradient — so the packed payload carries both the state and its
gradient across the boundary.

## Why the negative control is the point

Detaching the block-residual slots at creation emulates exactly what a two-tensor
payload would do: the slots travel forward, the next stage consumes them, and
their gradient never comes back (core's `backward_step` back-props only
`output_tensor[0]`, schedules.py:451-493).

Its signature here is worth stating plainly:

* **the loss is identical to seven decimal places** — the forward is untouched;
* **the gradients are wrong by 8.1e-4**.

A loss-only comparison would have called this run correct. That is review finding
A1 reproduced on demand, and it is why G20 asserts gradient flow rather than loss
agreement.

## Shapes core was told

| rank | layers (0-idx) | recv × | send × | hook bound |
|---:|---|---:|---:|---|
| 0 | 0–1 | 1 | 2 | yes |
| 1 | 2–3 | 2 | 3 | yes |

Neighbour-consistent by construction (`1 + ceil((last+1)/block)`), supplied
through core's own `adjust_tensor_shapes_fn` — no schedule was re-implemented,
and the hook is bound only because PP > 1.

## Two configuration traps found while making this deterministic

Both would have produced a *permissive* gate rather than a wrong answer, which is
worse — the measured floor would have hidden a real defect.

1. **Dropout defaults to 0.1.** With it on, two identical single-rank runs differ
   by ~1.3e-4, so a floor measured that way is drift, not kernel noise. The probe
   sets `hidden_dropout = attention_dropout = 0`.
2. **The router's expert-bias accumulator mutates during forward.** Same effect,
   same fix (`moe_router_enable_expert_bias=False`) — routing behaviour belongs to
   P6, not to a transport gate.

With both off, the model is bitwise reproducible and the gate can demand exact
equality instead of a tolerance.

## Scope

The block used here is transport-faithful and layer-approximate: it carries,
packs, unpacks and mixes exactly as specified, but wraps a stock
`TransformerLayer` instead of splitting the two mixes around the attention and
MLP halves. P5's `K3TransformerLayer` makes the placement faithful; nothing in
this gate depends on it, and **no numerical-parity gate may cite this block**.
