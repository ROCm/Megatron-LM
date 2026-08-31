# G26 (second half) — does `router_replay` pin routing at bf16 + EP?

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.router_replay_probe --preset 4L --ep 8 --seq 256`
> Raw: `results/raw/router_replay_raw.jsonl`. 4 L official, EP=8, bf16, `eval()` + `no_grad`,
> `moe_enable_routing_replay=True`. 3 replay instances for 3 MoE layers, as expected.

## The question

`results/ep_smoke.md` measured two identical forwards differing by max-abs
0.53–0.79 on logits of std 1.69, with **4.0 % of tokens routed to different
experts**. Finding **A18** located the source *inside* the MoE accumulation —
the first MoE router is bitwise identical on every rank, so the noise enters after
it and compounds through routing.

`router_replay` is core's mechanism for pinning routing: RECORD the top-k indices
on one forward, REPLAY_FORWARD them on the next. Three forwards on identical
input answer whether that removes the divergence.

## Result

| comparison | max-abs | rel-L2 | argmax agreement | bitwise |
|---|---:|---:|---:|---|
| plain repeat (A vs B) | 0.7051 | 0.01840 | 0.9883 | no |
| **replayed (A vs C)** | **0.0625** | **0.00498** | 0.9883 | **no** |
| improvement | **11.3x** | **3.7x** | none | — |

**Routing was the dominant amplifier, and replay pins it — but the model is still
not reproducible.** Max-abs falls 11-fold and rel-L2 3.7-fold, which confirms the
4.0 % re-routing was carrying most of the divergence. What remains, rel-L2
**5.0e-03**, is the accumulation noise A18 found, reaching the output by paths
replay does not cover: replay fixes *which experts a token goes to*, not the order
in which their contributions are summed.

## The number that did not move is the interesting one

**argmax agreement is 0.9883 in both** — 253 of 256 tokens, the same three
flipping either way. Replay cuts the L2 divergence by 3.7x and changes top-1
agreement not at all.

That says the tokens which flip are near-ties in the *output distribution*, not
victims of re-routing. Cutting routing noise does not help them, because their
margin was never routing-sized to begin with. A determinism claim resting on
argmax agreement would have looked identical before and after — and would have
concluded, wrongly, that replay did nothing.

## What this licenses

* **`router_replay` works and does what it says.** Use it when routing decisions
  must be identical across runs — A/B comparisons, debugging a divergence,
  isolating a routing change from a kernel change.
* **It is not a determinism switch.** It does not make bf16 + EP bitwise, and
  nothing in this project's exact-equality gates can move to it. G21's PP parity,
  G37's resume and G43's fused mixer remain valid only because they run in fp32,
  at tiny, or with expert bias off — that constraint (A18) is unchanged.
* **The residual has a floor worth quoting**: rel-L2 5.0e-03 at 4 L. It should
  grow with depth, since A18 showed the effect compounds layer over layer
  (0 → 4–8 → 15–29 tokens re-routed across three MoE layers).

## Scope

One geometry, forward only, eval mode. `REPLAY_BACKWARD` — which pins routing for
the recompute pass — is not exercised here, and matters for anyone wanting
reproducible *training* rather than reproducible inference.
