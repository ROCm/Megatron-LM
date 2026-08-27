# P11 — Performance baseline and AttnRes optimisation (PARTIAL, rescoped)

## 1. Objective

Find out where the time goes, then fix the top row. The first half happened; the
second half is not authorised by what the first half found.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/tools/proxy_ep8.py` | the proxy, its trace, and the ranked report |
| `kimi_k3/block/attn_res.py` | `attn_res_mix_fused`, and the mixer module's `fused` path |
| `kimi_k3/config/{k3_transformer_config,arguments}.py` | `--k3-attn-res-fused`, `--k3-attn-res-chunk` |
| `kimi_k3/tests/test_k3_p11_fused_attn_res.py` | 12 tests |
| `kimi_k3/develop/profile/profile-baseline-2026-08-27.md` | the report and the budgets |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G42** — proxy, trace, ranked report | **PARTIAL** | ran at official widths, **2 layers**, EP=4, seq 512: steady **1747.6 ms**, peak 140 GiB/rank, 213 k launches. EP=8 unplaceable on a shared node |
| **G43** — fused mixer parity | **GREEN** | forward and per-token gradients **bit-identical** at both tiers; the shared gain gradient differs by accumulation order only |
| **G44** — measured delta meets budget | **owed** | no perf fix is authorised (below) |
| **G45** — post-phase trace | **owed** | needs an unshared node |

## 4. The predicted bottleneck is not the bottleneck

The plan named the AttnRes mixer — ~120 GB of reads per forward — while insisting
the trace decides and the phase rescopes if they disagree. They disagree:

| | share of device time |
|---|---|
| `Optimizer.step#TensorParallelMuon.step` | **21.0 %** (one call, 960 ms) |
| `aten::addmm` | 12.3 % |
| collectives | 8.5 % |
| **`k3.attn_res`** | **0.1 %** |

Both directions of the caveat are in the report: the mixer is understated at 2
layers (`K <= 1` here against 8 at 93 L, and it is `O(K + 1)`), and the optimizer
is overstated (Muon scales with parameters, and only two layers of activation work
compete). Under **R9.4** — do not chase a row worth less than 10 % of steady
iteration time — nothing at this geometry authorises an AttnRes performance fix,
and none is claimed. T11.4 is rescoped rather than quietly dropped.

## 5. The chunked mixer stands on memory, not on this trace

It landed because `results/attn_res.md` measured 109.6 GiB per forward at
production shape, and because G43 shows the change is free: same numbers, less
memory. It ships **off** by default (R5.3) and turns on when G44 has a number.

The parity claim is split rather than loosened, which is the honest shape:

* forward — **bit-identical**;
* gradients of per-token tensors — **bit-identical**;
* gradient of `norm_weight`, the `[H]` gain shared by every token — differs by
  fp32 accumulation order, ~1e-5 relative, and a test shows **the error does not
  grow with chunk count**. That is what separates reordering a sum from losing
  precision.

## 6. The limit is arithmetic, not scheduling

Two AttnRes slots need 13 layers; 13 layers at official width is **484 GiB per
GPU** at EP=8, 1.7x a whole device. So `K = 1` is the ceiling for *any* single
node at *any* EP — EP=7 at 4 L was tried and OOM'd, which pins the analytic
model's edge. Ranking the mixer is structurally a multi-node measurement. Anyone
picking this up should size for pipeline parallelism across nodes rather than
waiting for the box to clear.

## 7. Owed

G44, G45, and the EP=8 run. Also two core warnings worth a decision rather than
suppression: `moe_router_dtype` (K3 already routes in fp32, so core is checking a
flag we do not set) and the `cuda_graph_scope` deprecation.
