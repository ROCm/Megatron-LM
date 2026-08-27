# 06 — Capacity and Parallelism

> Every number here is either (a) derived from the parameter table in
> [`../architecture/01-kimi-k3-architecture-deep-dive.md`](../architecture/01-kimi-k3-architecture-deep-dive.md) §8,
> or (b) an **estimate to be replaced by a measurement** from gates G5 (optimizer
> memory), G6 (AttnRes payload) and G28 (trainer memory report). Rule R9.1: no
> configuration ships with an unmeasured recipe. Estimates are labelled.

## 1. Where the parameters are

| Group | Params | Share |
|---|---:|---:|
| Routed experts (92 × 896 × 3 × 3584 × 3072) | 2 722.7 B | **98.0 %** |
| Everything else (attention, shared experts, latent projections, router, dense FFN, embeddings, norms) | 56.8 B | 2.0 % |
| **Total** | **2 779.5 B** | |

**Consequence:** expert parallelism is the only lever that matters for weight
memory. `EP` must divide 896 = 2⁷ × 7 → `EP ∈ {1, 2, 4, 7, 8, 14, 16, 28, 32,
56, 64, 112, 128, 224, 448, 896}`.

Per-GPU parameter count (TP × PP shard everything; EP shards only the routed
experts; DP replicates):

```
params_per_gpu = P_nonexpert / (TP·PP) + P_expert / (TP·PP·EP)
```

| TP | PP | EP | world (DP=1) | params/GPU | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 8 | 8 | (4 L config) 16.3 B | the single-node bring-up config (P7) |
| 1 | 8 | 8 | 64 | 49.6 B | 93 L — will not fit |
| 1 | 8 | 32 | 256 | 17.7 B | 93 L — tight |
| 1 | 16 | 16 | 256 | 14.2 B | 93 L — plausible |
| 1 | 16 | 32 | 512 | 8.9 B | 93 L — comfortable |

## 2. Optimizer memory — MEASURED (gate G5, 2026-08-27)

Bytes per parameter **per GPU**, for the parameters resident on that GPU.
Measured with `tools/opt_mem_probe.py` on a 99.0 M-parameter model carrying K3's
shape mix (MLA + MoE + norms), TP = PP = EP = 1, world size = DP, as the CUDA
allocator delta from before construction to after two optimizer steps. Full
tables: [`../results/opt_mem.md`](../results/opt_mem.md); raw per-rank rows:
`../results/opt_mem_raw.jsonl`.

| recipe | DP=1 | DP=2 | DP=4 | DP=8 | shards across |
|---|---:|---:|---:|---:|---|
| `adam` | 18.02 | 18.02 | 18.02 | 18.02 | nothing |
| `adam` + `--use-distributed-optimizer` | 18.02 | 12.02 | 9.03 | **7.52** | DP |
| the same + `--use-precision-aware-optimizer` | — | 11.02 | — | **7.27** | DP |
| `muon` | 15.17 | — | — | **15.17** | nothing |
| **`dist_muon`** | 15.17 | 11.00 | 8.91 | **7.87** | DP (whole tensors) |

What the measurements settle:

1. **A sharded Muon recipe exists and roughly halves optimizer memory.**
   `dist_muon` at DP = 8 costs **7.87 B/param** against plain `muon`'s 15.17 —
   1.93×. The incoming plan's flat 14 B/param is right only at DP = 1, where
   `dist_muon` and `muon` measure identically (15.17 both), because there is
   nothing to shard.
2. **The analytic model holds.** Fitting `6 + 8/DP + c` to the `dist_muon` row
   gives `c` = 1.17 / 1.00 / 0.91 / 0.87 at DP = 1 / 2 / 4 / 8 — that residual is
   the scalar (Adam) group plus allocator overhead. `adam` + dist-opt lands on
   its analytic `6 + 12/DP` to two decimals at DP = 2 and 4.
3. **`muon`'s 15.17 is not 14** because the parameters Muon does *not* manage —
   norms, biases, embeddings, and in K3 also the conv weights, `A_log` and
   `dt_bias` — carry Adam's 8 B/param. The split is computed for the real model
   by `tools/mem_budget.py:muon_group_split`.
4. **`dist_muon`'s balance is approximate.** Per-rank spread at DP = 8 is
   7.65 – 8.25 B/param (±4 %), because sharding assigns *whole tensors* in a
   ping-pong order by numel. With grouped-GEMM experts — a few very large tensors
   per layer — that imbalance can grow; P7 re-measures at the real shapes.
5. **Precision-aware Adam is only marginally cheaper than plain dist-opt Adam**
   at DP = 8 (7.27 vs 7.52) on this shape mix, so it is not by itself a reason to
   prefer Adam over `dist_muon`.

Constraints that remain true regardless: `--use-distributed-optimizer` is
rejected for every Muon variant (`arguments.py:1552`), and `dist_muon` is the
only Muon variant compatible with `--overlap-grad-reduce` /
`--overlap-param-gather` (`arguments.py:881-882, 1549-1550`).

Two caveats on transferring these numbers to the 93 L model: the probe runs
EP = 1, so expert parameters here are DP-replicated rather than EP-sharded; and
the probe's shape mix is K3-like but not K3-exact (the KDA modules land in P3).
G28 re-measures on the 4 L official config.

### 2.1 Cluster-level floor **[estimate — refine with G28]**

At DP = 1, cluster-wide weight + optimizer state is `P × 15.17 B ≈ 42.2 TB`
(using the measured `muon` figure rather than the analytic 14). A node is
8 × MI355X × 288 GB = 2.304 TB, so the **raw** floor is ≈ 19 nodes with zero
headroom — unusable as a target, since activations, the AttnRes payload,
fragmentation and communication buffers come out of the same budget. At 60–75 %
utilisation that is **25–31 nodes at DP = 1**, falling as DP rises because
`dist_muon` shards master weights and momentum while weights and gradients
replicate. `tools/mem_budget.py` computes the curve; no configuration quotes a
node count that is not backed by a row in `../results/opt_mem.md`.

## 3. AttnRes memory — MEASURED (gate G6, 2026-08-27)

Full tables: [`../results/attn_res.md`](../results/attn_res.md). Production
width, `H = 7168`, `S = 8192`, `B = 1`, bf16, PP = 8.

### 3.1 Pipeline payload

Packed payload per boundary is `(1 + K) × S × B × H × 2 B`, i.e. 224 MB at the
first boundary rising to 896 MB at the last. Under 1F1B the in-flight microbatch
count falls as the payload grows, so the product **peaks in the middle of the
pipeline**: 2.8 GB at stage 3, ≈ 5.6 GB counting saved input and output tensors.
Both the `[12]×7 + 9` and the even `[12,12,12,12,12,11,11,11]` layouts produce the
same multipliers, so pick the even one for compute balance.

Context parallelism divides `S` and therefore divides this line directly — it is
the first mitigation to reach for, ahead of reducing PP depth.

### 3.2 Mixer temporaries — the real cost

One mix at `K+1 = 9`: **7.1 GB** peak forward, **12.2 GB** peak forward+backward,
5.15 ms / 15.65 ms. Cost is linear in `K+1`.

Across the model — 186 mixes, mean `K+1` = 5.39 — that is **109.6 GiB read per
forward** and an eager mixer forward of **≈ 635 ms per microbatch**, against
≈ 800 ms for *all* the routed-expert GEMMs at 40 % MFU. The AttnRes mixer is the
single largest non-GEMM cost in the model, exactly as predicted, and now with a
number attached.

Consequences:

1. **Recompute is mandatory at production width.** Saving each mix's fp32 stack
   for backward would cost ≈ 236 GB per microbatch. With recompute the peak is
   one live pair (4–12 GB by `K`), and the traffic is paid twice.
2. **The fp32 upcast costs 2.3× memory and ≈1.5× time** (measured against the
   same mix kept in bf16). We keep fp32 — it is the release's semantics — but a
   fused kernel can accumulate in registers and pay neither.
3. **P11 budget, set here:** ≤ 10 % of the eager forward (≤ 64 ms per microbatch)
   and no fp32 stack resident in HBM.

## 4. Pipeline layout rules

1. **93 is not divisible by anything convenient** (93 = 3 × 31). Every PP config
   uses an explicit `pipeline_model_parallel_layout`; a config test asserts the
   layout sums to 93 and that no stage is empty (G46).
2. **Prefer boundaries at 0-indexed layer ≡ 11 (mod 12)** — the payload
   multiplier is `1 + ceil((l_last + 1) / 12)`, so ending a stage just before an
   append layer saves one slot per boundary.
3. **The last stage should be the short one** (it carries the output mix, final
   norm, head and loss).
4. **VPP is descoped** — the interleaved schedule asserts
   `adjust_tensor_shapes_fn is None` (`schedules.py:949`).
5. **PP = 1 must not bind the shape hook** — the no-pipelining schedule asserts
   it is `None` too (`schedules.py:631`).

## 5. Other parallelism notes

- **TP:** the MoE latent projections are built with `parallel_mode="duplicated"`
  (`moe_layer.py:252-274`), so each TP rank holds a full `7168 × 3584` pair per
  MoE layer (≈ 4.7 B parameters replicated per TP rank across 92 layers). Worth a
  measurement before choosing TP > 1, and a candidate upstream proposal later.
- **CP:** divides `S` and therefore both AttnRes terms in §3. `fla` has a KDA CP
  path (pre-gated `kg`/`qg` contract) — long-context phase only, not v1.
- **EP:** must divide 896; `moe_shared_expert_overlap` must stay **off** with
  latent projections (`moe_layer.py:422-424`) — surface the assertion, never mask it.
- **Sequence parallel:** interacts with the payload only through
  `get_tensor_shapes`, which already divides the sequence dimension by TP when
  `sequence_parallel` is on; our multiplier applies on top and needs no special
  handling — but it is asserted in G21.

## 6. What to measure, in what order

| Gate | Measures | Replaces |
|---|---|---|
| **G5** | bytes/param per group per recipe vs DP; `dist_muon` shard balance | §2's table |
| **G6** | payload bytes per boundary; mixer temporaries; eager mixer wall-clock | §3's estimates |
| **G28** | end-to-end peak and persistent memory for the 4 L config | §1's per-GPU table |
| **G42** | where the time actually goes at EP = 8 | §3.2's traffic estimate |

Until then, every table in this document is planning input, not a commitment.
