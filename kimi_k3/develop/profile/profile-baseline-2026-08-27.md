# P11 baseline — where the time actually goes, and why the phase rescoped

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 8 --seq 512 --iterations 8`
> and the same with `--fused-attn-res`. Raw: `results/raw/proxy_ep8_raw.jsonl`.
> Traces are gitignored run artefacts; the numbers below are what is kept.
>
> The first version of this report was taken at EP=4 / 2 layers, because the node
> was shared at the time. Both sets are here — the second is the gate, the first
> is now a depth cross-check that happens to confirm it.


> **Caveat added 2026-09-03 (G49).** A cold proxy run reads roughly 6x slow on
> this node: whichever arm runs first pays aiter `.co` kernel loads and
> hipBLASLt tuning, and the cost survives across processes rather than being
> absorbed by the 3 discarded warmup iterations. The TE 2.12 -> 2.18 + CK
> comparison below was taken on different days in different processes, so the
> wall-clock **1.44x may carry a warm-cache contribution** and should be
> re-measured with the arms adjacent and order-reversed. The 5.36x drop in
> kernel launches is structural and is not affected. See
> `results/weight_decay_ab.md`.


## The EP=8 baseline (added after the node was freed)

The full configuration the gate asks for: 4 L official, **EP=8**, seq 512,
`dist_muon`, full recompute, all eight GPUs.

| | EP=8, 4 layers | EP=4, 2 layers (first run) |
|---|---|---|
| cold iteration | 18.45 s | 11.48 s |
| **steady iteration** (median of 5) | **2653.5 ms** | 1747.6 ms |
| spread across the 5 | 2643–2659 ms | — |
| peak HBM per rank | **193.6 GiB** | 139.8 GiB |
| kernel launches | 321,116 | 213,113 |
| **`Optimizer.step#TensorParallelMuon.step`** | **21.1 %** | 21.0 % |
| `aten::addmm` | 12.3 % | 12.3 % |
| collectives | 6.6 % | 8.5 % |
| **`k3.attn_res`** | **0.09 %** | 0.1 % |

**The ranking is stable across a 2x change in depth.** That was not guaranteed —
the 2-layer report warned that the optimizer's 21 % was probably overstated,
because Muon scales with parameters while only two layers of activation work
competed with it. It was not overstated: doubling the layers doubled both, and
the ratio did not move. The caveat was reasonable a priori and the measurement
retired it.

What did *not* change is the conclusion: the AttnRes mixer is **0.09 %** of device
time, and the Muon optimizer step is the single largest row in the model.

## G44 / G45 — the chunked mixer, measured

Same configuration, `--k3-attn-res-fused`:

| | baseline | fused | delta |
|---|---|---|---|
| steady iteration | 2653.5 ms | **2649.6 ms** | −0.15 % (noise) |
| peak HBM | 193.63 GiB | **193.63 GiB** | **0.00** |
| kernel launches | 321,116 | 321,163 | +47 |

**No change, and none was possible.** The temporary the chunking removes is
`[T, K+1, H]` in fp32 — at 4 layers and seq 512 that is `512 x 2 x 7168 x 4` =
**29 MB**, against a 193 GiB peak. The 109.6 GiB figure it was built for is
*production* geometry: 93 layers (K = 8) at seq 8192, which is 8x the slots and
16x the sequence, and which does not fit on this node.

So **G45 is green** — no time regression, no memory regression, no new warnings —
and **G44 has nothing to meet**, because no budget was set for AttnRes under R9.4
once the trace put it at 0.1 %. The mixer keeps its default-off flag. This is the
honest closing position: the optimisation is correct (G43, bit-identical) and
currently pointless, and it becomes useful only at a geometry no single node can
run.

## What was actually run first, and what it is not

Official widths (hidden 7168, 896 experts, 96 heads) at **2 layers**, EP=4, seq
512, `dist_muon`, full recompute. Not the EP=8 configuration the gate asks for.

The node is shared. Another tenant holds ~232 GB on one GPU, which puts the 8-GPU
run out of reach: 4 L at EP=8 needs all eight devices, and at EP=4 it wants
277 GiB/rank and OOMs. Cutting to 2 layers fits in 140 GiB/rank on the four idle
devices. **G42 is therefore partial** — the harness is validated end to end and
the ranking below is real, but it is a ranking at 2 layers, and the sections
marked *not representative* say where that changes the answer.

| | |
|---|---|
| cold iteration | 11.48 s |
| steady iteration (median of 5) | **1747.6 ms** |
| peak HBM | 139.8–142.4 GiB per rank |
| kernel launches per iteration | 213,113 |
| collective time | 459.3 ms |

## Ranked, by self device time

| share | ms | calls | what |
|---|---|---|---|
| **21.0 %** | 960.1 | 1 | **`Optimizer.step#TensorParallelMuon.step`** |
| 12.3 % | 561.4 | 4,560 | `aten::addmm` |
| 5.7 % | 261.9 | 2,347 | `aten::mm` |
| 5.0 % | 229.7 | 14 | `ncclDevKernel_Generic` |
| 3.6 % | 165.5 | 9,614 | `aten::add_` |
| 3.5 % | 160.8 | 1 | `nccl:allreduce_coalesced` |
| 3.3 % | 148.8 | 896 | Tensile GEMM, MT192x224x64 |
| 3.1 % | 142.2 | 912 | Tensile GEMM, MT224x224x64 |
| 2.7 % | 123.2 | 1,136 | Tensile GEMM, MT256x224x64 |

## The prediction was wrong, and the phase rescopes

The plan named the AttnRes mixer as the predictable bottleneck — ~120 GB of reads
per forward at production shape — while noting that the trace decides, not the
prediction, and that the phase rescopes if they disagree (the DeepSeek-V4 P29
precedent). They disagree.

**`k3.attn_res` is 0.1 % of device time here.** The single largest row is the
**Muon optimizer step at 21 %** — one call, 960 ms, Newton-Schulz orthogonalising
every 2-D weight in the model.

Two honest qualifications, in opposite directions:

* **The mixer's share is structurally understated at 2 layers.** A residual slot
  is appended every 12 layers, so this proxy has at most **one** slot where the
  93 L model has **eight**. The mixer's cost is `O(K + 1)` in both reads and the
  fp32 stack, so its share can only grow with depth — by roughly 8x in the stack
  it holds. A 2-layer trace cannot rank it, and this one does not.
* **The optimizer's share is overstated at 2 layers.** Muon's cost scales with the
  *parameter* count, which at 2 layers is dominated by the MoE weights that every
  depth carries anyway, while the per-layer activation work that would compete
  with it is only 2 layers deep. It will still be a large row at depth; 21 % is
  not the number to quote.

So the ranked list is trustworthy about *this* geometry and is not yet the
production ranking. **Under R9.4 (do not chase a row worth < 10 % of steady
iteration time) nothing here authorises an AttnRes performance fix**, and none is
claimed. The chunked mixer that landed in P11 stands on its **memory** measurement
(`results/attn_res.md`: 109.6 GiB per forward, and G43's bit-identical forward),
not on this trace.

## Improvement budgets

Recorded so the later gates have something to be measured against, per R9.3.
These are set against the **EP=8** numbers above; the AttnRes row has been settled
(G44) and the rest are open.

| target | current | budget | why |
|---|---|---|---|
| Muon step | **21.1 % / 1503 ms** | −30 % of that row | **checked 2026-08-30, and the answer is slower**: per-head Newton-Schulz costs **0.57x** on KDA and **0.06x** on MLA `q_b` (`results/per_head_muon_cost.md`). The precision knob that would have been the other candidate is already applied — `muon_fp32_matmul_prec` defaults to `"medium"`, worth 7x over `"highest"`. The Triton SYRK path is ~2x slower on gfx950 and unreachable anyway (A20). **No lead remains on this row** short of a ROCm Newton-Schulz kernel |
| collectives | **6.6 % / 467 ms** | −20 % | overlap: `overlap_grad_reduce` is off in `build_optimizer` |
| `aten::add_`, 14,449 calls | 3.6 % | fewer launches | **321 k** launches per iteration is high; the AttnRes prefix accumulation is one contributor |
| AttnRes mixer | **0.09 % at EP=8** | no budget — **settled** | R9.4. G44 measured the chunked mixer at this geometry: 0.00 GiB and −0.15 % time, because the temporary it removes is 29 MB here |

## Warnings seen (G45)

Two, both from core and both worth a decision rather than suppression:

* `Using a large number of experts (>=32) without fp32 routing. Consider enabling
  moe_router_dtype` — K3 routes on **fp32 sigmoid** scores by construction
  (`moe/k3_router.py`), so this is core checking a flag we do not set rather than
  a real precision gap. Worth setting `moe_router_dtype` explicitly so the
  warning stops hiding a future real one.
* `full scope is deprecated. Use empty cuda_graph_scope` — from
  `recompute_granularity="full"`; cosmetic at the pin.

## No single-node proxy can rank the AttnRes mixer — at any EP

This is the more useful version of "the node was busy", and it is arithmetic
rather than scheduling. A residual slot is appended every 12 layers, so **two**
slots need at least **13** layers. At official width that is:

| layers | max slots | params/GPU at EP=8 | state + headroom | fits in 288 GiB |
|---|---|---|---|---|
| 2 | 1 | 7.85 B | 139.6 GiB | yes — what was run |
| 4 | 1 | 16.31 B | 201.5 GiB | yes at EP=8, **not** at EP=7 (measured: OOM) |
| **13** | **2** | 54.88 B | **484.3 GiB** | **no — 1.7x a whole GPU** |
| 25 | 3 | 106.24 B | 860.7 GiB | no |

So `K = 1` is the ceiling for any one node, and the mixer's cost is `O(K + 1)`.
Ranking it against the rest of the model is **structurally a multi-node
measurement**, not something a better-timed single-node run would fix. Anyone
picking this up should size for pipeline parallelism across nodes rather than
waiting for the box to clear.

## Owed

A >= 13-layer proxy **across nodes** — the only geometry at which the AttnRes
mixer can be ranked, or its chunking shown to pay. Everything runnable on one node
has now been run.
