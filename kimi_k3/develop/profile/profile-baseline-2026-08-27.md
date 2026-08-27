# P11 baseline — where the time actually goes, and why the phase rescoped

> `HIP_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m kimi_k3.tools.proxy_ep8 \`
> `    --preset 4L --ep 4 --layers 2 --seq 512 --iterations 8 --trace-dir ...`
> Raw: `results/raw/proxy_ep8_raw.jsonl`. Trace: untracked, see `notes/deletion-requests.md`.

## What was actually run, and what it is not

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

## Improvement budgets, for whenever the 8-GPU run is possible

Recorded so the later gates have something to be measured against, per R9.3:

| target | current | budget | why |
|---|---|---|---|
| Muon step | 21.0 % / 960 ms | −30 % of that row | per-head Newton-Schulz (P9) makes many small orthogonalisations out of few large ones; whether that is faster or slower on this part is **unmeasured** and is the first thing to check |
| collectives | 8.5 % / 390 ms | −20 % | overlap: `overlap_grad_reduce` is off in `build_optimizer` |
| `aten::add_`, 9,614 calls | 3.6 % | fewer launches | 213 k launches per iteration is high; the AttnRes prefix accumulation is one contributor |
| AttnRes mixer | 0.1 % **at 2 layers** | no budget | R9.4: not a target until a deeper trace ranks it |

## Warnings seen (G45's "no banned warnings")

Two, both from core and both worth a decision rather than suppression:

* `Using a large number of experts (>=32) without fp32 routing. Consider enabling
  moe_router_dtype` — K3 routes on **fp32 sigmoid** scores by construction
  (`moe/k3_router.py`), so this is core checking a flag we do not set rather than
  a real precision gap. Worth setting `moe_router_dtype` explicitly so the
  warning stops hiding a future real one.
* `full scope is deprecated. Use empty cuda_graph_scope` — from
  `recompute_granularity="full"`; cosmetic at the pin.

## Owed

The EP=8 run at 4 L, and a deeper proxy (≥ 24 layers, so at least two AttnRes
slots exist) before any AttnRes performance claim. Both need a node that is not
being shared. G44 and G45 stay open.
