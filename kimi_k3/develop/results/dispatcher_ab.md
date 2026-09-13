# G47 arms A and B — stock all-to-all vs allgather at EP=8

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 8 --seq 512 \
>     --iterations 8 --dispatcher {alltoall,allgather}`
> Raw: `results/raw/dispatcher_ab_raw.jsonl`. Same protocol as the P11 baseline, so the
> numbers are comparable rather than merely collected.

| arm | steady iteration | peak HBM | collective time | launches | starved experts |
|---|---|---|---|---|---|
| **A** `alltoall` | **2653.0 ms** | 193.61 GiB | 812.9 ms | 321,588 | 0 / 896 |
| **B** `allgather` | **2678.9 ms** | 193.63 GiB | **545.8 ms** | 321,166 | 0 / 896 |

**They are 1.0 % apart** — 26 ms, against a within-run spread of 16 ms. Just
outside noise, and not meaningfully more.

## The inversion is the interesting part

`allgather` spends **less** time in collectives (546 ms against 813) and still
finishes **slower**. It replicates tokens to every rank instead of routing them,
trading communication for redundant expert compute, and at EP=8 that trade is
roughly break-even and slightly unfavourable. A reading that looked only at
"comm_ms" would have picked the slower arm.

Routing load is effectively identical between them (max/mean 1.51 both, zero
starved of 896), which is the control that matters: neither arm is fast because
routing collapsed.

## What this says about running arms C–E

The sequencing rule in `plan-0/07-dispatcher-ab.md` was that if A and B cannot be
told apart, the harness is not measuring what it claims. It can tell them apart,
but only just — and the ceiling is the point: **collectives are ~11 % of device
time at this geometry**, so no dispatcher can win more than that, while the Muon
optimizer step sits at 21 %.

On this evidence, installing `deep_ep` or `mori` is not justified *at this
geometry*. It would be justified at one where communication actually dominates —
more layers per rank, longer sequences, or EP across nodes rather than within
one. That is the same multi-node condition the AttnRes ranking needs.

## A17, extended

Both arms resolved `moe_flex_dispatcher_backend = 'deepep'` while running
`alltoall`. It is inert — the field is read only when the dispatcher is `flex` —
but it means **switching a config to `flex` silently selects DeepEP**, with
nothing asked for and nothing logged. Combined with A17 (core's "cannot enable
both" guard being unreachable), the rule stands: every arm asserts the backend it
actually resolved to, and the harness records it.

---

# G60 — MoRI EP, measured. It does not remove the permute; it adds to it

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 8 \`
> `  --seq 512 --dispatcher flex --flex-backend mori [--triton-attn-res]`
> Raw: `results/raw/disp_*.jsonl`, `results/raw/combo_mori_triton.jsonl`.
> Arms C-E were listed as blocked on `deep_ep`/`mori` since P12. C is now unblocked.

## Getting MoRI installed

Three things cost time and none is in the Megatron docs:

* **The package is `amd_mori`, not `mori`.** `pip install mori` finds nothing,
  which is why this looked unavailable.
* **The version has to match the ROCm underneath.** `1.2.3` installs and then
  fails to import -- `libhipfile.so.0`, which ROCm 7.2.1 here does not ship and
  no apt package provides. `1.1.1` imports and then corrupts the heap at runtime
  (`free(): invalid pointer`, an ABI mismatch). **`1.2.2` works.**
* `libgrpc++1.51t64` from apt is required, and core needs
  `moe_router_dtype=fp32` (it warns, then aborts) plus an explicit
  `moe_mori_max_tokens_per_rank` (only the training entry point derives it).

**DeepEP is not available on this stack**: `ImportError`. It needs NVSHMEM.

## It is a regression at single-node EP=8

| arm | steady ms | peak GiB |
|---|---|---|
| A `alltoall` | **1858.2** | 190.96 |
| B `allgather` | 1849.9 | 190.99 |
| C `flex` + **MoRI** | 1881.2 | 190.97 |
| C + fused AttnRes | 1878.6 | 190.97 |

`k3.moe` region: **55.62 ms -> 141.79 ms**.

## Why -- the elementwise ops around the grouped GEMM

The expectation was that MoRI's fused dispatch/combine would absorb the torch
permute/unpermute. Measured, it does not:

| op | site | ms |
|---|---|---|
| `index_put_` | `token_dispatcher.py:1509` `_indices_to_multihot` | **0.40** |
| `index_select` | `token_dispatcher.py:1509` `_indices_to_multihot` | **0.18** |
| `index_select` | `moe_utils.py:352` `permute` | 0.13 |
| `scatter_add_` | `fused_a2a.py:785` (MoRI's own) | 0.09 |
| `scatter_add_` | `moe_utils.py:481` `unpermute` | 0.09 |

`unpermute`'s `scatter_add_` is **still there**, MoRI adds **another** in
`fused_a2a`, and `_indices_to_multihot` contributes 0.58 ms of new elementwise
work converting routing indices into the multi-hot form its dispatch kernel
wants. The permutation is not eliminated; a layer is added on top.

Expert GEMMs are unchanged -- still CK grouped, 49.39 ms over 24 launches. MoRI
changes dispatch, not the GEMM.

**Recommendation: do not enable MoRI on a single node.** Its case is inter-node
EP traffic, which this geometry cannot exercise; retest at multi-node.

## Two things the same trace settled

**SiTU is live in a real training step.** `aten::tanh` and `aten::sigmoid`
attribute to `kimi_k3/moe/situ.py:19 situ_glu`, there is **no `aten::gelu`
anywhere**, and nothing appears at `experts.py:310` -- core's GLU closure is
bypassed exactly as the `use_te_activation_func` wiring intends. The G53/G54
gates show `SituGLU` computes the right thing and is attached to the right
modules; only a trace shows core actually calls it.

**The fused AttnRes kernel works in-model**: `k3.attn_res` **7.65 ms -> 1.50 ms**,
5.1x, at a geometry (seq 512, K <= 1) where it has least to fuse.

## A lost baseline

The traced run overwrote `proxy_ep8_4L_rank0_shapes_stack.json`: the filename
keyed only on the profiler flags, not on the dispatcher or mixer, so the
alltoall + eager baseline was replaced by the MoRI + fused run and the
side-by-side diff is gone. The numbers survive here; the trace does not. Fixed --
the name now carries the configuration.
