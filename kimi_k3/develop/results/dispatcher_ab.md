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
