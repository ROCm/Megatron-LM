# G47 — the MoE dispatcher A/B matrix

> A plan, not an implementation (T12.4). No cluster job is launched from here (R10.1).
> Backend availability is asserted by `test_k3_p12_dispatchers.py`, so this document
> cannot drift from the pin without a test failing.

## The plan's premise has already been overtaken

The incoming plan framed this as "stock all-to-all vs a **DeepEP port** vs MoRI vs
a MoonEP evaluation". At the pinned Megatron SHA there is nothing to port:
`moe_flex_dispatcher_backend` already accepts **`deepep`** and **`mori`**, core
ships `moe/fused_a2a.py` with DeepEP's dispatch/combine, and the constraints are
enforced in `TransformerConfig.__post_init__`. The work is configuration and
measurement, not integration.

What is *not* there is the runtime: `deep_ep`, `mori` and `hybridep` are all
absent from this container, and `fused_a2a.HAVE_DEEP_EP` is `False`. So the A/B is
gated on installing backends, not on writing them.

## The matrix

| arm | `moe_token_dispatcher_type` | backend | present at the pin | runtime installed | notes |
|---|---|---|---|---|---|
| **A** baseline | `alltoall` | — | yes | yes | the P11 baseline was taken on this |
| **B** allgather | `allgather` | — | yes | yes | core's default; replicates tokens, so expected worse at EP ≥ 8 — included as a control, not a candidate |
| **C** DeepEP | `flex` | `deepep` | yes | **no** (`deep_ep`) | `moe_deepep_num_sms` is the tuning knob; incompatible with `moe_pad_expert_input_to_capacity` |
| **D** MoRI | `flex` | `mori` | yes | **no** (`mori`) | **requires `moe_mori_max_tokens_per_rank`** — core raises without it, and only the training entry point derives it, so any other caller must set it explicitly |
| **E** HybridEP | `flex` | `hybridep` | yes | **no** | mutually exclusive with `deepep` |
| **F** MoonEP | — | — | **no** | no | evaluation only; nothing in core references it |

## What each arm has to report

Same protocol as P11, so the numbers are comparable rather than merely collected:

1. steady iteration ms (median of five, after three warm-up steps);
2. collective device ms and its share, from the same `record_function` regions;
3. peak HBM per rank;
4. kernel launches per iteration — the baseline's 213 k is high, and dispatcher
   choice is one of the few things that moves it;
5. the quantile-balancing load report: per-expert token counts and the
   max/mean ratio, since a dispatcher that is fast because routing collapsed is
   not fast.

Point 5 is the one that makes this a K3 matrix rather than a generic one. K3
routes 16 of 896 experts with a **quantile-balanced** bias whose algorithm is
ours, not the release's (`moe/k3_router.py`), so an EP arm must be read together
with the load it produced.

## A trap in arm D: asking for MoRI can silently give you DeepEP

`TransformerConfig.__post_init__` handles `moe_enable_deepep` **first** and
overwrites `moe_flex_dispatcher_backend` with `"deepep"`. The MoRI branch below it
then never matches, so its "Cannot enable both DeepEP and MORI EP" error is
**unreachable**: set both and you get DeepEP, with only a deprecation warning
about an unrelated flag.

For this matrix that is not academic. Arms C and D would report identical numbers
and nothing in the output would say why. Every arm must therefore assert the
backend it actually got, from the config object after construction, rather than
trusting the flags it passed in. Register: finding **A17**.

## Sequencing

1. Run **A** and **B** at EP=8, 4 L official — no installs needed, and B is the
   control that shows the harness can tell dispatchers apart at all. If A and B
   are indistinguishable, the matrix is not measuring what it claims and nothing
   further is worth running.
2. Install `deep_ep`, run **C**, sweep `moe_deepep_num_sms`.
3. Install `mori`, run **D** with `moe_mori_max_tokens_per_rank` set explicitly to
   `micro_batch_size * seq_length`, adjusted for sequence/context parallelism.
4. **E** and **F** only if C or D shows a gap worth chasing under R9.4 (a row
   worth less than 10 % of steady iteration time is not a target).

## Precondition

All of it needs a node that is not shared. The P11 baseline could not place an
EP=8 run because another tenant held ~232 GB, and every arm above is an EP=8
measurement by definition.
