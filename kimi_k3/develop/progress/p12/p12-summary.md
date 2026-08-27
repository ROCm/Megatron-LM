# P12 — Scale-out preparation (COMPLETE except two owed items)

## 1. Objective

Work out what running 93 L would actually take, and hand a human something they
can launch. No cluster job is launched from here (R10.1).

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/config/scaleout.py` | PP layouts, layout legality, the EP ladder, node counts |
| `kimi_k3/develop/results/scaleout_93l.md` | the table and what it means |
| `kimi_k3/develop/plan-0/07-dispatcher-ab.md` | the dispatcher A/B matrix |
| `kimi_k3/tests/test_k3_p12_{scaleout,dispatchers}.py` | 19 tests |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G46** — 93 L configs validated analytically | **GREEN** | floor is **28 nodes**; layouts, parameter counts and memory all derived from measurements |
| **G47** — dispatcher A/B + templates | **GREEN as a plan** | `plan-0/07-dispatcher-ab.md`, with every claim about the pin asserted by a test |

## 4. What 93 L costs

**28 nodes**, at `pp8 x ep28`: 19.25 B parameters per GPU, 141 GiB of state at the
**measured** 7.87 bytes/param (`results/opt_mem.md`), plus 82 GiB of headroom
measured in the 4 L run. `pp4 x ep56` also reaches 28 nodes but leaves 13 GiB
spare against 65, so it is the fragile one. A test asserts the constants are the
measured ones, because a table that quietly reads an estimate is decoration.

## 5. PP cannot exceed 8

A layer appends a residual slot when its 0-indexed position is a multiple of 12,
and **the prefix sum resets at that moment**. A boundary there hands the next
stage a fresh prefix rather than a half-accumulated one. 93 layers give seven
whole blocks, hence eight aligned cut points — so every `pp16` candidate is
flagged rather than silently emitted.

It is not a correctness failure: G21 proved bitwise transport of a partial prefix
across a stage boundary. It is that a `pp16` layout leaves a block straddling two
stages, and any repartition or recompute then has to reason about a prefix that
began somewhere else.

Payload cost also grows with depth — the last boundary of a `pp8` layout carries
8x the hidden state against 2x at the first. **A balanced layer split is not a
balanced bandwidth split**, and the last stage carries the tail block, the extra
layer, and the final MLA pair.

## 6. The dispatcher plan's premise was already overtaken

T12.4 asked for a "DeepEP port". Core ships DeepEP *and* MoRI as flex backends at
the pin; what is missing is the runtime (`deep_ep`, `mori`, `hybridep` are all
absent, `HAVE_DEEP_EP` is False). The matrix is gated on installs, not integration.

Finding **A17** came out of writing it: core's "Cannot enable both DeepEP and MORI
EP" guard is **unreachable**, because `moe_enable_deepep` is processed first and
overwrites the backend. Ask for both and you get DeepEP. Harmless in training,
but it would have made two arms of the matrix report identical numbers with
nothing saying why — so every arm asserts the backend it actually got.

## 7. Owed

T12.5 (continued-pretrain flatness scripts), and the matrix's own precondition:
an unshared node, since every arm is an EP=8 measurement by definition.
