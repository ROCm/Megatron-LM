# G49 — AdamW vs Adam, and what a cold run costs

> `torchrun --nproc_per_node=4 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 4 \`
> `  --experts 112 --seq 512 --optimizer adam_dist [--coupled-weight-decay]`
> Raw: `results/raw/wd_ab4_*.jsonl` (original order), `wd_abrev_*.jsonl` (reversed).
> GPUs 0,1,2,4 — the node was shared; 3 and 5 belonged to another team.

## Why this was run

`build_optimizer` briefly forced `decoupled_weight_decay=False` whenever CPU
offload was off, silently downgrading `adam_dist` from AdamW to Adam. The fix
restores core's default (True). The question was whether the arms measured under
the bug — in particular `adam_dist` at 390 ms / 228.2 GiB (G47) — needed
re-running.

## The decay mode is free

| run order | arm | steady ms | peak GiB |
|---|---|---|---|
| original | AdamW (1st) | 2,146.6 | 92.97 |
| original | Adam (2nd) | 569.2 | 92.97 |
| **reversed** | **Adam (1st)** | **330.4** | 92.97 |
| **reversed** | **AdamW (2nd)** | **330.5** | 92.97 |

Reversed, the two arms differ by **0.03%**. Peak memory is identical to the
hundredth of a GiB in all four runs, which is what the mechanism predicts: the
flag changes *where* decay is applied, not the state tensors. Core passes it as
a single boolean into TE's fused kernel (`optimizer/__init__.py:545`).

**So G47 stands as an AdamW result and does not need re-running**, even though
it was measured with coupled decay.

## The real finding: first-run overhead is ~6x

The original ordering showed AdamW 3.8x slower than Adam. That was entirely
position, not the flag. Whichever arm ran first paid it.

It is not warm-up *within* a run — the proxy already discards 3 warmup
iterations before timing. It survives across processes, so it is disk- and
driver-level state: aiter `.co` kernel loads, hipBLASLt tuning. Note the first
position itself got faster between the two sweeps (2,146.6 -> 330.4 ms), so the
cache keeps filling across runs, and the first sweep was also contending with
another team's job on GPUs 3 and 5.

**A single cold proxy run at this geometry reads roughly 6x slow.** Any
comparison whose arms were not run back to back, in one script, on a quiet node,
is suspect.

### What this calls into question

Comparisons made *within* one sweep script are sound — every arm pays the same
warm cache. That covers the optimizer arms (G47), the dispatcher A/B and the
scale-out sweeps.

The **TE 2.12 -> 2.18 + CK baseline (1.44x)** is the exception: the two sides
were measured on different days in different processes. The 5.36x drop in kernel
launches is structural and cannot be a caching artifact, but the wall-clock
1.44x could carry some warm-cache contribution. Re-measure with the arms
adjacent and order-reversed before quoting it as a pure pin/kernel win.

## Method note

Order reversal is now the cheapest guard available against this, and it costs
one line in a sweep script. Worth doing by default for any two-arm claim.
