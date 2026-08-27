# G37 — save, resume, and the same losses, under PP=2

> `torchrun --nproc_per_node=2 -m kimi_k3.tools.pp_resume_probe --steps 20 --save-at 10 --fixed-batch`
> Raw: `results/raw/pp_resume_raw.jsonl`. Tiny preset, PP=2, fp32, `dist_muon`, fixed batch.

Resume is where the AttnRes payload could quietly break. The block residual is
*activation* state and rightly absent from the checkpoint; the optimizer, the
routing bias and the RNG are not. A resume that drops any of them still runs,
still descends, and produces a subtly different model. So the claim is the strict
one — **bitwise** identical losses for every step after the resume point — and
the comparison is against the tail of an uninterrupted run of the same length,
because only that run knows what those steps should have produced.

| run | dropout | bitwise | max Δ over 10 steps |
|---|---|---|---|
| **resume** | 0 | **yes** | **0.0** |
| **resume** | 0.1 | **yes** | **0.0** |
| control: fresh optimizer | 0 | no | 0.2422 |
| control: RNG not restored | 0.1 | no | 0.1292 |
| control: RNG not restored | 0 | *yes* | 0.0 — see below |

Both controls fail, so the gate can fail. Final loss at step 20 on the fixed
batch: 2.8172, reached identically by the continuous and the resumed run.

## The RNG control was inert until dropout was turned on

At the tiny preset dropout is 0 (it is set that way on purpose — see the G7
record), and nothing else on this path draws from the generator. Dropping the RNG
restore therefore changed nothing, and the control silently proved nothing. With
dropout at 0.1 it diverges by 0.129, and the real resume is *still* bitwise — which
is the statement worth having. A control that passes for the wrong reason is
indistinguishable from one that works.

## What the gate found: `dist_muon` could not be resumed at all

The first run did not reach the comparison. `ChainedOptimizer.state_dict()`
returns a **list** when more than one optimizer is chained, and `dist_muon` always
chains two; `LayerWiseDistributedOptimizer.load_state_dict` then calls `.values()`
on it and raises. The layer-wise optimizer cannot read what its own `state_dict()`
writes, so **every `dist_muon` resume fails as shipped** — on the optimizer this
project measured at 7.87 B/param and recommends for K3.

Core stays unmodified (R2.1). `kimi_k3/optim/resume.py` applies the list
conversion the override intends and calls `ChainedOptimizer.load_state_dict`
directly, and a tripwire fails once core fixes this so the workaround leaves
rather than lingers. Register: finding A15.

## Scope

Tiny width, PP=2, one node, 20 steps, no dist-checkpointing. It does not cover
`torch_dist` sharded checkpoints, a changed DP size across the resume, or
production geometry — all of which belong with the scheduled multi-node work.
