# 2026-08-27 — `fla` and `chunk_kda`: what is actually wrong (corrected)

`troubleshooting` · gate **G1** · owner: P0-T0.1
**This note replaces an earlier version that was wrong. The retraction is §1.**

## 1. Retraction

The first version of this note claimed that `fla`'s `chunk_kda` "silently ignores
`A_log`, `dt_bias` and `transpose_state_layout`", on the grounds that they are
absent from the signature while it ends in `**kwargs`. **That conclusion was
wrong.** It was drawn from reading the signature alone and never running the call.

`fla` `main` (0.6.0, commit `5e02dd3`) handles all three **deliberately**:

```python
# fla/ops/kda/chunk.py
if 'transpose_state_layout' in kwargs:
    state_v_first = kwargs.pop('transpose_state_layout')      # accepted alias
...
if use_gate_in_kernel:
    A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
    if A_log is None and lower_bound is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True` ...")
```

The docstring documents both, including the gate math
`-exp(A_log) * softplus(g + dt_bias)` and the `lower_bound` variant, and the
function *raises* rather than proceeding silently when `A_log` is missing.

Verified empirically on MI355X: the released Kimi-K3 call runs, and perturbing
`A_log` changes the output (`max|Δ| = 0.079` on a tiny shape), so it is plainly
not ignored.

**Correction to the method, too:** the earlier note said the check should compare
parameter *names* via `inspect.signature`. That check would have **failed on a
working library**, because `A_log` legitimately arrives through `**kwargs`. The
right check is functional: call the op the way the release does and compare
against the FP32 oracle.

## 2. What is actually blocking G1

Two real problems, neither of them an API mismatch.

### 2.1 The KDA **backward** does not compile on gfx950

Forward is fine. Backward dies inside Triton's AMD backend:

```
fla/ops/kda/chunk_intra.py:395:0: error: Failures have been detected while
processing an MLIR pass pipeline
  note: Pipeline failed while executing [`TritonAMDGPUPipeline` on 'builtin.module']
RuntimeError: PassManager::run failed
```

reached through `chunk_kda_bwd` → `chunk_kda_bwd_intra`, with triton 3.6.0 and
`gfx950`. So on this stack `fla` gives us inference-shaped KDA and no training
KDA.

This is the same risk the plan already carried (upstream fla KDA-backward issues
#807 / #785) arriving by a different route: a compiler failure rather than a
numerical one. It is loud, which is the good case.

### 2.2 The published wheel is unusable

`pip install flash-linear-attention==0.5.2` ships **only** `fla/layers` and
`fla/models` — 154 files, no `fla/ops` at all — so `fla.layers.kda`'s own
`from fla.ops.kda import chunk_kda` cannot resolve. Install from git.

## 3. Where this leaves the pin

| question | answer |
|---|---|
| Is there a usable revision? | **Yes for forward**: git `main`, 0.6.0 at `5e02dd3`. The released call is accepted as-is. |
| Can we train with it today? | **No** — the backward fails to compile on gfx950 / triton 3.6.0. |
| Does that block P2–P6? | No. `k3_kda_backend` stays `eager` (rule R5.3); the FP32 oracle is the contract, and it was always going to be permanent (rule R8.1). |
| What unblocks it? | In order of cheapness: (a) a different triton build — try the ROCm-vendored triton and any newer 3.7+; (b) prune the failing autotune config in `chunk_intra`; (c) upstream the reproducer to Triton/fla; (d) our own backward. |

## 4. Actions

1. Pin `fla` at git `5e02dd3` in `PINS.md` with the forward-only caveat recorded.
2. G1's check is **functional, not signature-based**: run the released call, assert
   the forward matches the FP32 oracle, and assert the backward either matches or
   fails loudly — never infer from `inspect.signature`.
3. File the Triton reproducer under `repro/` when P3 starts; it is a clean,
   minimal, and upstream-worthy failure.
4. Re-run this note's checks at every pin bump (rule R10.3).

## 5. Method lesson worth keeping

Reading a signature is not running a function. The earlier conclusion was
plausible, specific, and wrong, and it survived several documents before an
actual call disproved it. Anything asserted about a dependency's *behaviour*
gets executed before it is written down.
