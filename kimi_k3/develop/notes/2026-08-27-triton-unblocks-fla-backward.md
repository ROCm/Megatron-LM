# 2026-08-27 — triton 3.7.1 unblocks the KDA backward (G1 green)

`troubleshooting` · closes gate **G1** · follows
[`2026-08-27-fla-signature-check.md`](2026-08-27-fla-signature-check.md)

## The question asked

"Install a PyTorch ROCm version whose Triton unblocks the fla backward."

## The answer

**No PyTorch change is needed, and one would have been costly.** The container's
TransformerEngine, apex and AITER are all built against `torch 2.10.0+git94c6e04`;
replacing torch would rebuild-or-break every one of them, and the whole test suite
depends on TE for the MLA spec. The failure was never torch's — it was a Triton
MLIR pass, and Triton is a separate package.

Upgrading **Triton alone, from 3.6.0 to 3.7.1**, fixes it:

| | forward | backward |
|---|---|---|
| triton 3.6.0 | OK | `RuntimeError: PassManager::run failed` at `fla/ops/kda/chunk_intra.py:395` (`TritonAMDGPUPipeline`) |
| **triton 3.7.1** | OK | **OK** — all of `q, k, v, g, beta, A_log, dt_bias` receive finite, non-zero gradients |

Verified at `(T, H, K) = (128, 4, 64)` and `(2048, 8, 128)`.

## How it was done safely

1. Installed the candidate into an isolated directory (`pip install --target
   /tmp/triton371 --no-deps triton==3.7.1`) and ran the repro with `PYTHONPATH`
   shadowing — no change to the environment.
2. Re-ran the **whole** `kimi_k3` suite and a TE `Linear` forward+backward under
   the shadowed version: 52 passed, TE fine.
3. Only then installed it for real. Revert is `pip install triton==3.6.0`.

## Does the Triton upgrade break `torch.compile`?

A fair question, because inductor generates Triton code against Triton's internal
APIs — and because **Megatron puts `torch.compile` on the default training path**,
not an opt-in one: `megatron/core/jit.py` sets `jit_fuser = torch.compile` for
torch >= 2.2, and that decorator is applied across `fusions/fused_bias_swiglu`,
`fused_bias_dropout`, `activations`, `torch_norm`, `moe/router`,
`moe/token_dispatcher` and `ssm/gated_delta_net`. If torch and Triton disagreed,
ordinary K3 training would break, not just a future experiment.

Checked on torch 2.10.0+git94c6e04 + triton 3.7.1, all green:

| check | result |
|---|---|
| `torch.utils._triton.has_triton()` / `torch._inductor.runtime.triton_compat` | OK |
| compiled elementwise + reduction, forward and backward | OK, matches eager |
| compiled softmax | OK |
| **`mode="max-autotune"`** matmul, fwd+bwd | OK — 37 Triton MM choices autotuned with AMD-specific params (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`), max diff 1.95e-3 vs eager, i.e. bf16 noise |
| Megatron's `jit_fuser` path (`bias_swiglu_impl`) fwd+bwd | OK |
| the whole `kimi_k3` suite | 58 passed |

Our own code calls `torch.compile` nowhere today; the exposure is entirely
through core's fusions, and through P11 if the fused AttnRes mixer takes the
`torch.compile` route (which is the cheap baseline to try before a hand-written
kernel).

`tests/test_k3_p0_torch_compile_contract.py` keeps this checked continuously —
including an assertion that core still routes its fusions through
`torch.compile`, so if that changes we find out rather than assume.

## Caveats worth carrying

* **This is upstream PyPI Triton, not the ROCm-vendored build.** The ROCm index
  (`download.pytorch.org/whl/rocm7.2`) tops out at `pytorch-triton-rocm 3.5.1`,
  which is *older* than the 3.6.0 we started from, so it cannot help. The PyPI
  wheel ships the AMD backend and works here, but it is not the vendored
  configuration: re-validate whenever torch or ROCm moves, and confirm kernel
  quality (not just correctness) when P11 measures performance.
* **Correctness is asserted, not yet parity-checked.** The backward now compiles
  and produces finite non-zero gradients. Whether it is *right* is G15's job,
  against the FP32 oracle at production shapes — the fla KDA-backward bug history
  (#807, #785) is exactly why the oracle stays in tree permanently (rule R8.1).
* A gradient sanity check must use a `sum` loss, not `mean`: averaging over
  millions of elements puts gradient magnitudes near 1e-6, which reads as zeros
  at ordinary print precision. That cost one false alarm here.

## What is now a gate rather than a claim

`kimi_k3/tests/test_k3_p0_fla_contract.py` calls `chunk_kda` exactly as the
release does and asserts behaviour: forward runs; `A_log` demonstrably changes
the output; `transpose_state_layout` is honoured; a missing `A_log` raises rather
than being ignored; and the backward compiles with finite non-zero gradients at
two shapes. Functional, never signature-based (rule R3.6).
