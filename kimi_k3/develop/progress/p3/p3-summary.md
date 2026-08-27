# P3 — KDA (COMPLETE)

## 1. Objective

Ship Kimi Delta Attention: an FP32 oracle that *is* the contract, a module whose
parameters match the released checkpoint, a backend switch, and tolerances
derived from measurement.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/attention/kda_eager_fp32.py` | the oracle: gate, L2 norm, gated delta rule, state layout |
| `kimi_k3/attention/kda.py` | `KimiDeltaAttention` + `K3KDASelfAttention` (Megatron's interface) |
| `kimi_k3/attention/kda_backends.py` | `eager` \| `fla` dispatch on one shared kwarg set |
| `kimi_k3/numerics.py` | dtype promotion, shared by both oracles |
| `kimi_k3/tests/tolerance.py` | the three statistics and the measured bounds |
| `kimi_k3/tools/kda_parity_probe.py` | the G15 measurement |
| `kimi_k3/specs/layer_specs.py` | KDA layers now build real KDA |
| `kimi_k3/tests/test_k3_p3_kda_{oracle,parity}.py` | 24 tests |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G14** — oracle | **GREEN** | **bit-identical** to fla's own `naive_recurrent_kda`; 7e-7 vs `chunk_kda` and `fused_recurrent_kda` in fp32; fp64 `gradcheck` passes |
| **G15** — fla parity | **GREEN** at tiny/mid | fp32 7.0e-7 · bf16 4.3e-3 against a 3.3e-3 dtype floor · backward 2.9e-5 fp32 / 6.1e-3 bf16. Production geometry is nightly-owed |
| **G16** — state + checkpoint | **GREEN** | recurrent state carries across a split sequence; `transpose_state_layout` inverts; `sharded_state_dict` and `load_state_dict` round-trip |

## 4. The oracle is validated, not merely written

An oracle nobody cross-checks is a second implementation of the same
misunderstanding. This one is **bit-identical** to fla's `naive_recurrent_kda`
and within 7e-7 of two independent kernels, so the recurrence and the gate were
transcribed correctly rather than approximately. Both halves came from reading
source — the gate from `chunk.py`'s docstring and `fused_recurrent.py:163-176`,
the recurrence from `naive.py` — never from the paper.

Its parameter count also equals `mem_budget.kda_layer_params` exactly, and that
formula was checked against the released checkpoint's tensor shapes. So the
module, the analytic model and the real checkpoint agree three ways.

## 5. Findings

**Error does not grow with sequence length** (256 → 4096 moves the third
significant digit). The opposite is normally assumed for a recurrent model, and
the incoming plan assumed it too. The `lower_bound = -5` gate caps per-step decay
at `exp(-5) ≈ 0.0067`, so old state — and old error — is forgotten faster than it
accumulates. Pinned by a test, because it is the kind of property that would
otherwise be rediscovered by someone debugging a long-context run.

**`.float()` downcasts fp64**, again. The same trap that bit the AttnRes mixer in
P0 bit the KDA oracle in P3: transcribing the release's `.float()` literally
makes gradcheck compare a float64 numerical Jacobian against a float32 analytical
one. Promotion now lives in `kimi_k3/numerics.py` and both oracles import it, so
there is one place to get it right.

**The gate saturates at both ends.** `lower_bound * sigmoid(...)` is open at zero
mathematically but reaches exactly 0 and exactly -5 in floating point for large
inputs. The test asserts the closed interval and, separately, that ordinary
inputs stay strictly inside — otherwise the assertion would be checking floating
point rather than the gate.

**Megatron is sequence-first.** `K3KDASelfAttention` transposes `[s, b, h]` to
`[b, t, h]` once. Getting this wrong would still run — and would attend across
the batch instead of over time — so a test perturbs one batch element and checks
the others do not move.

## 6. Hand-off to P4

P4 replaces the MLA branch of `_layer_spec_for` exactly as P3 replaced the KDA
one. The pattern is established: read the release, build an oracle, validate it
against something independent, measure the floor, then set a bound.

## 7. Commit chain

| commit | scope |
| --- | --- |
| **157d86a18** | P3 T3.1–T3.5: oracle, module, backends, tolerances; G14/G15/G16 green |

**P3 closes here.** Next: P4 (gated MLA with NoPE).
