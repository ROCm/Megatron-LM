# P1 — Scaffold and environment (COMPLETE)

## 1. Objective

Make the core-isolation promise enforceable, and make an IFU break loudly rather
than silently.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/ci/no_core_diff_guard.sh` | the guard: merge-base by default, `--staged` for pre-commit, an explicit range for CI |
| `kimi_k3/tests/test_k3_p1_pin_contracts.py` | 8 pin contracts, one test each, plus 6 hermetic guard tests |
| `kimi_k3/model/core_patch.py` | contracts restructured into a named `PIN_CONTRACTS` table so a failure says which one |
| `kimi_k3/tests/test_env.py`, `kimi_k3/tools/capture_env.py` | dependency pins asserted and the environment table generated |
| `.github/workflows/kimi_k3.yml` | CI stages 0 and 1 (allowlisted root file) |
| `.pre-commit-config.yaml` | local `kimi-k3-no-core-diff` hook (allowlisted root file) |
| `kimi_k3/README.md` | package entry point |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G9** — guard | **GREEN** | passes on the branch (59 files); rejects a staged core edit (exit 1); honours exactly the two allowlisted root files; rejects any other root file; merge-base semantics proven with a simulated IFU |
| **G10** — env + pin contracts | **GREEN** | 8 contracts green; torch 2.10 / triton 3.7.1 / TE 2.12 / fla 0.6.0 / 8× MI355X |

`pytest kimi_k3/tests/ -q` → **80 passed, 1 skipped** (AITER K3 kernels, pending
the checkout bump P10 owns).

## 4. Two decisions worth recording

**The guard compares against the merge-base, not `rocm_dev`'s tip.** After an IFU
rebase the branch legitimately contains upstream's core commits; a tip comparison
would flag every one of them and the guard would be turned off within a week. A
test simulates exactly that (upstream moves core → we merge → we add `kimi_k3/`
work → guard still passes).

**The contracts are a named table, not one assertion block.** Each is its own
parametrised test, so an IFU failure names the mechanism — "MoESubmodules.router
+ MoELayer.postprocess" — and its message says what to re-check rather than only
that something changed. The eighth contract came out of P0: MLA passes
`k_channels` to a `core_attention` that core's own `DotProductAttention` rejects
(finding A13), which is why construction gates need TE and a GPU.

## 5. Handed to a human

The CI workflow's `runs-on` labels and container image are `TODO(maintainer)`:
this repository's runners are not visible from the branch. Everything else in it
is ready — both stages are `pytest` invocations plus the guard. The pre-commit
hook needs `pre-commit install` once, per the plan's convention that a human
registers hooks.

## 6. Hand-off to P2

P2 promotes the P0 config/preset/construction work from prototype to shipped
code and adds the layer-pattern and parameter-count gates (G11–G13). Both already
have passing implementations from P0; P2 is where they get their final form and
their own test files.

## 7. Commit chain

| commit | scope |
| --- | --- |
| **c96fad823** | P1 T1.1–T1.5: guard, pin contracts, env checks, CI wiring; G9 and G10 green |

**P1 closes here.** Next: P2 (config, presets, model construction; G11–G13).
