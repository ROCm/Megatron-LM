# P2 — Config, presets, model construction (COMPLETE)

## 1. Objective

Promote P0's prototypes to the shipped path, and pin the layer pattern — the
single likeliest place for a silent error in the whole project.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/config/arguments.py` | `add_kimi_k3_args`, plus `explicitly_set` / `apply_preset_defaults` so a preset fills gaps without overruling the command line |
| `kimi_k3/config/k3_transformer_config.py` | `k3_first_k_dense_replace` added |
| `kimi_k3/specs/layer_specs.py` | the layer **plan** (pure config) and the per-layer **specs**; core accepts heterogeneous `layer_specs` natively |
| `kimi_k3/model/build.py` | `build_k3_model(preset)` — one call from a preset name to a model, and the official-preset guard |
| `kimi_k3/tests/test_k3_p2_{specs,arguments,construction}.py` | 26 tests |

## 3. Gates

| Gate | Status | Numbers |
|---|---|---|
| **G11** — layer pattern | **GREEN** | 93 L: 69 KDA / 24 MLA matching the literal released list; layer 0 KDA + dense at 33792; the tail `M M` at 1-indexed 92/93; slots at 0/12/…/84 |
| **G12** — config + construction | **GREEN** | `build_k3_model` gives a `K3TransformerBlock` with heterogeneous layers; config type survives; mixers per layer + at the output |
| **G13** — parameter count | **GREEN** | 4L 94.0 B · 8L 214.7 B · 93L 2.779 T, each within 2 %; per-component subtotals cross-checked against the plan |

`pytest kimi_k3/tests/ -q` → **106 passed, 1 skipped**.

## 4. Two bugs the tests caught

**A preset was clobbering explicit flags.** `--k3-preset 93L --k3-attn-res-block-size 6`
silently produced 12. argparse cannot distinguish a defaulted value from an
explicitly passed one after the fact, so `apply_preset_defaults` now re-parses
with every default suppressed and skips whatever the caller actually typed.

**Meta-device construction does not avoid allocation** (finding **A14**).
Megatron and TE modules place parameters on `torch.cuda.current_device()`
explicitly, ignoring an ambient `torch.device("meta")` context: on the tiny
preset, 93 of 114 parameters land on the GPU anyway — only our own AttnRes
mixers honour it. The incoming plan's "official presets are validated by
meta-device construction + analytic parameter counting" is therefore half
unavailable; at 93 L it would try to materialise 2.78 T parameters.
`build_k3_model` now **refuses** a non-tiny preset without an explicit
`allow_official=True`, so rule R4.3 is enforced rather than merely written down.

## 5. Design note: plan before spec

`k3_layer_plan()` answers *what each layer is* from the config alone — no CUDA,
no distributed state, no model — so the pattern is pinned in CI stage 0 at
essentially zero cost. `get_k3_layer_specs()` turns that into modules, and P3/P4
change only that half when the real KDA and gated-MLA modules exist. Until then a
KDA layer is built from core's MLA spec, marked as a placeholder in the module
docstring; the *plan* is already correct, so the gate is real today.

## 6. Hand-off to P3

P3 (KDA) replaces one branch of `_layer_spec_for` and nothing else in this phase.
The FP32 oracle it must match is defined by the released call's semantics; the
fla backend is available (triton 3.7.1) but stays off by default until G15.

## 7. Commit chain

_pending review (rule R1.1)._
