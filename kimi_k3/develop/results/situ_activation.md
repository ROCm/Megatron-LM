# G52/G53 — the routed experts were running GeGLU, not SiTU

> `pytest kimi_k3/tests/test_k3_p6_moe.py -k situ_activation`
> Found while tracing the elementwise kernels around the backward grouped GEMM
> (G51): the launching op was `aten::gelu` from `moe/experts.py:310 glu`.

## The defect

The release ships `hidden_act: situ` (`activation_situ_beta 4.0`,
`activation_situ_linear_beta 25.0`). `kimi_k3/moe/situ.py` held a faithful
transcription of it, with unit tests that passed. **It was never wired into the
model.** Three paths, three activations, none of them K3's:

| path | activation | used by |
|---|---|---|
| `presets.py` — sets `gated_linear_unit` but never `activation_func`, so core's default applies | **GeGLU** | `build_k3_model`, the proxy, `ep_smoke` — every measurement on this branch |
| `k3_config_builder.py:42` under `--swiglu` | **SwiGLU** | the args path |
| release | **SiTU-GLU** | what it should be |

`grep` for `situ_glu` returned only `k3_qat_experts.py` (a reference module) and
tests. Core's `experts.py:310 glu` calls `self.config.activation_func`, whose K3
default was inherited `F.gelu`.

Severity: not performance. Any training run from this branch optimised the wrong
network — wrong activation on all 896 routed experts, wrong in a numerically
plausible way (GeGLU trains fine, it just is not K3), and a converted checkpoint
would have silently mismatched the release. The two config paths also disagreed,
so args runs and preset runs were not the same model.

## Why nothing caught it

Anchored parity (G32) covered gated MLA (5.82e-03), KDA (7.8/7.4e-03), AttnRes
(bit-identical) and routing (100% identical top-16) — but **never the expert FFN**.
`anchor_moe.py` only fetches expert weight tensors; it never compared an output.
The one component with a bespoke activation was the one component parity skipped.

`situ_glu`'s own unit tests passed throughout, because they test the function in
isolation against the release formula. The function was right; the wiring was
absent. **A correct-but-unreachable function with a green test.**

## The fix

Core's GLU cannot express SiTU. It computes
`activation_func(gate) * (up + glu_linear_offset)` (`mlp.py:324`,
`moe/experts.py:310`), giving the *up* branch no non-linearity — but SiTU applies
`linear_beta * tanh(up / linear_beta)` to it. An `activation_func` callable only
ever sees the gate half, so it is structurally incapable.

The seam that works is `config.use_te_activation_func`: on that path both
`MLP.forward` (`mlp.py:268-271`) and `TEGroupedMLP.bias_act_func`
(`moe/experts.py:274`) hand the **whole** `[gate | up]` tensor to
`self.activation_func` with no chunking — exactly what `situ_glu` wants. Core
already exposes the builder slot (`MLPSubmodules.activation_func`,
`GroupedMLPSubmodules.activation_func`), so this needs **no core diff and no
subclassing**.

- `moe/situ.SituGLU` — the activation as a core-shaped module.
- `k3_transformer_config.__post_init__` sets `use_te_activation_func=True`,
  `bias_activation_fusion=False`, `gated_linear_unit=True`. It also parks
  `activation_func=F.silu`: unused on this path, but core asserts the field is one
  of gelu/silu/relu (`transformer_config.py:1730`), so it must hold a legal value.
- `specs/layer_specs._use_situ` walks routed experts, shared experts and the
  leading dense FFN, since the release applies `situ` to all of them.
- `--k3-situ-activation` / `--no-...` exposes it.

Verified: 28 `SituGLU` instances in the tiny model (dense layer + every local
expert); `SituGLU` on the experts of all three MoE layers in the 4L grouped spec;
and `train_smoke` still descends, 8.3611 -> 0.0125 over 12 steps.

## The gate, and a near-miss inside it

`test_expert_ffn_matches_the_released_situ_activation[grouped=False,True]` builds a
real MoE layer, takes one expert's actual weights, and compares the full FFN
against the released formula written out independently (not by calling `situ_glu`,
so a regression there cannot cancel on both sides).

**The first version of the gate was nearly useless, and the teeth check is what
exposed it.** Measured rel-L2 against SiTU:

| input scale | \|gate\|max | ours | GeGLU | SwiGLU |
|---|---|---|---|---|
| x0.1 | 0.13 | **0.00e+00** | 1.69e-02 | **9.01e-05** |
| x2.0 | 2.82 | **0.00e+00** | 2.02e-01 | 3.38e-02 |
| x10 | 11.84 | **0.00e+00** | 5.77e-01 | 5.61e-01 |

SiTU is *defined* by tanh limiting at `beta = 4`. Below that the three activations
are all near-linear and SwiGLU sits within **9e-05** of SiTU — a gate run at small
scale passes with the wrong activation. The first draft used `x*0.1` and a `2e-2`
tolerance, which would have admitted GeGLU at 1.69e-02.

Fixed by testing in the regime the activation is defined by: `x*10`, tolerance
**1e-5** (both sides are the same fp32 arithmetic — measured 0.00e+00), teeth
threshold **1e-1**. The test also asserts `|gate|max > k3_situ_beta`, so it fails
loudly if a future width or init change drifts it back out of the discriminating
region rather than silently going blind.

## Follow-up

- Every convergence result on this branch predates the fix and was obtained with
  GeGLU experts: the twin runs, the QAT convergence study, the flatness probes.
  None is wrong as a *mechanical* result, but none describes K3's network.
- `anchor_moe.py` still does not compare expert output against released weights.
  The gate above anchors against the formula, not the checkpoint; a released-weight
  expert parity remains the stronger check and is not done.
