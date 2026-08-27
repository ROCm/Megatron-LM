# 2026-08-27 — `fla` `chunk_kda` signature does not match the release

`troubleshooting` · blocks gate **G1** · owner: P0-T0.1

## What the release calls

`modeling_kimi_linear.py`, `KimiDeltaAttention.forward`:

```python
chunk_kda(q=q, k=k, v=v, g=g, beta=beta,
          A_log=self.A_log, dt_bias=self.dt_bias,
          initial_state=recurrent_state, output_final_state=True,
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          safe_gate=self.gate_lower_bound is not None,
          lower_bound=self.gate_lower_bound,          # -5.0
          transpose_state_layout=True, cu_seqlens=cu_seqlens)
```

## What `fla` actually exposes

`fla/ops/kda/chunk.py` on `main`:

```python
def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False,
              use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              allow_neg_eigval=False, safe_gate=False, lower_bound=None,
              disable_recompute=False, return_intermediate_states=False,
              state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
              cp_context=None, **kwargs)
```

Three mismatches:

1. **no `A_log`** and **no `dt_bias`** — the decay parameters the release passes;
2. **`state_v_first`**, not `transpose_state_layout`;
3. the signature ends in **`**kwargs`**, so all three unknown arguments are
   **silently accepted and ignored**. Nothing raises. `fla`'s own
   `fla/layers/kda.py` calls the op with `state_v_first=True` and no `A_log`,
   confirming which spelling that revision expects.

The PyPI wheel `flash-linear-attention==0.5.2` ships only `fla/layers` and
`fla/models` — `fla/ops` is absent from the wheel entirely — so `pip install`
of that version cannot satisfy the import either.

## Why this matters more than a normal version skew

A wrong-but-accepted call runs KDA with the learned decay ignored. There is no
exception, no warning, and the loss still falls: exactly the class of failure the
project's oracle-first rule exists to catch. It would surface as an unexplained
parity gap in G15 — or not at all, if the fla backend were ever made the default
before that gate was green.

## Actions

1. **G1's check asserts by parameter name.** `tools/check_fla_signature.py`
   compares `inspect.signature(chunk_kda).parameters` against the released kwarg
   set and fails on any missing name — never "the call did not raise".
2. **Find the revision the release was built against.** Candidates: a Moonshot
   fork, or an fla commit predating the `transpose_state_layout` → `state_v_first`
   rename. The HF repo pins no fla version (no `requirements.txt`, no mention in
   the model card), so this has to be established by bisecting the fla history for
   a `chunk_kda` that takes `A_log`/`dt_bias`/`transpose_state_layout`.
3. **Until then `k3_kda_backend` stays `eager`** (rule R5.3). The FP32 oracle is
   defined by the *released call's semantics*, not by fla's current signature, so
   P3 can proceed and G15 becomes the gate that consumes the resolved pin.
4. If no such revision exists publicly, the fallback is to compute the decay from
   `A_log`/`dt_bias` **outside** the kernel and pass the pre-gated `g`, matching
   whatever `use_gate_in_kernel=False` expects — an eager/fla hybrid whose parity
   is then G15's problem. Record the decision in `PINS.md` either way.
