# 2026-08-27 — Release artefact audit (P0-T0.2, gate G2)

`analysis` · closes the four open ❓ items in
[`../architecture/01-kimi-k3-architecture-deep-dive.md`](../architecture/01-kimi-k3-architecture-deep-dive.md).

Source: `HF moonshotai/Kimi-K3` at revision
`a590ce090cb049c93a33dfe8c208ec652aa20503` (lastModified 2026-08-20). Read:
`config.json`, `configuration_kimi_k3.py`, `modeling_kimi_linear.py`,
`model.safetensors.index.json`, and four safetensors **headers** via HTTP range
requests (shards 1, 4, 13, 94) plus 512 bytes of tensor data for one `A_log`.
No shard was downloaded in full.

## (a) `A_log` is `[128]` with the last 32 exactly zero — CONFIRMED

| | |
|---|---|
| declared in modeling code | `Parameter(log(uniform(1, 16)))`, shape `[num_heads] = [96]` |
| in the checkpoint | `F32 [128]`, one per KDA layer (69 of them) |
| `A_log[96:128]` | all exactly `0.0` (verified on layer 12: `max|·| = 0.0`) |
| `exp(A_log[:96])` | 0.626 … 2.736 — a trained distribution, not the init |

So the release pads the head axis to 128 for kernel alignment. Converter rule:
**assert `A_log[96:] == 0` then trim `[128] → [96]` on import; zero-pad on export.**
`dt_bias` is `F32 [12288]` with no padding.

Also worth carrying: `A_log`, `dt_bias`, the three `*_conv1d.weight` tensors and
`o_norm.weight` are stored **F32**, not bf16.

## (b) The MLA LoRA norms use eps 1e-6, not `rms_norm_eps` — CONFIRMED

```python
class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):     # modeling_kimi_linear.py:227
```

`q_a_layernorm = KimiRMSNorm(self.q_lora_rank)` and
`kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` are constructed **without**
`eps`, so they take 1e-6 while every other K3 norm is passed
`eps=config.rms_norm_eps = 1e-5`. Review finding B7 is real. The config carries
`k3_mla_lora_norm_eps = 1e-6` and a test pins it.

## (c) Key layout — exact, and the vision skip-list is now a number

497,220 tensors across 96 shards. Top-level prefixes:

| prefix | tensors | disposition |
|---|---:|---|
| `language_model.model.*` | 497,051 | mapped |
| `language_model.lm_head.weight` | 1 | mapped |
| `vision_tower.*` | 165 | **explicitly skipped** |
| `mm_projector.*` | 3 | **explicitly skipped** |

so the converter's skip-list must account for exactly **168** tensors, and
"zero unmapped tensors" (G30) means 497,052 mapped + 168 skipped.

Per-layer names differ from the guesses in
[`../plan-0/02-target-architecture.md`](../plan-0/02-target-architecture.md) §9:

- **KDA and MLA both live under `self_attn`** — not `linear_attn`. Layer kind is
  told apart by which tensors exist (`A_log`/`q_proj` vs `q_a_proj`/`kv_b_proj`).
- MoE module is **`block_sparse_moe`**, not `mlp`; the dense layer 0 uses `mlp`.
- Routed experts: `block_sparse_moe.experts.{e}.{w1,w2,w3}.weight_packed` +
  `.weight_scale` (MXFP4 + e8m0). `w1`/`w3` packed `[3072, 1792]` = logical
  `[3072, 3584]`; `w2` packed `[3584, 1536]` = logical `[3584, 3072]`; scales
  `[3072, 112]` / `[3584, 96]` = one per 32-wide group.
- Shared experts: `shared_experts.{gate,up,down}_proj` at **`[6144, 7168]`** —
  hidden width, confirming finding B6.
- AttnRes: `self_attention_res_{norm,proj}` and `mlp_res_{norm,proj}` per layer
  (187 of each = 93 × 2 + 1 model-level `output_attn_res_*`), `res_proj` shape
  `[1, 7168]`.

The mapping table in `02-target-architecture.md` §9 has been corrected to match.

## (d) Tokenizer

From `config.json`: `vocab_size 163840`, `bos 163584`, `eos 163586`,
`pad 163839`, `media_placeholder_token_id 163605`. The repo ships
`tiktoken.model`, `tokenization_kimi.py`, `encoding_k3.py` and
`tokenizer_config.json`; P8-T8.2 reads ids from those rather than from the config.

## Fixture produced

`kimi_k3/tests/fixtures/release_shapes.json` — 50 tensors, shapes and dtypes
only (6 KB, no weight data), covering one KDA layer, one MLA layer, the MoE
non-expert tensors, one expert triple, the dense layer, the per-layer norms and
every model-level tensor. `test_k3_p0_param_count.py` checks every formula in
`tools/mem_budget.py` against it, so the analytic oracle is anchored to the real
checkpoint rather than to the paper.
