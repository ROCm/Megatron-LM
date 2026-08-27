# 02 — Target Architecture

> How Kimi K3 maps onto Megatron core's `config + spec + submodule + build_module`
> pattern, expressed as the interfaces we commit to. Ground truth for the model
> itself is [`../architecture/01-kimi-k3-architecture-deep-dive.md`](../architecture/01-kimi-k3-architecture-deep-dive.md);
> the file list is [`03-code-layout.md`](03-code-layout.md).
> Code below is **design sketch**, not final source — but every core symbol it
> names was verified to exist at the pin.

## 1. Module map

| K3 release (`modeling_kimi_linear.py`) | Our target | Megatron parent / mechanism | Phase |
|---|---|---|---|
| `KimiLinearConfig` (text_config) | `KimiK3TransformerConfig` | `MLATransformerConfig` (**never** built by `core_transformer_config_from_args`) | P2 |
| `KimiLinearForCausalLM` | `K3GPTModel` | `GPTModel` + scoped block rebinding | P2 |
| `KimiLinearModel` decoder loop + `_apply_output_attn_res` | `K3TransformerBlock` | `TransformerBlock` subclass | P5 |
| `KimiDecoderLayer._forward_attn_residual` | `K3TransformerLayer` + `AttnResMixer` | `TransformerLayer` subclass + spec submodule | P5 |
| `KimiDeltaAttention` | `KimiDeltaAttention` (ours) | `MegatronModule`, spec submodule of the layer | P3 |
| `chunk_kda` / eager oracle | `kda_backends.py` | runtime dispatch, eager default | P3 |
| `KimiMLAAttention` | `K3GatedMLA` | `MLASelfAttention` subclass | P4 |
| `KimiSparseMoeBlock` | `K3MoELayer` | `MoELayer` subclass (`postprocess` override) | P6 |
| `KimiMoEGate` | `QuantileBalancingRouter` | `TopKRouter` subclass via `MoESubmodules.router` | P6 |
| `SituAndMul` | `situ_glu()` | callable in `MLPSubmodules.activation_func` | P6 |
| `KimiBlockSparseMLP` (routed) | core `TEGroupedMLP` / `GroupedMLP` | unchanged, at latent width 3584 | P6 |
| `KimiMLP` (shared) | core `SharedExpertMLP` | unchanged, at hidden width 7168 | P6 |
| (quantized experts) | `KimiK3QATGroupedMLP` | expert submodule spec | P10 |
| — | `PerHeadMuon` | wraps core `get_megatron_muon_optimizer` + `clip_qk` | P9 |
| — | `K3StateDictAdapter` | new (`tools/mapping.py`) | P8 |

## 2. Config and its construction (P2)

```python
@dataclass
class KimiK3TransformerConfig(MLATransformerConfig):
    # --- KDA ---
    k3_kda_layers: tuple[int, ...] = ()        # 1-indexed, from linear_attn_config.kda_layers
    k3_kda_num_heads: int = 96
    k3_kda_head_dim: int = 128
    k3_kda_conv_size: int = 4
    k3_kda_gate_lower_bound: float = -5.0
    k3_kda_use_full_rank_gate: bool = True
    k3_kda_backend: str = "eager"              # eager | fla   (R8.1)
    # --- MLA ---
    k3_mla_use_nope: bool = True
    k3_mla_use_output_gate: bool = True
    k3_mla_lora_norm_eps: float | None = None  # None -> rms_norm_eps; set if B7 resolves otherwise
    k3_mla_fp32_attn_output: bool = True       # [report]
    # --- AttnRes ---
    k3_attn_res_block_size: int = 12
    k3_attn_res_fp32: bool = True
    k3_attn_res_fused: bool = False            # P11
    # --- MoE ---
    k3_routed_expert_hidden_size: int = 3584   # mirrored into moe_latent_size
    k3_latent_moe_use_norm: bool = True
    k3_situ_beta: float = 4.0
    k3_situ_linear_beta: float = 25.0
    k3_router_quantile_balancing: bool = True
    k3_qb_num_bins: int = 1024
    # --- QAT ---
    k3_qat_experts: bool = False               # P10
    k3_qat_stochastic_rounding: bool = False
```

`k3_config_builder.k3_config_from_args(args)` mirrors the *field-collection* loop
of `core_transformer_config_from_args` but never calls it — at
`arguments.py:1230-1232` core replaces any caller-supplied `config_class` with
`MLATransformerConfig` whenever `args.multi_latent_attention` is set, which would
silently discard our subclass. **G11** asserts `type(config) is
KimiK3TransformerConfig`, that the MLA fields are populated, and that
`core_transformer_config_from_args` never appears on the call stack (patched
sentinel during the test).

### 2.1 Field mapping from the released `config.json`

| Release field | Our config / Megatron arg |
|---|---|
| `hidden_size 7168` | `hidden_size` |
| `num_hidden_layers 93` | `num_layers` |
| `linear_attn_config.kda_layers` (1-indexed) | `k3_kda_layers`; layer kind resolved as `is_kda = (layer_idx + 1) in k3_kda_layers` |
| `q_lora_rank 1536 / kv_lora_rank 512 / qk_nope_head_dim 128 / qk_rope_head_dim 64 / v_head_dim 128` | MLA fields inherited from `MLATransformerConfig` |
| `mla_use_nope true` | `k3_mla_use_nope` (core `rope_type` is left at its default and **never applied** — see §5) |
| `num_experts 896 / num_experts_per_token 16` | `num_moe_experts`, `moe_router_topk` |
| `moe_intermediate_size 3072` | `moe_ffn_hidden_size` |
| `routed_expert_hidden_size 3584` | `moe_latent_size` **and** `k3_routed_expert_hidden_size` |
| `num_shared_experts 2` | `moe_shared_expert_intermediate_size = 2 * 3072 = 6144` (on **hidden**, finding B6) |
| `moe_router_activation_func sigmoid` | `moe_router_score_function = "sigmoid"` |
| `topk_method noaux_tc` | `moe_router_enable_expert_bias = True` |
| `moe_renormalize true` | `moe_router_topk_scaling_factor = 1.0` + renormalisation in the router |
| `num_expert_group 1 / topk_group 1` | `moe_router_num_groups = None` (no group routing) |
| `first_k_dense_replace 1` | `moe_layer_freq` pattern with layer 1 dense, `ffn_hidden_size = 33792` |
| `rms_norm_eps 1e-5` | `layernorm_epsilon` (plus `k3_mla_lora_norm_eps` if B7 says otherwise) |
| `attn_res_block_size 12` | `k3_attn_res_block_size` |
| `tie_word_embeddings false` | `untie_embeddings_and_output_weights = True` |

## 3. Model construction without touching core (P2)

`GPTModel.__init__` constructs `TransformerBlock` directly
(`gpt_model.py:209`), and the module-level import lives at `gpt_model.py:34`.
Rather than duplicating ~190 lines of `__init__` (rotary variants, MTP, output
layer, tying, offloading), we rebind the symbol for the duration of construction:

```python
# kimi_k3/model/core_patch.py
@contextlib.contextmanager
def k3_block_class(block_cls):
    """Rebind the symbol GPTModel resolves when it builds its decoder.

    No file under megatron/** is modified (R2.2). Scoped so nothing else in the
    process ever sees the rebound symbol. Guarded by test_k3_p1_pin_contracts.py,
    which fails if gpt_model stops resolving TransformerBlock at module scope.
    """
    import megatron.core.models.gpt.gpt_model as gm
    original = gm.TransformerBlock
    gm.TransformerBlock = block_cls
    try:
        yield
    finally:
        gm.TransformerBlock = original


class K3GPTModel(GPTModel):
    def __init__(self, config, transformer_layer_spec, **kw):
        with k3_block_class(K3TransformerBlock):
            super().__init__(config, transformer_layer_spec, **kw)
        assert isinstance(self.decoder, K3TransformerBlock)

    def set_input_tensor(self, input_tensor):
        # GPTModel asserts len(input_tensor) == 1 (gpt_model.py:288); we keep the
        # single-tensor contract but the tensor is the PACKED AttnRes payload.
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.decoder.set_input_tensor(input_tensor[0])
```

**G12** asserts (a) `model.decoder` is a `K3TransformerBlock`, (b) **no core
`TransformerBlock` instance was ever constructed** (constructor spy installed for
the duration of the test), (c) the rebinding is uninstalled afterwards.
The full-`__init__`-override variant stays documented as the fallback if a future
IFU makes the rebinding unsafe.

## 4. AttnRes: block, layer, and the pipeline payload (P5)

### 4.1 State contract

Two tensors flow through the decoder (architecture §6):

| name | shape (Megatron `[s, b, h]` layout) | meaning |
|---|---|---|
| `prefix_sum` | `[S, B, H]` | running sum inside the current block; **this is the hidden state** |
| `block_residual` | `[S, B, K, H]` | `K` frozen block outputs, `K = ceil(l / block_size)` before layer `l` |

`K` grows monotonically: a slot is appended by every layer whose 0-indexed global
index satisfies `l % 12 == 0`. For 93 layers: 8 slots.

### 4.2 Packing protocol (`attn_res_pp.py`)

Core's `backward_step` back-props **only** `output_tensor[0]`
(`schedules.py:451-493`), so a two-tensor payload would lose `block_residual`'s
gradient silently (finding A1). The payload is therefore **one tensor**,
concatenated along the sequence axis so the P2P shape stays a
`(seq, mbs, hidden)` triple that core's `p2p_communication` already understands:

```python
def pack(prefix_sum, block_residual):        # [S,B,H], [S,B,K,H] -> [(1+K)*S, B, H]
    S, B, K, H = block_residual.shape
    return torch.cat([prefix_sum, block_residual.permute(2, 0, 1, 3).reshape(K * S, B, H)], dim=0)

def unpack(packed, S, K):                    # inverse; autograd-transparent
    prefix_sum = packed[:S]
    block_residual = packed[S:].view(K, S, *packed.shape[1:]).permute(1, 2, 0, 3)
    return prefix_sum, block_residual

def slots_before(layer_idx, block_size):     # K seen by a layer on entry
    return math.ceil(layer_idx / block_size)
```

Both `cat` and `view/permute` are differentiable, so a single
`torch.autograd.backward` on the packed tensor delivers gradients to
`prefix_sum` **and** every slot. **G20 is an explicit gradient-flow test that
fails if a slot gradient is zero when it should not be.**

### 4.3 Per-stage shapes (`k3_schedule.py`)

Core already supports per-rank shape adjustment in the 1F1B schedule
(`schedules.py:2019, 2180-2183`); it is simply never bound outside NVIDIA-modelopt
distillation (`training.py:1815-1853`). We bind it from our own entry point:

```python
def k3_adjust_tensor_shapes(recv_shapes, send_shapes):
    l_first, l_last = local_layer_range()            # global 0-indexed
    k_in  = slots_before(l_first, block_size)
    k_out = slots_before(l_last + 1, block_size)
    return ([_scale_seq(recv_shapes[0], 1 + k_in)],
            [_scale_seq(send_shapes[0], 1 + k_out)])
```

Neighbour consistency is automatic: stage `s` sends with `1 + ceil((l_last+1)/12)`
and stage `s+1` receives with `1 + ceil(l_first'/12)` where `l_first' = l_last + 1`.

The binding is **conditional on `PP > 1`**: `adjust_tensor_shapes_fn` is asserted
`None` in the no-pipelining schedule (`schedules.py:631`) and in the interleaved
schedule (`:949`) — so VPP stays descoped and single-GPU tests keep working
(finding A3).

Payload bandwidth/memory multiplier is `1 + K` (up to **9×**); the measured
numbers land in `06-capacity-and-parallelism.md` from gate G6.

### 4.4 Block and layer

```python
class K3TransformerBlock(TransformerBlock):
    """Owns the AttnRes state. Keeps `self.layers` (core's clip_qk walks it)."""

    def set_input_tensor(self, packed):        # packed payload from P2P (or None on stage 0)
        self.input_tensor = packed

    def forward(self, hidden_states, ...):
        if self.input_tensor is not None:
            prefix_sum, block_residual = unpack(self.input_tensor, S, self.k_in)
        else:
            prefix_sum, block_residual = hidden_states, new_empty_slots()
        for layer in self.layers:
            prefix_sum, block_residual = layer(prefix_sum, block_residual, ...)
        if self.post_process:                  # last stage: collapse and finish
            hidden = self.output_attn_res(prefix_sum, block_residual)
            return self.final_layernorm(hidden)
        return pack(prefix_sum, block_residual)
```

`K3TransformerLayer` implements the exact sequence in architecture §6 (mix → maybe
append slot and reset `prefix_sum` → norm → attention → accumulate → mix → norm →
MoE → accumulate). Recompute wraps the layer with **both** state tensors as
checkpoint inputs; `k3_attn_res_fp32` controls the mixer dtype (R7.3).

## 5. KDA and gated MLA (P3, P4)

```python
@dataclass
class KimiDeltaAttentionSubmodules:
    q_proj: ModuleSpec; k_proj: ModuleSpec; v_proj: ModuleSpec
    q_conv1d: ModuleSpec; k_conv1d: ModuleSpec; v_conv1d: ModuleSpec
    f_a_proj: ModuleSpec; f_b_proj: ModuleSpec      # low-rank DECAY gate
    g_proj: ModuleSpec                              # full-rank OUTPUT gate
    b_proj: ModuleSpec                              # beta
    o_norm: ModuleSpec                              # gated RMSNorm(head_dim, sigmoid)
    o_proj: ModuleSpec
```

The backend call mirrors the release kwarg-for-kwarg (architecture §3), including
`use_qk_l2norm_in_kernel=True` (finding B4). `kda_eager_fp32.py` implements the
same math in FP32 with explicit L2-norm, `sigmoid(beta)`, `A_log`/`dt_bias` decay
and the `lower_bound=-5` clamp, so the oracle and the kernel share one contract.
`sharded_state_dict` covers `A_log`, `dt_bias` and the conv weights.

```python
class K3GatedMLA(MLASelfAttention):
    # Core MLA only knows rope_type in {"rope","yarn"} (multi_latent_attention.py:193-215)
    # and always applies rotary. K3 is NoPE:
    #   - rotary application is bypassed entirely (the 64 "rope" dims are shared content,
    #     produced once MQA-style and expanded across heads, never rotated)
    #   - softmax_scale = (qk_nope + qk_rope) ** -0.5 = 192 ** -0.5
    #   - full-rank sigmoid output gate multiplies attn_output BEFORE linear_proj
    #   - attention output kept in fp32 when k3_mla_fp32_attn_output  [report]
    #   - exposes `clip_qk` so core's optimizer/qk_clip.py:clip_qk() picks it up (finding A8)
```

## 6. MoE stack (P6)

```python
class K3MoELayer(MoELayer):
    """Core's LatentMoE minus one norm.

    moe_layer.py:505-515 does fc2_latent_proj(output) with no normalisation;
    K3 applies RMSNorm(3584, eps=rms_norm_eps) to the combined expert output
    BEFORE the up-projection (finding A10).
    """
    def postprocess(self, output, shared_expert_output):
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output = self.routed_expert_norm(output)      # <-- the delta
            output, _ = self.fc2_latent_proj(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output
```

Everything else is core: router on the 7168-wide hidden **before** the latent
down-projection (matching the release), grouped GEMM experts at 3584→3072→3584,
shared experts at 7168 with intermediate 6144, alltoall dispatcher.
`moe_shared_expert_overlap` must stay off (`moe_layer.py:422-424`).

```python
class QuantileBalancingRouter(TopKRouter):
    """Sigmoid scores; e_score_correction_bias added for SELECTION only; weights
    gathered from the unbiased scores; renormalised; scaled by 1.0.
    Quantile Balancing [report] updates the bias from a running per-expert score
    histogram (k3_qb_num_bins) so each expert's selection quantile tracks its
    target share. Injected through MoESubmodules.router (verified builder field).
    """
```

SiTU-GLU (fp32 math, cast back at the end):

```python
def situ_glu(gate_up, beta=4.0, linear_beta=25.0):
    gate, up = gate_up.chunk(2, dim=-1)
    gate, up = gate.float(), up.float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    return (a * (linear_beta * torch.tanh(up / linear_beta))).to(gate_up.dtype)
```

## 7. QAT expert module (P10)

```python
class KimiK3QATGroupedMLP(TEGroupedMLP):
    """W-MXFP4 / A-MXFP8 forward, high-precision STE backward.

    forward : quantise activations to MXFP8 (per_1x32, e8m0) and use the cached
              packed MXFP4 weights -> AITER a8w4 fused MoE (SiTU betas passed through).
    backward: STE. dgrad/wgrad run in bf16 / blockwise-FP8 grouped GEMM against
              DEQUANTISED weights. No FP4 backward exists (no transposed-scale
              weight copy for dgrad; no weight operand for wgrad).
    state   : fp32 masters; the packed MXFP4 cache is refreshed on optimizer step;
              packed data + e8m0 scales round-trip through the checkpoint.
    """
```

Validation is against an explicit BF16 fake-quant module with identical
quantisation and ordinary autograd — **never `gradcheck`** (R4.4).

## 8. What we consume from core unchanged

| Mechanism | Symbol | Why it matters |
|---|---|---|
| PP payload shapes | `schedules.adjust_tensor_shapes_fn` | removes the need for a custom schedule (finding A2) |
| Router injection | `MoESubmodules.router` | `QuantileBalancingRouter` drops in |
| LatentMoE | `config.moe_latent_size` + `fc1/fc2_latent_proj` | 7168↔3584 projections (TE required, `parallel_mode="duplicated"`) |
| Sigmoid + noaux_tc routing | `TopKRouter` + `moe_router_enable_expert_bias` | matches the release exactly |
| Grouped GEMM + dispatchers | `TEGroupedMLP`, `MoEAlltoAllTokenDispatcher` | 896 experts, EP |
| QK-clip | `optimizer/qk_clip.py:clip_qk` + `--qk-clip*` | Muon-recommended; we only expose the hook |
| Muon | `get_megatron_muon_optimizer`, `LayerWiseDistributedOptimizer` | `dist_muon` shards master+momentum across DP (finding A6) |
| Routing determinism | `moe/router_replay.py` | reproducible expert assignment in tests |
| Checkpointing | dist-checkpointing / `sharded_state_dict` | all new modules implement it |

## 9. State-dict mapping (P8, sketch)

Release keys are nested under the multimodal wrapper; the text tower is what we
map, and vision keys are **explicitly skipped, not ignored by omission**
(finding B8).

| Release key (per 0-indexed layer `i`) | Megatron key |
|---|---|
| `model.language_model.layers.{i}.self_attn.q_a_proj.weight` | `decoder.layers.{i}.self_attention.linear_q_down_proj.weight` |
| `…self_attn.q_a_layernorm.weight` | `…self_attention.q_layernorm.weight` |
| `…self_attn.q_b_proj.weight` | `…self_attention.linear_q_up_proj.weight` |
| `…self_attn.kv_a_proj_with_mqa.weight` | `…self_attention.linear_kv_down_proj.weight` |
| `…self_attn.kv_a_layernorm.weight` | `…self_attention.kv_layernorm.weight` |
| `…self_attn.kv_b_proj.weight` | `…self_attention.linear_kv_up_proj.weight` |
| `…self_attn.g_proj.weight` | `…self_attention.output_gate.weight` |
| `…self_attn.o_proj.weight` | `…self_attention.linear_proj.weight` |
| `…linear_attn.{q,k,v}_proj.weight` | `…self_attention.{q,k,v}_proj.weight` |
| `…linear_attn.{q,k,v}_conv1d.weight` | `…self_attention.{q,k,v}_conv1d.weight` |
| `…linear_attn.A_log` | `…self_attention.A_log` — **assert `[96:] == 0`, trim `[128]→[96]`; zero-pad on export** |
| `…linear_attn.dt_bias` | `…self_attention.dt_bias` (`[12288]`) |
| `…linear_attn.{f_a,f_b,g,b}_proj.weight` | `…self_attention.{f_a,f_b,g,b}_proj.weight` |
| `…linear_attn.o_norm.weight` | `…self_attention.o_norm.weight` |
| `…self_attention_res_{norm,proj}.weight` | `…attn_res_attn.{norm,proj}.weight` |
| `…mlp_res_{norm,proj}.weight` | `…attn_res_mlp.{norm,proj}.weight` |
| `…mlp.gate.weight` / `…gate.e_score_correction_bias` | `…mlp.router.weight` / `…mlp.router.expert_bias` |
| `…mlp.routed_expert_down_proj.weight` | `…mlp.fc1_latent_proj.weight` |
| `…mlp.routed_expert_norm.weight` | `…mlp.routed_expert_norm.weight` |
| `…mlp.routed_expert_up_proj.weight` | `…mlp.fc2_latent_proj.weight` |
| `…mlp.experts.{e}.w1/w3.weight` (MXFP4-packed + `weight_scale`) | `…mlp.experts.linear_fc1.weight{e}` (gate slot first, up second; dequantised on import) |
| `…mlp.experts.{e}.w2.weight` | `…mlp.experts.linear_fc2.weight{e}` |
| `…mlp.shared_experts.w1/w3/w2.weight` | `…mlp.shared_experts.linear_fc{1,1,2}.weight` |
| `model.language_model.embed_tokens.weight` / `norm.weight` / `lm_head.weight` | `embedding.word_embeddings.weight` / `decoder.final_layernorm.weight` / `output_layer.weight` |
| `model.language_model.output_attn_res_{norm,proj}.weight` | `decoder.output_attn_res.{norm,proj}.weight` |
| `vision_tower.*`, `mm_projector.*` | **skipped by an explicit rule**, counted and reported |

Exact key prefixes are read from the released `model.safetensors.index.json` in
P8-T1; the table above is the shape of the mapping, and the dry run
(**G30**) is what proves it.
