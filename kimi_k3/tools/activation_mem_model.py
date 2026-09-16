"""Phase A1 -- bottom-up analytic activation-memory model for Kimi K3 (v2).

WHY THIS FILE EXISTS
--------------------
`kimi_k3/config/scaleout.py` hard-codes

    NON_PARAM_HEADROOM_GIB = 82.0

as "everything that is not parameters, gradients and optimizer state:
activations, the AttnRes payload, fragmentation, NCCL buffers." That 82 GiB is a
SINGLE measured point (the 4 L official config on one node, gate G28) reused flat
for EVERY 93 L candidate, regardless of seq, micro-batch, CP, recompute policy,
or layers-per-stage. This module replaces the flat constant with a
component-wise f(seq, mbs, CP, recompute, layers_on_stage, mla_on_stage, K).

METHODOLOGY (v2, per PR #162 review)
------------------------------------
Every term here is derived BOTTOM-UP from released shapes and dtypes. No measured
quantity is used as a model INPUT:

  * optimizer state bytes come from the analytic `2 + 4 + 8/DP` dist_muon formula
    (mem_budget.OPTIMIZER_BYTES_PER_PARAM), NOT the measured 7.87 bytes/param;
  * the AttnRes mixer term is derived from the (K+1)-slot fp32 upcast shape, NOT
    anchored to G6's 7.1/12.2 GB;
  * the non-itemized residual (fragmentation, allocator slack) is an EXPLICIT
    fraction of the itemized total, NOT back-solved to reproduce 82 GiB.

Measured gates (G5 opt_mem 7.87 bytes/param, G6 attn_res 7.1/12.2 GB & 224 MiB
payload, G28 82 GiB headroom) are used ONLY as independent cross-checks -- see
`crosscheck_*` helpers -- so that a mismatch is a finding to chase, not a knob
that silently absorbs a modeling miss.

UNITS: `bytes` always spelled out for byte counts; `B` only ever means billion
(parameters). Storage in GiB (2**30 bytes).
"""

from dataclasses import dataclass


GIB = 2 ** 30
BF16 = 2  # bytes
FP32 = 4  # bytes


# --- released K3 dims (presets.py, _official_config) ------------------------
@dataclass(frozen=True)
class K3Dims:
    hidden: int = 7168
    num_attn_heads: int = 96  # MLA heads
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    num_experts: int = 896
    topk: int = 16
    moe_inter: int = 3072  # per-expert intermediate (routed)
    routed_latent: int = 3584
    shared_inter: int = 6144
    dense_ffn: int = 33792
    vocab: int = 163840
    attn_res_block: int = 12
    kda_num_heads: int = 96
    kda_head_dim: int = 128


DIMS = K3Dims()


# --- pipeline geometry helper ----------------------------------------------
def k_slots(layer_0idx: int, block: int = DIMS.attn_res_block) -> int:
    """AttnRes slots accumulated in the payload up to a 0-indexed layer.

    Shape-derived (scaleout.slots_at): K = ceil(layer / block).
    """
    if layer_0idx <= 0:
        return 0
    return -(-layer_0idx // block)


# --- individual activation components (all return bytes, all shape-derived) --
# Each is per micro-batch, per pipeline stage, at the local sequence length
# S_local = seq / CP.  bf16 unless a term is explicitly fp32.

def attn_res_payload_bytes(s_local, mbs, k):
    """Packed residual stream crossing a boundary. Shape-derived.

    payload = (1 + K) x S x B x H x 2 bytes.
    Cross-check (not input): k=1, S=8192, B=1 -> 2 x 8192 x 7168 x 2 = 224.0 MiB,
    which matches G6's quoted 224 MiB -- validation of the shape, not a fitted
    constant.
    """
    return (1 + k) * s_local * mbs * DIMS.hidden * BF16


# The AttnRes mixer upcasts the packed slots to fp32 and forms a small number of
# live fp32 temporaries. We enumerate them explicitly rather than borrow G6.
MIXER_ACCUM_FP32_TEMPS = 2  # running prefix accumulator + fp32 output (fwd)


def attn_res_mixer_bytes(s_local, mbs, k, with_backward,
                         accum_temps=MIXER_ACCUM_FP32_TEMPS):
    """One live AttnRes mixer under recompute. BOTTOM-UP from shapes.

    The mixer combines the (K+1) packed slots. Live fp32 set (forward):
      - (K+1) input slots upcast to fp32 : (K+1) x S x B x H x 4
      - `accum_temps` fp32 scratch tensors: accum_temps x S x B x H x 4
    Backward keeps the saved inputs plus the incoming grad, ~2x the forward set.

    NOTE (open item from PR #162): at the G6 geometry (K+1=9, S=8192, B=1) this
    yields ((9)+2) x 8192 x 7168 x 4 = 2.58 GB forward, whereas G6 MEASURED
    7.1 GB. The ~2.75x gap implies the CK AttnRes mixer holds more live fp32
    temporaries than the 2 modeled here; it is flagged for a source read of the
    mixer kernel (see crosscheck_mixer). We do NOT tune `accum_temps` to close
    the gap -- that would re-introduce the fitting this refactor removed.
    """
    slots_fp32 = (1 + k) + accum_temps
    fwd = slots_fp32 * s_local * mbs * DIMS.hidden * FP32
    return 2 * fwd if with_backward else fwd


def moe_dispatch_buffer_bytes(s_local, mbs):
    """EP all-to-all dispatch/combine + expert intermediates for ONE MoE layer.

    Shape-derived. Each token routes to topk=16 experts; the dispatch path holds:
      - gathered expert INPUT  : B x S x topk x H x 2         (send + recv -> x2)
      - expert INTERMEDIATE    : B x S x topk x moe_inter x 2 (w1/w3 GLU -> x2)
      - combine buffer         : B x S x topk x H x 2
    Uniform-load upper bound; the QB router / EPLB lowers the per-rank max but
    NOT this per-layer transient. Linear in S and in mbs.
    """
    tokens = mbs * s_local * DIMS.topk
    dispatch_in = 2 * tokens * DIMS.hidden * BF16
    expert_mid = 2 * tokens * DIMS.moe_inter * BF16
    combine = tokens * DIMS.hidden * BF16
    return dispatch_in + expert_mid + combine


def mla_attn_transient_bytes(s_local, mbs, qk_clip_score=False):
    """MLA attention transient for ONE MLA layer. Shape-derived.

    Default (production): FLASH / CK / AITER style -- the score [B, nh, S, S] is
    NEVER materialized in HBM; only the flash working set is resident:
        B x nh x S x (qk_head_dim + v_head_dim) x 2.
    This is LINEAR in S (quadratic only in TIME), so MLA memory does trade with
    seq and mbs like the other terms.

    Optional term (`qk_clip_score=True`): the MuonClip max-logit capture in the
    current EAGER/SDPA release recomputes a full fp32 score. SDPA is
    query-chunked so only one [B, nh, chunk, S] tile is resident at a time. This
    is a release-eager artifact, NOT the production fused path, hence default
    OFF. Modeled here only so the sensitivity study can show what turning it on
    would cost.
    """
    nh = DIMS.num_attn_heads
    flash_ws = mbs * nh * s_local * (DIMS.qk_head_dim + DIMS.v_head_dim) * BF16
    if not qk_clip_score:
        return flash_ws
    query_chunk = 2048
    resident_tile = mbs * nh * query_chunk * s_local * FP32
    return flash_ws + resident_tile


def layer_input_checkpoint_bytes(s_local, mbs, layers_on_stage):
    """With recompute, each layer stashes only its input [S, B, H]. Shape-derived.

    L_stage x S x B x H x 2. Grows with layers-per-stage -- the axis the flat
    82 GiB is blind to.
    """
    return layers_on_stage * s_local * mbs * DIMS.hidden * BF16


def head_logits_bytes(s_local, mbs):
    """LM-head logits + softmax/loss scratch, LAST stage only. Shape-derived.

    logits [S, B, vocab] bf16 + fp32 loss scratch/grad ~ B x S x vocab x (2+4).
    Linear in S; the single biggest edge-stage term at large vocab.
    """
    return mbs * s_local * DIMS.vocab * (BF16 + FP32)


def pp_p2p_buffer_bytes(s_local, mbs, k):
    """Pipeline-parallel point-to-point send/recv buffers. Shape-derived.

    1F1B holds a send and a recv copy of the boundary payload: 2 x payload.
    """
    return 2 * attn_res_payload_bytes(s_local, mbs, k)


# --- assembled per-stage activation peak ------------------------------------
@dataclass
class PipelineStageInputs:
    """Geometry of ONE pipeline stage fed to the activation estimator.

    Renamed from StageSpec (PR #162): it is the per-pipeline-stage input bundle,
    not a spec of the whole model. `stage_activation_gib` consumes one of these
    per PP rank; the whole-model headroom is the max over stages.
    """
    seq: int
    mbs: int
    cp: int
    layers_on_stage: int
    mla_layers_on_stage: int
    last_layer_0idx: int  # deepest layer on this stage (sets K)
    is_last_stage: bool  # carries the LM head
    in_flight: int = 1  # 1F1B microbatches live at this stage (Phase C refines)
    recompute: str = "full"  # "full" | "off"
    qk_clip_score: bool = False  # release-eager MuonClip rescore (default off)


# Fragmentation / allocator slack as an EXPLICIT fraction of the itemized live
# set. This is an ESTIMATED modeling assumption, not a back-solved absolute; the
# crosscheck_headroom against G28 tells us whether the fraction is right.
FRAG_FRACTION = 0.07  # ESTIMATED: caching-allocator slack, ~7% of live bytes


def stage_activation_gib(spec: PipelineStageInputs, frag_fraction=FRAG_FRACTION):
    """Peak activation set on ONE pipeline stage, GiB, broken down by component.

    "Stage" = the set of layers resident on one PP rank (not one layer, not the
    whole model). Sums input checkpoints over the stage's layers and, under
    recompute="full", adds only the single largest per-layer transient (one
    layer recomputed live at a time). The non-itemized residual is
    `frag_fraction` of the itemized total -- an explicit allocator-slack
    assumption, no measured 82 GiB injected.
    """
    s_local = spec.seq // spec.cp
    k = k_slots(spec.last_layer_0idx)

    checkpoints = layer_input_checkpoint_bytes(s_local, spec.mbs, spec.layers_on_stage)

    if spec.recompute == "full":
        # One layer recomputed live at a time -> the single largest transient.
        mla_t = 0
        if spec.mla_layers_on_stage > 0:
            mla_t = mla_attn_transient_bytes(s_local, spec.mbs, spec.qk_clip_score)
        moe_t = moe_dispatch_buffer_bytes(s_local, spec.mbs)
        transient = max(mla_t, moe_t)
        mixer = attn_res_mixer_bytes(s_local, spec.mbs, k, with_backward=True)
    else:
        # No recompute: every layer's stack is resident simultaneously.
        mla_total = spec.mla_layers_on_stage * mla_attn_transient_bytes(
            s_local, spec.mbs, spec.qk_clip_score
        )
        moe_layers = spec.layers_on_stage - spec.mla_layers_on_stage
        moe_total = moe_layers * moe_dispatch_buffer_bytes(s_local, spec.mbs)
        transient = mla_total + moe_total
        mixer = spec.layers_on_stage * attn_res_mixer_bytes(
            s_local, spec.mbs, k, with_backward=True
        )

    # AttnRes payload and per-microbatch stashes multiply by in-flight count.
    payload = spec.in_flight * attn_res_payload_bytes(s_local, spec.mbs, k)
    checkpoints = spec.in_flight * checkpoints
    logits = head_logits_bytes(s_local, spec.mbs) if spec.is_last_stage else 0
    nccl = pp_p2p_buffer_bytes(s_local, spec.mbs, k)

    itemized = {
        "checkpoints": checkpoints,
        "transient_recompute": transient,
        "attn_res_mixer": mixer,
        "attn_res_payload": payload,
        "head_logits": logits,
        "nccl_pp_p2p": nccl,
    }
    parts_gib = {name: val / GIB for name, val in itemized.items()}
    itemized_gib = sum(parts_gib.values())
    parts_gib["frag_allocator"] = frag_fraction * itemized_gib
    parts_gib["TOTAL"] = itemized_gib + parts_gib["frag_allocator"]
    return parts_gib


# --- optimizer/state bytes: analytic, measured used only to cross-check ------
def state_bytes_per_param(dp: int) -> float:
    """dist_muon state bytes per resident param. ANALYTIC.

    Mirrors mem_budget.OPTIMIZER_BYTES_PER_PARAM["dist_muon"] = 2 + 4 + 8/dp:
    bf16 weight (2) + fp32 grad (4) + fp32 master+momentum (8) sharded over DP.
    """
    return 2 + 4 + 8 / dp


# --- independent cross-checks against measured gates (NOT model inputs) ------
def crosscheck_state_bytes(dp: int = 8, measured=7.87):
    """Analytic dist_muon bytes/param vs G5 measured. Returns (analytic, gap)."""
    analytic = state_bytes_per_param(dp)
    return analytic, measured - analytic


def crosscheck_mixer(measured_fwd_gb=7.1, measured_bwd_gb=12.2):
    """Analytic AttnRes mixer at the G6 geometry vs G6 measured.

    Returns forward/backward analytic GB and the multiplicative gap to G6.
    """
    k = 9 - 1  # G6 quotes K+1 = 9 slots
    fwd = attn_res_mixer_bytes(8192, 1, k, with_backward=False) / 1e9
    bwd = attn_res_mixer_bytes(8192, 1, k, with_backward=True) / 1e9
    return {
        "fwd_analytic_gb": fwd,
        "fwd_measured_gb": measured_fwd_gb,
        "fwd_gap_x": measured_fwd_gb / fwd,
        "bwd_analytic_gb": bwd,
        "bwd_measured_gb": measured_bwd_gb,
        "bwd_gap_x": measured_bwd_gb / bwd,
    }


def crosscheck_headroom(anchor_seq=4096, measured_transient_gib=9.70):
    """Analytic 4 L-stage ACTIVATION vs the MEASURED transient. Returns
    (modelled_gib, measured_transient_gib, gap).

    The retired G28 "82 GiB headroom" was `peak - after_optimizer`. Because
    after_optimizer OMITS the lazily-allocated Muon state (~44.8 GiB, allocated
    only on the first step()), that headroom was NOT pure activation: it silently
    folded ~44.8 GiB of persistent optimizer state into what the model was asked
    to explain as activation -- essentially the whole +43 GiB the itemized model
    "could not account for".

    The honest activation cross-check compares the model against the true
    transient `peak - resident` at a MEASURED seq. Default anchor is the fused
    `fla` production transient (seq-invariant ~9.7 GiB), which the analytic model
    targets; pass the eager number (18/31/58 at seq 1024/2048/4096) only to see
    the backend artifact. Regime matches the proxy: recompute="full", PP=1, all 4
    layers on one stage, 1 MLA layer, in_flight=1.
    """
    anchor = PipelineStageInputs(
        seq=anchor_seq, mbs=1, cp=1,
        layers_on_stage=4, mla_layers_on_stage=1, last_layer_0idx=3,
        is_last_stage=True, in_flight=1, recompute="full", qk_clip_score=False,
    )
    modelled = stage_activation_gib(anchor)["TOTAL"]
    return modelled, measured_transient_gib, measured_transient_gib - modelled
