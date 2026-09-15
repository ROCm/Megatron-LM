"""Phase A1 -- analytic activation-memory model for Kimi K3.

WHY THIS FILE EXISTS
--------------------
`kimi_k3/config/scaleout.py` hard-codes

    NON_PARAM_HEADROOM_GIB = 82.0

as "everything that is not parameters, gradients and optimizer state:
activations, the AttnRes payload, fragmentation, NCCL buffers." That 82 GiB is a
SINGLE measured point: the 4 L official config on one node (G28), where the
measured 202 GiB peak minus 16.31 B params/rank x 7.87 bytes/param (119.6 GiB of
dist_muon@dp8 state) leaves 82.4 GiB. `scaleout.py` then reuses that flat 82 GiB
for EVERY 93 L candidate, regardless of seq, micro-batch, CP, recompute policy,
or layers-per-stage.

That is wrong in two independent directions, and both matter for the node count:

  1. The 4 L anchor has only 4 layers and exactly 1 MLA layer on the (single)
     stage, and at 4 L the whole model fits so recompute is effectively off.
     A 93 L PP=8 stage carries ~12 layers incl. ~3 MLA layers -- its live
     activation set is larger even before recompute enters.
  2. AttnRes payload and mixer temporaries grow with pipeline DEPTH:
     payload = (1+K) x S x B x H x 2 bytes with K = ceil(layer/12), so K rises
     from 1 at 4 L to 7 at the deepest 93 L boundary. The flat 82 GiB cannot see
     this at all.

So the flat constant is a 4 L / no-recompute / K=1 point, and reusing it flat
makes the 93 L node count (28 nodes, G46) OPTIMISTIC in the mid-pipeline stages
and mis-shaped versus seq/mbs. This module replaces the constant with a
component-wise f(seq, mbs, CP, recompute, layers_on_stage, mla_on_stage, K).

STATUS: local reconstruction. Formulas grounded in the released dims (presets.py)
and the MEASURED gates G5 (opt_mem) / G6 (attn_res) / G28 (trainer memory). To be
reconciled into kimi_k3/tools/mem_budget.py + scaleout.py when the mi355 checkout
is reachable again. Every term is tagged MEASURED or ESTIMATED.

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

    MEASURED formula (G6 / scaleout.slots_at): K = ceil(layer / block).
    """
    if layer_0idx <= 0:
        return 0
    return -(-layer_0idx // block)


# --- individual activation components (all return bytes) --------------------
# Each is per micro-batch, per pipeline stage, at the given local sequence
# length S_local = seq / CP.  bf16 unless a term is explicitly fp32.

def attn_res_payload_bytes(s_local, mbs, k):
    """Packed residual stream crossing a boundary. MEASURED (G6).

    payload = (1 + K) x S x B x H x 2 bytes.
    Validates: k=1, S=8192, B=1 -> 2 x 8192 x 7168 x 2 = 224.0 MiB (G6 says 224).
    """
    return (1 + k) * s_local * mbs * DIMS.hidden * BF16


def attn_res_mixer_pair_bytes(s_local, mbs, k, with_backward):
    """One live mixer pair under recompute. MEASURED (G6), scaled linearly.

    G6 anchor: K+1 = 9, S=8192, B=1 -> 7.1 GB forward, 12.2 GB forward+backward.
    Cost is linear in (K+1) x S x B (fp32 upcast is already inside the anchor).
    Recompute keeps ONE live pair; without recompute the whole fp32 stack would
    cost ~236 GB / micro-batch (G6) -- that regime is modelled by recompute="off".
    """
    anchor_gb = 12.2 if with_backward else 7.1  # GB (1e9), as quoted by G6
    anchor_bytes = anchor_gb * 1e9
    ref = (9) * 8192 * 1  # (K+1) x S x B at the anchor
    scale = ((1 + k) * s_local * mbs) / ref
    return anchor_bytes * scale


def moe_dispatch_buffer_bytes(s_local, mbs):
    """EP all-to-all dispatch/combine + expert intermediates for ONE MoE layer.

    ESTIMATED from dims. Each token is routed to topk=16 experts; the dispatch
    path holds:
      - gathered expert INPUT  : B x S x topk x H x 2         (send + recv -> x2)
      - expert INTERMEDIATE    : B x S x topk x moe_inter x 2 (w1/w3 GLU -> x2)
      - combine buffer         : B x S x topk x H x 2
    This is the term Ye asked to see itemized. It is the uniform-load upper
    bound; the QB router / EPLB would lower the per-rank max but NOT this
    per-layer transient (see project memory). Linear in S and in mbs.
    """
    tokens = mbs * s_local * DIMS.topk
    dispatch_in = 2 * tokens * DIMS.hidden * BF16
    expert_mid = 2 * tokens * DIMS.moe_inter * BF16
    combine = tokens * DIMS.hidden * BF16
    return dispatch_in + expert_mid + combine


def mla_attn_transient_bytes(s_local, mbs, qk_clip_recompute):
    """MLA attention transient for ONE MLA layer. ESTIMATED / oracle-tagged.

    Two regimes:
      - qk_clip_recompute=True  : the release recomputes a full [B, nh, S, S]
        fp32 score matrix every forward for MuonClip max-logit capture
        (STATUS T1). Quadratic in S. At S=8192, B=1: 96 x 8192 x 8192 x 4 =
        25.8 GiB -- but SDPA is query-CHUNKED, so only one chunk is resident.
        We model the resident chunk as 1/chunks of the full matrix.
      - qk_clip_recompute=False : fused-attn path, no fp32 score in HBM; only the
        flash working set ~ B x nh x S x (qk+v) x 2.
    The quadratic term is the reason MLA memory (unlike MoE) does NOT trade
    freely between seq and mbs.
    """
    nh = DIMS.num_attn_heads
    if qk_clip_recompute:
        query_chunk = 2048  # SDPA query chunk (STATUS: query-chunked recompute)
        chunks = max(1, s_local // query_chunk)
        full_score = mbs * nh * s_local * s_local * FP32
        return full_score / chunks
    flash_ws = mbs * nh * s_local * (DIMS.qk_head_dim + DIMS.v_head_dim) * BF16
    return flash_ws


def layer_input_checkpoint_bytes(s_local, mbs, layers_on_stage):
    """With recompute, each layer stashes only its input [S, B, H]. MEASURED-shape.

    L_stage x S x B x H x 2. Tiny relative to the transients; grows with
    layers-per-stage (the axis the flat 82 GiB is blind to).
    """
    return layers_on_stage * s_local * mbs * DIMS.hidden * BF16


def head_logits_bytes(s_local, mbs):
    """LM-head logits + softmax/loss scratch, LAST stage only. ESTIMATED.

    logits [S, B, vocab] bf16 + fp32 loss scratch + grad ~ B x S x vocab x 6.
    Linear in S; the single biggest edge-stage term at large vocab.
    """
    return mbs * s_local * DIMS.vocab * (BF16 + FP32)


# --- assembled per-stage activation peak ------------------------------------
@dataclass
class StageSpec:
    seq: int
    mbs: int
    cp: int
    layers_on_stage: int
    mla_layers_on_stage: int
    last_layer_0idx: int  # deepest layer on this stage (sets K)
    is_last_stage: bool  # carries the LM head
    in_flight: int = 1  # 1F1B microbatches live at this stage (Phase C refines)
    recompute: str = "full"  # "full" | "off"
    qk_clip_recompute: bool = True


def stage_activation_gib(spec: StageSpec, residual_gib: float):
    """Peak activation set on one stage, GiB, broken down by component.

    residual_gib = fragmentation + NCCL/collective buffers + allocator slack,
    calibrated once against the 4 L G28 anchor (see calibrate_residual).
    """
    s_local = spec.seq // spec.cp
    k = k_slots(spec.last_layer_0idx)

    checkpoints = layer_input_checkpoint_bytes(s_local, spec.mbs, spec.layers_on_stage)

    if spec.recompute == "full":
        # One layer recomputed live at a time -> the single largest transient.
        mla_t = 0
        if spec.mla_layers_on_stage > 0:
            mla_t = mla_attn_transient_bytes(s_local, spec.mbs, spec.qk_clip_recompute)
        moe_t = moe_dispatch_buffer_bytes(s_local, spec.mbs)
        transient = max(mla_t, moe_t)
        mixer = attn_res_mixer_pair_bytes(s_local, spec.mbs, k, with_backward=True)
    else:
        # No recompute: every layer's stack is resident simultaneously.
        mla_total = spec.mla_layers_on_stage * mla_attn_transient_bytes(
            s_local, spec.mbs, spec.qk_clip_recompute
        )
        moe_layers = spec.layers_on_stage - spec.mla_layers_on_stage
        moe_total = moe_layers * moe_dispatch_buffer_bytes(s_local, spec.mbs)
        transient = mla_total + moe_total
        mixer = spec.layers_on_stage * attn_res_mixer_pair_bytes(
            s_local, spec.mbs, k, with_backward=True
        )

    # AttnRes payload and per-microbatch stashes multiply by in-flight count.
    payload = spec.in_flight * attn_res_payload_bytes(s_local, spec.mbs, k)
    checkpoints = spec.in_flight * checkpoints
    logits = head_logits_bytes(s_local, spec.mbs) if spec.is_last_stage else 0

    parts = {
        "checkpoints": checkpoints,
        "transient_recompute": transient,
        "attn_res_mixer": mixer,
        "attn_res_payload": payload,
        "head_logits": logits,
    }
    parts_gib = {name: val / GIB for name, val in parts.items()}
    parts_gib["residual_frag_nccl"] = residual_gib
    parts_gib["TOTAL"] = sum(parts_gib.values())
    return parts_gib


def calibrate_residual(anchor_headroom_gib=82.0):
    """Back-solve residual (frag/NCCL/slack) so the 4 L G28 anchor reproduces.

    4 L anchor geometry: single stage, PP=1, all 4 layers resident, S=8192, B=1,
    CP=1, 1 MLA layer (layer idx 3, 1-indexed 4). At 4 L the model fits without
    recompute, so the anchor is the recompute="off" regime -- this is exactly the
    conflation the flat constant hides.
    """
    anchor = StageSpec(
        seq=8192,
        mbs=1,
        cp=1,
        layers_on_stage=4,
        mla_layers_on_stage=1,
        last_layer_0idx=3,
        is_last_stage=True,
        in_flight=1,
        recompute="off",
        qk_clip_recompute=True,
    )
    modelled = stage_activation_gib(anchor, residual_gib=0.0)["TOTAL"]
    residual = anchor_headroom_gib - modelled
    return residual, modelled, anchor
