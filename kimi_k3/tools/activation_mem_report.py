"""Phase A1 driver -- turns the flat NON_PARAM_HEADROOM_GIB into f(seq, mbs, ...).

Reconciliation point for `activation_mem_model.py`. Unlike the standalone
laptop driver, this imports the repo's OWN param/layout math
(`tools/mem_budget.params_per_gpu`, `config/scaleout.aligned_layout/boundaries`)
so the activation model sits directly on top of the measured parameter oracle
(G13) rather than a reimplementation.

Run in the training container (needs torch, via the config builder)::

    python -m kimi_k3.tools.activation_mem_report

Prints:
  1. the 4 L G28 calibration (what the flat 82 GiB actually decomposes into),
  2. the 93 L PP=8 per-stage headroom the flat constant is blind to,
  3. the node-count implication when headroom is no longer flat,
  4. a seq/mbs/CP/recompute sensitivity sweep on one stage.

The standalone laptop copy (no torch, math reimplemented) that produced the
numbers in the PR description lives at ~/workspace/kimi_k3/run_activation_model.py
and matches this to the decimal (params/gpu 19.25 B, state 141.1 GiB = G46).
"""

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset, OFFICIAL_FULL_ATTN_1IDX
from kimi_k3.tools.mem_budget import params_per_gpu
from kimi_k3.config import scaleout

from kimi_k3.tools.activation_mem_model import (
    StageSpec,
    stage_activation_gib,
    calibrate_residual,
    k_slots,
)


HBM_GIB = 288.0
BYTES_PER_PARAM = 7.87  # dist_muon@dp8, MEASURED (G5, opt_mem.md)


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_parts(parts):
    order = [
        "checkpoints",
        "transient_recompute",
        "attn_res_mixer",
        "attn_res_payload",
        "head_logits",
        "residual_frag_nccl",
        "TOTAL",
    ]
    for name in order:
        print(f"    {name:22s} {parts[name]:8.2f} GiB")


def main():
    mla_1idx = set(OFFICIAL_FULL_ATTN_1IDX)

    # -- 1. calibration against the 4 L G28 anchor --------------------------
    residual, modelled, anchor = calibrate_residual(scaleout.NON_PARAM_HEADROOM_GIB)
    banner("1. 4 L G28 anchor -- what the flat headroom decomposes into")
    print(f"  scaleout.NON_PARAM_HEADROOM_GIB = {scaleout.NON_PARAM_HEADROOM_GIB}")
    print(f"  geometry: seq={anchor.seq} mbs={anchor.mbs} cp={anchor.cp} "
          f"layers={anchor.layers_on_stage} mla={anchor.mla_layers_on_stage} "
          f"recompute={anchor.recompute}")
    print(f"  itemized components sum to {modelled:.2f} GiB")
    print(f"  residual (frag + NCCL + allocator slack) back-solved = {residual:.2f} GiB")
    fmt_parts(stage_activation_gib(anchor, residual_gib=residual))

    # -- 2. 93 L PP=8 per-stage headroom ------------------------------------
    spec_cfg = preset("93L")
    cfg = config_from_preset(spec_cfg["config"])
    vocab = spec_cfg["model"]["vocab_size"]
    pp = 8
    ep = 28  # the G46 floor config
    flat = scaleout.NON_PARAM_HEADROOM_GIB
    layout = scaleout.aligned_layout(cfg.num_layers, pp, cfg.k3_attn_res_block_size)
    bounds = [0] + scaleout.boundaries(layout)

    banner("2. 93 L PP=8 aligned -- per-stage activation the flat headroom cannot see")
    print(f"  layout (layers/stage) = {layout}")
    print("  seq=8192 mbs=1 cp=1 recompute=full  (production regime, G6-mandated)")
    print()
    header = ("stage", "layers", "mla", "K", "in_flight", "act_GiB", "vs flat")
    print("    " + "".join(f"{h:>10s}" for h in header))
    stage_totals = []
    for s in range(pp):
        first = bounds[s]
        n_layers = layout[s]
        last = first + n_layers - 1
        n_mla = sum(1 for i in range(first + 1, last + 2) if i in mla_1idx)
        in_flight = pp - s  # 1F1B steady-state microbatches at stage s
        spec = StageSpec(
            seq=8192, mbs=1, cp=1,
            layers_on_stage=n_layers,
            mla_layers_on_stage=n_mla,
            last_layer_0idx=last,
            is_last_stage=(s == pp - 1),
            in_flight=in_flight,
            recompute="full",
            qk_clip_recompute=True,
        )
        total = stage_activation_gib(spec, residual_gib=residual)["TOTAL"]
        stage_totals.append(total)
        row = (f"{s}", f"{n_layers}", f"{n_mla}", f"{k_slots(last)}", f"{in_flight}",
               f"{total:.1f}", f"{total - flat:+.1f}")
        print("    " + "".join(f"{c:>10s}" for c in row))

    worst = max(stage_totals)

    # -- 3. node-count implication ------------------------------------------
    banner("3. node-count implication")
    per_gpu = params_per_gpu(cfg, vocab, tp=1, pp=pp, ep=ep)
    state_gib = per_gpu * BYTES_PER_PARAM / (2 ** 30)
    print(f"  config: 93L tp1 pp{pp} ep{ep}  (G46 floor candidate)")
    print(f"  params/gpu = {per_gpu / 1e9:.2f} B   state = {state_gib:.1f} GiB")
    print(f"  flat headroom  {flat:.1f} -> total {state_gib + flat:.1f} GiB "
          f"({'FITS' if state_gib + flat <= HBM_GIB else 'OOM'} vs {HBM_GIB:.0f})")
    print(f"  modelled worst {worst:.1f} -> total {state_gib + worst:.1f} GiB "
          f"({'FITS' if state_gib + worst <= HBM_GIB else 'OOM'} vs {HBM_GIB:.0f})")
    print(f"  headroom delta on binding stage: {worst - flat:+.1f} GiB")

    # -- 4. seq / mbs / CP / recompute sensitivity --------------------------
    banner("4. sensitivity of one 93L stage (12 layers, 3 MLA, K=4)")
    print("  MLA-quadratic-in-time but chunked-linear-in-memory; MoE/payload linear")
    print()
    header = ("seq", "mbs", "cp", "recompute", "act_GiB", "transient", "mixer", "logits")
    print("    " + "".join(f"{h:>11s}" for h in header))
    base = dict(
        layers_on_stage=12, mla_layers_on_stage=3, last_layer_0idx=47,
        is_last_stage=True, qk_clip_recompute=True,
    )
    for seq in (4096, 8192, 16384):
        for mbs in (1, 2):
            for cp in (1, 2):
                for rc in ("full", "off"):
                    spec = StageSpec(seq=seq, mbs=mbs, cp=cp, in_flight=1, recompute=rc, **base)
                    p = stage_activation_gib(spec, residual_gib=residual)
                    row = (f"{seq}", f"{mbs}", f"{cp}", rc, f"{p['TOTAL']:.1f}",
                           f"{p['transient_recompute']:.1f}", f"{p['attn_res_mixer']:.1f}",
                           f"{p['head_logits']:.1f}")
                    print("    " + "".join(f"{c:>11s}" for c in row))
            print()


if __name__ == "__main__":
    main()
