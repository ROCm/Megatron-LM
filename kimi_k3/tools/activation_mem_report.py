"""Phase A1 driver (v2) -- turns the flat NON_PARAM_HEADROOM_GIB into f(seq, mbs, ...).

Reconciliation point for `activation_mem_model.py`. Unlike the standalone
laptop driver, this imports the repo's OWN param/layout math
(`tools/mem_budget.params_per_gpu`, `tools/mem_budget.OPTIMIZER_BYTES_PER_PARAM`,
`config/scaleout.aligned_layout/boundaries`) so the activation model sits
directly on top of the parameter oracle (G13) rather than a reimplementation.

v2 (PR #162): nothing measured is a model INPUT. State bytes come from the
analytic dist_muon formula OPTIMIZER_BYTES_PER_PARAM["dist_muon"](dp) = 2+4+8/dp;
the measured 7.87 bytes/param, G6's 7.1/12.2 GB mixer, and G28's 82 GiB headroom
appear only as independent cross-checks (section 0).

Run in the training container (needs torch, via the config builder)::

    python -m kimi_k3.tools.activation_mem_report

The standalone laptop copy (no torch, math reimplemented) that produced the
numbers in the PR description lives at ~/workspace/kimi_k3/run_activation_model.py
and matches this to the decimal (params/gpu 19.25 B, G46 layout).
"""

from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset, OFFICIAL_FULL_ATTN_1IDX
from kimi_k3.tools.mem_budget import params_per_gpu, OPTIMIZER_BYTES_PER_PARAM
from kimi_k3.config import scaleout

from kimi_k3.tools.activation_mem_model import (
    PipelineStageInputs,
    stage_activation_gib,
    state_bytes_per_param,
    crosscheck_state_bytes,
    crosscheck_mixer,
    crosscheck_headroom,
    k_slots,
)


HBM_GIB = 288.0
MEASURED_STATE_BYTES = 7.87  # G5 opt_mem.md -- cross-check ONLY, never an input


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
        "nccl_pp_p2p",
        "frag_allocator",
        "TOTAL",
    ]
    for name in order:
        print(f"    {name:22s} {parts[name]:8.2f} GiB")


def main():
    mla_1idx = set(OFFICIAL_FULL_ATTN_1IDX)
    dp = 8

    # -- 0. cross-checks: analytic vs measured (measured is NOT an input) ----
    banner("0. cross-checks -- analytic model vs measured gates (validation only)")
    a_state, gap_state = crosscheck_state_bytes(dp=dp, measured=MEASURED_STATE_BYTES)
    print(f"  state bytes/param  : analytic OPTIMIZER_BYTES_PER_PARAM['dist_muon']({dp})"
          f" = {a_state:.2f}  vs G5 measured {MEASURED_STATE_BYTES}  (gap {gap_state:+.2f})")
    mx = crosscheck_mixer()
    print(f"  AttnRes mixer fwd  : analytic {mx['fwd_analytic_gb']:.2f} GB  "
          f"vs G6 measured {mx['fwd_measured_gb']:.1f} GB  (gap x{mx['fwd_gap_x']:.2f})")
    print(f"  AttnRes mixer bwd  : analytic {mx['bwd_analytic_gb']:.2f} GB  "
          f"vs G6 measured {mx['bwd_measured_gb']:.1f} GB  (gap x{mx['bwd_gap_x']:.2f})")
    print("    ^ open item: gap => uncounted live fp32 temporaries in the CK")
    print("      AttnRes mixer; flagged for a source read (not tuned away).")
    modelled82, gap82 = crosscheck_headroom(scaleout.NON_PARAM_HEADROOM_GIB)
    print(f"  4 L headroom       : analytic {modelled82:.1f} GiB  vs G28 measured "
          f"{scaleout.NON_PARAM_HEADROOM_GIB} GiB  (gap {gap82:+.1f})")

    # -- 1. 93 L PP=8 per-stage headroom ------------------------------------
    spec_cfg = preset("93L")
    cfg = config_from_preset(spec_cfg["config"])
    vocab = spec_cfg["model"]["vocab_size"]
    pp = 8
    ep = 28  # the G46 floor config
    flat = scaleout.NON_PARAM_HEADROOM_GIB
    layout = scaleout.aligned_layout(cfg.num_layers, pp, cfg.k3_attn_res_block_size)
    bounds = [0] + scaleout.boundaries(layout)

    banner("1. 93 L PP=8 aligned -- per-stage activation the flat headroom cannot see")
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
        spec = PipelineStageInputs(
            seq=8192, mbs=1, cp=1,
            layers_on_stage=n_layers,
            mla_layers_on_stage=n_mla,
            last_layer_0idx=last,
            is_last_stage=(s == pp - 1),
            in_flight=in_flight,
            recompute="full",
            qk_clip_score=False,
        )
        total = stage_activation_gib(spec)["TOTAL"]
        stage_totals.append(total)
        row = (f"{s}", f"{n_layers}", f"{n_mla}", f"{k_slots(last)}", f"{in_flight}",
               f"{total:.1f}", f"{total - flat:+.1f}")
        print("    " + "".join(f"{c:>10s}" for c in row))

    worst = max(stage_totals)

    # -- 2. node-count implication (analytic state bytes) -------------------
    banner("2. node-count implication (analytic state bytes)")
    per_gpu = params_per_gpu(cfg, vocab, tp=1, pp=pp, ep=ep)
    bpp = state_bytes_per_param(dp)
    state_gib = per_gpu * bpp / (2 ** 30)
    print(f"  config: 93L tp1 pp{pp} ep{ep}  (G46 floor candidate)")
    print(f"  params/gpu = {per_gpu / 1e9:.2f} B   state = {state_gib:.1f} GiB "
          f"(analytic {bpp:.2f} bytes/param, dist_muon@dp{dp})")
    print(f"  flat headroom  {flat:.1f} -> total {state_gib + flat:.1f} GiB "
          f"({'FITS' if state_gib + flat <= HBM_GIB else 'OOM'} vs {HBM_GIB:.0f})")
    print(f"  modelled worst {worst:.1f} -> total {state_gib + worst:.1f} GiB "
          f"({'FITS' if state_gib + worst <= HBM_GIB else 'OOM'} vs {HBM_GIB:.0f})")
    print(f"  headroom delta on binding stage: {worst - flat:+.1f} GiB")

    # -- 3. seq / mbs / CP / recompute sensitivity --------------------------
    banner("3. sensitivity of one 93L stage (12 layers, 3 MLA, K=4)")
    print("  MLA flash-style (linear in mem); MoE/payload linear. qk_clip off.")
    print()
    header = ("seq", "mbs", "cp", "recompute", "act_GiB", "transient", "mixer", "logits")
    print("    " + "".join(f"{h:>11s}" for h in header))
    base = dict(
        layers_on_stage=12, mla_layers_on_stage=3, last_layer_0idx=47,
        is_last_stage=True, qk_clip_score=False,
    )
    for seq in (4096, 8192, 16384):
        for mbs in (1, 2):
            for cp in (1, 2):
                for rc in ("full", "off"):
                    spec = PipelineStageInputs(seq=seq, mbs=mbs, cp=cp, in_flight=1,
                                               recompute=rc, **base)
                    p = stage_activation_gib(spec)
                    row = (f"{seq}", f"{mbs}", f"{cp}", rc, f"{p['TOTAL']:.1f}",
                           f"{p['transient_recompute']:.1f}", f"{p['attn_res_mixer']:.1f}",
                           f"{p['head_logits']:.1f}")
                    print("    " + "".join(f"{c:>11s}" for c in row))
            print()


if __name__ == "__main__":
    main()
