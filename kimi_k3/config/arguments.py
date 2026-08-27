"""K3 command-line arguments.

Every field of `KimiK3TransformerConfig` that a run might want to change is
reachable from here, so `k3_config_from_args` can pick it up by name (its
collection loop is field-driven). Defaults are the released values, and the
help text says where each one came from -- these are the numbers a reader will
want to check against the model card.
"""

import argparse

from .k3_transformer_config import KimiK3TransformerConfig
from .presets import PRESETS, kda_layers_1idx


def add_kimi_k3_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="kimi-k3")

    group.add_argument("--k3-preset", type=str, default=None, choices=sorted(PRESETS),
                       help="start from a named preset (tiny / 4L / 8L / 93L)")

    # --- KDA ---
    group.add_argument("--k3-kda-num-heads", type=int, default=96)
    group.add_argument("--k3-kda-head-dim", type=int, default=128)
    group.add_argument("--k3-kda-conv-size", type=int, default=4,
                       help="short-convolution kernel width (release: 4)")
    group.add_argument("--k3-kda-gate-lower-bound", type=float, default=-5.0,
                       help="in-kernel gate lower bound (release: -5.0)")
    group.add_argument("--k3-kda-use-full-rank-gate", action=argparse.BooleanOptionalAction,
                       default=True, help="release: true; the output gate is full-rank")
    group.add_argument("--k3-kda-backend", type=str, default="eager", choices=["eager", "fla"],
                       help="eager is the FP32 oracle and stays the default until G15 (rule R5.3)")

    # --- MLA ---
    group.add_argument("--k3-mla-use-nope", action=argparse.BooleanOptionalAction, default=True,
                       help="release: true; the 64 'rope' dims are never rotated")
    group.add_argument("--k3-mla-use-output-gate", action=argparse.BooleanOptionalAction,
                       default=True)
    group.add_argument("--k3-mla-lora-norm-eps", type=float, default=1e-6,
                       help="q_a/kv_a layernorms take KimiRMSNorm's 1e-6 default, not rms_norm_eps")
    group.add_argument("--k3-mla-fp32-attn-output", action=argparse.BooleanOptionalAction,
                       default=True, help="[report] fp32 attention output during training")
    group.add_argument("--k3-attn-res-chunk", type=int, default=4096,
                       help="rows per chunk in the fused AttnRes mixer; memory only")
    group.add_argument("--k3-max-logit-chunk", type=int, default=1024,
                       help="query-block size for the QK-clip max-logit recompute; memory only")

    # --- AttnRes ---
    group.add_argument("--k3-attn-res-block-size", type=int, default=12,
                       help="release: 12, giving 8 residual slots over 93 layers")
    group.add_argument("--k3-attn-res-fp32", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--k3-attn-res-fused", action=argparse.BooleanOptionalAction, default=False,
                       help="P11 fused mixer; off until its parity gate is green")

    # --- MoE ---
    group.add_argument("--k3-routed-expert-hidden-size", type=int, default=3584,
                       help="latent width the routed experts run at (mirrored to moe_latent_size)")
    group.add_argument("--k3-latent-moe-use-norm", action=argparse.BooleanOptionalAction,
                       default=True, help="RMSNorm on the combined expert output before up-proj")
    group.add_argument("--k3-first-k-dense-replace", type=int, default=1)
    group.add_argument("--k3-situ-beta", type=float, default=4.0)
    group.add_argument("--k3-situ-linear-beta", type=float, default=25.0)
    group.add_argument("--k3-router-quantile-balancing", action=argparse.BooleanOptionalAction,
                       default=True)
    group.add_argument("--k3-qb-num-bins", type=int, default=1024)

    # --- QAT ---
    group.add_argument("--k3-qat-experts", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--k3-qat-stochastic-rounding", action=argparse.BooleanOptionalAction,
                       default=False, help="SR is unstated in the report; RNE by default")
    return parser


def k3_field_names() -> set:
    """Config fields this module can set, for the coverage test."""
    return {f for f in KimiK3TransformerConfig.__dataclass_fields__ if f.startswith("k3_")}


def explicitly_set(argv) -> set:
    """Which K3 dests the caller actually typed.

    A preset must fill gaps, not overrule the command line, and argparse does not
    record that distinction: after parsing, a defaulted value and an explicitly
    passed one look identical. Re-parsing with every default suppressed leaves
    only what was given.
    """
    probe = argparse.ArgumentParser(add_help=False)
    add_kimi_k3_args(probe)
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    known, _ = probe.parse_known_args(list(argv))
    return set(vars(known))


def apply_preset_defaults(args: argparse.Namespace, argv=None) -> argparse.Namespace:
    """Fill `args` from `--k3-preset` without clobbering explicit flags.

    Pass `argv` so explicit flags win; without it the preset overwrites
    everything, which is only safe when no K3 flags were given.
    """
    if not getattr(args, "k3_preset", None):
        return args
    explicit = explicitly_set(argv) if argv is not None else set()
    preset_cfg = PRESETS[args.k3_preset]["config"]
    for key, value in {**preset_cfg, **PRESETS[args.k3_preset]["model"]}.items():
        if key not in explicit:
            setattr(args, key, value)
    args.num_experts = preset_cfg.get("num_moe_experts")
    if args.k3_preset != "tiny" and "k3_kda_layers" not in explicit:
        args.k3_kda_layers = kda_layers_1idx(getattr(args, "num_layers"))
    return args
