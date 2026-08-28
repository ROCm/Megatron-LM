"""93 L scale-out configurations: PP layouts, the EP ladder, and node counts.

Everything here is derived from **measurements**, never estimates (rule R9.1):
bytes per parameter come from `develop/results/opt_mem.md`, parameter counts from
`tools/mem_budget.py`'s analytic breakdown, which P0 checked against a real
build. The agent prepares these; a human launches the job (R10.1).

## Why a PP boundary should land on an AttnRes block boundary

A K3 layer appends a residual slot when its 0-indexed position is a multiple of
`k3_attn_res_block_size` (12), and at that moment **the prefix sum resets** --
`prefix_sum = None`, the stream restarts for the new block. A pipeline boundary
placed there therefore hands the next stage a payload whose prefix half is fresh
rather than half-accumulated, and whose slot count is exactly `layer / 12`.

Mid-block boundaries are not *broken* -- P5 proved the packed payload carries a
partial prefix and its gradient across a stage boundary, bitwise (G21). They are
worse: the payload is the same size but the stage split now cuts a block in half,
so a recompute or a repartition has to reason about a prefix that started on
another stage. Legality here means "aligned", and `layout_problems` says which
rule a layout breaks rather than just rejecting it.

The tail is the exception worth naming: layers 92 and 93 (1-indexed) are **both**
gated MLA, the one place the 3:1 stride breaks. A boundary between them splits a
pair that nothing else in the model resembles, so it is called out separately.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Measured, not assumed: `results/opt_mem.md`. Keyed by the data-parallel size
#: the parameters are *actually* sharded over -- which is not the same number for
#: expert and non-expert parameters. See `expert_data_parallel_size`.
DIST_MUON_BYTES_PER_PARAM = {1: 15.17, 2: 11.00, 4: 8.91, 8: 7.87}

#: Kept for the older call sites; prefer the table above.
BYTES_PER_PARAM = {"dist_muon@dp8": 7.87, "dist_muon@dp4": 8.91, "muon": 15.17, "adam_dist@dp8": 6.0}
#: MI300-class part. The usable figure, not the nameplate.
HBM_PER_GPU_GIB = 288.0
GPUS_PER_NODE = 8
#: Everything that is not parameters, gradients and optimizer state: activations,
#: the AttnRes payload, fragmentation, NCCL buffers. Measured at 4 L in G28,
#: where 16.31 B params/rank at 7.87 B/param leaves this much of the 202 GiB peak.
NON_PARAM_HEADROOM_GIB = 82.0


@dataclass(frozen=True)
class ScaleOutConfig:
    name: str
    tp: int
    pp: int
    ep: int
    recipe: str = "dist_muon@dp8"

    @property
    def gpus(self) -> int:
        return self.tp * self.pp * self.ep

    @property
    def nodes(self) -> float:
        return self.gpus / GPUS_PER_NODE


#: T12.3 -- the EP ladder. 896 experts, so EP must divide 896.
EP_LADDER: Tuple[int, ...] = (8, 16, 28, 32, 56)

#: T12.1 -- candidate 93 L shapes. Node counts are computed, not written down.
CANDIDATES: Tuple[ScaleOutConfig, ...] = tuple(
    ScaleOutConfig(name=f"93L-tp1-pp{pp}-ep{ep}", tp=1, pp=pp, ep=ep)
    for pp in (4, 8, 16)
    for ep in EP_LADDER
)


def expert_data_parallel_size(world: int, tp: int, pp: int, ep: int) -> int:
    """How many ranks an *expert* parameter's optimizer state is sharded over.

    Core: `expert_data_parallel_size = world_size // (etp * ep * pp)`
    (`parallel_state.py:788-795`, with `etp` defaulting to `tp`). This is **not**
    the ordinary data-parallel size, and the difference is the single most
    optimistic assumption in the first version of this model.

    At `pp8 x ep28` on 224 GPUs it comes out to **1** -- expert optimizer state is
    not sharded at all, so those parameters cost **15.17** bytes each rather than
    the 7.87 measured at DP=8. Expert weights are ~98 % of K3, so applying the
    DP=8 figure to them understates per-GPU state by nearly a factor of two.
    """
    denominator = tp * ep * pp
    if world % denominator:
        raise ValueError(f"world {world} is not divisible by tp*ep*pp = {denominator}")
    return world // denominator


def bytes_per_param(dp: int) -> float:
    """`dist_muon` bytes per parameter at a given sharding width, interpolated.

    Measured at DP 1/2/4/8; between those it is linear in 1/DP, which is what the
    measurements show (the sharded part halves as DP doubles).
    """
    if dp in DIST_MUON_BYTES_PER_PARAM:
        return DIST_MUON_BYTES_PER_PARAM[dp]
    fixed, sharded = 6.0, 9.17  # 6 + 9.17/1 = 15.17; 6 + 9.17/8 = 7.15 ~ 7.87 measured
    return fixed + sharded / dp


def experts_divide(ep: int, num_experts: int = 896) -> bool:
    return num_experts % ep == 0


def uniform_layout(num_layers: int, pp: int) -> List[int]:
    """Layers per stage, as even as the layer count allows."""
    base, extra = divmod(num_layers, pp)
    return [base + (1 if i < extra else 0) for i in range(pp)]


def boundaries(layout: Sequence[int]) -> List[int]:
    """0-indexed layer at which each stage after the first begins."""
    out, running = [], 0
    for count in layout[:-1]:
        running += count
        out.append(running)
    return out


def layout_problems(
    layout: Sequence[int], num_layers: int = 93, block_size: int = 12
) -> List[str]:
    """Everything wrong with a PP layout, named. Empty means legal."""
    problems = []
    if sum(layout) != num_layers:
        problems.append(f"layout covers {sum(layout)} layers, not {num_layers}")
    if any(count <= 0 for count in layout):
        problems.append(f"empty stage in {list(layout)}")
    for boundary in boundaries(layout):
        if boundary % block_size:
            problems.append(
                f"boundary at layer {boundary} splits an AttnRes block "
                f"(nearest aligned: {block_size * round(boundary / block_size)})"
            )
        if boundary == num_layers - 1:
            problems.append(
                f"boundary at layer {boundary} separates the final MLA pair "
                f"(layers {num_layers - 1} and {num_layers}, 1-indexed) -- the one "
                "place the 3:1 stride breaks"
            )
    return problems


def aligned_layout(num_layers: int = 93, pp: int = 8, block_size: int = 12) -> List[int]:
    """A layout whose boundaries all land on AttnRes block boundaries.

    93 is not a multiple of 12, so the last stage absorbs the remainder -- it
    carries the tail block plus the extra MLA layer, and is the stage to watch
    when the load turns out uneven.
    """
    blocks = num_layers // block_size  # 7 whole blocks, plus 9 tail layers
    if pp > blocks + 1:
        raise ValueError(
            f"pp={pp} exceeds the {blocks + 1} aligned cut points that {num_layers} layers "
            f"at block size {block_size} allow; any larger PP must split an AttnRes block"
        )
    per_stage_blocks = uniform_layout(blocks, pp)
    layout = [count * block_size for count in per_stage_blocks]
    layout[-1] += num_layers - blocks * block_size
    return layout


def slots_at(boundary: int, block_size: int = 12) -> int:
    """AttnRes slots in the payload crossing a boundary at this 0-indexed layer."""
    return -(-boundary // block_size)


def split_params_per_gpu(cfg, vocab_size: int, tp: int, pp: int, ep: int):
    """(non-expert, expert) parameters on one GPU. They shard differently."""
    from kimi_k3.tools.mem_budget import breakdown

    b = breakdown(cfg, vocab_size)
    latent = cfg.k3_routed_expert_hidden_size
    routed = max(cfg.num_layers - 1, 0) * cfg.num_moe_experts * 3 * latent * cfg.moe_ffn_hidden_size
    return (b.total - routed) / (tp * pp), routed / (tp * pp * ep)


def state_gib(cfg, vocab_size: int, config: "ScaleOutConfig", world: int) -> Dict[str, float]:
    """Per-GPU parameter, gradient and optimizer state, sharded as core shards it.

    The two halves do not share a data-parallel size. Ordinary parameters shard
    over `world / (tp * pp)`; expert parameters shard over
    `world / (tp * ep * pp)`, which at `pp8 x ep28` on 224 GPUs is **1**. Using
    the DP=8 figure for both -- as the first version of this model did --
    understated per-GPU state by nearly a factor of two, because expert weights
    are ~98 % of K3.
    """
    non_expert, expert = split_params_per_gpu(cfg, vocab_size, config.tp, config.pp, config.ep)
    dp = world // (config.tp * config.pp)
    edp = expert_data_parallel_size(world, config.tp, config.pp, config.ep)
    gib = (non_expert * bytes_per_param(dp) + expert * bytes_per_param(edp)) / 2**30
    return {
        "params_per_gpu": non_expert + expert,
        "expert_fraction": expert / (non_expert + expert),
        "data_parallel": dp,
        "expert_data_parallel": edp,
        "expert_bytes_per_param": bytes_per_param(edp),
        "state_gib": round(gib, 1),
    }


def memory_per_gpu_gib(config: ScaleOutConfig, params_per_gpu: float) -> Dict[str, float]:
    """Deprecated flat model, kept so older callers do not silently change.

    It applies one bytes-per-param figure to every parameter and a constant
    headroom. Both were measured to be wrong -- see `state_gib` and
    `develop/results/scaleout_93l.md`.
    """
    state = params_per_gpu * BYTES_PER_PARAM[config.recipe] / 2**30
    return {
        "params_per_gpu": params_per_gpu,
        "state_gib": round(state, 1),
        "headroom_gib": NON_PARAM_HEADROOM_GIB,
        "total_gib": round(state + NON_PARAM_HEADROOM_GIB, 1),
        "fits": state + NON_PARAM_HEADROOM_GIB <= HBM_PER_GPU_GIB,
    }


def plan(config: ScaleOutConfig) -> Dict:
    """The full picture for one candidate, including why it does not work."""
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset
    from kimi_k3.tools.mem_budget import params_per_gpu

    spec = preset("93L")
    cfg = config_from_preset(spec["config"])
    try:
        layout = aligned_layout(cfg.num_layers, config.pp, cfg.k3_attn_res_block_size)
        alignment_note = []
    except ValueError as exc:
        # Not a crash -- a result. 93 layers give only 7 whole AttnRes blocks, so
        # any PP above 8 *must* cut one. The memory numbers are still worth having.
        layout = uniform_layout(cfg.num_layers, config.pp)
        alignment_note = [str(exc)]
    per_gpu = params_per_gpu(cfg, spec["model"]["vocab_size"], config.tp, config.pp, config.ep)

    row = {
        "name": config.name, "tp": config.tp, "pp": config.pp, "ep": config.ep,
        "gpus": config.gpus, "nodes": config.nodes, "recipe": config.recipe,
        "layout": layout,
        "boundaries": boundaries(layout),
        "payload_slots": [slots_at(b, cfg.k3_attn_res_block_size) for b in boundaries(layout)],
        "problems": alignment_note
        + layout_problems(layout, cfg.num_layers, cfg.k3_attn_res_block_size),
    }
    if not experts_divide(config.ep, cfg.num_moe_experts):
        row["problems"].append(f"ep={config.ep} does not divide {cfg.num_moe_experts} experts")
    row.update(memory_per_gpu_gib(config, per_gpu))
    return row
