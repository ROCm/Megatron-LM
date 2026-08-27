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

#: Measured, not assumed: `results/opt_mem.md`, dist_muon at DP=8.
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


def memory_per_gpu_gib(config: ScaleOutConfig, params_per_gpu: float) -> Dict[str, float]:
    """What one GPU has to hold, from the measured bytes/param."""
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
