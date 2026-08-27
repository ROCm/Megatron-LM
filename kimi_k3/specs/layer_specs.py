"""Per-layer module specs for Kimi K3.

Core supports heterogeneous layers natively: hand `TransformerBlock` a
`TransformerBlockSubmodules` whose `layer_specs` is a per-layer list and it
builds exactly those (`transformer_block.py:222-258`). K3 needs that, because no
two consecutive layers are alike: `KDA KDA KDA MLA` repeating, layer 1 dense and
every other layer MoE.

The plan is separated from the spec on purpose. `k3_layer_plan()` answers *what
each layer is* -- decided entirely by the config and testable on CPU without
building anything -- while `get_k3_layer_specs()` turns that into modules. P3 and
P4 change only the second half, when the real KDA and gated-MLA modules exist.

**Placeholder, and marked as one:** until P3/P4 land, a KDA layer is built from
core's MLA spec. The *plan* is already correct, so `test_k3_p2_specs.py` can pin
the pattern now and P3 flips one branch.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

KDA = "kda"
MLA = "mla"
DENSE = "dense"
MOE = "moe"


@dataclass(frozen=True)
class K3LayerPlan:
    """What one layer is. 0-indexed, like everything internal."""

    layer_idx: int
    attention: str  # KDA | MLA
    ffn: str  # DENSE | MOE
    ffn_hidden_size: int
    appends_attn_res_slot: bool
    slots_on_entry: int

    @property
    def is_kda(self) -> bool:
        return self.attention == KDA


def k3_layer_plan(config) -> Tuple[K3LayerPlan, ...]:
    """The whole model's plan, independent of pipeline placement."""
    plan = []
    for i in range(config.num_layers):
        is_dense = i < getattr(config, "k3_first_k_dense_replace", 1)
        plan.append(
            K3LayerPlan(
                layer_idx=i,
                attention=KDA if config.is_kda_layer(i) else MLA,
                ffn=DENSE if is_dense else MOE,
                ffn_hidden_size=config.ffn_hidden_size if is_dense else config.moe_ffn_hidden_size,
                appends_attn_res_slot=config.appends_attn_res_slot(i),
                slots_on_entry=config.attn_res_slots_before(i),
            )
        )
    return tuple(plan)


def local_layer_plan(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> Tuple[K3LayerPlan, ...]:
    """The slice of the plan this pipeline stage owns.

    `TransformerBlock` builds exactly `len(layer_specs)` layers, so a stage must
    be handed its own layers -- not the whole model's.
    """
    from megatron.core.transformer.transformer_block import get_num_layers_to_build

    plan = k3_layer_plan(config)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    count = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return plan[offset : offset + count]


def _layer_spec_for(config, entry: K3LayerPlan) -> ModuleSpec:
    """One layer's ModuleSpec.

    P3 replaces the KDA branch with the real `KimiDeltaAttention` submodules and
    P4 replaces the MLA branch with `K3GatedMLA`; the dense/MoE split below is
    already final.
    """
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=None if entry.ffn == DENSE else config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        multi_latent_attention=True,  # placeholder for KDA layers until P3
    )


def get_k3_layer_specs(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> List[ModuleSpec]:
    return [_layer_spec_for(config, e) for e in local_layer_plan(config, vp_stage, pp_rank)]


def get_k3_block_spec(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    """A block spec whose layers differ from one another, as K3's do."""
    from megatron.core.transformer.transformer_block import LayerNormImpl

    return TransformerBlockSubmodules(
        layer_specs=get_k3_layer_specs(config, vp_stage, pp_rank), layer_norm=LayerNormImpl
    )


def describe(config) -> str:
    """One line per block of 12, for eyeballing a preset."""
    plan = k3_layer_plan(config)
    block = config.k3_attn_res_block_size
    out = []
    for start in range(0, len(plan), block):
        chunk = plan[start : start + block]
        kinds = " ".join(("K" if e.is_kda else "M") + ("*" if e.appends_attn_res_slot else " ")
                         for e in chunk)
        out.append(f"  layers {chunk[0].layer_idx:>3}-{chunk[-1].layer_idx:<3} {kinds}")
    return "\n".join(out)
