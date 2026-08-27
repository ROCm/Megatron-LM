"""Per-head Muon for K3's attention projections, and the parameter-group policy.

Core's Muon already knows how to split a *fused* QKV matrix before
Newton-Schulz: `TensorParallelMuon.orthogonalize` keys on `param.is_qkv`, which
`get_megatron_muon_optimizer` sets for `linear_qkv.weight`, and it carries an
explicit `TODO(deyuf): support MLA`. K3 has no `linear_qkv.weight` at all -- KDA
and gated MLA both keep `q`/`k`/`v` as separate projections -- so core's split
never fires and every attention matrix would be orthogonalized as one big
`[num_heads * head_dim, hidden]` block. That is finding A7, and the head split
below is ours.

**Why the split matters.** Newton-Schulz drives a matrix's singular values toward
1, and `get_muon_scale_factor` then rescales by the matrix's shape. Applied to
the stacked matrix, both steps mix heads that are functionally independent: a
head with a large gradient is normalised against heads with small ones, and the
spectral scale is computed for a `[12288, 7168]` matrix that the model never uses
as one. Per-head, each `[128, 7168]` slice is orthogonalized and scaled on its
own terms.

**Tensor parallelism makes this simpler, not harder.** TP splits attention *by
head*, so a single head lives entirely on one rank. Every head slice is therefore
TP-local and is orthogonalized with `partition_dim=None` -- no cross-rank
reduction, no `tp_group`. This is the opposite of core's fused-QKV split, where
each slice still spans the sharded query-group axis and keeps its `partition_dim`.

**What is not split.** `f_a_proj`, `q_a_proj` and `kv_a_proj_with_mqa` project
into a shared latent and have no head axis. `b_proj` is `[num_heads, hidden]` --
one *row* per head, so a per-head slice would be `[1, hidden]`, where
orthogonalization is just normalisation; it goes through whole-matrix Muon
instead.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.optimizer.muon import TensorParallelMuon

#: `[num_heads * head_dim, in]` -- the head axis is the output axis.
OUT_AXIS = 0
#: `[out, num_heads * head_dim]` -- the head axis is the input axis (`o_proj`).
IN_AXIS = 1

#: Suffix -> which axis carries heads, for each attention kind.
KDA_SPLITS = {
    "q_proj.weight": OUT_AXIS,
    "k_proj.weight": OUT_AXIS,
    "v_proj.weight": OUT_AXIS,
    "f_b_proj.weight": OUT_AXIS,
    "g_proj.weight": OUT_AXIS,
    "g_b_proj.weight": OUT_AXIS,
    "o_proj.weight": IN_AXIS,
}
MLA_SPLITS = {
    "q_b_proj.weight": OUT_AXIS,
    "kv_b_proj.weight": OUT_AXIS,
    "g_proj.weight": OUT_AXIS,
    "o_proj.weight": IN_AXIS,
}

#: Sent to the scalar optimizer (Adam or Lion) by name, on top of core's own rule
#: that anything not 2-D and anything flagged `is_embedding_or_output_parameter`
#: goes there. Listed explicitly so the policy is a statement, not a side effect
#: of a shape check: `A_log` and `dt_bias` are 1-D *and* fp32 gains, the conv
#: weights are 3-D, and `expert_bias` is a routing statistic rather than a weight.
SCALAR_BY_NAME = ("A_log", "dt_bias", "expert_bias", "_conv1d_weight", "norm", "layernorm")


def head_split(name: str, param: torch.Tensor, config) -> Optional[Tuple[int, int]]:
    """`(num_heads, axis)` for a per-head attention matrix, else None."""
    if param.dim() != 2:
        return None
    if ".kda." in name:
        table, heads = KDA_SPLITS, config.k3_kda_num_heads
    elif ".mla." in name:
        table, heads = MLA_SPLITS, config.num_attention_heads
    else:
        return None
    axis = next((a for suffix, a in table.items() if name.endswith(suffix)), None)
    if axis is None:
        return None
    if param.shape[axis] % heads:
        raise ValueError(
            f"{name}: {list(param.shape)} axis {axis} is not divisible by {heads} heads"
        )
    return heads, axis


def param_role(name: str, param: torch.Tensor) -> str:
    """`"muon"` or `"scalar"`, following core's rule plus K3's named exceptions.

    The one judgment call is the MoE router *weight*: it is a 2-D matrix and goes
    to Muon, matching Moonshot's own published policy (AdamW for the embedding,
    the LM head and non-matrix parameters; Muon for the rest). Its companion
    `expert_bias` is a running statistic and does not.
    """
    if getattr(param, "is_embedding_or_output_parameter", False):
        return "scalar"
    if param.dim() != 2:
        return "scalar"
    if any(token in name for token in SCALAR_BY_NAME):
        return "scalar"
    return "muon"


def k3_param_policy(model) -> Dict[str, str]:
    """The whole policy for one model, as a plain dict, so a test can assert it."""
    return {name: param_role(name, param) for name, param in model.named_parameters()}


def tag_k3_heads(model, config) -> List[str]:
    """Mark every per-head matrix with `param.k3_head_split`. Returns the names."""
    tagged = []
    for name, param in model.named_parameters():
        split = head_split(name, param, config)
        if split is not None:
            param.k3_head_split = split
            tagged.append(name)
    return tagged


def orthogonalize_per_head(
    grad: torch.Tensor, num_heads: int, axis: int, orthogonalize_fn
) -> torch.Tensor:
    """Apply `orthogonalize_fn` to each head's slice independently.

    `orthogonalize_fn` takes a 2-D matrix and returns one of the same shape; the
    caller has already bound `partition_dim=None`, because a head slice is
    TP-local.
    """
    if axis == OUT_AXIS:
        heads = grad.view(num_heads, -1, grad.shape[1])
        out = torch.stack([orthogonalize_fn(h) for h in heads])
        return out.view_as(grad)
    heads = grad.view(grad.shape[0], num_heads, -1).transpose(0, 1)
    out = torch.stack([orthogonalize_fn(h) for h in heads])
    return out.transpose(0, 1).reshape(grad.shape)


class PerHeadMuon(TensorParallelMuon):
    """`TensorParallelMuon` that splits K3's attention matrices by head first.

    Everything else -- momentum, Nesterov, weight decay, the Newton-Schulz
    coefficients, the scale mode -- is core's, unchanged. Only the choice of
    *what matrix* goes into Newton-Schulz differs.
    """

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        split = getattr(p, "k3_head_split", None)
        if split is None:
            return super().orthogonalize(p, grad, **kwargs)
        num_heads, axis = split
        return orthogonalize_per_head(
            grad,
            num_heads,
            axis,
            # a head slice is complete on one rank: no tp_group, no partition_dim
            lambda h: self.scaled_orthogonalize_fn(h.contiguous(), None, None),
        )
