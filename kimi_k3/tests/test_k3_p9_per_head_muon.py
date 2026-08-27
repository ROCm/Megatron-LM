"""P9 / gate G35 -- per-head Muon, and the K3 parameter-group policy.

The parity claim is not tautological: one step on a `[num_heads * head_dim, in]`
matrix with the head split enabled must equal `num_heads` independent steps on
`[head_dim, in]` matrices taken by core's *unmodified* Muon. If the split, the
reshape, the per-slice scale factor or the momentum bookkeeping is wrong, the two
disagree.
"""

import pytest
import torch

from megatron.core.optimizer.muon import TensorParallelMuon

from kimi_k3.optim.per_head_muon import (
    IN_AXIS,
    OUT_AXIS,
    PerHeadMuon,
    head_split,
    k3_param_policy,
    param_role,
    tag_k3_heads,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MUON_KWARGS = dict(lr=0.1, momentum_beta=0.95, weight_decay=0.0, num_ns_steps=5)
HEADS, HEAD_DIM, IN_DIM = 4, 8, 16


def fixed_grad(shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).cuda()


@pytest.mark.parametrize("axis", [OUT_AXIS, IN_AXIS])
def test_one_split_step_equals_many_whole_steps(axis):
    """G35: the head split is exactly per-head Muon, not an approximation of it."""
    shape = (HEADS * HEAD_DIM, IN_DIM) if axis == OUT_AXIS else (IN_DIM, HEADS * HEAD_DIM)
    param = torch.zeros(*shape, device="cuda")
    param.k3_head_split = (HEADS, axis)
    grad = fixed_grad(shape, seed=3)

    split_opt = PerHeadMuon([param], **MUON_KWARGS)
    param.grad = grad.clone()
    split_opt.step()

    # the same gradient, head by head, through core's unmodified optimizer
    if axis == OUT_AXIS:
        slices = list(grad.view(HEADS, HEAD_DIM, IN_DIM))
    else:
        slices = list(grad.view(IN_DIM, HEADS, HEAD_DIM).transpose(0, 1))
    reference = []
    for head in slices:
        piece = torch.zeros_like(head)
        opt = TensorParallelMuon([piece], **MUON_KWARGS)
        piece.grad = head.contiguous().clone()
        opt.step()
        reference.append(piece)
    expected = (
        torch.stack(reference).view_as(param)
        if axis == OUT_AXIS
        else torch.stack(reference).transpose(0, 1).reshape(param.shape)
    )
    torch.testing.assert_close(param, expected, rtol=0, atol=0)


def test_the_split_actually_changes_the_update():
    """A parity test passes trivially if both paths are the same path."""
    shape = (HEADS * HEAD_DIM, IN_DIM)
    grad = fixed_grad(shape, seed=5)
    updates = {}
    for split in (None, (HEADS, OUT_AXIS)):
        param = torch.zeros(*shape, device="cuda")
        if split:
            param.k3_head_split = split
        param.grad = grad.clone()
        PerHeadMuon([param], **MUON_KWARGS).step()
        updates[split] = param
    assert not torch.allclose(updates[None], updates[(HEADS, OUT_AXIS)], atol=1e-5)


def test_untagged_parameters_take_core_s_path_unchanged():
    shape = (IN_DIM, IN_DIM)
    grad = fixed_grad(shape, seed=7)
    ours, theirs = torch.zeros(*shape, device="cuda"), torch.zeros(*shape, device="cuda")
    ours.grad, theirs.grad = grad.clone(), grad.clone()
    PerHeadMuon([ours], **MUON_KWARGS).step()
    TensorParallelMuon([theirs], **MUON_KWARGS).step()
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


def test_head_split_covers_the_projections_that_have_heads(single_rank_world, tiny_config):
    """Which K3 tensors get split, checked against a real model rather than a list."""
    from kimi_k3.model.build import build_k3_model

    model = build_k3_model("tiny")
    tagged = set(tag_k3_heads(model, tiny_config))
    by_name = dict(model.named_parameters())

    for name in tagged:
        heads, axis = by_name[name].k3_head_split
        assert by_name[name].shape[axis] % heads == 0

    # every attention output projection, of both kinds, must be in there
    o_projs = [n for n in by_name if n.endswith("o_proj.weight") and "attention" in n]
    assert o_projs and set(o_projs) <= tagged

    # the latent projections have no head axis and must not be
    for suffix in ("f_a_proj.weight", "q_a_proj.weight", "kv_a_proj_with_mqa.weight"):
        assert not any(n.endswith(suffix) for n in tagged), suffix

    # b_proj is [num_heads, hidden]: one row per head, so no useful split. The
    # suffix has to be exact -- `f_b_proj.weight` also ends in "b_proj.weight",
    # and it *is* per-head.
    assert not any(n.endswith(".kda.b_proj.weight") for n in tagged)
    assert any(n.endswith(".kda.f_b_proj.weight") for n in tagged)


def test_head_split_rejects_a_shape_that_does_not_divide(tiny_config):
    param = torch.zeros(HEADS * HEAD_DIM + 1, IN_DIM)
    with pytest.raises(ValueError, match="not divisible"):
        head_split("decoder.layers.0.self_attention.kda.q_proj.weight", param, tiny_config)


def test_parameter_group_policy(single_rank_world, tiny_config):
    """G35's second half: which tensors go to Muon and which to the scalar path."""
    from kimi_k3.model.build import build_k3_model

    policy = k3_param_policy(build_k3_model("tiny"))
    assert policy, "no parameters"

    for name, role in policy.items():
        if name.endswith(("A_log", "dt_bias", "expert_bias")) or "conv1d" in name:
            assert role == "scalar", name
        if "norm" in name.lower():
            assert role == "scalar", name
        if name.endswith("router.weight"):
            assert role == "muon", name  # deliberate: see param_role's docstring

    assert policy["embedding.word_embeddings.weight"] == "scalar"
    assert "muon" in policy.values() and "scalar" in policy.values()


def test_every_non_2d_parameter_is_scalar():
    """The floor under the policy: Newton-Schulz needs a matrix."""
    assert param_role("x", torch.zeros(8)) == "scalar"
    assert param_role("x", torch.zeros(8, 1, 4)) == "scalar"
    assert param_role("x.weight", torch.zeros(8, 4)) == "muon"
