"""P10 / gate G41 (part) -- QAT actually reaching a real model's experts.

`k3_qat_experts.py` proved the semantics at one expert. What is checked here is
that they reach the experts a real K3 model builds: the right modules, the right
weights, the master still a parameter, and the gradient still arriving at it.
"""

import pytest
import torch
import torch.nn.utils.parametrize as parametrize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def build(qat: bool):
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.moe.k3_qat_wiring import enable_qat_experts

    torch.manual_seed(0)
    model = build_k3_model("tiny")
    touched = enable_qat_experts(model) if qat else {"weights": 0}
    return model, touched


def test_it_finds_the_routed_experts_and_only_those(single_rank_world):
    from kimi_k3.moe.k3_qat_wiring import expert_linears

    model, touched = build(qat=True)
    assert touched["weights"] > 0, touched
    assert touched["activation_hooks"] == touched["modules"]

    names = {n for n, m in model.named_modules() if m in expert_linears(model)}
    assert names, "no expert linears found"
    assert not any("shared" in n for n in names), (
        "shared experts are bf16 in the release and must not be quantised"
    )


def first_expert_weight(model):
    """(module, attribute) for one routed expert weight, whichever layout is in use."""
    from kimi_k3.moe.k3_qat_wiring import expert_linears, expert_weight_names

    module = expert_linears(model)[0]
    return module, expert_weight_names(module)[0]


def test_the_master_stays_a_parameter_and_the_weight_becomes_quantised(single_rank_world):
    from kimi_k3.moe.k3_qat import fake_quantize_mxfp4

    model, _ = build(qat=True)
    module, name = first_expert_weight(model)
    assert parametrize.is_parametrized(module, name)

    master = getattr(module.parametrizations, name).original
    assert isinstance(master, torch.nn.Parameter)
    quantised = getattr(module, name)
    torch.testing.assert_close(quantised, fake_quantize_mxfp4(master.detach()), rtol=0, atol=0)
    assert not torch.equal(quantised, master), "quantisation did nothing"


def test_the_gradient_reaches_the_master(single_rank_world):
    """The STE's whole job. A parametrization that blocks it would train nothing."""
    model, _ = build(qat=True)
    module, name = first_expert_weight(model)
    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    model(input_ids=tokens, position_ids=None, attention_mask=None).sum().backward()

    master = getattr(module.parametrizations, name).original
    assert master.grad is not None
    assert float(master.grad.abs().max()) > 0


def test_enabling_twice_does_not_stack_quantisers(single_rank_world):
    """Quantising a quantised value is a different operation, not a no-op."""
    from kimi_k3.moe.k3_qat_wiring import enable_qat_experts

    model, first = build(qat=True)
    module, name = first_expert_weight(model)
    before = getattr(module, name).clone()
    second = enable_qat_experts(model)
    assert second["weights"] == 0, second
    assert second["modules"] == first["modules"], "the finder went blind to its own work"
    torch.testing.assert_close(getattr(module, name), before, rtol=0, atol=0)


def test_qat_changes_the_forward(single_rank_world):
    """Otherwise the twin run would be measuring nothing."""
    plain, _ = build(qat=False)
    quantised, _ = build(qat=True)
    tokens = torch.randint(0, 4096, (1, 16), device="cuda")
    with torch.no_grad():
        a = plain(input_ids=tokens, position_ids=None, attention_mask=None)
        b = quantised(input_ids=tokens, position_ids=None, attention_mask=None)
    assert not torch.allclose(a, b, atol=1e-6), "QAT produced an identical forward"


def test_the_state_dict_map_round_trips(single_rank_world):
    """Enabling QAT renames the tensors the converter writes; both ways must work."""
    from kimi_k3.moe.k3_qat_wiring import qat_state_dict_map

    plain, _ = build(qat=False)
    quantised, _ = build(qat=True)

    plain_keys = {k for k in plain.state_dict() if torch.is_tensor(plain.state_dict()[k])}
    qat_keys = {k for k in quantised.state_dict() if torch.is_tensor(quantised.state_dict()[k])}
    assert plain_keys != qat_keys, "expected the parametrization to rename something"

    mapped = set(qat_state_dict_map({k: torch.zeros(1) for k in plain_keys}, to_qat=True))
    assert mapped == qat_keys, sorted(mapped ^ qat_keys)[:5]

    back = set(qat_state_dict_map({k: torch.zeros(1) for k in qat_keys}, to_qat=False))
    assert back == plain_keys, sorted(back ^ plain_keys)[:5]


def test_a_plain_checkpoint_loads_into_a_qat_model(single_rank_world):
    """The reason the map exists: converted weights must reach a QAT run."""
    from kimi_k3.moe.k3_qat_wiring import qat_state_dict_map

    plain, _ = build(qat=False)
    quantised, _ = build(qat=True)
    source = {k: v for k, v in plain.state_dict().items() if torch.is_tensor(v)}

    missing, unexpected = quantised.load_state_dict(
        qat_state_dict_map(source, to_qat=True), strict=False
    )
    assert unexpected == [], unexpected[:5]
    assert [m for m in missing if torch.is_tensor(quantised.state_dict().get(m))] == []


def test_the_config_flag_turns_qat_on_by_itself(single_rank_world):
    """`--k3-qat-experts` was a field nothing read. It reads now."""
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.moe.k3_qat_wiring import expert_linears, expert_weight_names

    torch.manual_seed(0)
    model = build_k3_model("tiny", k3_qat_experts=True)
    module = expert_linears(model)[0]
    assert parametrize.is_parametrized(module, expert_weight_names(module)[0])

    plain = build_k3_model("tiny")
    other = expert_linears(plain)[0]
    assert not parametrize.is_parametrized(other, expert_weight_names(other)[0])
