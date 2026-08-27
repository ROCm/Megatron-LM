"""P3 / gates G15–G16 -- fla against the oracle, and the module round-trip.

Every bound here was measured before it was written down, and each is read
against the floor beside it (rule R4.4, `kimi_k3/tests/tolerance.py`). The
headline: in fp32 fla agrees with the oracle to 7e-7, and in bf16 the gap is
4.3e-3 against a *dtype floor of 3.3e-3* -- so bf16 disagreement is bf16, not
the kernel.
"""

import pytest
import torch

from kimi_k3.attention.kda import KimiDeltaAttention
from kimi_k3.attention.kda_backends import fla_available, kda_forward
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.tests.tolerance import (
    KDA_BWD_BF16,
    KDA_BWD_FP32,
    KDA_FWD_BF16,
    KDA_FWD_FP32,
    assert_within,
    compare,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not fla_available(), reason="fla is a pinned optional backend"),
]


def _inputs(B=1, T=256, H=4, K=32, dtype=torch.float32, grad=False, seed=0):
    torch.manual_seed(seed)
    mk = lambda *s, d=dtype: torch.randn(*s, device="cuda", dtype=d, requires_grad=grad)
    return dict(
        q=mk(B, T, H, K), k=mk(B, T, H, K), v=mk(B, T, H, K), g=mk(B, T, H, K),
        beta=mk(B, T, H, d=torch.float32),
        A_log=torch.rand(H, device="cuda", dtype=torch.float32).log_().requires_grad_(grad),
        dt_bias=mk(H * K, d=torch.float32),
    )


@pytest.mark.parametrize("seq", [128, 1024])
def test_forward_parity_fp32(seq):
    args = _inputs(T=seq)
    ours, our_state = kda_forward(**args, backend="eager", output_final_state=True)
    theirs, their_state = kda_forward(**args, backend="fla", output_final_state=True)
    assert_within(theirs, ours, KDA_FWD_FP32, f"fla fp32 forward @{seq}")
    assert_within(their_state, our_state, KDA_FWD_FP32, f"fla fp32 state @{seq}")


@pytest.mark.parametrize("seq", [128, 1024])
def test_forward_parity_bf16_against_its_own_dtype_floor(seq):
    """The comparison that matters is same-dtype; the floor is measured here too."""
    f32 = _inputs(T=seq, dtype=torch.float32)
    b16 = _inputs(T=seq, dtype=torch.bfloat16)

    eager32, _ = kda_forward(**f32, backend="eager")
    eager16, _ = kda_forward(**b16, backend="eager")
    floor = compare(eager16, eager32)["rel_l2"]
    assert floor <= KDA_FWD_BF16.floor * 1.5, (
        f"the bf16 dtype floor moved: {floor:.3e} vs the recorded {KDA_FWD_BF16.floor:.3e}"
    )

    fla16, _ = kda_forward(**b16, backend="fla")
    stats = assert_within(fla16, eager16, KDA_FWD_BF16, f"fla bf16 forward @{seq}")
    assert stats["rel_l2"] < 3 * floor, "fla is adding much more error than bf16 itself does"


@pytest.mark.parametrize("dtype,bound", [(torch.float32, KDA_BWD_FP32), (torch.bfloat16, KDA_BWD_BF16)])
def test_backward_parity(dtype, bound):
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")

    def grads(backend):
        args = _inputs(T=256, dtype=dtype, grad=True)
        out, _ = kda_forward(**args, backend=backend)
        out.float().pow(2).sum().backward()
        return {n: args[n].grad.detach() for n in names}

    ours, theirs = grads("eager"), grads("fla")
    for n in names:
        assert_within(theirs[n], ours[n], bound, f"fla {dtype} backward d{n}")


def test_error_does_not_grow_with_sequence_length():
    """The `lower_bound = -5` gate bounds error accumulation.

    Worth pinning: a recurrent model is normally expected to drift over long
    contexts, and the incoming plan assumed it would. It does not here, because
    a per-step decay of at most exp(-5) forgets old state faster than error
    accumulates in it.
    """
    seen = []
    for seq in (128, 512, 2048):
        args = _inputs(T=seq, dtype=torch.bfloat16)
        eager, _ = kda_forward(**args, backend="eager")
        fla, _ = kda_forward(**args, backend="fla")
        seen.append(compare(fla, eager)["rel_l2"])
    assert max(seen) < 2 * min(seen), f"error grew with length: {seen}"


# --- the module (G16) --------------------------------------------------------


@pytest.fixture()
def module():
    torch.manual_seed(0)
    cfg = config_from_preset(preset("tiny")["config"])
    return KimiDeltaAttention(cfg).cuda().float(), cfg


def test_module_backends_agree(module):
    m, cfg = module
    x = torch.randn(2, 64, cfg.hidden_size, device="cuda")
    out_e, st_e = m(x, output_final_state=True, backend="eager")
    out_f, st_f = m(x, output_final_state=True, backend="fla")
    assert_within(out_f, out_e, KDA_FWD_FP32, "module output")
    assert_within(st_f, st_e, KDA_FWD_FP32, "module state")


def test_module_parameter_count_matches_the_analytic_model(module):
    """Ties the module to the table that was checked against the checkpoint."""
    from kimi_k3.tools.mem_budget import kda_layer_params

    m, cfg = module
    assert sum(p.numel() for p in m.parameters()) == kda_layer_params(cfg)


def test_every_parameter_receives_gradient(module):
    m, cfg = module
    x = torch.randn(2, 32, cfg.hidden_size, device="cuda")
    out, _ = m(x)
    out.sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_state_round_trips_through_the_module(module):
    """Running two halves with a carried state equals running the whole sequence."""
    m, cfg = module
    x = torch.randn(1, 64, cfg.hidden_size, device="cuda")
    whole, _ = m(x, output_final_state=True)
    first, state = m(x[:, :32], output_final_state=True)
    second, _ = m(x[:, 32:], initial_state=state, output_final_state=True)
    torch.testing.assert_close(first, whole[:, :32], rtol=1e-5, atol=1e-5)
    # the short conv sees a 3-token-shorter history at the split, so compare past it
    torch.testing.assert_close(second[:, cfg.k3_kda_conv_size:],
                               whole[:, 32 + cfg.k3_kda_conv_size:], rtol=2e-4, atol=2e-4)


def test_sharded_state_dict_round_trips(module, single_rank_world):
    m, cfg = module
    sharded = m.sharded_state_dict(prefix="kda.")
    assert sharded, "sharded_state_dict is empty"
    assert all(k.startswith("kda.") for k in sharded), sorted(sharded)[:3]
    plain = m.state_dict()
    fresh = KimiDeltaAttention(cfg).cuda().float()
    fresh.load_state_dict(plain)
    for k, v in plain.items():
        assert torch.equal(fresh.state_dict()[k], v), k


# --- the seam with P2's spec factory ----------------------------------------


def test_kda_layers_in_a_real_model_use_kda(single_rank_world):
    """The plan said which layers are KDA; this proves the model agrees."""
    from kimi_k3.attention.kda import K3KDASelfAttention
    from kimi_k3.model.build import build_k3_model
    from kimi_k3.specs.layer_specs import k3_layer_plan

    model = build_k3_model("tiny")
    plan = k3_layer_plan(model.config)
    for i, layer in enumerate(model.decoder.layers):
        is_kda = isinstance(layer.self_attention, K3KDASelfAttention)
        assert is_kda == plan[i].is_kda, f"layer {i}: {type(layer.self_attention).__name__}"


def test_model_forward_and_backward_reaches_every_kda_parameter(single_rank_world):
    from kimi_k3.model.build import build_k3_model

    model = build_k3_model("tiny")
    tokens = torch.randint(0, 4096, (1, 32), device="cuda")
    out = model(input_ids=tokens, position_ids=None, attention_mask=None)
    out.float().pow(2).sum().backward()
    kda = [(n, p) for n, p in model.named_parameters() if ".kda." in n]
    assert len(kda) == 42, len(kda)
    missing = [n for n, p in kda if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_sequence_first_transpose_is_applied(single_rank_world):
    """Megatron hands `[s, b, h]`; KDA thinks in `[b, t, h]`. Off-by-a-transpose
    would still run and quietly attend across the batch instead of over time."""
    from kimi_k3.attention.kda import K3KDASelfAttention

    cfg = config_from_preset(preset("tiny")["config"])
    attn = K3KDASelfAttention(cfg).cuda().float()
    s, b = 12, 3
    x = torch.randn(s, b, cfg.hidden_size, device="cuda")
    out, bias = attn(x)
    assert bias is None and out.shape == x.shape

    # each batch element must be independent: perturbing one must not move another
    x2 = x.clone()
    x2[:, 1] += 3.0
    out2, _ = attn(x2)
    assert torch.equal(out[:, 0], out2[:, 0])
    assert not torch.equal(out[:, 1], out2[:, 1])
