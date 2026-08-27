"""P12 / gate G47 -- what the dispatcher A/B matrix can actually be run against.

The matrix in `plan-0/07-dispatcher-ab.md` makes claims about the pin: which
backends core supports, which constraints it enforces, and which runtimes are
installed. Those claims are asserted here so the document cannot quietly go stale.
"""

import dataclasses
import importlib

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig

BACKENDS = ("deepep", "mori", "hybridep")


def test_core_already_supports_the_backends_the_plan_wanted_ported():
    """The plan said 'DeepEP port'. There is nothing to port."""
    fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    assert {"moe_flex_dispatcher_backend", "moe_mori_max_tokens_per_rank",
            "moe_deepep_num_sms", "moe_enable_deepep"} <= fields

    import megatron.core.transformer.moe.fused_a2a  # noqa: F401


def test_the_dispatcher_types_are_the_three_the_matrix_names():
    hint = TransformerConfig.__dataclass_fields__["moe_token_dispatcher_type"].type
    for name in ("allgather", "alltoall", "flex"):
        assert name in str(hint), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_which_runtimes_are_installed(backend):
    """Records the answer rather than asserting one; the matrix reads this.

    If a backend appears here later, the corresponding arm becomes runnable and
    the plan's table needs its row updated.
    """
    module = {"deepep": "deep_ep", "mori": "mori", "hybridep": "hybridep"}[backend]
    try:
        importlib.import_module(module)
        installed = True
    except ImportError:
        installed = False
    # None are installed at the pin. When one is, this fails and the table gets fixed.
    assert not installed, f"{module} is now installed -- update plan-0/07-dispatcher-ab.md"


def test_mori_refuses_without_its_buffer_size():
    """The constraint the matrix calls out: core raises rather than defaulting."""
    with pytest.raises(ValueError, match="moe_mori_max_tokens_per_rank"):
        TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4,
            num_moe_experts=8, moe_ffn_hidden_size=64,
            moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="mori",
        )


def test_asking_for_mori_and_deepep_silently_gives_deepep():
    """Core's "Cannot enable both" guard is unreachable. Finding A17.

    `__post_init__` handles `moe_enable_deepep` first and *overwrites*
    `moe_flex_dispatcher_backend` with `"deepep"`. The MoRI branch below it then
    never matches, so the error it would have raised cannot fire. Ask for MoRI and
    DeepEP together and you get DeepEP, with only a deprecation warning about an
    unrelated flag.

    This matters for the A/B matrix specifically: arms C and D would report
    identical numbers and nothing would say why. The tripwire is deliberately
    written the way round it is -- when core fixes the ordering this test fails
    and the matrix's note comes out.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4,
            num_moe_experts=8, moe_ffn_hidden_size=64,
            moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="mori",
            moe_mori_max_tokens_per_rank=1024, moe_enable_deepep=True,
        )
    assert config.moe_flex_dispatcher_backend == "deepep", (
        "core now rejects or honours mori+deepep; drop the note from "
        "plan-0/07-dispatcher-ab.md and re-check arm D"
    )
