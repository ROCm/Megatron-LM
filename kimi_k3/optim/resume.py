"""Loading a `dist_muon` optimizer state back, which core cannot currently do.

`ChainedOptimizer.state_dict()` returns a **list** -- one entry per chained
optimizer -- whenever there is more than one, and `dist_muon` always has two (the
Muon optimizer and the scalar one). `LayerWiseDistributedOptimizer` overrides
`load_state_dict` to undo a dict-vs-list conversion it does for dist-checkpointing,
and that override does:

    wrapped_state_dict = state_dict          # a list
    for sd in wrapped_state_dict.values():   # AttributeError

so the layer-wise optimizer cannot read what its own `state_dict()` writes. Found
by gate G37 (`develop/results/pp_resume.md`); review-register finding A16.

Rule R2.1 keeps core unmodified, so the fix is here: apply the same list
conversion the override intends, then call `ChainedOptimizer.load_state_dict`
directly. `test_k3_p9_resume.py` holds a tripwire that fails once core fixes this,
so the workaround leaves rather than lingers.
"""

from typing import Any


def load_optimizer_state_dict(optimizer, state_dict: Any) -> None:
    """`optimizer.load_state_dict(state_dict)`, with the layer-wise case handled."""
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    if not (isinstance(optimizer, LayerWiseDistributedOptimizer) and isinstance(state_dict, list)):
        optimizer.load_state_dict(state_dict)
        return

    for entry in state_dict:
        params = entry.get("fp32_from_fp16_params")
        if isinstance(params, dict):
            entry["fp32_from_fp16_params"] = [v for _, v in sorted(params.items())]
    ChainedOptimizer.load_state_dict(optimizer, state_dict)


def core_layerwise_load_is_broken() -> bool:
    """True while core's override still cannot read a list. Drives the tripwire."""
    import inspect

    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

    source = inspect.getsource(LayerWiseDistributedOptimizer.load_state_dict)
    # it iterates `.values()` over whatever it was handed, and nothing in it ever
    # asks whether that was the list `ChainedOptimizer.state_dict` returns
    return "wrapped_state_dict.values()" in source and "isinstance(state_dict, list)" not in source
