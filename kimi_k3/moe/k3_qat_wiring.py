"""Turning QAT on for a real model's routed experts.

`k3_qat_experts.py` proved the *semantics* at one expert (G8, G40): fp32 masters,
an MXFP4 forward, an STE backward. This module applies them to the experts a real
K3 model actually builds, which are TE `GroupedLinear` modules holding one
`weight{i}` per local expert.

The hook is `torch.nn.utils.parametrize`, and it fits almost too well.
`TEGroupedLinear._get_weight_tensors` does `getattr(self, f"weight{i}")` on every
forward, so a parametrization is read every step rather than captured once. What
it gives is exactly the QAT contract, for free:

* the fp32 master stays a real parameter, at `parametrizations.weight{i}.original`,
  and it is what the optimizer updates;
* `module.weight{i}` becomes the fake-quantised value, recomputed each forward;
* autograd threads the STE through, because `ste_mxfp4` *is* the parametrization.

No packed cache is kept here. The cache in `k3_qat_experts.py` exists to model what
a kernel would read; in training the quantisation is recomputed from the master
every step anyway, so caching it would only add a way for the two to disagree.

**This changes the state dict.** `weight0` becomes
`parametrizations.weight0.original`, so a checkpoint written under QAT does not
load into a model built without it. `qat_state_dict_map` translates, and a test
holds both directions.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.utils.parametrize as parametrize

from .k3_qat import MX_GROUP, ste_mxfp4, ste_mxfp8

#: Shared experts are *not* quantised -- the release ships them in bf16, and only
#: the routed experts are MXFP4.
EXPERT_CONTAINER = "experts"
SHARED = "shared"


class FakeQuantMXFP4(torch.nn.Module):
    """Parametrization: the master goes in, the MXFP4 round trip comes out.

    Grouping is along the last axis, which for a `[out, in]` weight is the
    reduction axis -- the same axis the release quantises along, verified against
    the checkpoint's `[3072, 112]` scales for a `[3072, 3584]` weight.
    """

    def __init__(self, group_size: int = MX_GROUP):
        super().__init__()
        self.group_size = group_size

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[-1] % self.group_size:
            return weight
        return ste_mxfp4(weight, self.group_size)

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Loading a plain weight sets the master to it, unquantised."""
        return value


def _quantize_activations(group_size: int):
    def hook(_module, args):
        if not args or not torch.is_tensor(args[0]):
            return None
        x = args[0]
        if x.shape[-1] % group_size:
            return None
        return (ste_mxfp8(x, group_size),) + tuple(args[1:])

    return hook


def expert_weight_names(module) -> List[str]:
    """The weight parameters this expert linear owns, whichever layout it uses.

    Both appear in a real K3 model and the choice is not ours: with
    `moe_grouped_gemm` (the official presets) experts are a `TEGroupedMLP` holding
    `weight0..weightN`; without it (the tiny preset) they are a `SequentialMLP` of
    per-expert `MLP`s holding a single `weight`. A finder that knows only the
    grouped layout silently quantises nothing at tiny, and every test still passes.
    """
    names = [n for n, _ in module.named_parameters(recurse=False) if n.startswith("weight")]
    # Once parametrized the real parameter moves to `parametrizations.<name>.original`
    # and no longer appears in `named_parameters(recurse=False)`. Without this the
    # finder goes blind to exactly the modules it has already handled -- which
    # would make `enable_qat_experts` look idempotent by finding nothing at all.
    if parametrize.is_parametrized(module):
        names += list(module.parametrizations.keys())
    return sorted({n for n in names if n == "weight" or n[6:].isdigit()})


def expert_linears(model) -> List[torch.nn.Module]:
    """Every linear that belongs to a *routed* expert."""
    out = []
    for name, module in model.named_modules():
        if f".{EXPERT_CONTAINER}." not in f".{name}." or SHARED in name:
            continue
        if expert_weight_names(module):
            out.append(module)
    return out


def enable_qat_experts(
    model, group_size: int = MX_GROUP, quantize_activations: bool = True
) -> Dict[str, int]:
    """Put every routed expert weight under MXFP4 QAT. Returns what it touched.

    Idempotent: a module that is already parametrized is skipped, so calling this
    twice does not stack two quantisers on one weight (which would quantise the
    quantised value and is not the same operation).
    """
    touched = {"modules": 0, "weights": 0, "activation_hooks": 0}
    for module in expert_linears(model):
        for name in expert_weight_names(module):
            if parametrize.is_parametrized(module, name):
                continue
            parametrize.register_parametrization(module, name, FakeQuantMXFP4(group_size))
            touched["weights"] += 1
        if quantize_activations and getattr(module, "_k3_qat_handle", None) is None:
            module._k3_qat_handle = module.register_forward_pre_hook(
                _quantize_activations(group_size)
            )
            touched["activation_hooks"] += 1
        touched["modules"] += 1
    return touched


def disable_activation_quantisation(model) -> int:
    """Remove *our* activation hooks, by handle. Returns how many went.

    Serving runs quantised weights with unquantised activations, so measuring
    what QAT's activation path costs means turning exactly that off. Clearing
    `module._forward_pre_hooks` would do it too, and would also silently remove
    anything core had registered there -- which is why the handle is kept.
    """
    removed = 0
    for module in expert_linears(model):
        handle = getattr(module, "_k3_qat_handle", None)
        if handle is not None:
            handle.remove()
            module._k3_qat_handle = None
            removed += 1
    return removed


def qat_state_dict_map(state_dict: Dict[str, torch.Tensor], to_qat: bool) -> Dict[str, torch.Tensor]:
    """Translate between plain and parametrized expert keys.

    `to_qat=True`: `...linear_fc1.weight0` -> `...linear_fc1.parametrizations.weight0.original`.
    `to_qat=False`: the inverse. Needed because enabling QAT renames the very
    tensors the converter writes.
    """
    out = {}
    for key, value in state_dict.items():
        if to_qat and ".weight" in key and "parametrizations" not in key:
            head, _, tail = key.rpartition(".")
            routed = f".{EXPERT_CONTAINER}." in f".{head}." and SHARED not in head
            if routed and (tail == "weight" or (tail.startswith("weight") and tail[6:].isdigit())):
                key = f"{head}.parametrizations.{tail}.original"
        elif not to_qat and ".parametrizations." in key and key.endswith(".original"):
            head, _, rest = key.partition(".parametrizations.")
            key = f"{head}.{rest[: -len('.original')]}"
        out[key] = value
    return out
