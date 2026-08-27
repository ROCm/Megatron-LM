"""One routed expert under MXFP4/MXFP8 QAT.

The P0 prototype for gate G8. It owns the *semantics* the AITER a8w4 kernel path
has to match in P10:

* **fp32 masters** are the parameters the optimizer updates;
* a **packed MXFP4 cache** (uint8 data + e8m0 scales) is what a kernel would
  read, refreshed from the masters after every optimizer step;
* the forward consumes the cache, the backward is a **straight-through
  estimator** to the masters;
* both survive a checkpoint round-trip.

`gradcheck` is invalid here by construction -- an STE backward is deliberately
not the derivative of its forward (rule R4.4). The gate compares against an
explicit fake-quant reference instead.

The fast forward (`aiter.ops.opus.moe_stage{1,2}_a8w4`, tuned on gfx950 for
expert=896 / topk=16 / model_dim=3584 / `ActivationType.Situv2`) is P10's job;
this module runs the dequantised matmul in its place.
"""

from typing import Dict

import torch

from .k3_qat import MX_GROUP, dequantize_mxfp4, fake_quantize_mxfp4, quantize_mxfp4
from .situ import situ_glu


def ste_from_cache(master: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
    """Forward takes the cached (quantised) value, backward is identity to master.

    `master + (cached - master).detach()` is exactly the value of `cached` with
    the gradient of `master` -- the same contract as an explicit STE autograd
    function, without a second implementation to keep in sync.
    """
    return master + (cached - master).detach()


class KimiK3QATExpert(torch.nn.Module):
    """A single routed expert: ``w2( situ_glu([w1 x | w3 x]) )``."""

    WEIGHTS = ("w1", "w3", "w2")

    def __init__(self, hidden: int, inter: int, group_size: int = MX_GROUP, dtype=torch.float32):
        super().__init__()
        self.hidden, self.inter, self.group_size = hidden, inter, group_size
        # fp32 masters
        self.w1 = torch.nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        for w in (self.w1, self.w3, self.w2):
            torch.nn.init.normal_(w, std=0.02)
        # packed caches -- buffers so they ride the checkpoint
        for name in self.WEIGHTS:
            w = getattr(self, name)
            self.register_buffer(f"{name}_packed", torch.zeros(*w.shape[:-1], w.shape[-1] // 2,
                                                               dtype=torch.uint8))
            self.register_buffer(f"{name}_scale", torch.zeros(*w.shape[:-1],
                                                              w.shape[-1] // group_size,
                                                              dtype=torch.uint8))
        self.refresh_packed_cache()

    @torch.no_grad()
    def refresh_packed_cache(self) -> None:
        """Re-quantise the masters. Call after every optimizer step."""
        for name in self.WEIGHTS:
            packed, scale = quantize_mxfp4(getattr(self, name).data, self.group_size)
            getattr(self, f"{name}_packed").copy_(packed)
            getattr(self, f"{name}_scale").copy_(scale)

    def cached_weight(self, name: str) -> torch.Tensor:
        """The dequantised cache, carrying the master's gradient (STE)."""
        master = getattr(self, name)
        cached = dequantize_mxfp4(
            getattr(self, f"{name}_packed"), getattr(self, f"{name}_scale"), self.group_size
        ).to(master.dtype)
        return ste_from_cache(master, cached)

    def forward(self, x: torch.Tensor, quantize_activations: bool = True) -> torch.Tensor:
        from .k3_qat import ste_mxfp8

        h = ste_mxfp8(x, self.group_size) if quantize_activations else x
        gate = torch.nn.functional.linear(h, self.cached_weight("w1"))
        up = torch.nn.functional.linear(h, self.cached_weight("w3"))
        act = situ_glu(torch.cat([gate, up], dim=-1))
        act = ste_mxfp8(act, self.group_size) if quantize_activations else act
        return torch.nn.functional.linear(act, self.cached_weight("w2"))

    def cache_matches_masters(self) -> Dict[str, float]:
        """Max |dequant(cache) - fake_quant(master)| per weight; 0.0 when in sync."""
        out = {}
        for name in self.WEIGHTS:
            master = getattr(self, name)
            cached = dequantize_mxfp4(
                getattr(self, f"{name}_packed"), getattr(self, f"{name}_scale"), self.group_size
            )
            out[name] = (cached - fake_quantize_mxfp4(master.data, self.group_size)).abs().max().item()
        return out


class KimiK3FakeQuantExpertReference(torch.nn.Module):
    """The reference the STE path is checked against.

    Identical quantisation, ordinary autograd, no packed cache: the weight path
    is `fake_quantize(master)` computed inline every forward.
    """

    def __init__(self, expert: KimiK3QATExpert):
        super().__init__()
        self.group_size = expert.group_size
        for name in KimiK3QATExpert.WEIGHTS:
            setattr(self, name, torch.nn.Parameter(getattr(expert, name).detach().clone()))

    def forward(self, x: torch.Tensor, quantize_activations: bool = True) -> torch.Tensor:
        from .k3_qat import ste_mxfp4, ste_mxfp8

        h = ste_mxfp8(x, self.group_size) if quantize_activations else x
        gate = torch.nn.functional.linear(h, ste_mxfp4(self.w1, self.group_size))
        up = torch.nn.functional.linear(h, ste_mxfp4(self.w3, self.group_size))
        act = situ_glu(torch.cat([gate, up], dim=-1))
        act = ste_mxfp8(act, self.group_size) if quantize_activations else act
        return torch.nn.functional.linear(act, ste_mxfp4(self.w2, self.group_size))
