"""Measured tolerances, and the one comparison function every parity test uses.

Rule R4.4: bounds are established by measurement, in this order --

1. measure the **floor**: the same eager code in bf16 against itself in fp32;
2. measure the **signal**: the fast backend against the fp32 oracle, same dtype;
3. set the bound above the floor with recorded margin.

Every entry below carries the measurement that produced it, so a future reader
can tell a considered bound from a guessed one. Regenerate with
`python -m kimi_k3.tools.kda_parity_probe`; raw rows live in
`develop/results/kda_parity_raw.jsonl`.
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class Bound:
    rel_l2: float
    cosine: float
    measured: float
    floor: float
    note: str

    @property
    def margin(self) -> float:
        return self.rel_l2 / self.measured


def compare(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """rel-L2, max-abs and cosine, in fp32, flattened."""
    a, b = actual.float().flatten(), reference.float().flatten()
    return {
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-20)).item(),
        "max_abs": (a - b).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def assert_within(actual, reference, bound: Bound, what: str = "") -> Dict[str, float]:
    stats = compare(actual, reference)
    assert stats["rel_l2"] <= bound.rel_l2, (
        f"{what}: rel-L2 {stats['rel_l2']:.3e} exceeds {bound.rel_l2:.3e} "
        f"(measured {bound.measured:.3e}, floor {bound.floor:.3e}) -- {bound.note}"
    )
    assert stats["cosine"] >= bound.cosine, f"{what}: cosine {stats['cosine']:.6f} < {bound.cosine}"
    return stats


# --- KDA, measured 2026-08-27 on MI355X (tiny H=2 K=64 and mid H=8 K=128,
#     seq 256 / 1024 / 4096; every figure was flat across both axes) ----------

KDA_FWD_FP32 = Bound(
    rel_l2=1e-5, cosine=0.999999, measured=7.1e-7, floor=0.0,
    note="fla vs the fp32 oracle in fp32; ~14x margin. The kernel is right.",
)
KDA_FWD_BF16 = Bound(
    rel_l2=1e-2, cosine=0.9999, measured=4.3e-3, floor=3.3e-3,
    note="dominated by dtype, not by the kernel: the eager-vs-eager bf16 floor is "
         "itself 3.3e-3, so fla adds ~1e-3 on top of what bf16 costs anyway.",
)
KDA_BWD_FP32 = Bound(
    rel_l2=1e-3, cosine=0.999, measured=2.9e-5, floor=0.0,
    note="worst per-tensor gradient (dt_bias / g).",
)
KDA_BWD_BF16 = Bound(
    rel_l2=2e-2, cosine=0.999, measured=6.1e-3, floor=3.3e-3,
    note="worst per-tensor gradient (v).",
)
