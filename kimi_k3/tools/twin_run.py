"""Twin runs: is a code change inside the run-to-run noise, or did it move the model?

Two runs that *should* be equivalent -- eager vs `fla` KDA, recompute on vs off --
will not produce identical losses, because neither reassociates floating-point
arithmetic the same way. So "did this change anything" cannot be answered by
comparing to zero. It has to be answered against a **measured** band.

The band comes first, from runs that differ only by seed. That is the amount of
loss movement this configuration produces for no reason at all. A twin whose
statistics sit inside it has not been shown to change the model; a twin outside it
has.

Three statistics, because one hides things:

* `max_delta` -- the worst single step. Catches a spike that a mean would absorb.
* `mean_delta` -- the whole window. Catches a small constant offset.
* `final_delta` -- the mean over the last quarter. Catches slow divergence, which
  is the failure mode that matters and the one the first two miss.

`python -m kimi_k3.tools.twin_run --preset tiny --steps 40`
"""

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from ..training.pretrain_kimi_k3 import train_smoke


@dataclass
class Statistics:
    max_delta: float
    mean_delta: float
    final_delta: float

    def inside(self, band: "Statistics") -> bool:
        return all(
            getattr(self, f) <= getattr(band, f) for f in ("max_delta", "mean_delta", "final_delta")
        )


def compare(a: Sequence[float], b: Sequence[float]) -> Statistics:
    assert len(a) == len(b), f"{len(a)} vs {len(b)} steps"
    deltas = [abs(x - y) for x, y in zip(a, b)]
    tail = deltas[-max(1, len(deltas) // 4) :]
    return Statistics(
        max_delta=max(deltas),
        mean_delta=sum(deltas) / len(deltas),
        final_delta=sum(tail) / len(tail),
    )


def widest(stats: Sequence[Statistics]) -> Statistics:
    """The band is the worst each statistic gets across the seed pairs."""
    return Statistics(*(max(getattr(s, f) for s in stats) for f in ("max_delta", "mean_delta", "final_delta")))


def run(preset: str, steps: int, seed: int, **overrides) -> List[float]:
    """One run. Everything the run depends on is an argument -- nothing ambient."""
    return train_smoke(
        preset=preset,
        iterations=steps,
        seq_length=32,
        micro_batch_size=1,
        optimizer="dist_muon",
        lr=1e-4,
        bf16=False,
        seed=seed,
        fixed_batch=True,
        overrides=overrides or None,
    )


def noise_band(preset: str, steps: int, seeds: Sequence[int] = (0, 1, 2)) -> Dict:
    """Run the same configuration under several seeds and measure the spread."""
    curves = {seed: run(preset, steps, seed) for seed in seeds}
    pairs = {
        f"{a}v{b}": compare(curves[a], curves[b]) for a, b in itertools.combinations(seeds, 2)
    }
    return {
        "seeds": list(seeds),
        "pairs": {k: asdict(v) for k, v in pairs.items()},
        "band": asdict(widest(list(pairs.values()))),
        "curves": {str(k): v for k, v in curves.items()},
    }


#: Axes that should not change the model. Each is (name, overrides A, overrides B).
AXES = (
    ("kda_backend", {"k3_kda_backend": "eager"}, {"k3_kda_backend": "fla"}),
    ("recompute", {"recompute_granularity": None}, {"recompute_granularity": "full",
                                                   "recompute_method": "uniform",
                                                   "recompute_num_layers": 1}),
)


def axis_engaged(preset: str, name: str) -> Dict:
    """Evidence that the axis under test actually did something.

    The recompute twin is bitwise identical, which is the *correct* answer -- an
    activation checkpoint that is replayed correctly reproduces the same numbers.
    But "0.0" and "the flag did nothing" look the same from the outside, so the
    harness has to show the path fired rather than assume it. Without this the
    twin is a gate that cannot fail (review finding A1).
    """
    import torch
    from megatron.core import tensor_parallel

    from ..model.build import build_k3_model

    axis = next(a for a in AXES if a[0] == name)
    counts = []
    original = tensor_parallel.checkpoint
    try:
        for side in (axis[1], axis[2]):
            calls = []
            tensor_parallel.checkpoint = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
            model = build_k3_model(preset, **side)
            tokens = torch.randint(0, 64, (1, 8), device="cuda")
            model(input_ids=tokens, position_ids=None, attention_mask=None).sum().backward()
            counts.append(len(calls))
    finally:
        tensor_parallel.checkpoint = original
    return {"checkpoint_calls": counts, "differs": counts[0] != counts[1]}


def twin(preset: str, steps: int, name: str, seed: int = 0) -> Dict:
    """Both sides of one axis, same seed, same data."""
    axis = next(a for a in AXES if a[0] == name)
    a, b = run(preset, steps, seed, **axis[1]), run(preset, steps, seed, **axis[2])
    return {"axis": name, "seed": seed, "stats": asdict(compare(a, b)), "a": a, "b": b}


def _init_world() -> None:
    """A 1-rank world, because every run here is single-GPU by construction."""
    import torch
    from megatron.core import parallel_state

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--axes", nargs="*", default=[a[0] for a in AXES])
    ap.add_argument("--out")
    args = ap.parse_args()

    _init_world()

    report = {"preset": args.preset, "steps": args.steps}
    report["noise"] = noise_band(args.preset, args.steps, args.seeds)
    band = Statistics(**report["noise"]["band"])
    print(f"noise band over seeds {args.seeds}: {report['noise']['band']}")

    report["twins"] = {}
    for name in args.axes:
        try:
            result = twin(args.preset, args.steps, name)
        except Exception as exc:  # a missing backend is a result, not a crash
            report["twins"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: FAILED TO RUN -- {type(exc).__name__}: {exc}")
            continue
        result["inside_band"] = Statistics(**result["stats"]).inside(band)
        if name == "recompute":
            result["engaged"] = axis_engaged(args.preset, name)
        report["twins"][name] = result
        print(f"{name}: {result['stats']} inside_band={result['inside_band']}"
              + (f" engaged={result['engaged']}" if "engaged" in result else ""))

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
