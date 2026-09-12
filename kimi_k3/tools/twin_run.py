"""Twin runs: is a code change inside the run-to-run noise, or did it move the model?

Two runs that *should* be equivalent -- eager vs `fla` KDA, recompute on vs off --
will not produce identical losses, because neither reassociates floating-point
arithmetic the same way. So "did this change anything" cannot be answered by
comparing to zero.

**Method: a two-sample permutation test on the loss curves.** Each arm is run at
`--seeds-per-arm` seeds. The statistic is a summary of the difference between the
two arms' mean curves; its null distribution comes from relabelling which runs
belong to which arm. If the real labelling produces no larger a difference than a
random one, the change is not distinguishable from reseeding.

Three summaries, because one hides things:

* `max` -- the worst single step. Catches a spike that a mean would absorb.
* `mean` -- the whole window. Catches a small constant offset.
* `final` -- the mean over the last quarter. Catches slow divergence, which is the
  failure mode that matters and the one the first two miss.

Three tests, so p-values are corrected Holm-Bonferroni for the verdict.

## Why not the old "noise band" (G34-G55, superseded)

The original method measured pairwise deltas between seed runs and used the
**maximum** as a pass threshold. That was an ad-hoc rule of mine, never justified
and not a standard statistical method. Its defects, in the order they matter:

* The sample maximum is not a stable estimator -- it grows with the number of
  seeds, so the threshold depended on how many runs happened to be done, and
  *more* evidence made the test *more permissive*.
* The pairwise deltas are not independent: k seeds give k(k-1)/2 pairs but only k
  runs, so "15 pairs" was never 15 samples.
* Each twin was measured at a **single seed** and compared against that threshold,
  with no estimate of the twin statistic's own variability.
* No confidence level, no false-positive rate, no correction across the three
  statistics.

G55 is the concrete cost: `kda_backend` read outside a three-seed band and inside
a six-seed one, same number either way. A permutation test has none of these
properties -- it is distribution-free, uses the runs themselves as the null, and
reports a p-value at a stated level.

    python -m kimi_k3.tools.twin_run --preset tiny --steps 40 --seeds-per-arm 4
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


def mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    return [sum(step) / len(step) for step in zip(*curves)]


def group_statistic(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Statistics:
    """Summaries of the gap between the two arms' mean curves."""
    return compare(mean_curve(a), mean_curve(b))


def permutation_test(
    arm_a: Sequence[Sequence[float]], arm_b: Sequence[Sequence[float]], max_perms: int = 20000
) -> Dict:
    """Two-sample permutation test on the loss curves.

    Pools the runs and enumerates every way of splitting them into two groups of
    the original sizes. Label swaps give an identical statistic, so only half the
    splits are distinct; the observed labelling is one of them and is included in
    the count, which is what keeps the p-value valid (it can never be 0).

    Exact whenever the number of distinct splits fits under `max_perms`.

    Resolution is the binding constraint, and it interacts with the Holm correction
    in a way that is easy to get wrong: Holm multiplies the smallest of the three
    p-values by 3, so a verdict at alpha = 0.05 needs a raw p of 0.0167 or less,
    hence **at least 60 distinct splits**. 4 seeds per arm gives C(8,4)/2 = 35
    (min raw p 0.029, which Holm inflates to 0.086) and can therefore never reach
    significance at all. 5 per arm gives 126 splits (0.0079 -> 0.024) and can.
    """
    pooled = list(arm_a) + list(arm_b)
    n = len(arm_a)
    observed = group_statistic(arm_a, arm_b)

    index_sets = [frozenset(c) for c in itertools.combinations(range(len(pooled)), n)]
    seen, splits = set(), []
    for s in index_sets:                       # drop the label-swap duplicate
        key = min(s, frozenset(range(len(pooled))) - s, key=sorted)
        if key in seen:
            continue
        seen.add(key)
        splits.append(s)
    exact = len(splits) <= max_perms
    if not exact:
        import random

        rng = random.Random(0)
        splits = rng.sample(splits, max_perms)

    fields = ("max_delta", "mean_delta", "final_delta")
    counts = {f: 0 for f in fields}
    for s in splits:
        left = [pooled[i] for i in sorted(s)]
        right = [pooled[i] for i in range(len(pooled)) if i not in s]
        st = group_statistic(left, right)
        for f in fields:
            if getattr(st, f) >= getattr(observed, f) - 1e-12:
                counts[f] += 1
    p = {f: counts[f] / len(splits) for f in fields}

    # Holm-Bonferroni across the three statistics.
    order = sorted(fields, key=lambda f: p[f])
    adjusted, running = {}, 0.0
    for rank, f in enumerate(order):
        running = max(running, min(1.0, p[f] * (len(fields) - rank)))
        adjusted[f] = running
    return {
        "observed": asdict(observed),
        "p_value": p,
        "p_value_holm": adjusted,
        "splits": len(splits),
        "exact": exact,
        "min_attainable_p": 1.0 / len(splits),
        "significant_at_05": any(v <= 0.05 for v in adjusted.values()),
    }


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


def noise_band(preset: str, steps: int, seeds: Sequence[int] = (0, 1, 2, 3, 4, 5)) -> Dict:
    """Run the same configuration under several seeds and measure the spread.

    Six seeds, not three (G55). The band is a **max over pairs**, so few seeds
    sample that maximum badly and it reads low. Measured: the three-seed band puts
    `final_delta` at 0.0593 and the six-seed band at 0.0794, +34%, and **16 of the
    20 possible three-seed subsets would have wrongly failed the `kda_backend`
    twin** at 0.0754. Three seeds is three pairs; six is fifteen.
    """
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
    # Unlike the axes above, this one is *not* expected to be a no-op: per-head
    # Muon deliberately changes the update (G35's control asserts it does). The
    # band is still the right yardstick -- the question is whether a change that
    # costs 1.75x on the largest row moves the loss further than reseeding does.
    ("per_head_muon", {"k3_per_head_muon": False}, {"k3_per_head_muon": True}),
    # Also not a no-op, and deliberately so: this is the G52 fix. Before it, every
    # preset-built model ran GeGLU on the routed experts instead of the released
    # SiTU-GLU. Included as a control -- if wiring the correct activation moved the
    # loss no further than reseeding does, the gate would not be measuring anything.
    ("situ_activation", {"k3_situ_activation": False}, {"k3_situ_activation": True}),
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


def twin_arms(preset: str, steps: int, name: str, seeds: Sequence[int]) -> Dict:
    """Both sides of one axis, at every seed. This is what the old method lacked.

    G55 compared a **single** seed-0 twin against a threshold, so the twin statistic
    had no measured variability of its own -- 0.0754 could as easily have come out
    0.06 or 0.09. Running both arms across seeds is what makes a test possible.
    """
    axis = next(a for a in AXES if a[0] == name)
    return {
        "a": [run(preset, steps, sd, **axis[1]) for sd in seeds],
        "b": [run(preset, steps, sd, **axis[2]) for sd in seeds],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds-per-arm", type=int, default=5,
                    help="runs per arm. Holm multiplies the smallest of three p-values "
                         "by 3, so a verdict needs raw p <= 0.0167, i.e. >= 60 splits. "
                         "4 per arm gives only C(8,4)/2 = 35 (min raw p 0.029 -> 0.086 "
                         "after Holm) and can never reach alpha = 0.05; 5 gives 126 "
                         "splits (0.008 -> 0.024) and can.")
    ap.add_argument("--axes", nargs="*", default=[a[0] for a in AXES])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out")
    args = ap.parse_args()

    _init_world()
    seeds = list(range(args.seeds_per_arm))
    report = {"preset": args.preset, "steps": args.steps, "seeds_per_arm": args.seeds_per_arm,
              "alpha": args.alpha, "method": "two-sample permutation test, Holm-Bonferroni"}

    if args.seeds_per_arm < 5:
        print(f"WARNING: {args.seeds_per_arm} seeds/arm cannot reach alpha={args.alpha}; "
              "every verdict below will be 'not significant' by construction")

    report["twins"] = {}
    for name in args.axes:
        try:
            arms = twin_arms(args.preset, args.steps, name, seeds)
            result = permutation_test(arms["a"], arms["b"])
            result["curves"] = arms
        except Exception as exc:  # a missing backend is a result, not a crash
            report["twins"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: FAILED TO RUN -- {type(exc).__name__}: {exc}")
            continue
        if name == "recompute":
            result["engaged"] = axis_engaged(args.preset, name)
        report["twins"][name] = result
        moved = any(v <= args.alpha for v in result["p_value_holm"].values())
        obs = result["observed"]
        print(f"{name}: observed max={obs['max_delta']:.4f} mean={obs['mean_delta']:.4f} "
              f"final={obs['final_delta']:.4f}")
        print(f"    p(holm) max={result['p_value_holm']['max_delta']:.3f} "
              f"mean={result['p_value_holm']['mean_delta']:.3f} "
              f"final={result['p_value_holm']['final_delta']:.3f} "
              f"[{result['splits']} splits, exact={result['exact']}, "
              f"min p={result['min_attainable_p']:.3f}]")
        print(f"    -> {'MOVED THE MODEL' if moved else 'not distinguishable from reseeding'}"
              + (f"  engaged={result['engaged']}" if "engaged" in result else ""))

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
