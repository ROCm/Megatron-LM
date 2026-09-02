"""Gates G42 / G45 -- the EP=8 proxy, its trace, and the ranked bottleneck report.

    torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 \\
        --preset 4L --seq 512 --iterations 10 --trace-dir develop/profile/traces

Rule R9.3 is the whole point of this tool: **attribute before tuning**. Nothing in
P11 gets optimised until this has said what the time is actually going on, and
the ranked table below is what the later gates are measured against.

Attribution is by explicit `record_function` regions wrapped around K3's own
module classes, not by guessing from kernel names. Kernel names cannot separate
an `aten::mul` in the AttnRes mixer from one in the MoE, and a report that
guesses is worse than no report. The regions nest -- `k3.layer` contains the
others -- so the table reports both and derives the remainder rather than
pretending the parts sum to the whole.
"""

import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import torch

#: (region name, dotted import path). Wrapped in order; nesting is fine.
REGIONS = (
    ("k3.layer", "kimi_k3.block.k3_transformer_layer:K3TransformerLayer"),
    ("k3.attn_res", "kimi_k3.block.attn_res:AttnResMixer"),
    ("k3.kda", "kimi_k3.attention.kda:KimiDeltaAttention"),
    ("k3.mla", "kimi_k3.attention.gated_mla:K3GatedMLA"),
    ("k3.moe", "kimi_k3.moe.k3_moe_layer:K3MoELayer"),
)


def instrument() -> List:
    """Wrap each region's `forward` in a profiler marker. Returns undo handles."""
    from torch.profiler import record_function

    undo = []
    for name, path in REGIONS:
        module_path, class_name = path.split(":")
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        original = cls.forward

        def wrapped(self, *a, _original=original, _name=name, **k):
            with record_function(_name):
                return _original(self, *a, **k)

        cls.forward = wrapped
        undo.append((cls, original))
    return undo


def build(args, rank: int, world: int):
    from megatron.core import parallel_state, tensor_parallel

    from kimi_k3.model.build import build_k3_model
    from kimi_k3.training.pretrain_kimi_k3 import build_optimizer

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    overrides = {}
    if args.kda_backend:
        overrides["k3_kda_backend"] = args.kda_backend
    if not args.ck_grouped_gemm:
        overrides["k3_moe_ck_grouped_gemm"] = False
    if args.experts:
        overrides["num_moe_experts"] = args.experts
    if args.dispatcher:
        overrides["moe_token_dispatcher_type"] = args.dispatcher
    if args.flex_backend:
        overrides["moe_flex_dispatcher_backend"] = args.flex_backend
    if args.fused_attn_res:
        overrides.update(k3_attn_res_fused=True, k3_attn_res_chunk=args.attn_res_chunk)
    if args.layers:
        # k3_kda_layers is 1-indexed and preset-wide, so it has to be trimmed too
        from kimi_k3.config.presets import preset as get_preset

        kda = get_preset(args.preset)["config"]["k3_kda_layers"]
        overrides.update(num_layers=args.layers,
                         k3_kda_layers=tuple(n for n in kda if n <= args.layers))
    model = build_k3_model(
        args.preset, allow_official=args.preset != "tiny",
        expert_model_parallel_size=args.ep,
        recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1,
        **overrides,
    ).bfloat16()
    after_model = torch.cuda.memory_allocated()
    ddp, opt = build_optimizer(model, optimizer=args.optimizer, lr=1e-5, bf16=True,
                               cpu_offload=args.cpu_offload,
                               offload_fraction=args.offload_fraction,
                               decoupled_weight_decay=args.decoupled_weight_decay)
    resident = {
        "after_model_gib": round(after_model / 2**30, 2),
        "after_optimizer_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
    }
    return model, ddp, opt, resident


class _Done(Exception):
    """Early exit from the measurement block without going through the error path."""


def resolved_dispatcher(config) -> Dict:
    """What the config *ended up* with, not what was asked for.

    Finding A17: core's `__post_init__` silently overwrites
    `moe_flex_dispatcher_backend` when `moe_enable_deepep` is set, and its guard
    against the combination is unreachable. So every arm records the backend it
    actually got -- otherwise two arms can report identical numbers with nothing
    in the output saying why.
    """
    return {
        "dispatcher": config.moe_token_dispatcher_type,
        "flex_backend": getattr(config, "moe_flex_dispatcher_backend", None),
        "enable_deepep": getattr(config, "moe_enable_deepep", None),
    }


def one_step(ddp, opt, vocab, seq, step):
    from kimi_k3.training.pretrain_kimi_k3 import loss_func, mock_batch

    tokens, labels = mock_batch(vocab, seq, 1, seed=step)
    ddp.zero_grad_buffer()
    opt.zero_grad()
    loss, _ = loss_func(labels)(ddp(input_ids=tokens, position_ids=None, attention_mask=None))
    loss.backward()
    ddp.finish_grad_sync()
    opt.step()
    return float(loss.detach())


def summarise(prof, iteration_ms: float) -> Dict:
    """Region totals, the comm share, and the top kernels, from one traced step."""
    regions: Dict[str, float] = defaultdict(float)
    kernels: Dict[str, List] = defaultdict(lambda: [0.0, 0])
    comm_ms = 0.0
    launches = 0

    for event in prof.key_averages():
        cuda_ms = (getattr(event, "self_device_time_total", 0) or 0) / 1000.0
        total_ms = (getattr(event, "device_time_total", 0) or 0) / 1000.0
        name = event.key
        if name in dict(REGIONS):
            regions[name] += total_ms
            continue
        if name.startswith("Optimizer.step#"):
            # ROCTracer emits a "duplicate flow start" for the optimizer annotation
            # and charges the region's whole *wall* duration to self_device_time.
            # It is a real duration but not device time, and its children (the
            # Newton-Schulz addmm/mm) are counted again below -- so summing it into
            # device_ms inflated the total and every pct_device with it. Verified:
            # a toy region of known length reports wall, not kernel, time.
            regions[name] += cuda_ms or total_ms
            continue
        if cuda_ms <= 0:
            continue
        launches += event.count
        kernels[name][0] += cuda_ms
        kernels[name][1] += event.count
        if "nccl" in name.lower() or "allreduce" in name.lower() or "all_gather" in name.lower():
            comm_ms += cuda_ms

    ranked = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:15]
    device_ms = sum(v[0] for v in kernels.values())
    return {
        "iteration_ms": round(iteration_ms, 2),
        "device_ms": round(device_ms, 2),
        "comm_ms": round(comm_ms, 2),
        "kernel_launches": launches,
        "regions_ms": {k: round(v, 2) for k, v in sorted(regions.items(), key=lambda kv: -kv[1])},
        "top_kernels": [
            {"name": n, "ms": round(v[0], 2), "calls": v[1], "pct_device": round(100 * v[0] / device_ms, 1)}
            for n, v in ranked
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=None,
                    help="override the preset's layer count; the proxy trades depth "
                         "for fitting on the GPUs actually available")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--optimizer", default="dist_muon")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="offload optimizer state and the update to CPU; adam_dist only "
                         "(inert with Muon, finding A20)")
    ap.add_argument("--offload-fraction", type=float, default=1.0)
    ap.add_argument("--coupled-weight-decay", dest="decoupled_weight_decay",
                    action="store_false", default=True,
                    help="Adam rather than AdamW: decay enters the moments instead of "
                         "being applied to the weight directly. For A/B-ing the decay "
                         "mode; offload rejects it.")
    ap.add_argument("--dispatcher", choices=("alltoall", "allgather", "flex"), default=None,
                    help="MoE token dispatcher; the A/B matrix arms (plan-0/07-dispatcher-ab.md)")
    ap.add_argument("--kda-backend", choices=("eager", "fla"), default=None,
                    help="eager is the fp32 oracle and stores the recurrent state at every "
                         "timestep -- 109 GiB per layer at seq 8192 against fla's 10.2. Any "
                         "memory measurement meant to size a cluster must use fla.")
    ap.add_argument("--no-ck-grouped-gemm", dest="ck_grouped_gemm", action="store_false",
                    default=True, help="fall back to hipBLASLt for the routed experts, to "
                                       "isolate what the CK grouped GEMM contributes")
    ap.add_argument("--experts", type=int, default=None,
                    help="override num_moe_experts; used to measure how headroom scales "
                         "with the number of experts local to a rank")
    ap.add_argument("--flex-backend", default=None,
                    help="deepep | mori | hybridep, only with --dispatcher flex")
    ap.add_argument("--fused-attn-res", action="store_true",
                    help="run with the chunked AttnRes mixer (G44/G45)")
    ap.add_argument("--attn-res-chunk", type=int, default=4096)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--no-trace", action="store_true",
                    help="skip the profiled iteration; the profiler inflates peak memory, "
                         "so any memory-scaling measurement must not include it")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(var, None)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)

    instrument()
    from kimi_k3.config.presets import preset as get_preset

    row = {"preset": args.preset, "ep": args.ep, "seq": args.seq, "world": world,
           "rank": rank, "layers": args.layers, "fused_attn_res": args.fused_attn_res,
           "arm": args.dispatcher or "default", "experts": args.experts,
           "kda_backend": args.kda_backend, "ck_grouped_gemm": args.ck_grouped_gemm,
           "optimizer": args.optimizer, "cpu_offload": args.cpu_offload,
           "offload_fraction": args.offload_fraction,
           "decoupled_weight_decay": args.decoupled_weight_decay}
    torch.cuda.reset_peak_memory_stats()
    try:
        model, ddp, opt, resident = build(args, rank, world)
        # Headroom is peak minus what is *resident* after the optimizer exists,
        # measured rather than derived. Subtracting an analytic state estimate
        # imports every uncertainty in that estimate -- and the estimate is wrong
        # here anyway: with world=8 and EP=8 the expert parameters have an
        # expert-DP of 1, so their optimizer state is not sharded and the 7.87
        # B/param figure from G5 (DP=8) does not apply to them.
        row.update(resident)
        vocab = get_preset(args.preset)["model"]["vocab_size"]

        cold = time.perf_counter()
        one_step(ddp, opt, vocab, args.seq, 0)
        torch.cuda.synchronize()
        row["cold_iteration_s"] = round(time.perf_counter() - cold, 2)

        for step in range(1, args.warmup):
            one_step(ddp, opt, vocab, args.seq, step)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        steady = []
        for step in range(args.warmup, args.iterations):
            start = time.perf_counter()
            one_step(ddp, opt, vocab, args.seq, step)
            torch.cuda.synchronize()
            steady.append((time.perf_counter() - start) * 1000)
        row["steady_ms"] = [round(v, 1) for v in steady]
        row["steady_ms_median"] = round(sorted(steady)[len(steady) // 2], 1)

        # one traced steady iteration
        if args.no_trace:
            peak = torch.cuda.max_memory_allocated() / 2**30
            row["peak_gib"] = round(peak, 2)
            row["headroom_gib"] = round(peak - row["after_optimizer_gib"], 2)
            row["resolved"] = resolved_dispatcher(model.config)
            row["params_this_rank"] = sum(p.numel() for p in model.parameters())
            row["status"] = "ok"
            raise _Done()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            traced = time.perf_counter()
            one_step(ddp, opt, vocab, args.seq, args.iterations)
            torch.cuda.synchronize()
            traced_ms = (time.perf_counter() - traced) * 1000
        row["trace"] = summarise(prof, traced_ms)
        if args.trace_dir and rank == 0:
            os.makedirs(args.trace_dir, exist_ok=True)
            path = os.path.join(args.trace_dir, f"proxy_ep{args.ep}_{args.preset}_rank0.json")
            prof.export_chrome_trace(path)
            row["chrome_trace"] = path

        row["resolved"] = resolved_dispatcher(model.config)
        from kimi_k3.tools.ep_smoke import routing_load

        row["routing"] = routing_load(model)
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["status"] = "ok"
    except _Done:
        pass
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        row["status"] = type(exc).__name__
        row["error"] = str(exc)[:400]
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, row)
    if rank == 0:
        print(json.dumps(gathered[0], indent=2))
        for other in gathered[1:]:
            print(json.dumps({k: other[k] for k in ("rank", "status", "steady_ms_median", "peak_gib")
                              if k in other}))
        if args.json:
            with open(args.json, "a") as handle:
                for entry in gathered:
                    handle.write(json.dumps(entry) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
