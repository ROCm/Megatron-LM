# 2026-08-27 — How G5 measures optimizer memory (and one trap)

`analysis` · gate G5 · tooling: `kimi_k3/tools/opt_mem_probe.py`,
`kimi_k3/tools/opt_mem_report.py`

## Why measure rather than compute

The incoming plan derived every capacity number from a flat **14 B/param** for
Muon, reasoning from `assert not args.use_distributed_optimizer`
(arguments.py:1552) that Muon cannot shard optimizer state. `dist_muon` shards
by a different mechanism — `LayerWiseDistributedOptimizer` assigns whole tensors
to DP ranks (layer_wise_optimizer.py:105-158) — so the plan's table is right only
at DP = 1. G5 exists to replace the whole table with measurements.

## Method

One `torchrun` launch per recipe, world size = DP, TP = PP = EP = 1. The probe
model carries K3's *shape mix* rather than a stack of plain linears — MLA with
LoRA norms, MoE with routed and shared experts, RMSNorms — so the 2-D Muon group
and the scalar group are both populated (99.0 M parameters).

Bytes per parameter is the **CUDA allocator delta** from before model
construction to after two optimizer steps:

```
bytes_per_param = (memory_allocated_after_2_steps - memory_allocated_before_model) / n_params
```

Two steps, not one, because Adam moments are allocated lazily on first use. This
figure therefore includes bf16 weights, the fp32 gradient buffer, fp32 masters,
per-parameter state and any all-gather scratch — everything resident, not just
what the optimizer's `state` dict happens to name. A second pass walks the
(chained) optimizer and totals master weights and per-parameter state separately,
splitting by `param.dim() == 2`, so the Muon and scalar groups can be told apart.

Sanity check that validates the method: plain `adam` measures **18.02 B/param**
at every DP, against an analytic 2 + 4 + 4 + 8 = 18.

## The trap: DDP has to know about the distributed optimizer too

Setting `OptimizerConfig(use_distributed_optimizer=True)` alone is not enough.
`DistributedDataParallelConfig` needs the same flag, or DDP never allocates the
parameter buffers the distributed optimizer indexes into, and the **first step**
dies with:

```
File "megatron/core/optimizer/distrib_optimizer.py", line 2479, in copy_group_params
    shard_model_param = model_param_buffer.view(-1)[...]
AttributeError: 'NoneType' object has no attribute 'view'
```

Nothing validates the pair at construction time — the failure is deferred to the
first `optimizer.step()`, which for a real training job means it surfaces after
the whole model and dataloader are up. Worth remembering for P7's trainer
bring-up: if a run dies there, this mismatch is the first thing to check.

## Reproducing

```bash
torchrun --nproc_per_node=8 -m kimi_k3.tools.opt_mem_probe --optimizer dist_muon --out raw.jsonl
python -m kimi_k3.tools.opt_mem_report raw.jsonl > develop/results/opt_mem.md
```

Raw per-rank rows are kept in `develop/results/opt_mem_raw.jsonl` so the report
can be re-rendered without re-running the matrix, and so the per-rank spread
(whole-tensor sharding is only approximately balanced) stays inspectable.
