"""G48 -- do all ranks agree on the expert bias when they see different tokens?

    torchrun --nproc_per_node=8 -m kimi_k3.tools.router_sync_probe

Each rank observes only its own microbatch. Before `pooled_histogram()` the
quantile was estimated from that local view alone, so every rank derived a
different bias and the same token could be routed differently depending on
which rank held it -- with nothing to flag it. This feeds each rank a
deliberately different score distribution, which is the case that breaks.

`BYPASS_REDUCE=1` restores the pre-fix behaviour, so the probe can be shown to
fail when the reduce is absent rather than merely passing when it is present.
"""
import torch, torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    from megatron.core import parallel_state, tensor_parallel
    parallel_state.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    from kimi_k3.moe.k3_router import ScoreQuantileEstimator, quantile_balancing_bias

    experts, topk = 64, 8
    est = ScoreQuantileEstimator(experts, num_bins=1024, momentum=0.0).cuda()
    # Rank-dependent scores: without a reduce, every rank sees a different world.
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(512, experts, generator=g, device="cuda") * (0.3 + 0.1 * rank)
    est.update(scores)

    import os
    if os.environ.get("BYPASS_REDUCE") == "1":   # the pre-fix behaviour
        ScoreQuantileEstimator.pooled_histogram = lambda self: self.histogram
    bias = quantile_balancing_bias(est, topk, experts)
    gathered = [torch.zeros_like(bias) for _ in range(world)]
    dist.all_gather(gathered, bias)
    spread = max((gathered[i] - gathered[0]).abs().max().item() for i in range(world))

    local = est.histogram.sum().item()
    pooled = est.pooled_histogram().sum().item()
    if rank == 0:
        print(f"ranks={world}  local hist mass={local:.1f}  pooled={pooled:.1f}  ratio={pooled/local:.2f}")
        print(f"max bias disagreement across ranks = {spread:.3e}")
        print("PASS: all ranks agree" if spread < 1e-6 else "FAIL: biases diverge")
    dist.destroy_process_group()

main()
