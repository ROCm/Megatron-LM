"""Every parameter shape Muon orthogonalises, and every one it does not.

    torchrun --standalone --nproc_per_node=8 -m kimi_k3.tools.muon_shapes --preset 4L --ep 8

Muon's rule is in `megatron/core/optimizer/muon.py:295-300`: a parameter goes to
the orthogonalised group iff it is **2-D** and not flagged
`is_embedding_or_output_parameter`. Everything else -- norms, biases, the
embedding and the output layer -- goes to the nonlinear (Adam) group.

The cost of the split is the whole performance story of the step: Newton-Schulz
runs `num_ns_steps` iterations on every matrix in the first group, and the trace
puts that at ~81% of the iteration.
"""

import argparse
import json
import os
from collections import defaultdict

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4L")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args()
    for v in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(v, None)

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    from megatron.core import parallel_state, tensor_parallel

    parallel_state.initialize_model_parallel(expert_model_parallel_size=args.ep)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    from kimi_k3.model.build import build_k3_model

    model = build_k3_model(args.preset, allow_official=args.preset != "tiny",
                           expert_model_parallel_size=args.ep).bfloat16()

    muon, other = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        is_embed = getattr(p, "is_embedding_or_output_parameter", False)
        goes_to_muon = (not is_embed) and p.dim() == 2
        bucket = muon if goes_to_muon else other
        # role, not the numbered name: weight17 and weight83 are the same shape
        role = (name.replace("decoder.layers.", "L")
                    .split(".", 1)[-1] if name.startswith("decoder.layers.") else name)
        role = "".join(c for c in role if not c.isdigit()).replace("..", ".")
        key = (tuple(p.shape), role)
        bucket[key][0] += 1
        bucket[key][1] += p.numel()

    if rank == 0:
        def show(title, d):
            tot_n = sum(v[0] for v in d.values())
            tot_e = sum(v[1] for v in d.values())
            print(f"\n=== {title}: {tot_n} tensors, {tot_e/1e9:.3f} B params ===")
            print(f"{'shape':>22} {'count':>7} {'B params':>10}  role")
            for (shape, role), (n, e) in sorted(d.items(), key=lambda kv: -kv[1][1]):
                print(f"{str(shape):>22} {n:7d} {e/1e9:10.4f}  {role[:52]}")
        show("MUON (orthogonalised)", muon)
        show("NONLINEAR (Adam: norms, biases, embed/output)", other)
        nm = sum(v[0] for v in muon.values())
        print(f"\nNewton-Schulz work per step: {nm} matrices x 5 NS steps"
              f" = {nm*5} orthogonalisations, {nm*5*3} GEMM calls")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"muon": {f"{s}|{r}": v for (s, r), v in muon.items()},
                           "other": {f"{s}|{r}": v for (s, r), v in other.items()}}, f, indent=2)
    torch.distributed.destroy_process_group()


main()
