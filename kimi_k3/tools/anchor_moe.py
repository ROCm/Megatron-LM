"""Anchored parity for one MoE layer -- routing and the latent path, real weights.

Dequantised experts are ~59 GB per layer, so the release block and ours cannot both
be resident. The comparison is therefore staged: routing decisions first (which is
where a misreading would show), then the latent projections and shared experts.
"""
import os, sys, json, glob
sys.path.insert(0, "/tmp/tf4562")
REF = "/tmp/claude-0/-workspace/edf3a927-9465-4cd8-beed-9a2f60edf2a5/scratchpad/k3ref"
sys.path.insert(0, REF)
for v in ("NVTE_FLASH_ATTN","NVTE_FUSED_ATTN","NVTE_UNFUSED_ATTN"): os.environ.pop(v, None)
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","30065")
import torch
from safetensors import safe_open

P = "language_model.model.layers.1.block_sparse_moe."
w = {}
for f in sorted(glob.glob("/tmp/k3w/*.safetensors")):
    with safe_open(f, framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k.startswith(P) and ".experts." not in k:
                w[k[len(P):]] = fh.get_tensor(k)
print("non-expert MoE tensors:", sorted(w), flush=True)

from megatron.core import parallel_state, tensor_parallel
torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
torch.cuda.set_device(0); parallel_state.initialize_model_parallel(1,1)
tensor_parallel.model_parallel_cuda_manual_seed(1234)
from kimi_k3.moe.k3_router import released_gate_reference
from megatron.core.transformer.moe.router import TopKRouter
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
cfg = config_from_preset(preset("93L")["config"])

S = 128
torch.manual_seed(5)
x = (torch.randn(1, S, 7168, device="cuda") * 0.02)
gate_w = w["gate.weight"].cuda().float()
bias = w["gate.e_score_correction_bias"].cuda().float()

# the release's own gate, transcribed in P6 and checked against it there
idx_ref, wt_ref = released_gate_reference(x, gate_w, bias, cfg.moe_router_topk)

# core's topk_routing_with_score_function, the path our QuantileBalancingRouter inherits
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
logits = torch.nn.functional.linear(x.view(-1, 7168), gate_w)
probs, routing_map = topk_routing_with_score_function(
    logits, cfg.moe_router_topk, use_pre_softmax=True,
    score_function="sigmoid", expert_bias=bias,
)
core_sel = routing_map.nonzero()[:, 1].view(-1, cfg.moe_router_topk)
w_core = probs.gather(1, core_sel)
ref_sel, _ = idx_ref.sort(dim=-1)
core_sel, _ = core_sel.sort(dim=-1)
agree = (ref_sel == core_sel).all(-1).float().mean().item()
print(f"\nROUTING, real layer-1 gate weights + e_score_correction_bias, {S} tokens:")
print(f"  top-16 expert sets identical for {agree*100:.2f}% of tokens", flush=True)
wr, _ = wt_ref.sort(dim=-1); wc, _ = w_core.sort(dim=-1)
print(f"  gathered weights: rel-L2 {((wc-wr).norm()/wr.norm()).item():.3e}", flush=True)
print(f"  bias actually nonzero: {bool((bias != 0).any())}  |bias|max {bias.abs().max():.4f}", flush=True)

# latent projections + routed_expert_norm: shapes and the norm core lacks
for k in ("routed_expert_down_proj.weight", "routed_expert_up_proj.weight", "routed_expert_norm.weight"):
    print(f"  {k:34} {tuple(w[k].shape)}", flush=True)
print(f"  shared_experts.gate_proj.weight     {tuple(w['shared_experts.gate_proj.weight'].shape)}", flush=True)
