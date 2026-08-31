"""Anchored parity for gated MLA — layer 3, real released weights."""
import os, sys, json, glob
sys.path.insert(0, "/tmp/tf4562")
REF = "/tmp/claude-0/-workspace/edf3a927-9465-4cd8-beed-9a2f60edf2a5/scratchpad/k3ref"
sys.path.insert(0, REF)
for v in ("NVTE_FLASH_ATTN","NVTE_FUSED_ATTN","NVTE_UNFUSED_ATTN"): os.environ.pop(v, None)
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","30061")
import torch
from safetensors import safe_open

P = "language_model.model.layers.3."
raw = torch.load("/tmp/k3_layer3_mla.pt", weights_only=True)
w = {k[len(P + "self_attn."):]: v for k, v in raw.items() if k.startswith(P + "self_attn.")}
print("MLA tensors:", sorted(w), flush=True)

cfg_json = json.load(open(f"{REF}/config.json"))["text_config"]
d = dict(cfg_json); d.setdefault("_attn_implementation", "eager"); d.setdefault("attention_dropout", 0.0)
cfg = type("Cfg", (), d)()
from k3pkg.modeling_kimi_linear import KimiMLAAttention
ref = KimiMLAAttention(cfg, layer_idx=3).cuda().bfloat16().eval()
missing, unexpected = ref.load_state_dict({k: v.cuda() for k, v in w.items()}, strict=False)
print("release module: missing", [m for m in missing if 'rotary' not in m], "unexpected", unexpected, flush=True)

from megatron.core import parallel_state, tensor_parallel
torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
torch.cuda.set_device(0); parallel_state.initialize_model_parallel(1,1)
tensor_parallel.model_parallel_cuda_manual_seed(1234)
from kimi_k3.attention.gated_mla import K3GatedMLA
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
ours = K3GatedMLA(config_from_preset(preset("93L")["config"])).cuda().bfloat16().eval()
m2, u2 = ours.load_state_dict({
    "q_a_proj.weight": w["q_a_proj.weight"].cuda(), "q_a_layernorm": w["q_a_layernorm.weight"].cuda(),
    "q_b_proj.weight": w["q_b_proj.weight"].cuda(),
    "kv_a_proj_with_mqa.weight": w["kv_a_proj_with_mqa.weight"].cuda(),
    "kv_a_layernorm": w["kv_a_layernorm.weight"].cuda(),
    "kv_b_proj.weight": w["kv_b_proj.weight"].cuda(),
    "o_proj.weight": w["o_proj.weight"].cuda(), "g_proj.weight": w["g_proj.weight"].cuda(),
}, strict=False)
print("our module    : missing", [x for x in m2 if torch.is_tensor(ours.state_dict().get(x))], "unexpected", u2, flush=True)
print("params: release", sum(p.numel() for p in ref.parameters()), "ours", sum(p.numel() for p in ours.parameters()), flush=True)

torch.manual_seed(1)
S = 128
x = (torch.randn(1, S, 7168, device="cuda", dtype=torch.bfloat16) * 0.02)
pos = torch.arange(S, device="cuda").unsqueeze(0)
# The release hands attention_mask straight to eager_attention_forward, so None
# means *bidirectional* attention there while our _sdpa forces is_causal=True.
# Comparing without this mask measures the harness, not the model.
causal = torch.full((S, S), float("-inf"), device="cuda", dtype=torch.bfloat16).triu(1)
causal = causal.view(1, 1, S, S)
with torch.no_grad():
    try:
        r = ref(hidden_states=x, position_ids=pos, attention_mask=causal)
    except TypeError:
        r = ref(hidden_states=x, attention_mask=causal, position_ids=pos, past_key_value=None)
    r = r[0] if isinstance(r, tuple) else r
    o_te = ours(x, backend="te")
    o_sdpa = ours(x, backend="sdpa")
b = r.float()
print(f"\nANCHORED MLA PARITY vs the release's own module, real layer-3 weights, seq {S}:")
for label, got in (("te   (new default)", o_te.float()), ("sdpa (release workaround)", o_sdpa.float())):
    print(f"  {label:26} rel-L2 {(got-b).norm()/b.norm():.4e} | max-abs {(got-b).abs().max():.3e} | "
          f"cosine {torch.nn.functional.cosine_similarity(got.flatten(), b.flatten(), dim=0):.6f}")
print(f"  std: release {b.std():.6f} | te {o_te.float().std():.6f} | sdpa {o_sdpa.float().std():.6f}")
