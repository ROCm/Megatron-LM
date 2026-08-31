"""Anchored parity for the remaining kinds: KDA at layers 1-2, and the AttnRes mixer."""
import os, sys, json, glob
sys.path.insert(0, "/tmp/tf4562")
REF = "/tmp/claude-0/-workspace/edf3a927-9465-4cd8-beed-9a2f60edf2a5/scratchpad/k3ref"
sys.path.insert(0, REF)
for v in ("NVTE_FLASH_ATTN","NVTE_FUSED_ATTN","NVTE_UNFUSED_ATTN"): os.environ.pop(v, None)
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","30063")
import torch
from safetensors import safe_open

def load(prefix):
    out = {}
    for f in sorted(glob.glob("/tmp/k3w/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if k.startswith(prefix): out[k[len(prefix):]] = fh.get_tensor(k)
    return out

d = dict(json.load(open(f"{REF}/config.json"))["text_config"])
d.setdefault("_attn_implementation", "eager"); d.setdefault("attention_dropout", 0.0)
cfg = type("Cfg", (), d)()
from k3pkg.modeling_kimi_linear import KimiDeltaAttention, KimiRMSNorm

from megatron.core import parallel_state, tensor_parallel
torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
torch.cuda.set_device(0); parallel_state.initialize_model_parallel(1,1)
tensor_parallel.model_parallel_cuda_manual_seed(1234)
from kimi_k3.attention.kda import KimiDeltaAttention as OurKDA
from kimi_k3.config.k3_config_builder import config_from_preset
from kimi_k3.config.presets import preset
from kimi_k3.tools.convert import trim_a_log
K3 = config_from_preset(preset("93L")["config"])

def stats(a, b):
    a, b = a.float(), b.float()
    return (f"rel-L2 {(a-b).norm()/b.norm():.4e} | max-abs {(a-b).abs().max():.3e} | "
            f"cosine {torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0):.6f}")

S = 128
torch.manual_seed(1)
x = (torch.randn(1, S, 7168, device="cuda", dtype=torch.bfloat16) * 0.02)

for L in (1, 2):
    w = load(f"language_model.model.layers.{L}.self_attn.")
    ref = KimiDeltaAttention(cfg, layer_idx=L).cuda().bfloat16().eval()
    w["A_log"] = trim_a_log(w["A_log"])
    m1, u1 = ref.load_state_dict({k: v.cuda() for k, v in w.items()}, strict=False)
    ours = OurKDA(config_from_preset(preset("93L")["config"], k3_kda_backend="fla")).cuda().bfloat16().eval()
    m2, u2 = ours.load_state_dict({
        "q_proj.weight": w["q_proj.weight"].cuda(), "k_proj.weight": w["k_proj.weight"].cuda(),
        "v_proj.weight": w["v_proj.weight"].cuda(), "o_proj.weight": w["o_proj.weight"].cuda(),
        "q_conv1d_weight": w["q_conv1d.weight"].cuda(), "k_conv1d_weight": w["k_conv1d.weight"].cuda(),
        "v_conv1d_weight": w["v_conv1d.weight"].cuda(),
        "f_a_proj.weight": w["f_a_proj.weight"].cuda(), "f_b_proj.weight": w["f_b_proj.weight"].cuda(),
        "g_proj.weight": w["g_proj.weight"].cuda(), "b_proj.weight": w["b_proj.weight"].cuda(),
        "A_log": w["A_log"].cuda(), "dt_bias": w["dt_bias"].cuda(),
        "o_norm_weight": w["o_norm.weight"].cuda(),
    }, strict=False)
    assert not u1 and not u2, (u1, u2)
    with torch.no_grad():
        r = ref(hidden_states=x); r = r[0] if isinstance(r, tuple) else r
        o, _ = ours(x)
    print(f"KDA layer {L}: params {sum(p.numel() for p in ref.parameters())} vs {sum(p.numel() for p in ours.parameters())} | {stats(o, r)}", flush=True)
    del ref, ours; torch.cuda.empty_cache()

# --- AttnRes mixer, layer 0's two sites, against the release's own formula ---
from kimi_k3.block.attn_res import attn_res_mix
res = load("language_model.model.layers.0.")
for site, nk, pk in (("attention", "self_attention_res_norm.weight", "self_attention_res_proj.weight"),
                     ("mlp", "mlp_res_norm.weight", "mlp_res_proj.weight")):
    nw, pw = res[nk].cuda().float(), res[pk].cuda().float()
    torch.manual_seed(3)
    prefix = torch.randn(S, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
    slots = torch.randn(S, 1, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
    ours_out = attn_res_mix(prefix, slots, nw, pw, K3.layernorm_epsilon)
    # the release's formula, transcribed inline from KimiLinearModel._apply_attn_res
    v = torch.cat((slots, prefix.unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + K3.layernorm_epsilon)
    scores = (k * (nw * pw.squeeze(0))).sum(-1)
    ref_out = torch.matmul(scores.softmax(-1).unsqueeze(1), v).squeeze(1).to(prefix.dtype)
    print(f"AttnRes {site:9} site, real layer-0 norm+proj: {stats(ours_out, ref_out)}", flush=True)
