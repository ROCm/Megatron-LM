"""G54 -- anchored parity for a routed expert, against released weights.

    python -m kimi_k3.tools.anchor_expert --expert 0 --weights /tmp/k3_expert0.pt

G32's anchored parity covered gated MLA, KDA, AttnRes and routing but never the
expert FFN -- `anchor_moe.py` fetches expert weight tensors and compares none of
them, because dequantised experts are ~59 GB per layer and the release block and
ours cannot both be resident. That gap is how G52 (every routed expert running
GeGLU instead of SiTU) survived for five phases.

It is affordable one expert at a time: the released expert is MXFP4, so w1/w3/w2
plus scales is ~17 MiB by ranged read (`tools/fetch_release_tensors.py`). This
runs the **release module** (`KimiBlockSparseMLP` + `SituAndMul`, HF
moonshotai/Kimi-K3) and our activation on the same dequantised weights and the
same input, and compares outputs.
"""

import argparse
import json
import os
import sys

import torch

REF = "/tmp/claude-0/-workspace/edf3a927-9465-4cd8-beed-9a2f60edf2a5/scratchpad/k3ref"
#: The release module asserts transformers >= 4.56.0; the image has 4.55.0, so a
#: newer one is staged here rather than upgrading the environment out from under
#: everything else. Same arrangement as anchor_mla.py / anchor_rest.py.
TF = "/tmp/tf4562"


def load_released_expert(path: str, expert: int):
    """Dequantise the released MXFP4 expert to fp32."""
    from kimi_k3.moe.k3_qat import dequantize_mxfp4

    raw = torch.load(path)
    pre = f"experts.{expert}."
    out = {}
    for w in ("w1", "w3", "w2"):
        packed = next(v for k, v in raw.items() if k.endswith(f"{pre}{w}.weight_packed"))
        scale = next(v for k, v in raw.items() if k.endswith(f"{pre}{w}.weight_scale"))
        out[w] = dequantize_mxfp4(packed.cuda(), scale.cuda()).float()
    return out


def release_module(w, beta: float, linear_beta: float):
    """The released expert, weights loaded, on GPU."""
    sys.path.insert(0, TF)
    sys.path.insert(0, REF)          # k3pkg/ wraps the release files so their
    from k3pkg.modeling_kimi_linear import KimiBlockSparseMLP   # relative imports resolve

    cfg_all = json.load(open(os.path.join(REF, "config.json")))["text_config"]

    class _Cfg:
        hidden_act = cfg_all["hidden_act"]
        activation_situ_beta = beta
        activation_situ_linear_beta = linear_beta
        hidden_size = w["w1"].shape[1]
        intermediate_size = w["w1"].shape[0]

    mod = KimiBlockSparseMLP(_Cfg(), hidden_size=_Cfg.hidden_size,
                             intermediate_size=_Cfg.intermediate_size).cuda().float()
    with torch.no_grad():
        mod.w1.weight.copy_(w["w1"])
        mod.w3.weight.copy_(w["w3"])
        mod.w2.weight.copy_(w["w2"])
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/tmp/k3_expert0.pt")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="input std. SiTU only separates from SwiGLU once |gate| "
                         "reaches beta, so a small value makes this check blind (G53).")
    ap.add_argument("--json")
    args = ap.parse_args()
    for v in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(v, None)

    w = load_released_expert(args.weights, args.expert)
    beta, linear_beta = 4.0, 25.0
    ref = release_module(w, beta, linear_beta)

    from kimi_k3.moe.situ import SituGLU

    torch.manual_seed(0)
    hidden = w["w1"].shape[1]
    x = torch.randn(args.tokens, hidden, device="cuda") * args.scale

    with torch.no_grad():
        want = ref(x)
        gate_up = torch.cat([torch.nn.functional.linear(x, w["w1"]),
                             torch.nn.functional.linear(x, w["w3"])], dim=-1)
        got = torch.nn.functional.linear(SituGLU(None)(gate_up), w["w2"])
        gate = gate_up[..., : gate_up.shape[-1] // 2]

        rel = ((got - want).norm() / want.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten().unsqueeze(0), want.flatten().unsqueeze(0)).item()
        # what the pre-G52 model actually computed, for contrast
        g, u = torch.chunk(gate_up, 2, dim=-1)
        wrong = {
            nm: ((torch.nn.functional.linear(fn(g) * u, w["w2"]) - want).norm()
                 / want.norm()).item()
            for nm, fn in (("geglu", torch.nn.functional.gelu),
                           ("swiglu", torch.nn.functional.silu))
        }

    row = {"expert": args.expert, "tokens": args.tokens, "input_scale": args.scale,
           "gate_abs_max": round(gate.abs().max().item(), 3), "situ_beta": beta,
           "rel_l2": rel, "cosine": cos, "wrong_activation_rel_l2": wrong,
           "discriminating": gate.abs().max().item() > beta}
    print(json.dumps(row, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(row, f, indent=2)


main()
