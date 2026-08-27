"""Fetch named tensors from the released checkpoint without downloading a shard.

A safetensors file is a JSON header followed by packed tensor data, so a single
range request gets the header and one more gets any tensor in it. Layer 0's whole
KDA block is 847 MiB that way, against ~16 GiB for the shard -- which is what
makes an anchored parity check affordable at all, and keeps it inside the
"never download >10 GB without asking" rule.

    python -m kimi_k3.tools.fetch_release_tensors \\
        --match layers.0.self_attn --out /tmp/layer0_self_attn.pt
"""

import argparse
import json
import struct
import subprocess
from typing import Dict, List

import torch

BASE_URL = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "U8": torch.uint8}


def _range(url: str, start: int, end: int, timeout: int = 900) -> bytes:
    out = subprocess.run(
        ["curl", "-sSL", "-r", f"{start}-{end}", url], capture_output=True, timeout=timeout
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode()[:200])
    return out.stdout


def read_header(shard: str) -> Dict:
    url = BASE_URL + shard
    size = struct.unpack("<Q", _range(url, 0, 7))[0]
    return json.loads(_range(url, 8, 8 + size - 1)), 8 + size


def fetch(shard: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    import numpy as np

    header, base = read_header(shard)
    url = BASE_URL + shard
    out = {}
    for key in keys:
        meta = header[key]
        start, end = meta["data_offsets"]
        raw = _range(url, base + start, base + end - 1)
        buf = np.frombuffer(raw, dtype=np.uint8).copy()
        out[key] = torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).view(*meta["shape"]).clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="model-00001-of-000096.safetensors")
    ap.add_argument("--match", required=True, help="substring the tensor name must contain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gib", type=float, default=2.0,
                    help="refuse to fetch more than this, so a typo cannot pull a whole shard")
    args = ap.parse_args()

    header, _ = read_header(args.shard)
    keys = [k for k in header if k != "__metadata__" and args.match in k]
    total = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in keys)
    print(f"{len(keys)} tensors, {total / 2**30:.2f} GiB")
    if total / 2**30 > args.max_gib:
        raise SystemExit(f"refusing: {total / 2**30:.2f} GiB exceeds --max-gib {args.max_gib}")
    torch.save(fetch(args.shard, sorted(keys)), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
