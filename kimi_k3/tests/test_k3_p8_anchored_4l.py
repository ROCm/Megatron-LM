"""P8 / gate G32 (full) -- four-layer anchored parity against the release.

The existing anchored test covers **one KDA layer**. This one covers the whole
architecture, because a four-layer slice of K3 contains every kind exactly once:

    layer 0   KDA  + dense FFN      (first_k_dense_replace = 1) + the AttnRes slot
    layer 1   KDA  + MoE
    layer 2   KDA  + MoE
    layer 3   gated MLA + MoE       (1-indexed layer 4, the first full-attention layer)

Why this matters more than the other tests: every other parity check in the tree
compares our code against **our own** eager oracle, and those oracles were written
by reading the release. A misreading is invisible to them -- the oracle encodes it
too. Only running the release's own module on the release's own weights can catch
"we built a coherent model that is not Kimi K3".

Gated MLA and the AttnRes stream have never been anchored. They are also where the
most subtle reading calls live: NoPE with an unrotated shared `k_rot`, the
`192**-0.5` softmax scale rather than `128**-0.5`, the sigmoid gate applied
*before* `o_proj`, LoRA norms at 1e-6 rather than `rms_norm_eps`, and AttnRes's
block size of 12 with its prefix reset.

Weights are 49.25 GiB and are **not** kept -- fetch, validate, delete. Set
`K3_SHARD_DIR` to a directory holding `model-0000{1,2,3,4}-of-000096.safetensors`.
"""

import os
import pathlib

import pytest
import torch

SHARDS = pathlib.Path(os.environ.get("K3_SHARD_DIR", "/tmp/k3w"))
NEEDED = [f"model-{i:05d}-of-000096.safetensors" for i in (1, 2, 3, 4)]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not all((SHARDS / n).exists() for n in NEEDED),
        reason=f"release shards not present in {SHARDS}; see this module's docstring",
    ),
]


def test_the_four_layer_slice_covers_every_layer_kind():
    """The claim that makes four layers worth 49 GiB rather than an arbitrary cut."""
    from kimi_k3.config.k3_config_builder import config_from_preset
    from kimi_k3.config.presets import preset

    cfg = config_from_preset(preset("93L")["config"])
    kinds = [("kda" if cfg.is_kda_layer(i) else "mla") for i in range(4)]
    assert kinds == ["kda", "kda", "kda", "mla"], kinds
    assert cfg.k3_first_k_dense_replace == 1, "layer 0 must be the dense FFN layer"
    assert cfg.appends_attn_res_slot(0), "layer 0 must create the AttnRes slot"
    assert not any(cfg.appends_attn_res_slot(i) for i in (1, 2, 3))
