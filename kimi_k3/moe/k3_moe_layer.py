"""Kimi K3's MoE layer: core's LatentMoE, plus the one norm it is missing.

Core supports the latent projections K3 needs (`moe_latent_size`, giving
`fc1_latent_proj` 7168->3584 and `fc2_latent_proj` 3584->7168), and it runs the
router on the full-width hidden state before the down-projection, exactly as
`KimiSparseMoeBlock.forward` does.

What it does not have is `routed_expert_norm`: the release applies
`RMSNorm(3584, eps=rms_norm_eps)` to the **combined expert output** before the
up-projection, and core's `MoELayer.postprocess` goes straight from combine to
`fc2_latent_proj` (moe_layer.py:505-515). That is review finding A10, and this
override is the whole of the fix.

Shared experts stay at the hidden width (7168, intermediate 2 x 3072 = 6144) --
only the *routed* experts run at the latent width (finding B6).
"""

from typing import Optional

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer


class K3MoELayer(MoELayer):
    """LatentMoE with the release's norm before the up-projection."""

    def __init__(self, config, submodules=None, layer_number: Optional[int] = None, **kwargs):
        super().__init__(config, submodules=submodules, layer_number=layer_number, **kwargs)
        self.latent_norm_eps = config.layernorm_epsilon
        if config.k3_latent_moe_use_norm and config.moe_latent_size:
            self.routed_expert_norm = torch.nn.Parameter(torch.ones(config.moe_latent_size))
        else:
            self.routed_expert_norm = None

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combine, **normalise**, up-project, then add the shared experts."""
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            if self.routed_expert_norm is not None:
                output = self._latent_norm(output)
            output, _ = self.fc2_latent_proj(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def _latent_norm(self, x: torch.Tensor) -> torch.Tensor:
        from ..numerics import to_hi

        xf = to_hi(x)
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.latent_norm_eps)
        return (self.routed_expert_norm.to(xf.dtype) * normed).to(x.dtype)
