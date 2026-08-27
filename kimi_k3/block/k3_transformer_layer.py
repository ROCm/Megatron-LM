"""The Kimi K3 decoder layer.

`TransformerLayer`'s own forward is a pre-norm residual stream: normalise,
sublayer, add back. K3 replaces *both* of those adds with an attention-residual
mix, so this overrides `forward` rather than hooking into it. Everything the
layer is built from -- `input_layernorm`, `self_attention`, `pre_mlp_layernorm`,
`mlp` -- is core's, constructed by core's spec machinery.

The sequence is `KimiDecoderLayer._forward_attn_residual`, verbatim:

    prefix_sum = hidden
    if slots:            hidden = mix_attn(prefix_sum, slots)
    if block boundary:   slots = cat(slots, prefix_sum); prefix_sum = None
    hidden = input_layernorm(hidden)
    hidden = self_attention(hidden)
    prefix_sum = hidden if prefix_sum is None else prefix_sum + hidden
    hidden = mix_mlp(prefix_sum, slots)
    hidden = pre_mlp_layernorm(hidden)
    hidden = mlp(hidden)
    prefix_sum = prefix_sum + hidden
    return prefix_sum, slots

Two details that look like details and are not:

* **the prefix sum is reset at a block boundary** (`prefix_sum = None`), so the
  residual stream restarts each block and the old stream is frozen into a slot.
  A layer that appends a slot therefore starts its own stream from its attention
  output, not from its input.
* **the slot is the prefix sum *before* this layer's attention**, captured after
  the attention-site mix has already read the older slots.
"""

from typing import Optional, Tuple

import torch

from megatron.core.transformer.transformer_layer import TransformerLayer

from .attn_res import AttnResMixer


class K3TransformerLayer(TransformerLayer):
    """A decoder layer whose residual stream is the AttnRes stream."""

    def __init__(self, config, submodules, layer_number: int = 1, **kwargs):
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)
        eps = config.layernorm_epsilon
        fp32 = getattr(config, "k3_attn_res_fp32", True)
        fused = getattr(config, "k3_attn_res_fused", False)
        chunk = getattr(config, "k3_attn_res_chunk", 4096)
        self.attn_res_attn = AttnResMixer(config.hidden_size, eps, fp32, fused, chunk)
        self.attn_res_mlp = AttnResMixer(config.hidden_size, eps, fp32, fused, chunk)
        self.block_size = config.k3_attn_res_block_size

    @property
    def global_layer_index(self) -> int:
        """0-indexed; `layer_number` is 1-indexed and already global."""
        return self.layer_number - 1

    @property
    def appends_slot(self) -> bool:
        return self.global_layer_index % self.block_size == 0

    def _mix(self, mixer: AttnResMixer, prefix: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Apply one mix to `[s, b, h]` state against `[s, b, k, h]` slots."""
        if slots.shape[2] == 0:
            return prefix
        s, b, h = prefix.shape
        out = mixer(prefix.reshape(s * b, h), slots.reshape(s * b, slots.shape[2], h))
        return out.view(s, b, h)

    def _sublayer_output(self, out) -> torch.Tensor:
        """Core sublayers return either a tensor or an `(output, bias)` pair."""
        if isinstance(out, tuple):
            output, bias = out
            return output if bias is None else output + bias
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        *,
        block_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s, b, h = hidden_states.shape
        slots = (
            block_residual
            if block_residual is not None
            else hidden_states.new_zeros(s, b, 0, h)
        )

        prefix_sum: Optional[torch.Tensor] = hidden_states
        mixed = self._mix(self.attn_res_attn, prefix_sum, slots)

        if self.appends_slot:
            slots = torch.cat([slots, prefix_sum.unsqueeze(2)], dim=2)
            prefix_sum = None  # the stream restarts for this block

        attn_out = self._sublayer_output(
            self.self_attention(self.input_layernorm(mixed), attention_mask=attention_mask)
        )
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        mixed = self._mix(self.attn_res_mlp, prefix_sum, slots)
        mlp_out = self._sublayer_output(self.mlp(self.pre_mlp_layernorm(mixed)))
        prefix_sum = prefix_sum + mlp_out

        return prefix_sum, slots
