"""K3 decoder block — owns the attention-residual state.

Scope note: this is the **transport-faithful, layer-approximate** version built
for gate G7. It carries `(prefix_sum, block_residual)` through the layer loop,
appends a slot at every block boundary, packs the pair into a single tensor at a
pipeline stage boundary and unpacks it on the far side — all exactly as the
release and as `attn_res_pp` specify.

What it does *not* yet do is split the two mixes around the attention and the MLP
halves of each layer: it wraps a stock `TransformerLayer` and mixes before and
after it instead. P5 (`K3TransformerLayer`) makes the placement faithful. The
distinction does not affect anything G7 tests — payload shapes, packing, and
gradient flow are identical either way — but it does mean **this block is not yet
numerically K3**, and no parity gate may cite it.

`self.layers` keeps its inherited name: core's `optimizer/qk_clip.py:clip_qk`
walks `decoder.layers` and skips modules without a `clip_qk` attribute.
"""

from typing import Optional, Tuple

import torch

from megatron.core.transformer.transformer_block import TransformerBlock

from .attn_res import AttnResMixer
from .attn_res_pp import pack, slots_before, unpack


class K3TransformerBlock(TransformerBlock):
    """TransformerBlock carrying the Kimi K3 attention-residual state."""

    #: Marker asserted by the block-injection test (G4).
    k3_block = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        hidden = cfg.hidden_size
        eps = cfg.layernorm_epsilon
        fp32 = getattr(cfg, "k3_attn_res_fp32", True)
        n = len(self.layers)
        self.attn_res_attn = torch.nn.ModuleList(
            [AttnResMixer(hidden, eps, fp32) for _ in range(n)]
        )
        self.attn_res_mlp = torch.nn.ModuleList(
            [AttnResMixer(hidden, eps, fp32) for _ in range(n)]
        )
        # The model-level mix lives on the last stage only.
        self.output_attn_res = AttnResMixer(hidden, eps, fp32) if self.post_process else None
        self._detach_slots_for_test = False

    # --- state helpers ------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self.config.k3_attn_res_block_size

    def global_layer_index(self, local_index: int) -> int:
        """0-indexed global layer number (``layer_number`` is 1-indexed and global)."""
        return self.layers[local_index].layer_number - 1

    def slots_on_entry(self) -> int:
        return slots_before(self.global_layer_index(0), self.block_size)

    def slots_on_exit(self) -> int:
        return slots_before(self.global_layer_index(len(self.layers) - 1) + 1, self.block_size)

    def _mix(self, mixer, prefix: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Apply one mix on ``[S, B, H]`` state against ``[S, B, K, H]`` slots."""
        if slots.shape[2] == 0:
            return prefix
        s, b, h = prefix.shape
        out = mixer(prefix.reshape(s * b, h), slots.reshape(s * b, slots.shape[2], h))
        return out.view(s, b, h)

    # --- pipeline plumbing --------------------------------------------------

    def set_input_tensor(self, input_tensor: torch.Tensor):
        """Receive the packed payload from the previous stage."""
        self.input_tensor = input_tensor

    def _initial_state(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_process or self.input_tensor is None:
            s, b, h = hidden_states.shape
            return hidden_states, hidden_states.new_zeros(s, b, 0, h)
        s = self.input_tensor.shape[0] // (1 + self.slots_on_entry())
        return unpack(self.input_tensor, s, self.slots_on_entry())

    # --- forward ------------------------------------------------------------

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        prefix, slots = self._initial_state(hidden_states)

        for i, layer in enumerate(self.layers):
            layer_idx = self.global_layer_index(i)

            mixed = self._mix(self.attn_res_attn[i], prefix, slots)

            if layer_idx % self.block_size == 0:
                new_slot = prefix.unsqueeze(2)
                if self._detach_slots_for_test:
                    # Emulates the failure mode of a two-tensor payload: the slot
                    # travels forward but its gradient never comes back.
                    new_slot = new_slot.detach()
                slots = torch.cat([slots, new_slot], dim=2)
                prefix = None

            out = layer(mixed, attention_mask, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            prefix = out if prefix is None else prefix + out
            prefix = self._mix(self.attn_res_mlp[i], prefix, slots)

        if not self.post_process:
            return pack(prefix, slots)

        hidden = self._mix(self.output_attn_res, prefix, slots)
        if self.final_layernorm is not None:
            hidden = self.final_layernorm(hidden)
        return hidden

    # --- introspection used by the probes and gates -------------------------

    def payload_multipliers(self) -> Tuple[int, int]:
        return 1 + self.slots_on_entry(), 1 + self.slots_on_exit()
