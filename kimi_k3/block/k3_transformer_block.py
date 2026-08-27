"""K3 decoder block — owns the attention-residual state.

It carries `(prefix_sum, block_residual)` through the layer loop, packs the pair
into a single tensor at a pipeline stage boundary and unpacks it on the far side
— exactly as `attn_res_pp` specifies — and applies the model-level output mix on
the last stage.

The mixes themselves live in `K3TransformerLayer`, placed around the attention
and MLP halves the way the release places them: the block owns the *state*, the
layer owns the *math*.

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
        # Per-layer mixes belong to K3TransformerLayer; only the model-level mix
        # is the block's, and it lives on the last stage.
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
        """Apply the model-level mix on ``[S, B, H]`` against ``[S, B, K, H]``."""
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

        recompute = (
            self.training
            and self.config.recompute_granularity == "full"
            and torch.is_grad_enabled()
        )
        for layer in self.layers:
            if self._detach_slots_for_test and slots.shape[2] > 0:
                # Emulates the failure mode of a two-tensor payload: the slots
                # travel forward but their gradient never comes back (gate G20).
                slots = slots.detach()
            if recompute:
                prefix, slots = self._checkpointed_layer(layer, prefix, slots, attention_mask, kwargs)
            else:
                prefix, slots = layer(prefix, attention_mask, block_residual=slots, **kwargs)

        if not self.post_process:
            return pack(prefix, slots)

        hidden = self._mix(self.output_attn_res, prefix, slots)
        if self.final_layernorm is not None:
            hidden = self.final_layernorm(hidden)
        return hidden

    def _checkpointed_layer(self, layer, prefix, slots, attention_mask, kwargs):
        """Recompute one layer, carrying **both** state tensors.

        Core's own `_checkpointed_forward` assumes a single hidden-state tensor,
        so it cannot be reused: the block residual would be dropped from the
        saved inputs and silently recomputed from the wrong thing. AttnRes is
        recompute-mandatory at production width (the mixer's fp32 stacks would
        cost ~236 GB per microbatch to keep -- see develop/results/attn_res.md),
        so this path is not optional.
        """
        from megatron.core import tensor_parallel

        def run(prefix_in, slots_in):
            return layer(prefix_in, attention_mask, block_residual=slots_in, **kwargs)

        return tensor_parallel.checkpoint(run, False, prefix, slots)

    # --- introspection used by the probes and gates -------------------------

    def payload_multipliers(self) -> Tuple[int, int]:
        return 1 + self.slots_on_entry(), 1 + self.slots_on_exit()
