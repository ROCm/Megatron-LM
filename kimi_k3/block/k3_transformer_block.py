"""K3 decoder block.

P0 skeleton: it exists so the block-injection mechanism can be proven end to end
(gate G4) before any AttnRes code lands. P5 fills in the real state handling:
the block owns ``prefix_sum`` and ``block_residual``, unpacks the payload from
the previous pipeline stage, and packs it again on the way out
(see kimi_k3/develop/plan-0/02-target-architecture.md §4).
"""

from megatron.core.transformer.transformer_block import TransformerBlock


class K3TransformerBlock(TransformerBlock):
    """TransformerBlock carrying the Kimi K3 attention-residual state.

    ``self.layers`` is inherited unchanged and must keep that name: core's
    ``optimizer/qk_clip.py:clip_qk`` walks ``decoder.layers`` and skips modules
    without a ``clip_qk`` attribute.
    """

    #: Set by P5. Present now so tests can assert the injection reached us.
    k3_block = True

    def attn_res_slots_before(self, layer_idx: int) -> int:
        """Slots visible on entry to a 0-indexed global layer index."""
        return self.config.attn_res_slots_before(layer_idx)
