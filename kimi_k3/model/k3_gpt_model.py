"""K3 top-level model.

``K3GPTModel`` reuses ``GPTModel.__init__`` wholesale and swaps only the decoder
class, via the scoped rebinding in ``core_patch``. The alternative -- copying the
~190 lines of ``GPTModel.__init__`` (embedding, four rotary variants, MTP, output
layer, weight tying, offloading) -- was rejected as a permanent IFU liability;
see develop/plan-0/00-review-findings.md finding A5.
"""

from megatron.core.models.gpt.gpt_model import GPTModel

from ..block.k3_transformer_block import K3TransformerBlock
from .core_patch import k3_block_class


class K3GPTModel(GPTModel):
    """GPTModel whose decoder is a K3TransformerBlock."""

    def __init__(self, config, transformer_layer_spec, *args, **kwargs):
        with k3_block_class(K3TransformerBlock):
            super().__init__(config, transformer_layer_spec, *args, **kwargs)
        assert isinstance(self.decoder, K3TransformerBlock), (
            "block injection did not take effect -- check core_patch pin contracts"
        )

    def set_input_tensor(self, input_tensor):
        """Accept the packed AttnRes payload from the previous pipeline stage.

        Core sends and receives a list of tensors; K3 keeps the list length at 1
        because ``backward_step`` back-props only ``output_tensor[0]``, so the
        payload is a single packed tensor rather than several
        (develop/notes/2026-08-26-attn-res-pp-transport.md).
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, (
            f"K3 expects one packed payload tensor, got {len(input_tensor)}"
        )
        self.decoder.set_input_tensor(input_tensor[0])
