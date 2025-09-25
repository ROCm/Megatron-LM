from unittest import mock

import pytest

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestRoPEFusionCompatibility:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            add_bias_linear=False,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        yield
        Utils.destroy_model_parallel()

    def test_rope_fusion_with_unavailable_flag(self):
        """Test error handling when RoPE fusion is unavailable."""
        with mock.patch(
            'megatron.core.models.common.embeddings.rope_utils.HAVE_APPLY_ROPE_FUSION', False
        ):
            # Should raise error when trying to enable
            with pytest.raises(ValueError, match="apply_rope_fusion is not available"):
                TransformerConfig(
                    num_layers=2,
                    hidden_size=128,
                    num_attention_heads=4,
                    use_cpu_initialization=True,
                    apply_rope_fusion=True,
                )

    def test_rope_fusion_with_available_flag(self):
        """Test that RoPE fusion can be enabled when available."""
        with mock.patch(
            'megatron.core.models.common.embeddings.rope_utils.HAVE_APPLY_ROPE_FUSION', True
        ):
            config = TransformerConfig(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=4,
                use_cpu_initialization=True,
                apply_rope_fusion=True,
            )
            assert config.apply_rope_fusion is True

    def test_compatibility_layer_with_new_api(self):
        """Test that the compatibility fix works with new TE API."""
        import transformer_engine.pytorch.attention as te_attn

        # Verify expected state: new API present, old API absent
        if hasattr(te_attn, 'apply_rotary_pos_emb') and not hasattr(te_attn, 'FusedRoPEFunc'):
            import megatron.core.extensions.transformer_engine as te_ext
            from megatron.core.models.common.embeddings.rope_utils import HAVE_APPLY_ROPE_FUSION

            # Compatibility layer should provide the functions
            assert hasattr(te_ext, 'fused_apply_rotary_pos_emb')
            assert te_ext.fused_apply_rotary_pos_emb is not None
            assert HAVE_APPLY_ROPE_FUSION is True
        else:
            pytest.skip("Test requires new TE API without old API")

    def test_rope_without_fusion(self):
        """Test that RoPE works without fusion."""
        from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

        assert apply_rotary_pos_emb is not None

        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            apply_rope_fusion=False,
        )
        assert config.apply_rope_fusion is False
