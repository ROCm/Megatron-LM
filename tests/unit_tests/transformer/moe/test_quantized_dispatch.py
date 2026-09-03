# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for TE recipe mapping and quantized MORI dispatch helpers."""

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import fused_permute_with_probs, permute
from megatron.core.transformer.moe.quantized_dispatch import (
    dispatch_block_size,
    is_quantized_tensor,
    make_dispatch_quantizer,
    mori_scale_dim,
    quantize_hidden_for_dispatch,
    should_quantize_dispatch,
    unpack_rowwise,
    view_payload_for_mori,
    wrap_rowwise,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _te_recipes():
    pytest.importorskip("transformer_engine")
    from transformer_engine.common import recipe as te_recipe

    return te_recipe


def _mori_config(**kwargs):
    defaults = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="mori",
        moe_mori_max_tokens_per_rank=16,
        num_moe_experts=8,
        moe_quantized_dispatch=True,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


class TestScaleDimMapping:
    def test_tensorwise_and_delayed_are_per_token(self):
        te_recipe = _te_recipes()
        hidden = 4096
        assert mori_scale_dim(te_recipe.Float8CurrentScaling(), hidden) == 1
        assert mori_scale_dim(te_recipe.DelayedScaling(), hidden) == 1
        assert dispatch_block_size(te_recipe.Float8CurrentScaling()) == 1

    def test_blockwise_hidden_over_128(self):
        te_recipe = _te_recipes()
        hidden = 4096
        assert mori_scale_dim(te_recipe.Float8BlockScaling(), hidden) == hidden // 128
        assert dispatch_block_size(te_recipe.Float8BlockScaling()) == 128

    def test_mxfp8_hidden_over_32(self):
        te_recipe = _te_recipes()
        hidden = 4096
        assert mori_scale_dim(te_recipe.MXFP8BlockScaling(), hidden) == hidden // 32
        assert dispatch_block_size(te_recipe.MXFP8BlockScaling()) == 32

    def test_blockwise_rejects_unaligned_hidden(self):
        te_recipe = _te_recipes()
        with pytest.raises(ValueError, match="divisible"):
            mori_scale_dim(te_recipe.Float8BlockScaling(), 100)

    def test_config_rejects_unaligned_hidden_for_mori_fp8(self):
        with pytest.raises(ValueError, match="divisible"):
            _mori_config(hidden_size=100, fp8="e4m3", fp8_recipe="blockwise")

    def test_config_rejects_fp4_quantized_dispatch(self):
        with pytest.raises(ValueError, match="does not support FP4"):
            _mori_config(fp4="e2m1", moe_quantized_dispatch=True)

    def test_should_quantize_dispatch_requires_autocast(self):
        cfg = _mori_config(fp8="e4m3", fp8_recipe="blockwise")
        assert should_quantize_dispatch(cfg) is False

    def test_dual_dtype_cache_keys_differ(self):
        # FP8 dispatch (scale_dim>0) and BF16 combine (scale_dim=0) must not share an op.
        hidden, experts, topk, max_tok = 4096, 8, 8, 16
        fp8_key = (hidden, experts, topk, max_tok, hidden // 128, 4, "InterNodeV1")
        bf16_key = (hidden, experts, topk, max_tok, 0, 1, "InterNodeV1")
        assert fp8_key != bf16_key


class TestFirstLastBf16Layers:
    def test_first_last_context_disables_quantized_dispatch(self):
        pytest.importorskip("transformer_engine")
        from megatron.core.fp8_utils import get_fp8_context, is_first_last_bf16_layer

        cfg = _mori_config(
            num_layers=4,
            fp8="e4m3",
            fp8_recipe="blockwise",
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
        )
        assert is_first_last_bf16_layer(cfg, 0)
        with get_fp8_context(cfg, 0):
            assert should_quantize_dispatch(cfg) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestQuantizeWrapRoundtrip:
    def test_blockwise_quantize_unpack_wrap(self):
        te_recipe = _te_recipes()
        recipe = te_recipe.Float8BlockScaling()
        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        assert data.dtype == torch.uint8
        mori_payload = view_payload_for_mori(data)
        assert mori_payload.dtype != torch.uint8
        assert data.shape[0] == 8
        assert scales.shape[0] == 8
        assert scales.shape[1] == mori_scale_dim(recipe, 256)
        wrapped = wrap_rowwise(mori_payload, scales, meta)
        assert is_quantized_tensor(wrapped)
        recon = wrapped.dequantize().to(torch.bfloat16)
        q2 = make_dispatch_quantizer(recipe, x.device)(x)
        ref = q2.dequantize().to(torch.bfloat16)
        torch.testing.assert_close(recon, ref, rtol=0, atol=0)

    def test_tensorwise_replicate_per_row_scales(self):
        te_recipe = _te_recipes()
        recipe = te_recipe.Float8CurrentScaling()
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        assert scales.shape == (4, 1)
        wrapped = wrap_rowwise(data, scales, meta)
        assert is_quantized_tensor(wrapped)

    def test_delayed_quantize_wrap(self):
        te_recipe = _te_recipes()
        recipe = te_recipe.DelayedScaling()
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        assert scales.shape == (4, 1)
        wrapped = wrap_rowwise(data, scales, meta)
        assert is_quantized_tensor(wrapped)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(fused_permute_with_probs is None, reason="TE fused permute not available")
class TestPermuteQuantizedTokens:
    def test_permute_keeps_row_alignment(self):
        te_recipe = _te_recipes()
        recipe = te_recipe.Float8BlockScaling()
        T, H, E = 6, 256, 2
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        q = wrap_rowwise(data, scales, meta)
        routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda")
        routing_map[:, 0] = True
        routing_map[0, 1] = True
        probs = routing_map.float()
        num_out = int(routing_map.sum().item())
        permuted, _, _, _, _ = permute(
            q, routing_map, probs=probs, num_out_tokens=num_out, fused=True
        )
        assert is_quantized_tensor(permuted)
        _, _, sorted_idx, _, _ = permute(
            data, routing_map, probs=probs, num_out_tokens=num_out, fused=False
        )
        pdata, pscales, _ = unpack_rowwise(permuted)
        assert pdata.shape[0] == num_out
        assert pscales.shape[0] == num_out
        torch.testing.assert_close(pdata, data.index_select(0, sorted_idx), rtol=0, atol=0)
        torch.testing.assert_close(pscales, scales.index_select(0, sorted_idx), rtol=0, atol=0)

    def test_padding_rows_repeat_existing_tokens(self):
        """Router padding gathers extra copies of live tokens, not MORI capacity padding."""
        te_recipe = _te_recipes()
        recipe = te_recipe.Float8BlockScaling()
        T, H, E = 4, 128, 2
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        q = wrap_rowwise(data, scales, meta)
        routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda")
        routing_map[:, 0] = True
        # Extra pad row: duplicate token 0 onto expert 1 (same as pad_routing_map).
        routing_map[0, 1] = True
        probs = routing_map.float()
        num_out = int(routing_map.sum().item())
        permuted, _, _, _, _ = permute(
            q, routing_map, probs=probs, num_out_tokens=num_out, fused=True
        )
        _, _, sorted_idx, _, _ = permute(
            data, routing_map, probs=probs, num_out_tokens=num_out, fused=False
        )
        pdata, pscales, _ = unpack_rowwise(permuted)
        assert pdata.shape[0] == T + 1
        torch.testing.assert_close(pdata, data.index_select(0, sorted_idx), rtol=0, atol=0)
        torch.testing.assert_close(pscales, scales.index_select(0, sorted_idx), rtol=0, atol=0)
        # Duplicate payload row matches the source token, not uninitialized memory.
        torch.testing.assert_close(pdata[-1], data[sorted_idx[-1]], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBf16CombineContract:
    def test_mori_combine_and_dispatch_bwd_force_scale_dim_zero(self):
        import inspect

        from megatron.core.transformer.moe.fused_a2a import MoriCombine, MoriDispatch

        bwd_src = inspect.getsource(MoriDispatch.backward)
        comb_src = inspect.getsource(MoriCombine.forward)
        comb_bwd_src = inspect.getsource(MoriCombine.backward)
        assert "scale_dim=0" in bwd_src
        assert "fp8_dispatch=False" in bwd_src
        assert "scale_dim=0" in comb_src
        assert "scale_dim=0" in comb_bwd_src
        assert "fp8_dispatch=False" in comb_src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGroupedLinearAlreadyQuantized:
    def test_skips_split_quantize(self, monkeypatch):
        pytest.importorskip("transformer_engine")
        import transformer_engine.pytorch.module.grouped_linear as grouped_linear_mod
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Float8BlockScaling

        if fused_permute_with_probs is None:
            pytest.skip("TE fused permute not available")

        calls = {"n": 0}
        orig = grouped_linear_mod.tex.split_quantize

        def counted(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(grouped_linear_mod.tex, "split_quantize", counted)

        recipe = Float8BlockScaling()
        in_features, out_features = 128, 256
        m_splits = [8, 8]
        x = torch.randn(16, in_features, device="cuda", dtype=torch.bfloat16)
        q = make_dispatch_quantizer(recipe, x.device)(x)
        assert is_quantized_tensor(q)

        layer = te.GroupedLinear(
            2,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            y_q = layer(q, m_splits)
            n_after_q = calls["n"]
            y_x = layer(x, m_splits)
        assert n_after_q == 0
        assert calls["n"] >= 1
        assert y_q.shape == y_x.shape
        torch.testing.assert_close(y_q, y_x, rtol=0.25, atol=0.25)

    def test_fused_permute_used_for_quantized_tokens(self):
        if fused_permute_with_probs is None:
            pytest.skip("TE fused permute not available")
        te_recipe = _te_recipes()
        recipe = te_recipe.Float8BlockScaling()
        x = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
        data, scales, meta = quantize_hidden_for_dispatch(x, recipe)
        q = wrap_rowwise(data, scales, meta)
        routing_map = torch.ones(6, 1, dtype=torch.bool, device="cuda")
        permuted, _, _, _, _ = permute(
            q, routing_map, probs=routing_map.float(), num_out_tokens=6, fused=True
        )
        assert is_quantized_tensor(permuted)
        assert permuted.shape[0] == 6

