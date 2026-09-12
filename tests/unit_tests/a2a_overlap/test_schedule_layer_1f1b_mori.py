# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dedicated MORI coverage for the transformer/MTP layer 1F1B overlap schedule.

Extracted from ``test_schedule_layer_1f1b.py`` so MORI runs in its own fresh
``torchrun`` process, never interleaved with the sub-node (EP=4) alltoall/DeepEP/
ncclep cases. MORI's ``EpDispatchCombineHandle`` needs a node-spanning expert
communicator, and its symmetric memory is process-scoped (init once, finalize once
at session end via ``conftest.py``); mixing it with the sub-node parametrized cases
corrupts the shared process-group state and desyncs collectives.
"""
from contextlib import nullcontext

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_layer_1f1b import (
    run_mtp_layer_a2a_overlap_with_capture,
    run_mtp_layer_ref_with_capture,
    run_transformer_layer_a2a_overlap_with_capture,
    run_transformer_layer_ref_with_capture,
)
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    build_data,
    compare_captures,
    deterministic_mode,
    get_compare_tolerances,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    reinitialize_model_parallel_for_mori,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


class TestA2AOverlapLayerMori:
    """Run process-scoped MORI layer/MTP coverage in a fresh torchrun invocation."""

    def teardown_method(self, method):
        # Drop the per-case op; keep shmem alive. MORI shmem is process-scoped and
        # cannot be finalized then reinitialized, so finalize is owned solely by the
        # session-scoped conftest fixture (finalize once at session end).
        from megatron.core.transformer.moe.fused_a2a import reset_mori_op

        reset_mori_op()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.skipif(
        get_valid_flex_dispatcher_backend() != "mori", reason="MORI is not available"
    )
    def test_transformer_layer_overlap_mori(self):
        """MORI variant of ``test_transformer_layer_overlap``.

        Re-initializes a node-spanning expert-parallel layout that it fully owns; MORI's
        ``EpDispatchCombineHandle`` requires the expert (ETPxEP) communicator to span whole
        nodes. Skipped when the node's GPU count is not a power of two.
        """
        ep_size = reinitialize_model_parallel_for_mori()
        extra_kwargs = apply_flex_backend_kwargs({}, "flex", "mori")
        extra_kwargs["expert_model_parallel_size"] = ep_size
        config = get_test_config(extra_kwargs=extra_kwargs)
        atol, rtol = get_compare_tolerances("mori")
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, 0) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(
                capture_ref, capture_a2a_overlap, True, atol=atol, rtol=rtol
            )
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.skipif(
        get_valid_flex_dispatcher_backend() != "mori", reason="MORI is not available"
    )
    def test_mtp_layer_overlap_mori(self):
        """MORI variant of ``test_mtp_layer_overlap``. Owns its full-node re-initialization for
        the same node-spanning-expert-communicator reason as the transformer-layer variant."""
        ep_size = reinitialize_model_parallel_for_mori()
        extra_kwargs = apply_flex_backend_kwargs({}, "flex", "mori")
        extra_kwargs["expert_model_parallel_size"] = ep_size
        extra_kwargs["mtp_num_layers"] = 1
        extra_kwargs["mtp_loss_scaling_factor"] = 1.1
        # Config qk_layernorm must match the spec's qk_layernorm=True below, else MLA
        # self-attention rejects the layernorm'd q_up_proj. The non-MORI
        # test_mtp_layer_overlap sets this; the original MORI copy omitted it but never
        # ran (its base file is flaky_in_dev / deselected), so the mismatch was latent.
        extra_kwargs["qk_layernorm"] = True
        config = get_test_config(extra_kwargs=extra_kwargs)
        atol, rtol = get_compare_tolerances("mori")
        microbatches = 1
        seq_len = 32
        with deterministic_mode():
            # init models
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=16,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                multi_latent_attention=True,
            )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, True)
            if mtp_block_spec is None:
                # only last rank has mtp block
                assert True
                return
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                mtp_block_spec=mtp_block_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            gpt_model.decoder.final_layernorm = None
            gpt_model.cuda()
            params = reset_model(gpt_model)

            # build input data
            data = list(range(seq_len))
            hidden_states = [build_data(seq_len) for _ in range(microbatches)]
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            labels = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            position_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda()
            # get rotary pos emb
            _, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, _, _padding_mask = (
                gpt_model._preprocess(input_ids, position_ids)
            )
            # reset model
            params = reset_model(gpt_model)

            # run reference implementation
            capture_ref = run_mtp_layer_ref_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_mtp_layer_a2a_overlap_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            comp_res = compare_captures(
                capture_ref, capture_a2a_overlap, True, True, atol=atol, rtol=rtol
            )
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
