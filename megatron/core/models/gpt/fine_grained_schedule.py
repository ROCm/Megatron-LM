import contextlib
import weakref
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.pipeline_parallel.combined_1f1b import (
    AbstractSchedulePlan,
    ScheduleNode,
    get_com_stream,
    get_comp_stream,
    make_viewless,
)
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState


def weak_method(method):
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


class PreProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self):

        gpt_model = self.gpt_model
        decoder_input = self.model_chunk_state.decoder_input
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        inference_params = self.model_chunk_state.inference_params
        packed_seq_params = self.model_chunk_state.packed_seq_params

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif gpt_model.pre_process:
            decoder_input = gpt_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = gpt_model.decoder.input_tensor

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if (
            gpt_model.position_embedding_type == 'rope'
            and not gpt_model.config.multi_latent_attention
        ):
            if not gpt_model.training and gpt_model.config.flash_decode and inference_params:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = gpt_model.rotary_pos_emb_cache.setdefault(
                    inference_params.max_sequence_length,
                    gpt_model.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
                )
            else:
                rotary_seq_len = gpt_model.rotary_pos_emb.get_rotary_seq_len(
                    inference_params,
                    gpt_model.decoder,
                    decoder_input,
                    gpt_model.config,
                    packed_seq_params,
                )
                rotary_pos_emb = gpt_model.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        if (
            (gpt_model.config.enable_cuda_graph or gpt_model.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_params
        ):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # saved for later use
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):

        # Final layer norm.
        if self.gpt_model.decoder.final_layernorm is not None:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = transformer_layer.make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        gpt_model = self.gpt_model
        runtime_gather_output = self.model_chunk_state.runtime_gather_output
        labels = self.model_chunk_state.labels
        output_weight = None
        if gpt_model.share_embeddings_and_output_weights:
            output_weight = gpt_model.shared_embedding_or_output_weight()
        logits, _ = gpt_model.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if labels is None:
            # [s b h] => [b s h]
            return float16_to_fp32(logits.transpose(0, 1).contiguous())
        loss = float16_to_fp32(gpt_model.compute_language_model_loss(labels, logits))
        return loss


class TransformerLayerNode(ScheduleNode):

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False):
        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_inputs=free_inputs,
        )
        # layer state
        self.common_state = common_state
        # model chunk state
        self.chunk_state = chunk_state
        self.layer = layer
        self.detached = tuple()
        self.before_detached = tuple()

    def detach(self, t):
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def backward_impl(self, outputs, output_grad):
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self.before_detached = None
        self.detached = None
        # return grads for record stream
        return grads


class MoeAttnNode(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset

        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            (
                hidden_states,
                pre_mlp_layernorm_output,
                tokens_per_expert,
                permutated_local_input_tokens,
                probs,
            ) = self.layer._submodule_attention_router_compound_forward(
                hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        self.common_state.tokens_per_expert = tokens_per_expert

        # detached here
        self.common_state.probs = self.detach(probs)
        self.common_state.residual = self.detach(hidden_states)
        self.common_state.pre_mlp_layernorm_output = self.detach(pre_mlp_layernorm_output)

        return permutated_local_input_tokens

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_router_compound_dw()


class MoeDispatchNode(TransformerLayerNode):

    def forward_impl(self, permutated_local_input_tokens):
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            inputs = permutated_local_input_tokens
            dispatched_input = token_dispatcher.dispatch_all_to_all(permutated_local_input_tokens)
            # release tensor not used by backward
            # inputs.untyped_storage().resize_(0)
        return dispatched_input


class MoeMlPNode(TransformerLayerNode):
    def forward_impl(self, dispatched_input):
        pre_mlp_layernorm_output = self.common_state.pre_mlp_layernorm_output
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            expert_output, shared_expert_output, mlp_bias = self.layer._submodule_moe_forward(
                dispatched_input, self.common_state.tokens_per_expert, pre_mlp_layernorm_output
            )
            assert mlp_bias is None
            assert mlp_bias is None

        # pre_mlp_layernorm_output  used
        self.common_state.pre_mlp_layernorm_output = None
        return expert_output, shared_expert_output

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_mlp_dw()


class MoeCombineNode(TransformerLayerNode):
    def forward_impl(self, expert_output, shared_expert_output):
        # TODO(lhb): if dw use grad of residual and probs, necessary synchronization should be add
        residual = self.common_state.residual
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            permutated_local_input_tokens = token_dispatcher.combine_all_to_all(
                expert_output
            )
            output = self.layer._submodule_post_combine_forward(
                permutated_local_input_tokens, shared_expert_output, None, residual
            )
        cur_stream = torch.cuda.current_stream()
        self.common_state.residual.record_stream(cur_stream)
        self.common_state.probs.record_stream(cur_stream)
        self.common_state.residual = None
        self.common_state.probs = None
        return output


class DenseAttnNode(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset

        hidden_states = self.layer._submodule_attention_forward(
            hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return hidden_states


class FakeScheduleNode:

    def forward(self, inputs):
        return inputs

    def backward(self, outgrads):
        return outgrads


class DenseMlpNode(TransformerLayerNode):
    def forward_impl(self, hidden_states):
        return self.layer._submodule_dense_forward(hidden_states)


def build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream):
    common_state = TransformerLayerState()
    attn = DenseAttnNode(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = FakeScheduleNode()
    mlp = DenseMlpNode(chunk_state, common_state, layer, comp_stream, event)
    combine = FakeScheduleNode()
    return TransformerLayerSchedulePlan(attn, dispatch, mlp, combine)


def build_layer_schedule_plan(layer, event, chunk_state, comp_stream, com_stream):
    if not isinstance(layer.mlp, MoELayer):
        return build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream)
    common_state = TransformerLayerState()
    attn = MoeAttnNode(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = MoeDispatchNode(chunk_state, common_state, layer, com_stream, event, True)
    dispatch.name = "dispatch"
    mlp = MoeMlPNode(chunk_state, common_state, layer, comp_stream, event, True)
    mlp.name = "mlp"
    combine = MoeCombineNode(chunk_state, common_state, layer, com_stream, event, True)
    combine.name = "combine"
    return TransformerLayerSchedulePlan(attn, dispatch, mlp, combine)


class TransformerLayerState(MoEAlltoAllPerBatchState):
    pass


class ModelChunkSate:
    pass


class TransformerLayerSchedulePlan:

    def __init__(self, attn, dispatch, mlp, combine):
        self.attn = attn
        self.dispatch = dispatch
        self.mlp = mlp
        self.combine = combine


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    def __init__(self):
        super().__init__()
        self._pre_process = None
        self._post_process = None
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []
        self._event = torch.cuda.Event()

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):

        return schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad=grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    @property
    def event(self):
        return self._event

    def record_current_stream(self):
        stream = torch.cuda.current_stream()
        self.event.record(stream)

    def wait_current_stream(self):
        stream = torch.cuda.current_stream()
        self.event.wait(stream)

    @property
    def pre_process(self):
        return self._pre_process

    @pre_process.setter
    def pre_process(self, value):
        self._pre_process = value

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, value):
        self._post_process = value

    def get_layer(self, i):
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def num_layers(self):
        return len(self._transformer_layers)

    def add_layer(self, layer):
        self._transformer_layers.append(layer)

    @property
    def state(self):
        return self._model_chunk_state


def schedule_layer_1f1b(
    f_layer,
    b_layer,
    f_input=None,
    b_grad=None,
    pre_forward=None,
    pre_backward=None,
    pre_backward_dw=None,
    f_context=None,
    b_context=None,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()


    if pre_forward is not None:
        assert f_input is None
        # combine from last iter
        f_input = pre_forward()
        del pre_forward


    if pre_backward is not None:
        # attn backward from last iter
        assert b_grad is None
        b_grad = pre_backward()
        del pre_backward


    if b_layer is not None:
        with b_context:
            b_grad = b_layer.combine.backward(b_grad)

    if pre_backward_dw is not None:
        pre_backward_dw()
        del pre_backward_dw


    if f_layer is not None:
        with f_context:
            f_input = f_layer.attn.forward(f_input)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.dispatch.forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.mlp.backward(b_grad)
            b_grad = b_layer.dispatch.backward(b_grad)
            b_layer.mlp.dw()


    if f_layer is not None:
        with f_context:
            f_input = f_layer.mlp.forward(f_input)



    def next_iter_pre_forward():
        if f_layer is not None:
            with f_context:
                output = f_layer.combine.forward(f_input)
                return output

    def next_iter_pre_backward():
        if b_layer is not None:
            with b_context:
                grad = b_layer.attn.backward(b_grad)
                return grad

    def next_iter_pre_backward_dw():
        if b_layer is not None:
            with b_context:
                b_layer.attn.dw()


    if f_layer and b_layer:
        return next_iter_pre_forward, next_iter_pre_backward, next_iter_pre_backward_dw
    else:
        return next_iter_pre_forward(), next_iter_pre_backward(), next_iter_pre_backward_dw()


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()


    if f_schedule_plan:
        # pp output send/receive sync
        if pre_forward is not None:
            with f_context:
                pre_forward()
        f_schedule_plan.record_current_stream()

    if b_schedule_plan:
        b_schedule_plan.record_current_stream()

    f_input = None

    def layer_pre_forward():
        tmp = f_input
        if f_schedule_plan is not None:
            tmp = f_schedule_plan.pre_process.forward()
        return tmp

    def layer_pre_backward():
        tmp = grad
        if b_schedule_plan is not None:
            assert grad is not None
            if b_schedule_plan.post_process is not None:
                with b_context:
                    tmp = b_schedule_plan.post_process.backward(grad)

            if pre_backward is not None:
                # pp grad send receive sync here, safe for now, maybe not safe in the future
                with torch.cuda.stream(get_com_stream()):
                    b_schedule_plan.wait_current_stream()
                    with b_context:
                        pre_backward()
                    b_schedule_plan.record_current_stream()

        return tmp

    def layer_pre_backward_dw():
        pass

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        layer_pre_forward, layer_pre_backward, layer_pre_backward_dw = schedule_layer_1f1b(
            f_layer,
            b_layer,
            pre_forward=layer_pre_forward,
            pre_backward=layer_pre_backward,
            pre_backward_dw=layer_pre_backward_dw,
            f_context=f_context,
            b_context=b_context,
        )
        torch.cuda.nvtx.range_pop()

    # tail backward
    grad = layer_pre_backward()
    del layer_pre_backward
    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            tmp, grad, _ = schedule_layer_1f1b(None, b_layer, b_grad=grad)
            torch.cuda.nvtx.range_pop()

        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(grad)

    # tail forward
    f_input = layer_pre_forward()
    del layer_pre_forward
    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, tmp, _ = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)

    # output pp send receive, overlapped with attn backward
    if f_schedule_plan is not None and post_forward is not None:
        with f_context:
            f_schedule_plan.wait_current_stream()
            post_forward(f_input)

    # pp grad send / receive, overlapped with attn dw of cur micro-batch and forward attn of next micro-batch
    if b_schedule_plan is not None and post_backward is not None:
        with b_context:
            b_schedule_plan.wait_current_stream()
            post_backward(grad)

    # The last wgrad of attention
    layer_pre_backward_dw()
    del layer_pre_backward_dw

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()
    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()

    return f_input


def build_model_chunk_schedule_plan(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_params=None,
    packed_seq_params=None,
    extra_block_kwargs=None,
    runtime_gather_output: Optional[bool] = None,
):

    comp_stream = torch.cuda.current_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state
    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.decoder_input = decoder_input
    state.labels = labels
    state.inference_params = inference_params
    state.packed_seq_params = packed_seq_params
    state.extra_block_kwargs = extra_block_kwargs
    state.runtime_gather_output = runtime_gather_output
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, state, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:

        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    return model_chunk_schedule_plan
