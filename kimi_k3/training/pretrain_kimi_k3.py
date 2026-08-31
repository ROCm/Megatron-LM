"""Kimi K3 training entry point.

Two surfaces, deliberately:

* `model_provider` / `forward_step` have the shapes `megatron.training.pretrain`
  expects, so the stock training loop can drive K3 once its argument plumbing is
  wired (P7 follow-up);
* `train_smoke` is a self-contained loop -- build, forward, backward, step, N
  times -- which is what the gates actually run. It exercises the pieces that can
  break silently (the schedule binding, the optimizer split, the AttnRes payload)
  without a dataset or a full argument namespace in the way.

The one thing this file must own is *assembly order*, because getting it wrong
fails late and confusingly:

1. build the config with our builder, never core's (it would substitute the
   config class);
2. build the model inside the block injection;
3. bind `adjust_tensor_shapes_fn` **only when PP > 1** -- the other schedules
   assert it is `None`;
4. give `DistributedDataParallelConfig` the same `use_distributed_optimizer`
   value as `OptimizerConfig`, or the first step dies in
   `_copy_main_params_to_model_params` (recorded in the G5 note).
"""

from typing import Callable, List, Optional

import torch

from ..model.build import build_k3_model


def model_provider(pre_process: bool = True, post_process: bool = True, preset: str = "tiny", **kw):
    """Megatron's `model_provider` shape."""
    return build_k3_model(preset, pre_process=pre_process, post_process=post_process, **kw)


def mock_batch(vocab_size: int, seq_length: int, micro_batch_size: int, device="cuda", seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(
        0, vocab_size, (micro_batch_size, seq_length + 1), generator=generator
    ).to(device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def loss_func(labels: torch.Tensor) -> Callable:
    def _loss(output_tensor: torch.Tensor):
        logits = output_tensor.float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        return loss, {"lm loss": loss.detach()}

    return _loss


def forward_step(data_iterator, model):
    """Megatron's `forward_step` shape: returns `(output_tensor, loss_func)`."""
    tokens, labels = next(data_iterator)
    output = model(input_ids=tokens, position_ids=None, attention_mask=None)
    return output, loss_func(labels)


def build_optimizer(model, *, optimizer: str = "dist_muon", lr: float = 1e-4, bf16: bool = True):
    """Wrap the model in DDP and build the optimizer, consistently.

    `dist_muon` is the sharded Muon path -- 7.87 B/param at DP=8 against plain
    `muon`'s 15.17, measured in G5. `--use-distributed-optimizer` is rejected for
    every Muon variant, which is *not* the same as Muon being unable to shard.
    """
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.muon import get_megatron_muon_optimizer

    use_dist_opt = optimizer == "adam_dist"
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            # must match OptimizerConfig, or the first step dies in
            # _copy_main_params_to_model_params (G5 note)
            use_distributed_optimizer=use_dist_opt,
        ),
        model,
    )
    opt_config = OptimizerConfig(
        optimizer="adam" if use_dist_opt else optimizer,
        lr=lr,
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        use_distributed_optimizer=use_dist_opt,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    if "muon" in opt_config.optimizer:
        build = lambda: get_megatron_muon_optimizer(
            opt_config, [ddp], layer_wise_distributed_optimizer="dist" in opt_config.optimizer
        )
        if getattr(model.config, "k3_per_head_muon", False):
            from ..optim.per_head_muon import per_head_muon

            with per_head_muon(model, model.config):
                return ddp, build()
        return ddp, build()
    return ddp, get_megatron_optimizer(opt_config, [ddp])


def train_smoke(
    preset: str = "tiny",
    iterations: int = 10,
    seq_length: int = 32,
    micro_batch_size: int = 1,
    optimizer: str = "dist_muon",
    lr: float = 1e-4,
    bf16: bool = False,
    seed: int = 0,
    fixed_batch: bool = False,
    overrides: Optional[dict] = None,
) -> List[float]:
    """Run `iterations` real training steps and return the losses.

    Returns rather than prints, so a gate can assert on the trajectory instead of
    on a log line.

    With `fixed_batch`, every step sees the same tokens, so the loss must fall --
    that is the difference between "the optimizer ran" and "the model learns".
    On fresh random tokens it should instead sit near `ln(vocab_size)`, which is
    chance and is its own useful check.
    """
    from megatron.core import tensor_parallel

    from ..config.presets import preset as get_preset

    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)

    spec = get_preset(preset)
    model = build_k3_model(preset, **(overrides or {}))
    if bf16:
        model = model.bfloat16()
    ddp, opt = build_optimizer(model, optimizer=optimizer, lr=lr, bf16=bf16)

    losses = []
    for step in range(iterations):
        tokens, labels = mock_batch(
            spec["model"]["vocab_size"], seq_length, micro_batch_size,
            seed=seed if fixed_batch else seed + step,
        )
        ddp.zero_grad_buffer()
        opt.zero_grad()
        output = ddp(input_ids=tokens, position_ids=None, attention_mask=None)
        loss, _ = loss_func(labels)(output)
        loss.backward()
        ddp.finish_grad_sync()
        opt.step()
        losses.append(float(loss.detach()))

        for module in model.modules():
            update = getattr(module, "update_expert_bias", None)
            if callable(update):
                update()

    return losses
