# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import transformer_engine as te
from packaging import version
from torch.nn.functional import mse_loss
from torch.optim import Adam

from tests.unit_tests.test_utilities import Utils
from megatron.core.utils import is_te_min_version
logger = logging.getLogger(__name__)

HSDP = "hsdp"
DP = "dp"
DP_SHARD = "dp_shard"
DP_OUTER = "dp_outer"
CP = "cp"
DP_SHARD_CP = "dp_shard_cp"
TP = "tp"
NO_SHARD = "no_shard"
OPTIM = "optim"
OPTIM_GRADS = "optim_grads"
OPTIM_GRADS_PARAMS = "optim_grads_params"
CNN = "cnn"
TRANSFORMER = "transformer"
TE_TRANSFORMER = "te_transformer"
DIM_SIZE = 4
NUM_LAYERS = 2
NUM_STEPS = 2
DELAYED_FP8_RECIPE = "fp8_delayed_scaling"
CURRENT_FP8_RECIPE = "fp8_current_scaling"
BLOCKWISE_FP8_RECIPE = "fp8_blockwise_scaling"
MXFP8_BLOCKWISE_RECIPE = "mxfp8_blockwise"

# Needed for `torch.distributed.checkpoint.{save,load}` because
# multiple processes need to write to the same directory.
SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"


def destroy_device_mesh(device_mesh):

    # Get all process groups from the mesh before deleting it
    process_groups = []
    try:
        # Try to get process groups from the mesh
        if hasattr(device_mesh, 'get_group'):
            # Get all the groups including flattened ones
            for dim_name in [DP, DP_SHARD, CP, TP, DP_OUTER, DP_SHARD_CP, HSDP]:
                try:
                    group = device_mesh[dim_name].get_group() if hasattr(device_mesh[dim_name], 'get_group') else None
                    if group is not None and group not in process_groups:
                        process_groups.append(group)
                except:
                    pass
    except Exception as e:
        print(f"  [Cleanup] Warning: Could not enumerate process groups: {e}")
    
    # Teardown device mesh
    del device_mesh
    
    try:
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.child_to_root_mapping.clear()
        _mesh_resources.root_to_flatten_mapping.clear()
        _mesh_resources.mesh_stack.clear()
        _mesh_resources.mesh_dim_group_options.clear()
        _mesh_resources.flatten_name_to_root_dims.clear()
    except Exception as e:
        # Global _MeshEnv is on a convoluted deprecation path.
        # Attempt to clean the global state, otherwise skip.
        logger.warning(f"Did not clean the deprecated DeviceMesh global state. Skipping...\n{e}")
        pass

    # Now try to destroy the process groups
    # NOTE: This may not work on all PyTorch versions - the groups might be 
    # reference-counted and destroying them explicitly can cause issues
    for group in process_groups:
        try:
            # Only destroy non-default world groups
            if group != torch.distributed.group.WORLD:
                torch.distributed.destroy_process_group(group)
        except Exception as e:
            # Destroying process groups can fail, just log and continue
            print(f"  [Cleanup] Warning: Could not destroy process group: {e}")


class ToyCNN(torch.nn.Module):
    """Toy CNN model for testing Megatron-FSDP sharding for high-rank Tensor parameters and inputs."""

    def __init__(
        self,
        channels: int = 3,
        height: int = 10,
        width: int = 10,
        kernel_size: int = 3,
        output_dim: int = 10,
        bias: bool = True,
        num_layers: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.bias = bias
        self.num_layers = num_layers
        self.cnn_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(channels, channels, kernel_size, padding="same", bias=bias)
                for _ in range(num_layers)
            ]
        )
        self.dense = torch.nn.Linear(channels, 1, bias)

    def forward(self, x: torch.Tensor):
        """Toy forward pass for the CNN, where input and output shapes match."""
        x = x.broadcast_to(1, self.channels, self.height, self.width)
        for layer in self.cnn_layers:
            x = layer(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.dense(x).reshape(1, self.height, self.width)
        return x


class ToyTransformer(torch.nn.Module):
    """Toy Transformer model for testing Megatron-FSDP."""

    def __init__(self, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc_out = torch.nn.Linear(model_dim, output_dim)

    def forward(self, x, y):
        x = self.transformer(x, y)
        x = self.fc_out(x)
        return x


class ToyTETransformer(torch.nn.Module):
    """Toy Transformer model for testing Megatron-FSDP with Transformer Engine."""

    def __init__(
        self,
        model_dim,
        num_heads,
        num_layers,
        output_dim,
        fuse_qkv_params=False,
        params_dtype=torch.float32,
        device="cuda",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                te.pytorch.TransformerLayer(
                    hidden_size=model_dim,
                    ffn_hidden_size=model_dim,
                    num_attention_heads=num_heads,
                    fuse_qkv_params=fuse_qkv_params,
                    params_dtype=params_dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = te.pytorch.Linear(
            model_dim, output_dim, params_dtype=params_dtype, device=device
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


def build_toy_model(model_type: str, init_model_with_meta_device: bool, seed=None):
    """
    Helper function to build a toy model for testing Megatron-FSDP.
    """
    # Set the seed to make sure the same model is initialized on all ranks.
    if seed is not None:
        torch.manual_seed(seed)
    # Initialize on meta device or CUDA device. For CPU, use nullcontext() instead,
    # but for these tiny models we can just move everything to CUDA immediately.
    with torch.device("meta") if init_model_with_meta_device else torch.device("cuda"):
        if model_type == CNN:
            toy_model = ToyCNN(
                channels=3,
                height=DIM_SIZE,
                width=DIM_SIZE,
                kernel_size=3,
                output_dim=DIM_SIZE,
                bias=True,
                num_layers=NUM_LAYERS,
            )
            fsdp_unit_modules = [torch.nn.Conv2d, torch.nn.Linear]
        elif model_type == TRANSFORMER:
            toy_model = ToyTransformer(
                model_dim=DIM_SIZE, num_heads=2, num_layers=NUM_LAYERS, output_dim=DIM_SIZE
            )
            fsdp_unit_modules = [torch.nn.Transformer]
        elif model_type == TE_TRANSFORMER:
            toy_model = ToyTETransformer(
                model_dim=DIM_SIZE,
                num_heads=2,
                num_layers=NUM_LAYERS,
                output_dim=DIM_SIZE,
                device="meta" if init_model_with_meta_device else "cuda",
            )
            fsdp_unit_modules = [te.pytorch.TransformerLayer]

    # Return the toy model, optimizer, and FSDP unit modules.
    return toy_model, fsdp_unit_modules


def build_distributed_environment(mesh_dim_config: tuple):
    """
    Helper function to build a distributed environment for testing Megatron-FSDP.
    Order of dimensions is (DP_OUTER, DP_SHARD, CP, TP).
    """
    from torch.distributed.device_mesh import init_device_mesh

    # Construct device mesh.
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=mesh_dim_config, mesh_dim_names=(DP_OUTER, DP_SHARD, CP, TP)
    )
    # DP: Only relevant when using HSDP, where we need the flattened DP group for data parallelism. (Otherwise, just pass dp_shard.)
    device_mesh[(DP_OUTER, DP_SHARD)]._flatten(DP)
    # DP-Shard-CP: Only required if using CP. Otherwise, just pass dp_shard to FSDP.
    device_mesh[(DP_SHARD, CP)]._flatten(DP_SHARD_CP)
    # HSDP (DP-CP): Only required if using HSDP. Otherwise, don't pass hybrid_fsdp_group to Megatron-FSDP.
    device_mesh[(DP_OUTER, DP_SHARD, CP)]._flatten(HSDP)

    # Return the device mesh.
    return device_mesh


class TestMegatronFsdpFullyShard:
    """
    Test the fully_shard API for Megatron-FSDP.

    FIXME(@cspades): Megatron-FSDP leaves behind corrupted NCCL state that affects other tests.
    Until this is repaired, this test must be run in a separate bucket / container.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.4.0'),
        reason="Requires DTensor and DeviceMesh support in (approximately) PyTorch 2.4.0 or later. Should not be run on 2.2.0a0+81ea7a4 (LTS).",
    )
    @pytest.mark.parametrize("model_type", [CNN, TRANSFORMER, TE_TRANSFORMER])
    @pytest.mark.parametrize(
        "dp_shard_strategy", [NO_SHARD, OPTIM, OPTIM_GRADS, OPTIM_GRADS_PARAMS]
    )
    @pytest.mark.parametrize("dp_outer_strategy", [None, NO_SHARD, OPTIM])
    @pytest.mark.parametrize(
        "mesh_dim_config",
        [
            # (DP_OUTER, DP_SHARD, CP, TP)
            (2, 2, 2, 1),
            (1, 2, 2, 2),
            # TODO(@cspades, @boxiangw): Add a DTensor-based TP model
            # case to test strided sharding when using HSDP + TP.
            (2, 2, 1, 2),
        ],
    )
    @pytest.mark.parametrize(
        "common_args",
        [
            {
                "preserve_fp32_weights": True,
                "init_model_with_meta_device": True,
                "torch_compile": True,
            },
            {
                "preserve_fp32_weights": False,
                "init_model_with_meta_device": False,
                "torch_compile": False,
            },
        ],
    )
    def test_fully_shard(
        self, model_type, dp_shard_strategy, dp_outer_strategy, mesh_dim_config, common_args
    ):
        """
        Test the fully_shard API with different configurations.
        Does NOT test for performance or convergence.

        NOTE(@cspades): This test is combinatorially large,
        don't add any new parameters unless absolutely necessary,
        or if some combinations can be flattened or simplified.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard

        preserve_fp32_weights = common_args["preserve_fp32_weights"]
        init_model_with_meta_device = common_args["init_model_with_meta_device"]
        torch_compile = common_args["torch_compile"]

        # Skip due to lack of functionality.
        if init_model_with_meta_device and dp_shard_strategy == NO_SHARD:
            pytest.skip(
                "Meta device initialization (init_model_with_meta_device=True) is not "
                "supported or necessary for the 'no_shard' / 0 sharding strategy."
            )
        elif dp_outer_strategy == OPTIM:
            if dp_shard_strategy != OPTIM_GRADS_PARAMS:
                # TODO(@shjwudp, @cspades): Requires various modifications to support.
                # [default0]:FAILED tests/unit_tests/distributed/test_mfsdp_fully_shard.py
                # [False-True-True-True-mesh_dim_config0-optim-optim-cnn]
                # [False-True-True-True-mesh_dim_config0-optim-optim_grads-cnn]
                pytest.skip(
                    f"dp_outer sharding strategy {dp_outer_strategy} requires "
                    "zero_dp_strategy to be full-sharded ('optim_grads_params', 3)."
                )

        # Construct device mesh.
        device_mesh = build_distributed_environment(mesh_dim_config)

        # Construct toy model.
        toy_model, fsdp_unit_modules = build_toy_model(model_type, init_model_with_meta_device)
        toy_adam = Adam(params=toy_model.parameters(), lr=0.01)

        # Wrap in fully_shard.
        model, optimizer = fully_shard(
            module=toy_model,
            optimizer=toy_adam,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD_CP,
            dp_outer_dim=DP_OUTER if dp_outer_strategy is not None else None,
            tp_dim=TP,
            hybrid_fsdp_group=(
                device_mesh[HSDP].get_group() if dp_outer_strategy is not None else None
            ),
            fsdp_unit_modules=fsdp_unit_modules,
            zero_dp_strategy=dp_shard_strategy,
            outer_dp_sharding_strategy=(
                dp_outer_strategy if dp_outer_strategy is not None else "no_shard"
            ),
            preserve_fp32_weights=preserve_fp32_weights,
            grad_reduce_in_fp32=False,
            init_model_with_meta_device=init_model_with_meta_device,
        )
        model = torch.compile(model) if torch_compile else model

        # Mock input and target.
        toy_input = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")
        toy_target = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")

        for step in range(NUM_STEPS):
            # Synchronize model parameters and gradients on the final training step only.
            if step == NUM_STEPS - 1:
                model.set_model_auto_sync(True)
            else:
                model.set_model_auto_sync(False)

            # Forward pass.
            if model_type == CNN or model_type == TE_TRANSFORMER:
                output = model(toy_input)
            elif model_type == TRANSFORMER:
                output = model(toy_input, toy_input)

            # Loss.
            loss = mse_loss(output, toy_target)

            # Backward pass.
            loss.backward()

            # Validate gradients exist in the Torch Module, i.e. non-None and non-zero.
            grads_exist = any(
                isinstance(p.grad, torch.Tensor) and p.grad.to_local().count_nonzero().item() > 0
                for p in model.parameters()
            )
            sharding_group = (
                device_mesh[HSDP].get_group()
                if dp_outer_strategy == OPTIM
                else device_mesh[DP_SHARD_CP].get_group()
            )
            if dp_shard_strategy != NO_SHARD:
                # Because of uneven sharding, we need to gather the result from all ranks
                # to verify if any gradients exist or not at this step of training.
                grads_exist_gathered = [None] * sharding_group.size()
                torch.distributed.all_gather_object(
                    object_list=grads_exist_gathered, obj=grads_exist, group=sharding_group
                )
                # Gradients exist on at least one of the optimizer sharding ranks.
                grads_exist = any(grads_exist_gathered)

            # Gradients do not exist until synchronization is activated.
            if step == NUM_STEPS - 1:
                assert grads_exist, "Root module gradients should exist on final microbatch."
            else:
                assert (
                    not grads_exist
                ), "Root module gradients should not exist prior to optimization step."
            torch.distributed.barrier()

            # Optimizer step. Apply accumulated gradients to the model weights.
            if step == NUM_STEPS - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Required to reset the parallelism environment.
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("shard_strategy", [OPTIM_GRADS_PARAMS, OPTIM_GRADS, OPTIM, NO_SHARD])
    def test_fully_shard_ez(self, shard_strategy):
        """
        Test fully_shard(device_mesh=None). Represents the easiest entrypoint to Megatron-FSDP.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            fully_shard_model,
            fully_shard_optimizer,
        )

        # Construct toy model.
        toy_model, fsdp_unit_modules = build_toy_model(TRANSFORMER, False)

        # Fully-shard the model.
        mfsdp_model = fully_shard_model(
            module=toy_model, fsdp_unit_modules=fsdp_unit_modules, zero_dp_strategy=shard_strategy
        )

        # Initialize the distributed optimizer on the MegatronFSDP model.
        toy_adam = Adam(params=mfsdp_model.parameters(), lr=0.01)
        optimizer = fully_shard_optimizer(optimizer=toy_adam)

        # Mock input and target.
        toy_input = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")
        toy_target = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")

        for step in range(NUM_STEPS):

            # Forward pass.
            output = mfsdp_model(toy_input, toy_input)

            # Loss.
            loss = mse_loss(output, toy_target)

            # Backward pass.
            loss.backward()

            # Optimizer step.
            optimizer.step()
            optimizer.zero_grad()

    @pytest.mark.skipif(
        not is_te_min_version("2.10.0"),
        reason="TE >= 2.10.0 is required for test_fully_shard_te_quantized",
    )
    @pytest.mark.parametrize("init_model_with_meta_device", [True, False])
    @pytest.mark.parametrize(
        "te_recipe",
        [DELAYED_FP8_RECIPE, CURRENT_FP8_RECIPE, BLOCKWISE_FP8_RECIPE, MXFP8_BLOCKWISE_RECIPE],
    )
    def test_fully_shard_te_quantized(self, init_model_with_meta_device, te_recipe):
        """
        Test Megatron-FSDP with FP8 activations and parameters via TransformerEngine.
        """
        if te_recipe == MXFP8_BLOCKWISE_RECIPE:
            # TODO(@cspades, @ko3n1g): Add this test case in.
            pytest.skip(f"[Megatron CI/CD] MXFP8 requires Blackwell nodes to test.")

        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            fully_shard_model,
            fully_shard_optimizer,
        )

        # Build FP8 recipe.
        te_quant_recipe = None
        if te_recipe == MXFP8_BLOCKWISE_RECIPE:
            te_quant_recipe = te.common.recipe.MXFP8BlockScaling(
                fp8_format=te.common.recipe.Format.HYBRID
            )
        elif te_recipe == DELAYED_FP8_RECIPE:
            te_quant_recipe = te.common.recipe.DelayedScaling()
        elif te_recipe == CURRENT_FP8_RECIPE:
            te_quant_recipe = te.common.recipe.Float8CurrentScaling()
        elif te_recipe == BLOCKWISE_FP8_RECIPE:
            pytest.skip("FP8 block scaling is not supported on ROCM")
            te_quant_recipe = te.common.recipe.Float8BlockScaling()

        # Construct toy model compatible with FP8.
        with (
            te.pytorch.quantized_model_init(
                recipe=te_quant_recipe,
                # Needed for FP8 parameters with Megatron-FSDP.
                preserve_high_precision_init_val=True,
            )
            if te_quant_recipe is not None
            else nullcontext()
        ):
            # Fused QKV, BF16 precision for high-precision weights,
            # and hidden dimension divisibility by 32 is required
            # for some FP8 recipes such as MXFP8.
            toy_model = ToyTETransformer(
                model_dim=64,
                num_heads=2,
                num_layers=2,
                output_dim=64,
                fuse_qkv_params=True,
                params_dtype=torch.bfloat16,
                device="meta" if init_model_with_meta_device else "cuda",
            )

        # Fully-shard the model.
        mfsdp_model = fully_shard_model(
            module=toy_model,
            fsdp_unit_modules=[te.pytorch.TransformerLayer, te.pytorch.Linear],
            # Only ZeRO-3 / FSDP supports FP8 parameters.
            zero_dp_strategy=3,
            init_model_with_meta_device=init_model_with_meta_device,
            # Required for FP8 parameter support, except for MXFP8 which has
            # its own row-wise and col-wise (transpose) buffer management
            # schedule that is natively managed by Megatron-FSDP.
            keep_fp8_transpose_cache=True,
            # Required for FP8 parameters. The optimizer state (and gradients)
            # are never quantized, as TE produces high-precision wgrad and
            # dgrad from FP8 weights and activations. Already defaults to True.
            preserve_fp32_weights=True,
        )

        # Initialize the distributed optimizer on the MegatronFSDP model.
        toy_adam = Adam(params=mfsdp_model.parameters(), lr=0.01)
        optimizer = fully_shard_optimizer(optimizer=toy_adam)

        # Mock input and target. Requires 2^N batch size for (MX)FP8 kernels.
        toy_input = torch.randn(16, 64, 64, dtype=torch.bfloat16).to("cuda")
        toy_target = torch.randn(16, 64, 64, dtype=torch.bfloat16).to("cuda")

        for step in range(NUM_STEPS):

            # Forward pass.
            with (
                te.pytorch.autocast(recipe=te_quant_recipe)
                if te_quant_recipe is not None
                else nullcontext()
            ):
                output = mfsdp_model(toy_input)

            # Loss.
            loss = mse_loss(output, toy_target)

            # Backward pass.
            loss.backward()

            # Optimizer step.
            optimizer.step()
            optimizer.zero_grad()
