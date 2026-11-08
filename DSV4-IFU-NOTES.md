# r0.20.0 IFU + DSv4-Flash: resolution notes

Branch `dsv4-flash`, merging `upstream/core_r0.20.0` into `sudhu/core_r0.19_0_IFU`
(`cf85e22cc`). 42 conflicts; 1,984 files auto-merged.

**The DSv4-Flash surface lands as clean adds** -- no conflicts:
`deepseek_v4_hybrid_attention{,_module_specs}.py`, `csa.py`, `csa_utils/`,
`hyper_connection.py`, `fused_mhc_kernels.py`, `hybrid_hyper_connection.py`,
`training/models/deepseek_v4.py`, `dsa_layer_config.py`, `dsv4_layer_config.py`.
This is why a merge beats cherry-picking: DSv4 depends on `csa.py` + mHC, neither
of which exists in r0.19.

## MUST FIX BEFORE MERGE -- ROCm fixes destroyed by r0.20.0

1. **`fork` -> `spawn` in async checkpointing.** ROCm's only change to
   `dist_checkpointing/strategies/async_utils.py` was
   `mp.get_context('fork')` -> `'spawn'`. Upstream deleted the file and
   `TemporalAsyncCaller` no longer exists -- but `mp.get_context('fork')` still
   does, now at `megatron/training/checkpointing.py:1267`. The hazard is
   unchanged; relocate the fix there.

2. **Stale checkpoint-metadata cache (correctness).** ROCm's 28-line fix to
   `cached_metadata_filesystem_reader.py` keyed the cache on
   `(abs_path, st_mtime_ns, st_size)` instead of path alone, because writing a
   new checkpoint to the same directory returned stale metadata and failed the
   load. r0.20.0 deletes the file and imports the class from the external
   `nvidia_resiliency_ext` package, so the fix is lost AND we inherit an NVIDIA
   dependency. Decide: re-apply on top of the package, keep ROCm's local reader
   and patch `strategies/torch.py`, or upstream the fix.
   Directly relevant to LumenRL, which saves and reloads checkpoints to the same
   path every step.

## MUST FIX 1-4: resolved

**1. `fork` -> `spawn`: cannot be fixed in this repo, and still live.** Upstream moved
async checkpointing wholesale into the external `nvidia_resiliency_ext` package
(`serialization.py:50` imports `AsyncRequest` from it; `strategies/nvrx.py` imports the
async core). The fork ROCm patched now lives at
`nvidia_resiliency_ext/checkpointing/async_ckpt/core.py:243`, still
`mp.get_context('fork')` in 0.4.1. The fix must go to NVIDIA or to a local pin of that
package -- there is nothing to patch in Megatron. The one surviving in-tree `fork`
(`training/checkpointing.py:1267`) is a different path: async checkpoint *deletion*,
whose child is filesystem-only and documented as running without torch.distributed, so
ROCm's GPU-context concern does not transfer.

**2. Stale metadata cache: not applicable.** The package's
`CachedMetadataFileSystemReader` caches **per instance** (`self._cached_metadata`), not
in a class-level dict keyed by path, so the cross-checkpoint staleness ROCm fixed cannot
occur. ROCm's `(abs_path, st_mtime_ns, st_size)` key was fixing Megatron's own
class-level `_metadata_cache`, which no longer exists.

**3. MLA padding: this was a real defect, now fixed.** Removing ROCm's pad left the
feature internally inconsistent, not merely narrowed: `__init__` (`:330`) still sizes
`core_attention` with `v_channels == q_head_dim`, and the trim (`:625`) still fires, so
with `fused_padded_mla_attention=True` the kernel would have received `v_head_dim`-wide
values while configured for `q_head_dim` and then had its output trimmed as though
padded. Upstream's `_prepare_mla_core_attention_value` does not cover this path -- it is
reached only from the `cache_mla_latents` branch and gated on thd. The pad is restored
with a shape test (`value.shape[-1] != self.q_head_dim`) making it idempotent, so it
cannot double-pad if upstream's helper ever reaches here. Verified:
`test_multi_latent_attention.py` 47 passed, 0 failed.

**4. Orphaned `chat_template_utils.py`: wrong call, no action.** It has a second
consumer, `megatron/core/tokenizers/text/libraries/chat_template.py`. The earlier "0
consumers" claim came from grepping only for `token_ids_from_chat_template_output`.

### New upstream defect (3rd one found)

Megatron r0.20 calls
`CachedMetadataFileSystemReader(checkpoint_dir, cache_metadata=cache_metadata)`
(`strategies/torch.py:837`), but the latest released `nvidia-resiliency-ext` (0.4.1) has
`__init__(self, path)` -- no `cache_metadata`. That is a `TypeError` against every
published version (0.1.3 .. 0.4.1), so `ckpt_assume_constant_structure=True` cannot
work with a released package. Upstream is targeting an unreleased build. Relevant to
LumenRL, which saves and reloads checkpoints every step.

## ~~MUST FIX 5~~ -- RESOLVED: ROCm's router padding patch was redundant

`megatron/core/transformer/moe/router.py` was never a conflict: it auto-merged and
kept ROCm's 9-line addition, which **zeroes padding tokens' probs**

    if padding_mask is not None:
        valid_mask = (~padding_mask).unsqueeze(-1)
        probs = probs * valid_mask

while r0.20.0's new `test_router_with_padding_mask` asserts the opposite --
"Padded rows stay routed (top-k routes with nonzero weights) like any other row":

    assert torch.all(torch.count_nonzero(probs_with_mask[padding_rows], dim=-1) == topk)

Git merged ROCm's code with upstream's test and they disagree, so
`TestTop2Router::test_router_with_padding_mask[False|True]` passed before the merge
and fails after it. This is the only test regression attributable to the IFU.

**Resolved by removing ROCm's 8-line block.** Upstream r0.20 now does this in the
dispatcher (`token_dispatcher.py:1126-1139`) and does it more completely -- masking
both `probs` and `routing_map` for padded rows, gated on `dropless`:

    dropless = self.capacity_factor is None and self.moe_expert_rank_capacity_factor is None
    if padding_mask is not None and dropless:
        padding_rows = padding_mask.transpose(0, 1).reshape(-1)
        probs = probs.masked_fill(padding_rows.unsqueeze(-1), 0)
        routing_map = routing_map.reshape(num_tokens, self.num_experts) & (~padding_rows).unsqueeze(-1)

Four reasons removal beat marking the tests `failing_on_rocm`:

1. Upstream's dispatcher subsumes it, and masks `routing_map` too.
2. `git log -S` traces ROCm's block to `6f6bfd498 "IFU core_r0.17.0 Cleanup"` -- it
   was a previous IFU's patch over this same area, not a deliberate ROCm feature.
3. ROCm added **no** router tests; nothing asserts padded probs are zero.
4. Its own rationale is now contradicted: the comment says "do NOT mask
   `routing_map` (the dropless argsort permute needs
   `routing_map.sum() == num_tokens * topk`)", but upstream masks `routing_map` in
   exactly that dropless case, so the permute has evidently been adapted.

**One real behaviour change:** ROCm's block was unconditional while upstream's is
`dropless`-gated, so padded probs are no longer zeroed on the capacity-factor path.
Upstream does not zero them there either, so this aligns us with upstream -- but it
is a difference, not a no-op.

Result: `test_routers.py` 9 failed/126 passed -> **135 passed, 0 failed**.
`router.py` now differs from upstream only by one (unused) import.

Verified clean afterwards with the same harness: `test_multi_latent_attention.py`
47 passed, `test_moe_layer.py` 63 passed, `test_a2a_token_dispatcher.py` 41 passed
(the path that now owns the masking), `test_router_replay.py` 6, `test_moe_unpermute.py`
1, attention-variant specs 46. **Zero failures.**

Superseded reasoning below, kept for history:

ROCm's comment says the zeroing is load-bearing ("the dropless argsort permute needs
`routing_map.sum() == num_tokens * topk`"), so it is not cosmetic. Decide: keep the
ROCm behaviour and mark the upstream test `failing_on_rocm`, or drop the 9 lines if
upstream's semantics now supersede. Needs the router patch's author.

## Resolved test failures (all attributable ones)

Three real failures came out of the 116; the other 113 are missing optional
packages, missing `/opt/data` fixtures, or test files absent from the baseline.

| file | before | after | cause |
|---|---|---|---|
| `transformer/moe/test_routers.py` | 9 failed | 135 passed | MUST FIX 5 -- ROCm's router padding block, redundant with upstream's dispatcher |
| `transformer/moe/test_grouped_mlp.py` | 10 failed | 52 passed | ROCm's `tokens_per_expert_gpu` vs upstream's 2-arg stubs; plus an upstream stub defect (`wgrad_store`) |
| `distributed/test_param_and_grad_buffer.py` | 4 failed | 97 passed | my error: kept a `mock.patch` of `log_on_each_pipeline_stage`, which upstream deleted |
| `distributed/mfsdp_v1/test_mfsdp_fully_shard.py` | DCP cases failed | 237 passed (30 DCP cases pass, 12 legitimately skipped) | stale DTensor sharding cache across parametrizations |

### The DCP failure was not a ROCm support gap

`test_dcp_checkpoint_save_and_load` (new in r0.20) failed with
`RuntimeError: Could not resolve the process group registered under the name 4303`.
Not a size-1 mesh-dim problem: a plain-torch 2.10 repro building the same
`DeviceMesh((dp_outer=1, dp_shard_cp=8))` resolves both dims fine
(`_dim_group_names = ['1', '0']`). The failing name was a high counter value on a
*flattened* submesh -- a stale group.

ROCm had already diagnosed this in `cf85e22cc`: "DeviceMesh equality ignores
process-group names, so a mesh rebuilt over new groups hits cache entries holding
the old mesh. DTensors created afterwards then reference destroyed groups." Open
upstream bug: https://github.com/pytorch/pytorch/issues/169814. The mitigation,
`Utils._clear_dtensor_sharding_caches()`, ran only from
`Utils.destroy_model_parallel()` i.e. at `teardown_class` -- too coarse for a test
with ~48 parametrizations that each build and destroy their own mesh.

Fixed by adding a `teardown_method` to `TestMegatronFsdpFullyShard` that calls the
existing helper per test. Test-file only; no new helper, no `megatron/core`
divergence. **A `failing_on_rocm` marker would have been wrong** -- it would have
permanently hidden a feature that works.

### Upstream defects found, worth reporting

1. `experts.py:729` reads `self.linear_fc2.wgrad_store` while upstream's own
   `FakeGroupedLinear` stubs never define it. Not ROCm-specific.
2. `apply_router_token_dropping` has no padding awareness, so with
   `moe_expert_capacity_factor` set and a padding mask present, padded tokens
   consume expert capacity (the dispatcher's masking is `dropless`-gated). Nothing
   in-tree exercises the combination and LumenRL is always dropless.

## Full-suite triage (`run_unit_tests.sh`, 605 files)

| | baseline `ifu` | merged |
|---|---|---|
| test files in tree | 372 | 605 (r0.20 adds 233) |
| files failed | 50 | 116 |

Of the 66 merged-only failures, **41 are test files that do not exist on the
baseline** (new r0.20 coverage), leaving 26 candidates. Comparing file counts
without intersecting the file sets is how the first attempt at this went wrong.

Note the image: `lumenrl-megatron-ci:pytestcsv` (the LumenRL release image plus
pytest-csv, Dockerfile in `/data/rl_data/img-pytestcsv/`). It is **not** ROCm's CI
image and lacks MambaSSM, FLA, causal_conv1d, emerging-optimizers,
`nvidia_resiliency_ext`, flash_mla and cudnn-frontend, so **54 of the 116 failures
are missing-dependency noise**. The absent `nvidia_resiliency_ext` is incidentally
hard evidence for MUST FIX 2.

Of the clusters touching these resolutions:

| file | cause | verdict |
|---|---|---|
| `dist_checkpointing/test_async_utils_shutdown.py` | `No module named 'nvidia_resiliency_ext'` | environmental |
| `dist_checkpointing/test_serialization.py` | same | environmental |
| `test_argument_utils.py` | `flash_mla`, `cudnn-frontend` absent | environmental |
| `transformer/moe/test_paged_stashing.py` | `RuntimeError: PassManager::run failed` (Triton) | environmental |
| `tokenizers/test_tokenizer.py` | `/opt/data/tokenizers` absent from the image | environmental (fails on baseline too; upstream added 13 more hardcoded paths, 7 -> 20) |
| `transformer/moe/test_grouped_mlp.py` | two under-specified upstream test doubles | **fixed, see below** |

### test_grouped_mlp.py: 10 failed -> 52 passed

Two distinct causes, only one of which is about ROCm:

1. `_DoublingGroupedLinear.forward(self, hidden_states, tokens_per_expert)` could not
   accept ROCm's third argument. `tokens_per_expert_gpu` is **ROCm-only** (0 at the
   merge base, 0 in r0.20) and comes from `8dd45f5a5 "TE AITER groupedgemm
   optimizations in Megatron (#109)"` -- a real aiter performance feature, not IFU
   cleanup, with no upstream equivalent. So unlike the router block (MUST FIX 5)
   there is nothing upstream to defer to: removing the argument would undo the
   optimization. Fixed on the stub instead (`tokens_per_expert_gpu=None`).
2. **An upstream defect we merely inherited:** upstream's `experts.py:729` does
   `op.wgrad_store = self.linear_fc2.wgrad_store`, but upstream's own
   `FakeGroupedLinear` stubs never define `wgrad_store`. Nothing ROCm-specific --
   this would fail on NVIDIA too wherever that path is hit. Added
   `self.wgrad_store = None` to the four stubs. **Worth reporting upstream.**

Both edits are confined to one upstream *test* file, with in-code comments. No
`megatron/core` divergence, so the aiter grouped-GEMM path is untouched.

Still untriaged: the other ~20 candidate regressions (mfsdp_v2, inference, mimo,
test_attention, test_model_configs, test_param_norm, ...), and the MORI tests
(`test_token_dispatcher_mori.py`, `test_token_dispatcher_cold_rank_mori.py`,
`a2a_overlap/test_schedule_*_1f1b_mori.py`) which need a node-spanning EP group and
remain the real check on the token_dispatcher/moe_utils/experts resolutions.

## Local test results

Run with **`./run_unit_tests.sh`'s invocation** -- `torchrun --standalone
--nproc_per_node=8 -m pytest -m "(not flaky and not flaky_in_dev and not
failing_on_rocm and not failing_on_upstream or test_on_rocm) and not
experimental"` -- on image `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260921b`.
Use that harness, not bare pytest: bare pytest ignores `failing_on_rocm` (so it
runs tests ROCm already knows fail) and runs single-process (so distributed tests
error instead of running). Measured the difference: bare pytest reported
34 failed / 36 errors on code the real harness reports as 9 failed / 0 errors.

| file | baseline `ifu` | merged |
|---|---|---|
| `transformer/moe/test_routers.py` | 49 passed, **0 failed** | **9 failed**, 126 passed, 14 skipped |
| `transformer/test_multi_latent_attention.py` | 37 passed, 45 deselected | 47 passed, 2 skipped, 45 deselected |
| `transformer/moe/test_router_replay.py` | 6 passed | 6 passed |
| `transformer/moe/test_moe_unpermute.py` | 1 passed | 1 passed |
| `models/test_experimental_attention_variant_module_specs.py` | 41 passed | 41 passed |
| **total failures** | **0** | **9** |

**The baseline is green, so the 9 failures are entirely introduced by this merge.**
All are in `test_routers.py` and all are MUST FIX 5: two are
`test_router_with_padding_mask[False|True]`, which passed pre-merge; seven are the
new `test_padding_mask_never_changes_routes` (absent at baseline). Neither test
carries `failing_on_rocm`, so ROCm CI will go red.

Everything else is clean. In particular `test_multi_latent_attention.py` is fully
green and gained 10 passing tests, which establishes that dropping ROCm's
fused-padded-MLA block (MUST FIX 3) broke nothing in these tests -- though the
narrowed gate is still a behaviour change worth a decision.

Also verified: 19/19 targeted module imports succeed, including the new DSv4/mHC
surface.

Not run: the MORI tests (`test_token_dispatcher_mori.py`,
`test_token_dispatcher_cold_rank_mori.py`) need a node-spanning EP group and are
the real check on the `token_dispatcher`/`moe_utils`/`experts` resolutions. Also
not run: the other ~200 files `run_unit_tests.sh` covers.

## Resolutions (42/42 -- merge complete)

| file | resolution | why |
|---|---|---|
| `.gitlab/{stages/01.build,stages/04.functional-tests,scripts/build.sh}`, `.github/workflows/community-bot.yml` | kept deleted (ours) | ROCm deliberately dropped NVIDIA's GitLab CI |
| `dist_checkpointing/strategies/async_utils.py` | accepted deletion | see MUST FIX 1 |
| `dist_checkpointing/strategies/cached_metadata_filesystem_reader.py` | accepted deletion | see MUST FIX 2 |
| `models/gpt/fine_grained_callables.py` | theirs | upstream threads `padding_mask` as a param and moved ROCm's `.transpose(0,1).bool()` into `route()` (`moe_layer.py:511`). Keeping ours would double-transpose. |
| `training/global_vars.py` | theirs + cleanup | upstream moved `update_num_microbatches` to `core/num_microbatches_calculator`. Auto-merge left 3 dangling `_GLOBAL_NUM_MICROBATCHES_CALCULATOR` refs (a NameError); removed them. `training.py:83` already imports from core. |
| `training/__init__.py` | theirs | drops the `update_num_microbatches` re-export, adds `get_train_state`; consistent with the above |
| `training/argument_utils.py` | theirs | ROCm's `kw_args["use_fsdp2"] = ...` sat **after** `config = config_class(**kw_args)`, so it never took effect; `use_fsdp2` is also not an upstream config field. Dead line dropped. |
| `transformer/transformer_block.py` | both | copyright only -- kept NVIDIA 2026 + AMD 2024 |
| `transformer/transformer_layer.py` | theirs + ROCm's `**kwargs` | upstream added an OTel span and extracted `attention_bda_and_cross_attention`; ROCm's only real change was forwarding `**kwargs` to `self_attention`, re-applied. The merged `_forward_attention` signature already accepts it. |

## Source resolutions, continued

| file | resolution | why |
|---|---|---|
| `transformer/transformer_config.py` | **unioned** + both | `moe_flex_dispatcher_backend` Literal unioned to `deepep/deepepv2/hybridep/mori/ncclep` -- taking upstream's side would have **dropped `'mori'` and disabled MORI outright**. ROCm's `fused_padded_mla_attention`/`attention_sink_k` and upstream's `output_projection_*` are different fields; kept both. |
| `transformer/multi_latent_attention.py` | both imports; sink kept; padding dropped | see MUST FIX 3 |
| `extensions/transformer_engine.py` | unioned typing imports | ROCm needed `import inspect` (9 uses), upstream `Sequence` (4 uses); `Type` had 0 uses so it was dropped |
| `training/training.py` (5 hunks) | theirs + ROCm's memory tracer re-applied | upstream migrated `args.*` -> `cfg.*` (671 `args.` / 111 `cfg.` coexist -- the migration is incremental upstream too). Took `cfg.logger.log_throughput` and `cfg.profiling.*`, kept ROCm's mem-usage logging and `profile_memory=True`, kept both import groups, restored `memory_stats_collector` in both the `train_step` signature and its relocated call inside upstream's new OTel span. |
| `moe/token_dispatcher.py` (7 hunks) | **both classes kept whole** | Not really a conflict: ROCm added `_MoriManager` (320 lines) and upstream added `_DeepepV2Manager` (108 lines) in the same region, so git interleaved them. Resolved by replacing the whole interleaved span with each class extracted intact from its own side, then keeping ROCm's `elif == "mori"` branch alongside upstream's 3-entry hybridep `cudagraph_attrs`, and unioning the backend list in the error message. |
| `moe/experts.py` | both | kept ROCm's 3rd arg `tokens_per_expert_gpu` to `linear_fc2` (it originates in `token_dispatcher.py:805-807` and is already consumed at `experts.py:1017`, so dropping it here would be internally inconsistent) with upstream's `setup_activation_checkpoint` local |
| `moe/moe_utils.py` (2 hunks) | both | ROCm's cold-EP-rank `fused` downgrades (permute + unpermute) and upstream's batch-invariant assertions are independent. Upstream's asserts placed first as input validation; ROCm's downgrade only ever sets `fused=False`, so it cannot invalidate them. |
| `moe/fused_a2a.py` | both | ROCm's `import os` (1 use) + upstream's DeepEP-v2 `ElasticBuffer` probe |
| `models/gpt/fine_grained_callables.py` | theirs | see above |
| `tokenizers/.../multimodal_tokenizer.py` | theirs | see MUST FIX 4 |
| `examples/multimodal/dataloader_provider.py` | theirs | pure reformatting, identical semantics |
| `tests/unit_tests/conftest.py` | both | kept ROCm's `MAMBA_DETERMINISTIC` determinism fix (SSM Triton autotune must be pinned before any `megatron.core.ssm.*` import) + upstream's `import sys` and 3 new `test_utilities` imports |
| `tests/unit_tests/utils/test_experimental_log_once.py` | both imports | `safe_get_rank` (3 uses) and `Utils` (4 uses) both live |

## MUST FIX (continued)

3. **ROCm's fused padded MLA loses padding outside THD.** ROCm padded `value`
   whenever `config.fused_padded_mla_attention` was set. Upstream implemented
   the same idea generically (`_prepare_mla_core_attention_value` +
   `_trim_mla_core_attention_output`, which ROCm only had as a comment), but
   its gate is NARROWER:
   `qkv_format == "thd" and experimental_attention_variant is None`.
   So the ROCm fused MLA kernel no longer receives padded `value` outside THD,
   and `fused_padded_mla_attention` becomes an inert config field. Kept ROCm's
   separate attention-sink-bias block. Decide: re-widen the gate on
   `fused_padded_mla_attention`, or confirm the ROCm path is THD-only.

4. **`chat_template_utils.py` is orphaned.** It is ROCm-only (absent upstream)
   and `multimodal_tokenizer.py` was its sole consumer. Upstream's rewrite
   (`skip_chat_template`, `_apply_chat_template_to_text`, `_encode_with_markers`,
   thinking traces) is too large to graft ROCm's 3 lines onto, so it was taken
   whole and the helper now has 0 consumers. Either delete the module or verify
   upstream's new path handles whatever HF-output shape quirk it was written for.
   Not on the DSv4/FP8 path.

## Test resolutions (20 files)

Read individually rather than bulk-resolved by category: a "does ROCm's side
mention rocm?" heuristic scored `conftest.py` at zero and it held a real ROCm
determinism fix (`MAMBA_DETERMINISTIC`), and `test_distributed_data_parallel.py`
at zero while ROCm had deliberately removed `rtol=0, atol=0`.

ROCm-specific content preserved:

- `test_token_dispatcher.py`: ROCm's `dispatcher_dropless_multi_layer_multi_iter_test`
  (MORI op reuse across layers/iters), `is_mori_available`,
  `require_node_spanning_mori_ep`, `moe_mori_max_tokens_per_rank`. Upstream's
  `moe_layer_variant_parity_test` supersedes ROCm's `moe_layer_zero_copy_parity_test`
  (a rename + generalisation), so each was extracted whole from its own side.
- `test_submodule_callables.py`: upstream's combined parametrize and `flaky_in_dev`
  marks, but with ROCm's `_dispatcher_types_for_submodule_callables()` -- it drops
  `flex` when the backend is MORI, and upstream's bare `get_valid_token_dispatcher_types()`
  would re-enable a dispatcher that cannot share this file's EP group.
- `test_distributed_data_parallel.py`: ROCm's loosened `assert_close` (base had
  `rtol=0, atol=0`; ROCm removed it) plus upstream's new scaling-factor test class.
- `test_weighted_squared_relu_fusion.py`: ROCm's looser `weight_tols` (the fused jit
  path reorders the fp32 sum) plus upstream's new test.
- `test_param_and_grad_buffer.py`: ROCm's `mock.patch('torch.cuda.synchronize')` --
  the fork synchronizes unconditionally during buffer construction, which raises
  under a mocked CPU device.
- `mfsdp_v2/profiler_utils.py`: ROCm's skip when Kineto lacks `linked_correlation_id`,
  rebased onto upstream's `groups_by_correlation` (ROCm's `matching_correlations` was
  the stale name).
- `mfsdp_v2/test_fully_shard.py`: ROCm's `test_root_backward_returns_to_resting_memory`
  and `test_overlaps_communication_and_compute` re-appended whole; upstream's
  `test_hfsdp_reduce_scatters_dp_outer_on_last_microbatch` kept coherent.
- `test_cuda_graphs.py`, `test_hybrid_prefix_caching_e2e.py`: ROCm's
  `failing_on_rocm` markers kept alongside upstream's parametrize changes.
- `test_hybrid_block.py`: upstream renamed `HAVE_FLA` -> `HAVE_GDN`; ROCm's FLA skip
  was rebound to the new name.
- `conftest.py`, `test_dynamic_engine.py`: ROCm's `MAMBA_DETERMINISTIC` pin.

## Verification

A three-way audit over all **1088** merged python files -- definitions present in
`ifu` but NOT at the merge base (i.e. genuinely ROCm-added), checked against the
merged tree -- reports exactly two losses, both deliberate:

- `_metadata_signature` (MUST FIX 2 above: upstream deleted the whole file)
- `update_num_microbatches` in `training/global_vars.py` (upstream moved it to
  `megatron/core/num_microbatches_calculator.py`, which `training.py:83` imports)

Nothing else ROCm added was dropped. Note the audit's first form --
`defs(ifu) - defs(upstream)` -- was wrong: it conflates "ROCm added" with
"upstream deleted" and flagged 12 phantom losses in `test_dynamic_engine.py`
(two test classes that exist at the merge base and were deleted upstream).

No conflict markers remain and every existing python file parses.
**Still unvalidated:** nothing has been run. The ROCm CI and the MORI/DeepEP
owners' review are the next gates; the MoE resolutions in particular are a
proposal, not an answer.
