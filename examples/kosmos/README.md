# KOSMOS result regeneration

One script set that regenerates every KOSMOS result on one 8-GPU gfx950 node (MI350X or MI355X):

- the KOSMOS microbenchmarks (KOSMOS `bench/run_all.sh`: MoE forward / backward, cold and warm, against the PyTorch
  baseline (torch.distributed RCCL all-to-all + per-expert torch GEMMs); TP+SP every config, BF16 and MXFP8, against
  the PyTorch baseline in TE's non-overlapped call order (torch.distributed RCCL collectives + torch.mm /
  F.scaled_mm); forward gate and backward checks; optionally the KOSMOS unfused ablation and, internal only,
  MegaMoE);
- the Llama 3 8B / 70B / 405B TP+SP end-to-end runs;
- the Qwen3-235B and DeepSeek-V3 MoE (all-to-all) end-to-end runs, with every backend, in BF16 and MXFP8.

Megatron's printed end-of-run numbers are then collected into tables.

```
./run_all.sh manual                     # print the hand steps
./run_all.sh setup                      # verify (read-only)
./run_all.sh setup --build              # build KOSMOS (make all python) and TE in place against it
DRYRUN=1 ./run_all.sh micro tpsp a2a    # print every command, the run count and the GPU-time estimate
./run_all.sh micro tpsp a2a collect     # run
```

## Layout

| file | what |
|---|---|
| `config.sh` | every knob: repo paths, branches and commits, dependency paths, container names, steps' arms, repeats, ports, guards. Any variable can be set in the environment |
| `run_all.sh [STEP [--flag]]...` | the driver. Default `setup micro tpsp a2a collect`. Each step can run alone |
| `steps/{manual,setup,micro,tpsp,a2a,collect}.sh` | the steps |
| `tools/llama_run.sh`, `tools/moe_run.sh`, `tools/primus_run.sh` | one training run inside a container: hard timeout, no-log-growth watchdog (rc 86), sweep of leftover ranks |
| `tools/routing_stats.sh` | a run's measured routing load and KOSMOS capacity need |
| `tools/gen_scripts.sh` | derived copies of this tree's train scripts, written to `OUT_ROOT` (see "Not on the branches") |
| `lib/common.sh` | `DRYRUN`, container exec, GPU-idle and host-load guards, optional `flock`, shuffle, ports, plan |

`config.sh` first reads a site file, `KOSMOS_SITE` (default `~/.config/kosmos/site.sh`), for node-local paths and
container names; without one, the defaults are the three trees side by side, dependencies under `~/kosmos_deps`,
containers `kosmos` and `primus`, results under `~/kosmos_results/<GPU>` (the GPU name, e.g. `MI350X`, is appended to
any `OUT_ROOT` that does not already end in it, so each GPU model keeps its own `runs.tsv`). GPU name, arch, count and
power cap are read from sysfs. Nothing writes into the three trees except `setup --build` (build outputs) and `setup --checkout`.

## A new node

The site file is outside the repo and holds whatever differs from the defaults. A minimal one:

```
# ~/.config/kosmos/site.sh
KOSMOS_IMAGE=<registry>/<TE CI image>   # required: TheRock ROCm 7.14, torch 2.12, Open MPI (no usable default)
TE_DIR=/path/to/TransformerEngine       # default: next to this Megatron-LM tree
KOSMOS_DIR=/path/to/KOSMOS              # default: next to this Megatron-LM tree
DEPS_ROOT=/path/to/kosmos_deps          # default: ~/kosmos_deps
OUT_ROOT=/path/to/kosmos_results        # default: ~/kosmos_results (<GPU> appended)
KOSMOS_CT=kosmos PRIMUS_CT=primus       # container names (KOSMOS_CT= empty: run on the host)
CT_MOUNTS="-v /path:/path"              # docker run volumes (default -v $HOME:$HOME); must cover all of the above
LOCK_FILE=                              # optional flock per GPU invocation
```

`DEPS_ROOT` contents (each path can also be set on its own, see `config.sh`; `setup` checks every one):

| path | what | made by |
|---|---|---|
| `deepep/` (`build.sh`, `pylib/deep_ep`) | Primus-Turbo intranode DeepEP | by hand, then `setup --build-deps` |
| `Primus-Turbo/` | Primus-Turbo `4b0f01b9`, built in place | `run_all.sh manual`, `setup --build-deps` |
| `megamoe/{pt_shim,pyenv,flydsl_autotune}` | MegaMoE shim + FlyDSL 0.2.4 (megamoe arm only) | `run_all.sh manual` |
| `apex_shim/fused_weight_gradient_mlp_cuda.py` | APEX stand-in on TE `general_gemm` (only for the optional GA-fusion runs, `TPSP_GA=1`) | by hand |
| `numa_bind.sh` | per-rank `numactl` wrapper (numa ablation only) | by hand |
| `huggingface/` | Qwen3-235B-A22B and DeepSeek-V3 tokenizers | `run_all.sh manual` |
| `Primus/`, `primus_hf/` | Primus `4969fd51` (Primus-workflow arm only) | `run_all.sh manual` |

Then: `./run_all.sh manual` (containers, clones, tokenizers), `./run_all.sh setup --build --build-deps` (KOSMOS
`make all python`, TE built in place against it with `NVTE_KOSMOS_ROOT` / `NVTE_KOSMOS_LIB`, DeepEP, Primus-Turbo),
`./run_all.sh setup` until it reports OK. The GPU-time estimates below are MI350X numbers.

## Steps

| step | what | GPU time (MI350X) |
|---|---|---|
| `manual` | prints the image pulls, `docker run` lines, clones, the MegaMoE shim and the tokenizer downloads | none |
| `setup` | checks each tree's branch and commit (git log / diff only), the KOSMOS build is current (`make -q`), the `make python` module exists, TE's `libtransformer_engine.so` contains the KOSMOS backend and was built against this KOSMOS, and every dependency. Exits non-zero on any problem. `--build`: `make -j all python` in KOSMOS, then `python setup.py build_ext --inplace` in TE with `NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=<arch> NVTE_KOSMOS_ROOT=<KOSMOS> NVTE_KOSMOS_LIB=<build>/libkosmos_tpsp.a`. `--build-deps`: DeepEP and Primus-Turbo. `--checkout`: switch a clean tree to its branch | none (CPU) |
| `micro` | KOSMOS `bench/run_all.sh` in `KOSMOS_CT`, three GPU invocations: `env md5 gate check` (forward gate, backward checks against the CPU reference); `moe` (`bench/moe/run.sh`: the directions and the routings of `MICRO_DISTS`, default `0 1`: balanced, and skewed = the bench's dist 1, the logits of the first E/8 experts, all of rank 0's, raised by log 3, so rank 0 receives 2.32-2.39x the mean rows and the hottest expert 2.37-2.47x the mean expert rows; per direction and routing a KOSMOS process (`moe_bench`) and a PyTorch baseline process (`baseline.py`, torchrun with `CT_PYTHON`), all in a random order, each walking the 7 shapes `MICRO_REPS=3` times, cold and warm; KOSMOS checked against the CPU reference and the baseline against an fp32 PyTorch reference in rep 0); `tpsp md5end` (`bench/tpsp/run.sh`, every config, BF16 and MXFP8, `REPS=3`, a KOSMOS and a PyTorch process per config and precision); then the report on the CPU, which joins the KOSMOS and baseline CSVs per shape. MoE and TP+SP alike: per iteration the slowest rank, the median over iterations, the median over the reps. `MICRO_ABLATIONS` (default `unfused`; empty leaves it out): KOSMOS unfused as one more arm (MoE and TP+SP), in the results' Ablations section. `MICRO_MEGA=1` (internal only): `run_all.sh mega` in `PRIMUS_CT` (Primus-Turbo does not import in the KOSMOS container), `MICRO_REPS` sweeps per direction, routing and shape, on the bench's routing files (`bench/megamoe/gen_routing.sh`), added to the results' MoE tables. Output `OUT_ROOT/micro/<GPU>.md` (`MICRO_RESULTS_IN_TREE=1`: KOSMOS `results/<GPU>.md`) | ~80 min, 3 invocations, unfused arm included (balanced only, `MICRO_DISTS=0`: ~55 min); `MICRO_ABLATIONS=` -11 min (balanced -7); `MICRO_MEGA=1` +18 min, 1 invocation. Estimates: these runners are not yet timed on a GPU |
| `tpsp` | Llama 3 BF16, `train_llama3.sh`, SEQ 8192, TP8, 10 iterations, GBS 64 everywhere: 8B MBS 1/2/4/8 (32 layers); 70B MBS1/80L, MBS2/80L, MBS4/60L, MBS8/50L; 405B MBS1/2/4/8 at 24 layers (`RECOMPUTE=1`). Arms `base_noovl` (no overlap) and `kos_tpsp` (KOSMOS for every overlapped op), 3 runs each, order shuffled per (config, GA, repeat). GA fusion off (`TPSP_GA=0`, default). Optional GA-on set (`TPSP_GA="0 1"`, APEX stand-in): 8B MBS1 and the four 70B configs, both arms. Stall watchdog 600 s, 1200 s for 405B (`TPSP_STALL_S_405`; its GBS-64 iterations take ~4-8 min). Hang policy (`TPSP_HANG_SKIP=1`): a run that stalls (rc 86), hits the hard timeout, or fails with an out-of-memory / launch-resource error stops its (model, arm, GA): its remaining repeats and every larger MBS are recorded as skipped rows ("skipped: smaller size hung/failed") and the other arms go on; on a rerun the state comes from `runs.tsv`. Any other failure is a failed row and the queue goes on | 81 runs, ~7.8 h: 8B 36 runs, 70B 36 runs, 405B 9 runs; the GA-on set +30 runs if the published failures repeat (`TPSP_EST_FAIL`: base_noovl hangs at MBS1, kos_tpsp completes MBS1/2 and crashes at MBS4); hang arms +4 runs, ~40 min (each hangs at its first MBS) |
| `a2a` | Qwen3-235B proxy (24 layers, recompute 3) and DeepSeek-V3 proxy with `MOE_LAYER_FREQ=1` (3 MoE layers), BF16 and MXFP8 (see "MXFP8 MoE runs"), `TRAIN_ITERS=10`, `PROFILE=false`, GA fusion off (`GA_FUSION=false`). `GEMM_TUNING=0` in every arm (the scripts force it off under TE grouped GEMM). Arms: `kosmos` (WGCAP 256, DDP gate, fused router), `grouped` (TE grouped GEMM + all-to-all), `deepep`, `seq` (SequentialMLP), `pt_rccl` (PyTorch + RCCL), `megamoe` (internal table only, never published); 3 runs each. Routing (`A2A_ROUTING=balanced\|skewed\|both`, default both): balanced = the proxies' `FORCE_BALANCE=true`; skewed = natural, `FORCE_BALANCE=false` (Megatron's router and aux loss from initialization, the fused router for KOSMOS; skew unknown before a run). Skewed runs record their routing (`MOE_ROUTING_STATS`) and set the KOSMOS arena (`KOSMOS_CF_natural_<model>`: 4.1 / 3.0). Each round runs every (model, arm, routing, precision) once, shuffled. Precisions (`A2A_PRECISIONS`, default `bf16 mxfp8`): every arm in BF16; in MXFP8 the arms of `A2A_MXFP8_ARMS` (`kosmos grouped deepep seq megamoe`; `pt_rccl` cannot run MXFP8 and is refused). Optional (BF16 and balanced only): `A2A_ABLATIONS` (`kosmos:rtroff`, `kosmos:gateoff`, `kosmos:noovl`, `<arm>:numa`), `A2A_PRIMUS=1` (Primus's own workflow with MegaMoE, `--train_iters 10`; DSv3 with 3 MoE layers, recompute 1, no MTP) | 144 runs, ~3.4 h: BF16 84 runs (balanced 42, natural skew +42; `megamoe` 12 of them), MXFP8 60 runs (5 arms; balanced 30, natural +30; estimated as BF16, not yet timed); `A2A_PRECISIONS=bf16` 84 runs, ~2.0 h; each ablation +6 runs, ~8 min; Primus workflow 6 runs, ~10 min |
| `collect` | `OUT_ROOT/RESULTS_TPSP.md`, `RESULTS_A2A.md` (no MegaMoE, no ablations) and `RESULTS_A2A_internal.md`: per configuration and arm the median of the valid runs of Megatron's printed TFLOP/s/GPU, ms/iter, tokens/GPU/s (and mem), the ms/iter min / max, the valid-run count (fewer than 3 is marked), the speedup against KOSMOS from the medians, then every run with its iteration-1 loss / grad norm, run-window host load and log path. A2A: one section per precision (BF16, MXFP8) and routing (balanced, skewed natural), speedups against KOSMOS at the same precision, each skewed table with the routing the runs measured (busiest rank and hottest expert over their means, the KOSMOS capacity factor needed and set; `tools/routing_stats.sh`). TP+SP: skipped and failed configs carry their note (hang, timeout, oom, skipped). Valid = exit 0, all 10 iterations and the summary printed, mean host load below `LOAD_MAX`, and for KOSMOS every rank's exit status report ok (no over-capacity call) and, in MXFP8, no MoE layer that fell back to bf16 experts. Every A2A run lists the grouped-GEMM fallback lines TE printed (TE grouped GEMM arms) | none |

Total with the defaults (`DRYRUN=1 ./run_all.sh micro tpsp a2a`): about 12.3 h of GPU time (228 GPU invocations:
micro 69 min, tpsp 7.8 h, a2a 3.4 h); balanced only (`MICRO_DISTS=0 A2A_ROUTING=balanced`) about 10.2 h (156); without
405B about 10.4 h; BF16 A2A only (`A2A_PRECISIONS=bf16`) 1.4 h less (60 runs). `DRYRUN=1` prints each step's count and estimate.
The estimates are MI350X wall times per run at 10 iterations, measured on the published 20-iteration runs: first log
line to iteration 10 plus the teardown after the last iteration, plus 0.5-0.6 min of runner overhead per run (container
exec, poll interval, `POST_RUN_SLEEP_S`; the guard waits are 0 on an idle node). Skewed runs are estimated as the
balanced ones. TP+SP `base_noovl` runs 10-35% longer than the overlapped arms and has its own estimate. 405B never
completed on MI350X: a completing run is estimated from its hard timeout ((timeout - 600 s) / 3, 3.3 to 12.4 min), and
the DRYRUN plan applies the hang policy to the published failures (`TPSP_EST_FAIL`): the overlapped arms crashed (MBS
1/2/8 within 1-2 min, MBS4 after 3-9 iterations) and `base_noovl` hung (4.4 min at the 180 s watchdog, 11.4 at the
former 600 s).

## Skewed routing and the KOSMOS arena

- KOSMOS reserves `KOSMOS_CAPACITY_FACTOR x T x k` arena rows per layer and rank (adapter default 1.5). A call that
  routes more rows to one rank (each expert padded to 256) gets an empty layout and undefined outputs, and sets a status
  bit on the device that stays set until read. The adapter reads every layer's status once at exit (one device sync)
  and each rank prints `[KOSMOS] plan status rank R: ok ...` or `... OVERFLOW ...`; `collect` marks a KOSMOS run
  invalid unless all 8 ranks printed ok.
- Every skewed run writes its routing (`MOE_ROUTING_STATS`: tokens per expert per router call and rank) to
  `<run>/routing`; `collect` reports the measured skew and the factor each run needed, which sets
  `KOSMOS_CF_natural_<model>` once the natural skew is known. Natural Qwen3: 4.1, which just covers the measured need
  of at most 4.07 (10 iterations; the routing collapses onto one expert by iteration 3) with no margin, at mem usage
  ~0.97 (MI350X); the ceiling (every token's experts on one rank) is 8.07, ~247 GB of KOSMOS memory, which does not
  fit. Natural DSv3: 3.0 (needs 1.27).

## MXFP8 MoE runs

- `PR=fp8 FP8_RECIPE=mxfp8` (`--fp8-recipe mxfp8 --fp8-format e4m3`, every layer FP8, `NVTE_ROCM_ENABLE_MXFP8=1` set by
  the scripts). Tags and logs carry the precision: `<model>_<arm>_mxfp8[_natural]_r<n>`, `train_mxfp8.log`; `runs.tsv`
  has it as its last column.
- `kosmos`: `KOSMOS_MXFP8=1`, the four expert GEMMs in MXFP8 inside the KOSMOS launches; MXFP8 layers use Megatron's
  router and shared expert (the fused router is BF16 only). Needs the `make python` module with `quantize_mxfp8`
  (`setup` checks it); a layer that falls back to bf16 experts prints `[KOSMOS] layer N: bf16 experts`, which makes the
  run invalid.
- `grouped`, `deepep` (TE grouped GEMM): `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1 NVTE_USE_CK_GROUPED_GEMM=0` (the
  HipKittens MXFP8 grouped GEMM; CK=1 alone would turn it off). `NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1` in these
  arms at both precisions, so every HipKittens decline (`[HK-grouped]` / `[HK-wgrad] falling back: ...`) and CK ->
  hipBLASLt fallback is in the log; `collect` counts them per run.
- `seq`: SequentialMLP on TE's MXFP8 linears (dense GEMMs, experts padded to 128 rows by Megatron).
- `megamoe`: Primus-Turbo's staged MXFP8 op (`fused_mega_moe_fp8_stage1` / `stage2`, Primus-Turbo 30e6de87 through
  `MEGAMOE_SHIM_MX`); the weights stay BF16 parameters and the op's MXFP8 weight cache is invalidated once per iteration.
- `pt_rccl` (Megatron's local linears stay BF16) has no MXFP8 run: an MXFP8 row for it would be a BF16 run under an
  MXFP8 label.
- A2A runs use the scripts' `GPU_MAX_HW_QUEUES=2` (`A2A_HWQ_ALL` overrides it). On the rebased `cosmic_crisp` Megatron,
  DeepEP and MegaMoE hung with 2; on the old base (`c6bf60ce7`) they did not. TP+SP keeps 1 for 405B `base_noovl` only
  (`TPSP_HWQ`).
- Expert padding. HipKittens needs every expert's rows to be 0 or a multiple of 256 (fprop N, wgrad M >= 256). The
  scripts pass `--moe-router-padding-for-quantization` under mxfp8, which pads the routing to Megatron's MXFP8 alignment,
  128, and skips TE's `Fp8Padding`; an expert with an odd multiple of 128 rows then makes HipKittens decline and CK MX
  run the call. With `A2A_MXFP8_ROUTER_PAD=0` (default) the MXFP8 runs use the derived scripts (`tools/gen_scripts.sh`,
  `MOE_ROUTER_PADDING_FOR_QUANT=false`) without that flag: Megatron's TE grouped MLP then pads each expert after the
  dispatch with TE's `Fp8Padding`, whose MXFP8 alignment is 256 under `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` on gfx950
  (zero rows, probability 0; no padded tokens travel in the all-to-all). The same scripts are used for every MXFP8 arm.
  `A2A_MXFP8_ROUTER_PAD=1` runs the scripts as they are.
- Ablations and the Primus workflow are BF16 only. KOSMOS's skewed-routing arena (`KOSMOS_CF_natural_<model>`) is the
  BF16 one.

## 10-iteration runs

- Llama (`train_llama3.sh`): the end-of-run lines average iterations 3 to 9 (`lines[2:-1]`), 7 iterations (3 to 19 at
  20). The LR schedule does not depend on `TOTAL_ITERS` (no warmup, decay over 320000 iterations).
- Qwen3 / DSv3 (`--skip-first 2`, profiler steps skipped only with `PROFILE=true`): iterations 3 to 10, 8 iterations (3 to
  11 and 14 to 20 at 20). `PROFILE=false` is set after the proxy: the proxies' profiler window (12-13) lies past
  iteration 10.
- Qwen's LR schedule is unchanged (the proxy sets `LR_DECAY_ITERS=320000`). DSv3's script sets
  `LR_DECAY_ITERS=TRAIN_ITERS-2` itself (the proxy's `null` is overwritten), so its decay is 8 iterations instead of 18;
  iteration 1 is inside the 2-iteration warmup in both, so iteration-1 loss / grad norm stay comparable to the 20-iteration
  runs.

Every GPU invocation first waits until no process on the node holds a GPU context (`/sys/class/kfd/kfd/proc`) and the
host 1-min load average is below `LOAD_MAX` (20); both waits are logged, and the load is sampled during every step so
that `collect` keeps only runs whose mean load stayed below `LOAD_MAX`. `LOCK_FILE` adds one `flock` per invocation.
A completed valid run is skipped when a step is rerun. Ports are checked free per run.

## By hand

- Containers: the TE CI image (TheRock ROCm 7.14, torch 2.12, Open MPI) and `rocm/primus:v26.3`, with this directory,
  the trees, `DEPS_ROOT` and `OUT_ROOT` mounted at the same path; TE installed editable from `TE_DIR` in the first.
- Clones and network installs: Primus-Turbo `4b0f01b9`, Primus `4969fd51` (only for the Primus-workflow arm), the tokenizers, `numactl` (only for the NUMA
  ablation).
- The MegaMoE shim in the KOSMOS container (`MEGAMOE_SHIM`, `MEGAMOE_PYENV`), the APEX stand-in (`APEX_SHIM_DIR`) and
  the NUMA wrapper (`NUMA_BIND_SCRIPT`).

`./run_all.sh manual` prints the commands.

## Not on the branches

- `setup` checks the trees have: Megatron `moe_utils.py` (`MOE_ROUTING_STATS`), `router.py` (records the routing)
  and `kosmos_moe.py` (exit status report, routing record); KOSMOS `bench/moe/run.sh` and the `moe` step of
  `bench/run_all.sh` (the MoE runner), and the PyTorch baselines `bench/moe/baseline.py`, `bench/tpsp/baseline.py`.
- `train_llama3.sh` has no `NUM_LAYERS_OVERRIDE` (the 70B MBS4 / MBS8 configs use 60 / 50 layers), and its
  `GRADIENT_ACCUMULATION_FUSION` is not tied to the kit's `GA_FUSION`. `tools/gen_scripts.sh llama` writes
  `OUT_ROOT/tpsp/train_llama3_e2e.sh` with `NUM_LAYERS_OVERRIDE` and `GRADIENT_ACCUMULATION_FUSION=${GA_FUSION:-0}` (GA
  fusion off unless `GA_FUSION=1` with the APEX stand-in; the container APEX has no `fused_weight_gradient_mlp_cuda`).
  The script's other default-on toggles (DP-average in the collective, CE fusion, fused QKV+RoPE, BF16 grad reduce) stay
  on, so these runs are not like-for-like with the published ones, which predate those toggles.
- The `noovl` ablation needs `DDP_OVERLAP=false`, and the MXFP8 runs `MOE_ROUTER_PADDING_FOR_QUANT=false` (no
  `--moe-router-padding-for-quantization`), which the MoE train scripts do not carry: `tools/gen_scripts.sh qwen|dsv3`
  writes copies with both switches, `OUT_ROOT/a2a/e2e_train_qwen3.sh` and `e2e_train_deepseekv3.sh` (with neither set
  they run as the originals). The `numa` ablation uses the scripts' `PRETRAIN_SCRIPT` as is.
- The no-overlap TP+SP baseline (`TP_COMM_OVERLAP=0`) runs on the KOSMOS TE build; set `TE_BASE_DIR` to a stock dev
  build to run it on that instead. TE's own overlap path is not run.
