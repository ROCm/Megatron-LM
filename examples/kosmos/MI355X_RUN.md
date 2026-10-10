<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
     License for AMD contributions = MIT. See LICENSE for more information -->

# KOSMOS benchmark run on MI355X: instructions for the run agent

This is the complete KOSMOS benchmark: the microbenchmarks, Llama 3 TP+SP end-to-end and MoE (A2A) end-to-end, all in
BF16 and MXFP8, on one 8x MI355X node. It is the same protocol as the finished MI350X run. Follow it exactly; don't
change settings to make something pass. If something fails, stop that step, collect the evidence (see "When
something fails") and report.

## 1. Rules

- **One GPU job at a time.** Run every GPU command under one lock file (`flock <lockfile> <command>`), and check that
  no other process holds a GPU context first (`ls /sys/class/kfd/kfd/proc` is empty). Never kill processes you did not
  start.
- **Quiet node.** Record foreign GPU or CPU load during the runs. For example, every 30 s log any KFD process outside
  your two containers and any non-container process above one core. Note the times, so that affected runs can be
  identified afterwards.
- **Do not change the protocol:**
  - GA fusion stays off.
  - `GEMM_TUNING=0`.
  - No hipBLASLt algorithm pins (`TE_HIPBLASLT_ALGO_LOAD` etc.).
  - 10 iterations and 3 runs per (config, arm).
  - GBS 64 for Llama.
  - The 405B runs use 24 layers.
  - MORI is not part of the run.
- **Megatron trees.** TP+SP runs from `cosmic_crisp`. MoE (A2A) runs from `alemagro/kosmos_oldbase`, which carries the
  same KOSMOS integration on the older Megatron base `c6bf60ce7`. On the rebased base, the TE grouped-GEMM and DeepEP
  MoE baselines hang or run 2.5-3.5x slower. The kit takes one tree per step (`TPSP_MLM_*` and `A2A_MLM_*` in
  `config.sh`), so nothing is swapped or rebuilt between steps.
- **No commits or pushes** unless the user asks. Results are files under `OUT_ROOT`.

## 2. Code

| repo | branch | notes |
|---|---|---|
| KOSMOS | `main` (`AMD-BRAIN-Internal/KOSMOS`) | `3rdparty/hipkittens` submodule |
| TransformerEngine | `alemagro/kosmos_final` | `cosmic_crisp` 4e331aff0 + the MXFP8 KOSMOS dispatch (the MI350X build); built in place against KOSMOS (`NVTE_KOSMOS_ROOT`) |
| Megatron-LM | `cosmic_crisp` | TP+SP; this kit is `examples/kosmos/` |
| Megatron-LM | `alemagro/kosmos_oldbase` | MoE; second worktree of the same clone. Local branch on the MI350X node, not pushed: push it (or fetch it from that clone) first |

```
git clone --recursive -b main <KOSMOS url> KOSMOS
git clone -b alemagro/kosmos_final <TE url> TransformerEngine && git -C TransformerEngine submodule update --init --recursive
git clone -b cosmic_crisp <Megatron-LM url> Megatron-LM
git -C Megatron-LM worktree add ../Megatron-LM-oldbase alemagro/kosmos_oldbase
```

Record the commit of every repo (`git log -1 --format=%H`) and keep it with the results.

## 3. Containers and site file

`./run_all.sh manual` prints every hand step: the second Megatron worktree, the dependency clones (Primus-Turbo
`4b0f01b9` for MegaMoE bf16 and DeepEP, Primus-Turbo `30e6de87` for MegaMoE MXFP8, Primus for the optional
Primus-workflow arm), the image pulls, the `docker run` lines, `transformers` in the KOSMOS container and the tokenizer
downloads. Run them in order. `setup --build-deps` then builds the rest (below).

- **KOSMOS container:** the TE CI image (TheRock ROCm 7.14, torch 2.12, Open MPI).
- **Primus container:** `rocm/primus:v26.3`.
- **Mounts:** both containers mount the home directory at the same path.

Write a site file (`~/.config/kosmos/site.sh`, or point `KOSMOS_SITE` at it) with this node's paths. The variables
are listed in `config.sh` and the README. At least:

```
KOSMOS_DIR=...  TE_DIR=...  DEPS_ROOT=...  OUT_ROOT=...
TPSP_MLM_DIR=<path>/Megatron-LM          TPSP_MLM_BRANCH=cosmic_crisp
A2A_MLM_DIR=<path>/Megatron-LM-oldbase   A2A_MLM_BRANCH=alemagro/kosmos_oldbase
KOSMOS_CT=<name>  KOSMOS_IMAGE=<TE CI image>  PRIMUS_CT=<name>
LOAD_MAX=0        # no pre-run host-load wait; record foreign load yourself (section 1)
GPU_NAME=MI355X   # results name (OUT_ROOT/<GPU>, KOSMOS GPU=), if sysfs does not already give it
```

Then:

```
cd Megatron-LM/examples/kosmos
./run_all.sh setup --build --build-deps    # KOSMOS (+ python module), TE in place + editable, Primus-Turbo x2,
                                           # DeepEP, the MegaMoE shims (bf16, MXFP8), FlyDSL 0.2.4
./run_all.sh setup                          # must end with "setup: OK"
```

Also record the node's power cap per GPU and the ROCm / amdgpu versions (`amd-smi`, `rocm-smi --showpower`).

## 4. Validation before the full run (about 1 h)

Do these in order. Only continue when each one passes.

1. **TE MXFP8 TP+SP on KOSMOS** (in the KOSMOS container, `tests/pytorch/distributed` of TE):
   ```
   export PYTHONPATH=$TE_DIR NVTE_ROCM_ENABLE_MXFP8=1 NVTE_USE_KOSMOS=1 NVTE_KOSMOS_BULK=1 NVTE_KOSMOS_LOG=1
   C="--seed=42 --seq-length=1024 --batch-size=2 --num-heads=32 --head-dim=48"
   torchrun --nproc_per_node=8 run_gemm_with_overlap.py $C --check-numerics --comm-type=AG --fused --p2p --quantization=mxfp8
   torchrun --nproc_per_node=8 run_layer_with_overlap.py $C --layer-type=LayerNormLinear --linear-parallel-mode=column \
       --num-layers=1 --use-bf16-params --no-bias --out-features=12288 --fp8 --quantization=mxfp8
   torchrun --nproc_per_node=8 run_layer_with_overlap.py $C --layer-type=Linear --linear-parallel-mode=row \
       --num-layers=1 --use-bf16-params --no-bias --in-features=12288 --fp8 --quantization=mxfp8
   python -m pytest -v test_rocm_fused_overlap.py
   ```
   - **Pass:** every check prints `NUMERICAL CHECK PASSED`, there is no `non-overlapped` line, and every op logs
     `KOSMOS -- MXFP8`.
   - **Determinism:** run the two layer tests twice; their `OUTPUT HASH` lines must match.
2. **Llama smoke** (8B MBS1, both arms, both precisions; a scratch `OUT_ROOT`):
   ```
   OUT_ROOT=<scratch> TPSP_CONFIGS="8:1:64:32:1450:2.6:3.0" E2E_REPS=1 ./run_all.sh tpsp
   ```
   - All 4 runs reach iteration 10 with rc 0.
   - `grep '\[KOSMOS\]' .../output_perf.log | sort -u` in the kos_tpsp runs shows every overlapped op on KOSMOS: in
     MXFP8 every line ends `KOSMOS -- MXFP8`; nothing reads `HipKittens` or `non-overlapped`.
3. **MoE smoke** (every arm once, balanced routing, BF16 and MXFP8; then the arms most likely to hang on skewed
   routing):
   ```
   OUT_ROOT=<scratch> E2E_REPS=1 A2A_ROUTING=balanced ./run_all.sh a2a
   OUT_ROOT=<scratch> E2E_REPS=1 A2A_ROUTING=skewed A2A_ARMS="grouped deepep megamoe" ./run_all.sh a2a
   ```
   - Every run reaches iteration 10 with rc 0.
   - KOSMOS prints `[KOSMOS] plan status rank R: ok` on all 8 ranks; in MXFP8 it says MXFP8 experts, never bf16
     experts.
   - On MI350X, DeepEP, MegaMoE and the qwen natural-routing grouped run completed with the scripts' default
     `GPU_MAX_HW_QUEUES=2` on this Megatron tree.
   - If a run stalls (rc 86, the log silent for minutes, GPUs at 100%), report it with the last 50 log lines. Do not
     switch hardware-queue settings on your own.
4. **Microbenchmark gate:** `./run_all.sh micro` runs the forward gate and the backward checks first (BF16 and
   MXFP8). Every line must read `FWD GATE PASS` / `CHECK PASS`.

## 5. The full run

```
cd Megatron-LM/examples/kosmos
MICRO_MEGA=1 flock <lockfile> ./run_all.sh micro tpsp a2a collect > run_$(date +%Y%m%d_%H%M).log 2>&1
```

- **MICRO_MEGA=1:** adds MegaMoE, bf16 (Primus-Turbo `PT_DIR`) and MXFP8 (`PT_MX_DIR`), to the microbenchmarks
  (internal comparison only).
- **Resuming:** the kit skips any run whose log already has its summary. After an interruption, rerun the same
  command; it resumes.
- **Expected time on MI350X** (MI355X should be faster; per-run times in section 9):
  - micro about 3.5 h;
  - TP+SP about 9 h BF16 and 5 to 6 h MXFP8;
  - A2A about 2.6 h;
  - about 21 h in total, plus about 1.5 h for the two ablations (section 8).

**Outputs** (`OUT_ROOT/<GPU>/`):
- `micro/<GPU>.md`;
- `RESULTS_TPSP.md`;
- `RESULTS_A2A.md` (public: no MegaMoE);
- `RESULTS_A2A_internal.md`;
- the per-run logs under `tpsp/runs/` and `a2a/runs/`, and the `runs.tsv` files.

## 6. Known behaviors (from MI350X; not failures)

- **Llama 405B runs exit with rc 1 after the summary** (a segfault at process exit). Collect counts a run as valid
  when all 10 iterations and the summary were printed.
- **405B no-overlap uses one hardware queue** (`TPSP_HWQ="405:base_noovl:1"`); with 2 it hangs at MBS 1.
- **qwen natural routing is extreme:** the busiest rank receives 4.0x the mean rows. The KOSMOS arena is 4.1x
  (`KOSMOS_CF_natural_qwen`); 4.5x ran out of memory in BF16. MegaMoE MXFP8 uses a pool of 5x
  (`MEGAMOE_POOL_natural_qwen`).
- **The TE MXFP8 grouped arms log `[HK-grouped] non-contiguous activation/output ... falling back`** about 480 times
  per run. Report the counts; collect lists them.
- **MegaMoE MXFP8 tunes a combine-kernel setting online on first use.** The microbenchmark locks it before timing;
  the e2e runs pin per-config values (`MEGAMOE_CU_*`).

## 7. When something fails

- **Evidence:** keep the run directory. Report the tag, rc, last iteration printed, the first error line per rank
  (`grep -a 'Error\|Traceback' | sort | uniq -c`) and, for a stall, the last 50 lines.
- **Retrying:** a failed run is retried by deleting its row from `runs.tsv` and rerunning the step. Do this only
  after the cause is understood.
- **Out of memory:** don't change batch sizes, layer counts or arena factors to work around an OOM; report it.
- **Report back:**
  - the four results files;
  - the commits of all repos;
  - the node info (power cap, ROCm/amdgpu versions);
  - the foreign-load log;
  - a list of every run that was invalid or retried, and why.

## 8. Ablations (after the full run)

Only two: role policy and clock/power. Run them in the KOSMOS container, one at a time under the same lock, after section 5 finishes. Each writes to
`KOSMOS/ablations/<name>/results*/`; copy those directories back as `ablations/<name>/results_MI355X/` (the figure
scripts read MI350X from `results/` and MI355X from `results_MI355X/`).

```
cd KOSMOS
export CT=<KOSMOS container>
ablations/clock/run.sh build && ablations/clock/run.sh check && ablations/clock/run.sh time    # GPU 0 only
ablations/policy/run.sh build peak check time summary     # peak re-measures C_peak, the push curves and alpha here
```

- **clock:** the MI355X power cap is 1400 W (MI350X: 1000 W). Record `amd-smi` power cap per GPU before the run.
- **policy:** `peak` must run on this node; the MI350X crew sizes, curves and alpha do not carry over.
- **Ownership:** the scripts chown results to uid 12377 (`OWNER` in `clock/run.sh`); set it to your uid if that
  user does not exist on this node.
- Every phase must end without a FAIL line; `check` phases exit non-zero on any failure. Report the same way as
  section 7.

## 9. Monitoring: health check and expected durations

**Health check (every 2 h while GPU work runs).** Never kill anything during a check; the kit's watchdogs handle
stalls. Per check:

1. Tail the run log (the `run_*.log` of section 5). It prints one `GPU <tag>` line per run.
2. `runs.tsv` under `OUT_ROOT/<GPU>/{tpsp,a2a}/`: count the rows added since the last check by return code and list
   every non-zero one with the first error line of its log (`grep -a -m1 'Error\|Traceback' <log>`).
   - `tpsp/runs.tsv` columns: date, tag, model, MBS, GBS, layers, arm, GA, rep, **rc (10)**, load, log, class (13),
     precision. `a2a/runs.tsv`: date, tag, model, arm, -, rep, **rc (7)**, load, log, routing, precision.
   - Valid: rc 0, and rc 1 for 405B runs that printed the summary (segfault at exit). `skip` rows are runs the hang
     policy skipped after a failure of the same (model, arm); report them.
   - rc 86 = stall (no log growth for 600 s, 1200 s for 405B), 124/137 = hard timeout, class `oom` = out of memory.
3. Foreign load: GPU processes outside your containers (`/sys/class/kfd/kfd/proc`) and CPU hogs, with times.
4. Append a 5-line status (time, current run, rc counts, foreign load, ETA) to a status file and report it.

**Stuck** means no new line in the run log for more than 60 min. Even the longest single run takes under
25 min on MI350X (table below). Then look at the current run's `output_perf.log` and `runner.log` and report;
don't kill or relaunch on your own.

**Expected durations, measured on MI350X** (1000 W; the MI355X at 1400 W should be the same or faster). Per run, time
between consecutive runs, so including start-up and teardown:

| step | per run (median / max) | runs | step total |
|---|---|--:|---|
| micro (BF16 + MXFP8, MoE and TP+SP) | - | - | about 3.5 h |
| TP+SP 8B, BF16 / MXFP8 | 2.2 / 3.5 min, 2.0 / 2.5 min | 24 + 24 | |
| TP+SP 70B, BF16 / MXFP8 | 8.5 / 10.0 min, 5.7 / 7.2 min | 24 + 24 | |
| TP+SP 405B (24 layers), BF16 / MXFP8 | 11.5 / 11.8 min, 7.1 / 7.2 min | 24 + 24 | |
| TP+SP all | | 144 | about 9 h BF16 + 5 to 6 h MXFP8 |
| MoE (A2A), all arms and precisions | 1.2 to 1.4 min (outliers 4 to 24 min) | 96 | about 2.6 h |
| ablation clock | | 90 processes | about 20 min |
| ablation policy (build, peak, check, time, summary) | | | about 1 h (MoE-only pass on MI350X: 25 min) |

Everything in sequence takes about 22 h. A run that takes more than 2x its row's maximum is worth a look in the next
health check; it is not by itself a reason to stop the step.

**A foreign GPU process during the run.** Never kill it (it is not yours). Note when it appeared and left (the
foreign-load log of section 1), let the current step go on, and afterwards rerun every run that overlapped that window:

- **End-to-end (TP+SP, MoE).** Each run directory `OUT_ROOT/<GPU>/{tpsp,a2a}/runs/<tag>/` has `t_start` and `t_end`
  (epoch seconds). A run overlaps if `t_start < foreign end` and `t_end > foreign start`. For each one:
  1. move its directory aside, e.g. to `OUT_ROOT/<GPU>/{tpsp,a2a}/contaminated/<tag>/` (the kit skips any run whose
     log has a summary, so leaving it in place means it is not rerun);
  2. delete its row from that step's `runs.tsv`;
  3. once the full run is done, rerun the same command (section 5) under the lock. Only the moved runs run again;
     then rerun `collect`.
- **Microbenchmarks.** These have no per-run resume. `find OUT_ROOT -name TIMES.txt` gives the start and end of each
  phase (checks, moe, tpsp). If the foreign process overlapped one, rerun `./run_all.sh micro` (about 3.5 h); it
  rewrites the micro results.
- **Ablations.** If it overlapped the clock ablation, rerun `ablations/clock/run.sh time`; it writes a new
  `results/time_*` directory and the figure reads the newest. If it overlapped the policy ablation, rerun
  `ablations/policy/run.sh time summary`.
- **Report it.** List the moved runs, the foreign process (PID, user, container or cgroup, command) and its time
  window. Keep the contaminated directories; do not delete them.
