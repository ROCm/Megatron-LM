<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
     License for AMD contributions = MIT. See LICENSE for more information -->

# KOSMOS benchmark run on MI355X: instructions for the run agent

This is the complete KOSMOS benchmark: the microbenchmarks, Llama 3 TP+SP end-to-end and MoE (A2A) end-to-end, all in
BF16 and MXFP8, on one 8x MI355X node, plus two ablations. It is the same protocol as the finished MI350X run
(2026-10-22/23). Follow it exactly; don't change settings to make something pass. If something fails, stop that step,
collect the evidence (section 7) and report.

## 1. Rules

- **One GPU job at a time.** Pass one lock file to the kit (`LOCK_FILE=<lockfile>`; the kit takes it around every GPU
  invocation) and take the same lock for anything else you run on the GPUs (`flock <lockfile> <command>`). Check
  that no other process holds a GPU context first (`ls /sys/class/kfd/kfd/proc` is empty). Never kill processes you
  did not start.
- **Quiet node.** Record foreign GPU or CPU load during the runs. For example, every 30 s log any KFD process outside
  your two containers and any non-container process above one core. Note the times, so that affected runs can be
  identified afterwards.
- **Do not change the protocol.** The kit defaults are the protocol; set nothing in section 5 beyond what is shown.
  - End-to-end: 20 iterations, 1 run per (config, arm) (`E2E_ITERS=20`, `E2E_REPS=1`, the defaults). ms/iter is
    the mean over iterations 3-19 (Llama) / 3-20 (MoE).
  - A2A arms: KOSMOS, Megatron base (TE grouped GEMM + all-to-all, i.e. RCCL) and DeepEP (`A2A_ARMS` default). No
    MegaMoE anywhere (`MICRO_MEGA=0`, the default). MORI is not part of the run.
  - GA fusion stays off. `GEMM_TUNING=0`. No hipBLASLt algorithm pins (`TE_HIPBLASLT_ALGO_LOAD` etc.).
  - GBS 64 for Llama. The 405B runs use 24 layers.
  - Hardware queues: `GPU_MAX_HW_QUEUES=2`, except every 405B arm uses 1 (`TPSP_HWQ_405=1`, the MI350X
    single-rank grad reduce-scatter hang workaround; kept on MI355X so the protocol is identical).
  - MoE routing: Qwen natural routing uses aux-loss coefficient 1e-2 for every arm (`A2A_AUX_natural_qwen`); the
    KOSMOS arena is 4.1 (BF16) / 5.5 (MXFP8) for Qwen natural and 3.0 for DSv3 natural (`KOSMOS_CF_*`). DSv3 uses
    top-k scaling factor 2.5 (in `train_deepseekv3.sh`).
- **Megatron trees.** TP+SP runs from `cosmic_crisp`. MoE (A2A) runs from `alemagro/kosmos_oldbase`, which carries the
  same KOSMOS integration on the older Megatron base `c6bf60ce7`. On the rebased base, the TE grouped-GEMM and DeepEP
  MoE baselines hang or run 2.5-3.5x slower. The kit takes one tree per step (`TPSP_MLM_*` and `A2A_MLM_*` in
  `config.sh`), so nothing is swapped or rebuilt between steps.
- **No commits or pushes** unless the user asks. Results are files under `OUT_ROOT`.

## 2. Code

| repo | branch | notes |
|---|---|---|
| KOSMOS | `main` (`AMD-BRAIN-Internal/KOSMOS`) | `3rdparty/hipkittens` submodule |
| TransformerEngine | `alemagro/kosmos_final` | `origin/dev` merged + the KOSMOS dispatch + the HipKittens MXFP8 grouped-GEMM workspace chunking; built in place against KOSMOS (`NVTE_KOSMOS_ROOT`, `NVTE_CK_JIT=0`) by `setup --build` |
| Megatron-LM | `cosmic_crisp` | TP+SP; this kit is `examples/kosmos/` |
| Megatron-LM | `alemagro/kosmos_oldbase` | MoE; second worktree of the same clone |

```
git clone --recursive -b main <KOSMOS url> KOSMOS
git clone -b alemagro/kosmos_final <TE url> TransformerEngine && git -C TransformerEngine submodule update --init --recursive
git clone -b cosmic_crisp <Megatron-LM url> Megatron-LM
git -C Megatron-LM fetch origin alemagro/kosmos_oldbase
git -C Megatron-LM worktree add ../Megatron-LM-oldbase alemagro/kosmos_oldbase
```

`config.sh` pins the oldest accepted commit of each repo (`*_COMMIT`); `setup` refuses a tree older than its pin.
Record the commit of every repo (`git log -1 --format=%H`) and keep it with the results.

## 3. Containers and site file

`./run_all.sh manual` prints every hand step: the second Megatron worktree, the dependency clones (Primus-Turbo
`4b0f01b9`, which DeepEP is built from, and `30e6de87`), the image pulls, the `docker run` lines, `transformers` in
the KOSMOS container and the tokenizer downloads. Run them all in order, including the MegaMoE-related clones:
`setup --build-deps` builds them unconditionally, although no MegaMoE arm runs. `setup --build-deps` then builds the
rest.

- **KOSMOS container:** the TE CI image (TheRock ROCm 7.14, torch 2.12, Open MPI).
- **Primus container:** `rocm/primus:v26.3` (DeepEP is built and run there).
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
./run_all.sh setup --build --build-deps    # KOSMOS (+ python module), TE in place + editable, Primus-Turbo, DeepEP
./run_all.sh setup                          # must end with "setup: OK"
DRYRUN=1 ./run_all.sh                       # prints the plan: micro 3, tpsp 48, a2a 24 GPU runs
```

(The dry-run plan models the MI350X outcomes, so it may list the two 405B MBS 8 KOSMOS runs as skipped; the real
run attempts them.) Also record the node's power cap per GPU and the ROCm / amdgpu versions (`amd-smi`,
`rocm-smi --showpower`).

## 4. Validation before the full run (about 1 h)

Do these in order, each under the lock. Only continue when each one passes.

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
   OUT_ROOT=<scratch> LOCK_FILE=<lockfile> TPSP_CONFIGS="8:1:64:32:1450:2.6:3.0" ./run_all.sh tpsp
   ```
   - All 4 runs reach iteration 20 with rc 0.
   - `grep '\[KOSMOS\]' .../output_perf.log | sort -u` in the kos_tpsp runs shows every overlapped op on KOSMOS: in
     MXFP8 every line ends `KOSMOS -- MXFP8`; nothing reads `HipKittens` or `non-overlapped`.
3. **MoE smoke** (every arm once, balanced routing, BF16 and MXFP8; then the arms most likely to hang on skewed
   routing):
   ```
   OUT_ROOT=<scratch> LOCK_FILE=<lockfile> A2A_ROUTING=balanced ./run_all.sh a2a
   OUT_ROOT=<scratch> LOCK_FILE=<lockfile> A2A_ROUTING=skewed A2A_ARMS="grouped deepep" ./run_all.sh a2a
   ```
   - Every run reaches iteration 20 with rc 0.
   - KOSMOS prints `[KOSMOS] plan status rank R: ok` on all 8 ranks; in MXFP8 it says MXFP8 experts and fused
     router, never bf16 experts.
   - If a run stalls (rc 86, the log silent for minutes, GPUs at 100%), report it with the last 50 log lines. Do not
     switch hardware-queue settings on your own.
4. **Microbenchmark gate:** `./run_all.sh micro` runs the forward gate and the backward and training checks first
   (BF16 and MXFP8). Every line must read `FWD GATE PASS` / `CHECK PASS` / `TRAIN CHECK PASS`. (This is the start of
   the full micro step; if it passes you can let it continue as part of section 5.)

## 5. The full run

```
cd Megatron-LM/examples/kosmos
SEED=<any fixed integer, recorded> LOCK_FILE=<lockfile> ./run_all.sh micro tpsp a2a collect > run_$(date +%Y%m%d_%H%M).log 2>&1
```

- **Resuming:** the kit skips any run whose log already has its summary. After an interruption, rerun the same
  command; it resumes.
- **Measured on MI350X** (1000 W; MI355X should be the same or faster): micro 2.25 h, TP+SP 8.0 h, A2A 0.5 h,
  about 10.75 h in total; plus about 0.5 h for the two ablations (section 8).

**Outputs** (`OUT_ROOT/<GPU>/`):
- `micro/<GPU>.md` and `micro/moe/SUMMARY_final.md`;
- `RESULTS_TPSP.md`;
- `RESULTS_A2A.md` and `RESULTS_A2A_internal.md`;
- the per-run logs under `tpsp/runs/` and `a2a/runs/`, and the `runs.tsv` files.

## 6. Known behaviors (from MI350X; not failures)

- **Llama 405B runs may exit with rc 1 after the summary** (a segfault at process exit). Collect counts a 405B run as
  valid when all iterations and the summary were printed.
- **405B MBS 8 MXFP8 KOSMOS ran out of memory on MI350X** (`torch.OutOfMemoryError` at iteration 1; the HBM is the
  same 288 GB on MI355X). Collect shows it as `OOM`. Report it; do not change anything to make it fit.
- **Qwen natural routing is extreme and differs by arm:** each arm trains its own router from initialization. On
  MI350X the busiest rank reached 4.0x (KOSMOS BF16), about 5.0x (MXFP8, every arm) and 5.8-6.0x (BF16 baselines)
  the mean rows. KOSMOS BF16 at 5.1 and 5.5 ran out of memory, so BF16 stays at 4.1. A KOSMOS run whose plan status
  says `OVERFLOW` is invalid; collect prints the arena each run needed. Report an overflow; do not change the arena.
- **The TE MXFP8 grouped arms may log `[HK-grouped] non-contiguous activation/output ... falling back`.** With this
  TE build the NT (wgrad) calls should not print it; any count is reported in the "HK grouped-GEMM notices" column
  and is not a validity criterion.
- **The BF16 PyTorch MoE baseline (micro) may print `HIPBLAS_STATUS_INTERNAL_ERROR ... recover by calling cublas`**
  for some kimi27 and dsv4pro GEMMs (MI350X); the micro report counts them as fallback lines. Report the counts.

## 7. When something fails

- **Evidence:** keep the run directory. Report the tag, rc, last iteration printed, the first error line per rank
  (`grep -a 'Error\|Traceback' | sort | uniq -c`) and, for a stall, the last 50 lines.
- **Retrying:** a failed run is retried by moving its directory aside and deleting its row from `runs.tsv`, then
  rerunning the step. Do this only after the cause is understood.
- **Out of memory:** don't change batch sizes, layer counts or arena factors to work around an OOM; report it.
- **Report back:**
  - the results files of section 5;
  - the commits of all repos;
  - the node info (power cap, ROCm/amdgpu versions);
  - the foreign-load log;
  - a list of every run that was invalid or retried, and why.

## 8. Ablations (after the full run)

Only two: clock/power and role policy. Run them from the host after section 5 finishes, under the same lock:

```
cd KOSMOS
CT=<KOSMOS container> BUILD=1 ONLY="clock policy" flock <lockfile> ablations/run_paper_ablations.sh
```

- This builds each ablation, then runs clock `time` (GPU 0 only) and policy `peak check time summary` (`peak`
  re-measures C_peak, the push curves and alpha on this node; the MI350X values do not carry over). Results go to
  `KOSMOS/ablations/<name>/results/<STAMP>/` and the queue log to `KOSMOS/work/paper_ablations_<STAMP>.log`.
- Copy each `results/<STAMP>` directory back as `ablations/<name>/results_MI355X/<STAMP>/`.
- **clock:** the MI355X power cap is 1400 W (MI350X: 1000 W). Record `amd-smi` power cap per GPU before the run.
- **Ownership:** results are chowned to the owner of the KOSMOS checkout.
- Every phase must end without a FAIL line; the queue log prints `rc=` per ablation and `failed: none` at the end.
  Report the same way as section 7.

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

**Stuck** means no new line in the run log for more than 60 min (the longest single run took 20 min on MI350X). Then
look at the current run's `output_perf.log` and `runner.log` and report; don't kill or relaunch on your own.

**Measured on MI350X** (1000 W), per run from start to end, 20 iterations:

| step | per run, mean / max | runs | step total |
|---|---|--:|---|
| micro (BF16 + MXFP8: gate, checks, MoE, TP+SP) | - | - | 2.25 h |
| TP+SP 8B, BF16 / MXFP8 | 3.2 / 4.0 min, 2.8 / 3.5 min | 8 + 8 | |
| TP+SP 70B, BF16 / MXFP8 | 12.8 / 16.5 min, 9.5 / 13.2 min | 8 + 8 | |
| TP+SP 405B (24 layers), BF16 / MXFP8 | 18.7 / 20.0 min, 11.3 / 13.3 min | 8 + 8 | |
| TP+SP all | | 48 | 8.0 h |
| MoE (A2A), all arms, BF16 / MXFP8 | 1.1 / 1.3 min, 1.2 / 1.7 min | 24 | 0.5 h |
| ablation clock | | | about 8 min |
| ablation policy (peak, check, time, summary) | | | about 15-30 min |

A run that takes more than 2x its row's maximum is worth a look in the next health check; it is not by itself a
reason to stop the step.

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
  phase (checks, moe, tpsp). If the foreign process overlapped one, rerun `./run_all.sh micro` (about 2.25 h); it
  rewrites the micro results.
- **Ablations.** If it overlapped one, rerun that ablation alone (`ONLY=clock` or `ONLY=policy`, without `BUILD=1`).
- **Report it.** List the moved runs, the foreign process (PID, user, container or cgroup, command) and its time
  window. Keep the contaminated directories; do not delete them.
