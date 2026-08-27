# Deletion requests — awaiting review

`decision` · standing instruction from 2026-08-27: **do not remove files.**
Anything that looks like it should go gets listed here instead, with a reason,
for a human to decide.

## Already removed (before the instruction arrived) — disclosed

| path | what it was | why | when |
|---|---|---|---|
| `/workspace/kimi_k3/` (outside the repo) | an empty package dir containing only an `__init__.py` I had just written by mistake, plus its `__pycache__` | I ran `mkdir -p kimi_k3/attention` while the shell's cwd had reset to `/workspace` instead of `/workspace/Megatron-LM`, creating a stray tree outside the repository. It was never tracked by git, never part of the project, and contained nothing but the file I had accidentally put there seconds earlier. Removed to keep the real package at `Megatron-LM/kimi_k3/`. | 2026-08-27, ~05:14 |

Nothing under `Megatron-LM/` was touched, and nothing tracked by git was removed.

## Proposed for deletion — none

No files are currently proposed for removal.

## Candidates a human may want to prune later (not deletions I am asking for)

| path | note |
|---|---|
| `/tmp/claude-0/.../scratchpad/k3ref/` | 60 MB of downloaded release artefacts (`model.safetensors.index.json` and the modeling sources). Outside the repo, scratch only, regenerable with `curl`. Left in place. |
| `/tmp/fla`, `/tmp/triton371` | reference clone and the isolated Triton used to test the upgrade. Outside the repo. Left in place. |

## 2026-08-27 — chrome trace, 241 MB (P11)

`kimi_k3/develop/profile/traces/proxy_ep4_4L_rank0.json` — 241 MB, produced by an
EP=4 proxy run that then OOM'd, so the trace is of an incomplete iteration and has
no value. It was briefly committed and rejected by GitHub's 100 MB limit; it is
now **untracked and gitignored**, but per the no-deletion rule it is still on
disk. Safe to delete. Any later `traces/` contents are the same: run artefacts,
with the report in `develop/profile/` being what is kept.
