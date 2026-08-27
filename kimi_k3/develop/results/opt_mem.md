# G5 — measured optimizer memory

> Produced by `kimi_k3/tools/opt_mem_probe.py`; regenerate with
> `kimi_k3/tools/opt_mem_report.py`. Bytes are per parameter **resident on
> one rank**, measured as the CUDA allocator delta from before model
> construction to after two optimizer steps, so masters, moments, gradient
> buffers and any all-gather scratch are all included.

Probe model: 99.0 M parameters (MLA + MoE + norms, bf16 weights, fp32 grad reduce), TP=1 PP=1 EP=1, world size = DP.

| recipe | DP | B/param (mean) | spread across ranks | param MB | grad MB | opt state MB | peak MB |
|---|---:|---:|---|---:|---:|---:|---:|
| `adam` | 1 | **18.02** | — | 191 | 378 | 1133 | 1734 |
| `adam` | 2 | **18.02** | 18.02 – 18.02 | 191 | 378 | 1133 | 1734 |
| `adam` | 4 | **18.02** | 18.02 – 18.02 | 191 | 378 | 1133 | 1734 |
| `adam` | 8 | **18.02** | 18.02 – 18.02 | 191 | 378 | 1133 | 1734 |
| `adam+distopt` | 1 | **18.02** | — | 191 | 378 | 1133 | 1734 |
| `adam+distopt` | 2 | **12.02** | 12.02 – 12.02 | 191 | 378 | 567 | 1168 |
| `adam+distopt` | 4 | **9.03** | 9.02 – 9.03 | 191 | 378 | 284 | 884 |
| `adam+distopt` | 8 | **7.52** | 7.52 – 7.53 | 191 | 378 | 142 | 758 |
| `adam+distopt+precaware` | 2 | **11.02** | 11.02 – 11.02 | 191 | 378 | 472 | 1073 |
| `adam+distopt+precaware` | 8 | **7.27** | 7.27 – 7.28 | 191 | 378 | 118 | 758 |
| `dist_muon` | 1 | **15.17** | — | 191 | 378 | 864 | 1465 |
| `dist_muon` | 2 | **11.00** | 10.49 – 11.51 | 191 | 378 | 470 | 1228 |
| `dist_muon` | 4 | **8.91** | 8.66 – 9.17 | 191 | 378 | 273 | 1034 |
| `dist_muon` | 8 | **7.87** | 7.65 – 8.25 | 191 | 378 | 174 | 935 |
| `muon` | 1 | **15.17** | — | 191 | 378 | 864 | 1465 |
| `muon` | 8 | **15.17** | 15.17 – 15.17 | 191 | 378 | 864 | 1465 |

## Optimizer state split (mean per rank, MB)

> The 2-D column is empty for the distributed optimizer by construction: it
> keeps state in flattened shard buffers, so the parameters its `state` dict
> is keyed on are 1-D views rather than the original matrices. Read the
> split only for the non-distributed recipes.

| recipe | DP | fp32 masters | state on 2-D params | state on other params |
|---|---:|---:|---:|---:|
| `adam` | 1 | 378 | 756 | 0.1 |
| `adam` | 2 | 378 | 756 | 0.1 |
| `adam` | 4 | 378 | 756 | 0.1 |
| `adam` | 8 | 378 | 756 | 0.1 |
| `adam+distopt` | 1 | 378 | 0 | 755.6 |
| `adam+distopt` | 2 | 189 | 0 | 377.8 |
| `adam+distopt` | 4 | 94 | 0 | 188.9 |
| `adam+distopt` | 8 | 47 | 0 | 94.5 |
| `adam+distopt+precaware` | 2 | 0 | 0 | 472.3 |
| `adam+distopt+precaware` | 8 | 0 | 0 | 118.1 |
| `dist_muon` | 1 | 378 | 410 | 0.1 |
| `dist_muon` | 2 | 189 | 205 | 0.1 |
| `dist_muon` | 4 | 94 | 102 | 0.0 |
| `dist_muon` | 8 | 47 | 51 | 0.0 |
| `muon` | 1 | 378 | 410 | 0.1 |
| `muon` | 8 | 378 | 410 | 0.1 |
