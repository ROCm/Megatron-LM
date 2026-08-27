# Kimi K3 in ROCm/Megatron-LM

Fork-local integration of Kimi K3 training. **Everything lives under `kimi_k3/`**
so that IFU merges from upstream never conflict with it; `megatron/**` is
read-only and a guard enforces it.

```
kimi_k3/
├── PINS.md          dependency SHAs, licenses, and the observed environment
├── config/          KimiK3TransformerConfig, the builder, presets
├── model/           K3GPTModel + the scoped block injection and its pin contracts
├── block/           AttnRes mixer, the packed pipeline payload, the K3 block
├── moe/             MXFP4/MXFP8 QAT, SiTU-GLU, the QAT expert
├── pipeline/        the adjust_tensor_shapes_fn binding
├── tools/           measurement probes and the analytic memory model
├── tests/           the gates
├── ci/              no_core_diff_guard.sh
└── develop/         plans, rules, notes, results  (docs only, no production code)
```

Start at [`develop/README.md`](develop/README.md); the working rules are
[`develop/rules/rule.md`](develop/rules/rule.md) and the plan is
[`develop/plan-0/`](develop/plan-0/).

## Running the gates

```bash
python -m pytest kimi_k3/tests/ -q          # everything (1 GPU for most of it)
kimi_k3/ci/no_core_diff_guard.sh            # nothing changed outside kimi_k3/
python -m kimi_k3.tools.capture_env         # the table for PINS.md §5
```

Measurement probes, which produce `develop/results/`:

```bash
torchrun --nproc_per_node=8 -m kimi_k3.tools.opt_mem_probe --optimizer dist_muon
python -m kimi_k3.tools.attn_res_probe --width production
bash kimi_k3/develop/progress/p0/run_g7.sh
```
