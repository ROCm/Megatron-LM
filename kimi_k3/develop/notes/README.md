# Development Notes

> Ad-hoc investigations, decision records and troubleshooting notes produced
> during development. Design decisions that outlive an investigation graduate
> into `../plan-<n>/` or `../rules/rule.md`.

## Naming convention

`YYYY-MM-DD-<topic>.md`, e.g. `2026-08-26-attn-res-pp-transport.md`.

## Tags (put one at the top of each note)

- `decision` — architecture / interface decision
- `analysis` — deep dive into the release or into core
- `troubleshooting` — an issue and how it was resolved
- `experiment` — record of a comparison
- `convergence` — loss curves / numerical alignment records

## Existing notes

| note | tag | summary |
|---|---|---|
| [`2026-08-26-attn-res-pp-transport.md`](2026-08-26-attn-res-pp-transport.md) | `decision` | Why the AttnRes pipeline payload is a single packed tensor and how per-stage shapes are derived |
| [`2026-08-27-release-audit.md`](2026-08-27-release-audit.md) | `analysis` | P0-T0.2 / gate G2: `A_log` padding, the 1e-6 LoRA-norm epsilon, exact key layout, tokenizer ids |
| [`2026-08-27-fla-signature-check.md`](2026-08-27-fla-signature-check.md) | `troubleshooting` | Why G1 is red: `chunk_kda` silently ignores `A_log` / `dt_bias` / `transpose_state_layout` |
| [`2026-08-27-optimizer-memory-method.md`](2026-08-27-optimizer-memory-method.md) | `analysis` | How G5 measures bytes/param, and the DDP-config trap that kills the first optimizer step |
| [`2026-08-27-triton-unblocks-fla-backward.md`](2026-08-27-triton-unblocks-fla-backward.md) | `troubleshooting` | G1 green: triton 3.6.0 → 3.7.1 makes the KDA backward compile; no torch change needed |
