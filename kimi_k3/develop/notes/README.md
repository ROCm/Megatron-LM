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
