# G32 — anchored parity against the release, on real weights

> Reproduce:
> ```
> python -m kimi_k3.tools.fetch_release_tensors --match layers.0.self_attn \
>     --out /tmp/k3_layer0_self_attn.pt
> pip install --target /tmp/tf4562 --no-deps transformers==4.56.2 "tokenizers>=0.22,<=0.23"
> K3_REFERENCE_PATH=<dir with the release's modeling files> PYTHONPATH=/tmp/tf4562 \
>     pytest kimi_k3/tests/test_k3_p8_anchored.py -q
> ```

## What was compared

Layer 0's KDA block — 14 tensors, **847 MiB**, fetched by HTTP range request
rather than pulling the ~16 GiB shard, which keeps this inside the project's
"never download >10 GB without asking" rule. Both our `KimiDeltaAttention` and
the **release's own** `KimiDeltaAttention` (its `modeling_kimi_linear.py`, run
under transformers 4.56.2 installed to a `--target` directory) were loaded with
those weights and given the same input.

## Result

| check | outcome |
|---|---|
| our module loads the released layout | **0 missing, 0 unexpected** |
| the release module loads it | 0 missing, 0 unexpected |
| parameter count, both modules | **443,740,384** — and the analytic `kda_layer_params` agrees exactly |
| forward, seq 128, bf16 | **rel-L2 7.36e-3 · max-abs 5.96e-6 · cosine 0.999973** |
| output scale | std 0.00011 both |

The relative figure sits inside the measured bf16 bound (1e-2, floor 3.3e-3;
`results/kda_parity.md`) and the absolute one is 6e-6 against activations of
order 1e-4 — bf16 round-off, not a structural difference. The remaining gap is
the gated RMSNorm: ours is plain torch, the release calls fla's
`FusedRMSNormGated`.

## The invariant, confirmed from the other side

The checkpoint stores `A_log` as `[128]` with the last 32 entries zero, while the
**release's own module declares `[96]`** — so loading the released checkpoint into
the released code fails without a trim. The padding is real and handling it is
not optional. P0 measured this from the shard header; this is independent
confirmation from the modeling side, and it is now a test.

## Scope

One layer, one kind (KDA), forward only, seq 128. It establishes that the
parameter layout and the KDA math match the release on **real** weights. It does
not cover gated MLA, the AttnRes stream, the MoE stack, or a whole truncated
model — that is the four-layer slice the plan describes, and it needs several
shards plus a truncated HF model to run against.
