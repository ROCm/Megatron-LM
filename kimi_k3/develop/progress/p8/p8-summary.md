# P8 — Converter and anchored parity (COMPLETE)

## 1. Objective

Read the released checkpoint into Megatron and write it back, and then check the
result against the release itself rather than against our own expectations.

## 2. What changed

| File | What |
|---|---|
| `kimi_k3/tools/mapping.py` | released keys <-> Megatron keys; layer-kind inference; invariant checks; `dry_run` over an index |
| `kimi_k3/tools/convert.py` | `hf_to_mcore` / `mcore_to_hf`, `A_log` trim/pad, MXFP4 dequantisation, fused `linear_fc1` |
| `kimi_k3/tools/fetch_release_tensors.py` | range-fetch named tensors from a shard, so a parity check costs 847 MiB and not 16 GiB |
| `kimi_k3/tests/test_k3_p8_convert.py` | 13 tests, no network |
| `kimi_k3/tests/test_k3_p8_anchored.py` | 3 tests against real weights, skipped by default |
| `kimi_k3/tests/fixtures/release_index_patterns.json` | 4.9 KB of patterns that expand to an equivalent 497,220-key index |

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| **G30** — every released tensor is accounted for | **GREEN** | 497,220 keys: **497,052 mapped, 168 skipped by an explicit rule, 0 unmapped**. 247,296 MXFP4 pairs, 896 experts/layer, 69 KDA + 24 MLA inferred from which tensors exist |
| **G31** — round-trip | **GREEN** | bf16 keys and values round-trip exactly; `A_log` trims only when the padding is really zero; exporting MXFP4 experts **refuses** rather than writing bf16 under a `weight_packed` name |
| **G32** — anchored parity | **GREEN for one KDA layer on real weights** | our module and the **release's own** both load layer 0 with 0 missing / 0 unexpected, 443,740,384 params each, forward agrees to **rel-L2 7.4e-3, cosine 0.999973** (`results/anchored_parity.md`) |
| **G33** — tokenizer round-trip | **GREEN** (landed in P7) | special ids asserted against the released `config.json` |

`pytest kimi_k3/tests/ -q` -> **183 passed, 2 skipped**.

## 4. "Zero unmapped" only means something if the skipped ones are named

168 tensors are skipped: the vision tower and the multimodal projector. That
number is asserted, not reported -- if a future revision adds a tensor we do not
handle, the count moves and `dry_run` fails. A converter that quietly drops what
it does not recognise passes every test it has and produces a broken model.

## 5. Three things that are not renames

* **`A_log` is padded.** The checkpoint stores `[128]` with the tail zero; the
  real width is 96. P0 measured this from the shard header. P8 confirmed it from
  the other side: the **release's own module declares `[96]`**, so its own
  checkpoint does not load into its own code without a trim.
* **Routed experts are MXFP4**, so import dequantises and the honest claim is
  "matches the dequantised release" -- the original weights were never published.
* **`linear_fc1` is fused** (gate first), while the release keeps `w1`/`w3` apart.

## 6. The bug the test parametrisation caught

Expert `w1` carries **two** markers: it is the gate half of a fused `linear_fc1`
*and* MXFP4-packed, so its mapped name ends `@gate@scale`. Splitting on the last
`@` sent the packed data and its scale into different buckets, where they never
met. A `w2`-only test would not have noticed, because `w2` has no half marker.
It was caught by parametrising over `w1` **and** `w2`, and the fix is
`_parse_target`, which looks for each marker by name instead of by position.

## 7. What G32 does not cover

One layer, one kind, forward only, seq 128. Gated MLA, the AttnRes stream, the
MoE stack and a whole truncated model are the four-layer slice the plan
describes; it needs several shards and a truncated HF model to run against, and
is owed alongside the other nightly items.
