# G50 — the TE pin and CK grouped GEMM, measured warm. Retracts the 1.44x

> `torchrun --nproc_per_node=8 -m kimi_k3.tools.proxy_ep8 --preset 4L --ep 8 \`
> `  --seq 512 --iterations 8 --optimizer dist_muon [--no-ck-grouped-gemm]`
> Eight runs, one script, EP=8 — the geometry the original claim was made at.
> Raw: `results/raw/teab_*.jsonl`.

## Retraction

**Reported: 2,653.5 -> 1,845.0 ms, a 1.44x speed-up from TE 2.18 + CK.**
**Measured warm: 1,912.6 -> 1,842.3 ms, a 1.038x speed-up.**

TE 2.12 does not take 2,653.5 ms. It takes **1,912.6 ms**. The missing ~740 ms
was cold-cache overhead in the original baseline (G49): aiter `.co` kernel loads
and hipBLASLt tuning, paid once per kernel set and persisting across processes.
The old baseline and the new measurement were taken on different days in
different processes, so only one side of the comparison ever paid it.

## Protocol

Order reversal alone would not have caught this. The two TE versions ship
different kernels, so a warm 2.18 cache does nothing for 2.12 — reversal fixes
*within*-version comparisons, and a discarded first run per configuration is
what makes the *cross*-version comparison honest. Both were used.

TE 2.12 was selected by `PYTHONPATH` shadowing of a private extraction, never by
reinstalling: the node is shared, and swapping the venv would have silently
handed 2.12 to anyone else's next job. Verified that both `te.__file__` and
`importlib.metadata.version` report 2.12 — the metadata matters because
`is_te_min_version` gates real behaviour in core, so a version mismatch would
have measured the wrong thing while looking correct.

| arm | steady ms | peak GiB |
|---|---|---|
| TE2.18 +CK — warm-up (discarded) | 1,844.4 | 190.99 |
| TE2.18 +CK — forward | 1,842.9 | 190.99 |
| TE2.18 +CK — reversed | 1,841.8 | 190.99 |
| TE2.18 hipBLASLt — forward | 1,865.4 | 190.99 |
| TE2.18 hipBLASLt — reversed | 1,865.6 | 190.99 |
| TE2.12 hipBLASLt — warm-up (discarded) | 1,916.6 | 191.92 |
| TE2.12 hipBLASLt — run A | 1,911.2 | 191.92 |
| TE2.12 hipBLASLt — run B | 1,913.9 | 191.92 |

Forward and reversed agree to **0.2 ms** on hipBLASLt and **1.1 ms** on CK; the
two independent TE 2.12 runs agree to **2.7 ms**. The effects below are small,
but they are repeatable rather than noise.

## Attribution, which was also backwards

| comparison | warm |
|---|---|
| CK grouped GEMM vs hipBLASLt (TE 2.18) | **1.013x** |
| TE 2.12 -> 2.18 (both hipBLASLt) | **1.025x** |
| end-to-end | **1.038x** |

The original analysis predicted CK was the primary driver, reasoning from the
5.36x collapse in kernel launches and `aten::add_` falling 14,449 -> 2,177 — a
pattern read as many small per-expert kernels merging into one grouped call.
**Wrong.** CK is worth 1.3%; the pin is worth 2.5%. The launch collapse comes
from TE 2.18's fusions, not from grouped-GEMM consolidation.

## What survives

The **5.36x launch reduction and 3.19x collective drop are structural.** They
come from a profiled trace rather than wall-clock and cannot be caching
artifacts. TE 2.18 genuinely does far less launch work — it simply does not
convert into wall-clock here, because at this geometry the iteration is **81%
Muon** (1,501 of 1,845 ms) and the optimizer is untouched by any of this.
Shaving the remaining 344 ms cannot move the total much, which is exactly why
all three ratios land between 1.01 and 1.04.

Moving the pin still stands on its own terms: TE 2.18 fixed the CK
fused-attention backward NaN (A21) that blocked the TE attention path outright.
That was never a performance argument.

**The conclusion the phase actually rests on is unchanged, and better supported:
the optimizer is the bottleneck, not the kernels.**
