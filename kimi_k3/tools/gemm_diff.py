"""Diff the GEMM kernel tables of base / batch48 / batch48+syrk from one traced step."""
import json, sys
OUT = "/tmp/g67g"

def load(name):
    r = [x for x in map(json.loads, open(f"{OUT}/{name}.json")) if x["rank"] == 0][0]
    t = r["trace"]
    return t, {k["name"]: (k["ms"], k["calls"]) for k in t["top_kernels"]}

arms = {}
for n in ("base", "b48", "b48syrk"):
    try:
        arms[n] = load(n)
    except Exception as e:
        print(f"{n}: {e}")

for n, (t, _) in arms.items():
    print(f"=== {n:8s} iter={t['iteration_ms']:8.1f}  device={t['device_ms']:8.1f}  "
          f"muon={t['regions_ms'].get('Optimizer.step#TensorParallelMuon.step', 0):8.1f}  "
          f"launches={t['kernel_launches']}")

names = sorted({n for _, k in arms.values() for n in k})
GEMM = ("Cijk", "nt_pp", "bmm", "addmm", "mm")
print(f"\n{'kernel':<60} " + " ".join(f"{a:>18}" for a in arms))
for n in names:
    if not any(g in n for g in GEMM):
        continue
    row = "".join(f"{k.get(n, (0,0))[0]:12.1f}/{k.get(n, (0,0))[1]:<6d}" for _, k in arms.values())
    print(f"{n[:60]:<60} {row}")

print("\n--- GEMM device-time totals (kernel names only, excludes aten:: wrappers) ---")
for a, (_, k) in arms.items():
    tot = sum(v[0] for n, v in k.items() if n.startswith(("Cijk", "nt_pp")))
    print(f"{a:10s} {tot:9.1f} ms")
