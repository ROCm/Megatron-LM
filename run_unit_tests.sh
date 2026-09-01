#!/bin/bash

set -u -o pipefail
set -x

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "Number of GPUs: $NUM_GPUS"

OUT_DIR=output
mkdir -p "$OUT_DIR"

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

# Per-test-file timeout in seconds (30 minutes).
# Prevents a single hung torchrun from consuming the entire CI time budget.
# --kill-after=300: if SIGTERM is ignored, SIGKILL after 5 more minutes.
TEST_TIMEOUT=${TEST_TIMEOUT:-1800}
LONG_TEST_TIMEOUT=${LONG_TEST_TIMEOUT:-3600}
KILL_AFTER=${KILL_AFTER:-300}

# Number of attempts per test file. Flaky RCCL/NCCL collective stalls on ROCm
# abort the whole torchrun (see ROCm/rccl#2022); a fresh attempt almost always
# passes. We retry on ANY non-zero exit, so a genuinely failing file is still
# reported as failed (after exhausting its attempts) rather than masked.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

# Fail fast on a wedged collective so each retry is cheap: abort the NCCL/RCCL
# watchdog at 180s instead of the 600s default, and skip the slow debug dump.
export TORCH_NCCL_TIMEOUT=${TORCH_NCCL_TIMEOUT:-180}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}

# Find all test files recursively
TEST_FILES=$(find tests/unit_tests -type f -name "test_*.py")

ANY_FAIL=0

for file in $TEST_FILES; do
    # Create unique filename by replacing slashes with underscores and removing tests/unit_tests/ prefix
    # E.g., tests/unit_tests/dist_checkpointing/test_optimizer.py -> dist_checkpointing_test_optimizer
    test_name=$(echo "$file" | sed 's|tests/unit_tests/||' | sed 's|/|_|g' | sed 's|\.py$||')

    csv_file="$OUT_DIR/test_report_${test_name}.csv"
    xml_file="$OUT_DIR/junit_report_${test_name}.xml"

    file_timeout=$TEST_TIMEOUT
    case "$file" in
        tests/unit_tests/dist_checkpointing/models/test_moe_experts.py | \
        tests/unit_tests/dist_checkpointing/test_layer_wise_optimizer.py)
            # These heavily parametrized files exceed 30 minutes on MI325X.
            file_timeout=$LONG_TEST_TIMEOUT
            ;;
    esac

    echo "Running test file: $file"
    attempt=1
    rc=0
    while (( attempt <= MAX_ATTEMPTS )); do
        if (( attempt > 1 )); then
            echo "RETRY: $file attempt ${attempt}/${MAX_ATTEMPTS} (previous rc=$rc)"
            # Reap any orphaned ranks a hung/aborted torchrun may have left behind
            # so they don't hold GPUs or NCCL groups for the next attempt.
            pkill -9 -f "pytest.*$(basename "$file")" 2>/dev/null || true
            sleep 5
        fi

        timeout --kill-after="$KILL_AFTER" "$file_timeout" \
            torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
                --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" \
                --csv "$csv_file" \
                --junitxml "$xml_file" \
                $file
        rc=$?

        # exit code 124 = SIGTERM timeout; 137 = SIGKILL (--kill-after triggered)
        if [[ $rc -eq 124 ]] || [[ $rc -eq 137 ]]; then
            echo "TIMEOUT: $file exceeded ${file_timeout}s (kill-after=${KILL_AFTER}s) on attempt ${attempt}."
        elif [[ $rc -ne 0 ]]; then
            echo "Test failed in $file on attempt ${attempt} (rc=$rc)."
        fi

        # Success — stop retrying this file.
        [[ $rc -eq 0 ]] && break
        (( attempt++ ))
    done

    # Record the authoritative final exit code for this file. The JUnit merge
    # below uses it to detect crashes: a rank that aborts (NCCL watchdog SIGABRT),
    # hits a GPU fault, or is SIGKILLed on timeout never writes its XML, so the
    # merged report would otherwise falsely show "passed" from the ranks that
    # survived. rc is the only reliable signal for those cases.
    echo "$rc" > "$OUT_DIR/status_${test_name}.rc"

    if [[ $rc -ne 0 ]]; then
        echo "FAILED: $file did not pass after ${MAX_ATTEMPTS} attempt(s) (final rc=$rc)."
        ANY_FAIL=1
    fi
done

if [[ $ANY_FAIL -ne 0 ]]; then
    echo "One or more test files failed."
else
    echo "All test files passed successfully."
fi

# Merge per-rank JUnit reports so a testcase is reported as failed if it failed
# on ANY rank. Each rank writes junit_report_<name>.rank<N>.xml (see conftest).
# The merge is driven by the per-file exit code (status_<name>.rc), NOT just the
# XML files, because a crashed/hung rank (NCCL SIGABRT, GPU fault, SIGKILL on
# timeout) never writes its XML. Without this, the merged report would falsely
# show "passed" from the surviving ranks. When a run crashed or a rank's report
# is missing, we inject a synthetic <error> testcase so the CI reporter goes red.
# Rank files are moved into output/per_rank/ for debugging; the reporter/artifact
# glob (junit_report_*.xml, top level only) then sees the merged, correct file.
EXPECTED_RANKS="$NUM_GPUS" python - <<'EOF'
import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

output_dir = "output"
per_rank_dir = os.path.join(output_dir, "per_rank")
os.makedirs(per_rank_dir, exist_ok=True)
expected_ranks = int(os.environ.get("EXPECTED_RANKS", "0") or 0)

rank_re = re.compile(r"\.rank\d+\.xml$")


def _outcome(tc):
    if tc.find("error") is not None:
        return "error"
    if tc.find("failure") is not None:
        return "failure"
    if tc.find("skipped") is not None:
        return "skipped"
    return "passed"


# Worse outcomes win when the same test ran on multiple ranks.
_ORDER = {"passed": 0, "skipped": 1, "failure": 2, "error": 3}

# Drive off the status files so every attempted test file is represented, even
# if it produced zero XML (all ranks hung and were killed).
test_names = {
    os.path.basename(p)[len("status_"):-len(".rc")]
    for p in glob.glob(os.path.join(output_dir, "status_*.rc"))
}
# Also cover any stray rank XML that somehow lacks a status file.
for p in glob.glob(os.path.join(output_dir, "junit_report_*.rank*.xml")):
    name = rank_re.sub("", os.path.basename(p))[len("junit_report_"):]
    test_names.add(name)

if not test_names:
    print("No test files to merge JUnit reports for.")

for test_name in sorted(test_names):
    base = f"junit_report_{test_name}.xml"
    rank_files = sorted(
        glob.glob(os.path.join(output_dir, f"junit_report_{test_name}.rank*.xml"))
    )
    rc_path = os.path.join(output_dir, f"status_{test_name}.rc")
    try:
        with open(rc_path) as fh:
            rc = int((fh.read().strip() or "0"))
    except (OSError, ValueError):
        rc = 0  # no status recorded; fall back to XML contents only

    cases = OrderedDict()
    for path in rank_files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue  # truncated/partial XML from a rank killed mid-write
        for ts in root.iter("testsuite"):
            for tc in ts.findall("testcase"):
                key = (tc.get("classname"), tc.get("name"))
                cur = cases.get(key)
                if cur is None or _ORDER[_outcome(tc)] > _ORDER[_outcome(cur)]:
                    cases[key] = tc

    # A crash is any of: non-zero exit code, fewer rank reports than expected,
    # or unparseable/empty reports. In those cases a rank died without writing
    # its results, so the surviving XML cannot be trusted to reflect failure.
    missing_ranks = (
        expected_ranks > 0 and len(rank_files) < expected_ranks
    )
    crashed = rc != 0 or missing_ranks
    already_red = any(_outcome(tc) in ("failure", "error") for tc in cases.values())
    if crashed and not already_red:
        tc = ET.Element("testcase", classname=test_name, name="distributed_run_integrity")
        err = ET.SubElement(tc, "error", message="rank crash / timeout")
        err.text = (
            f"Run integrity check failed: exit_code={rc}, "
            f"rank_reports={len(rank_files)}/{expected_ranks}. "
            "At least one rank crashed, hung, or was killed before writing its "
            "JUnit report (e.g. NCCL/RCCL watchdog SIGABRT, GPU fault, or "
            "timeout SIGKILL). The surviving ranks' results cannot be trusted."
        )
        cases[("integrity", "distributed_run_integrity")] = tc

    tests = failures = errors = skipped = 0
    suite = ET.Element("testsuite", name="pytest")
    for tc in cases.values():
        suite.append(tc)
        tests += 1
        o = _outcome(tc)
        failures += o == "failure"
        errors += o == "error"
        skipped += o == "skipped"
    suite.set("tests", str(tests))
    suite.set("failures", str(failures))
    suite.set("errors", str(errors))
    suite.set("skipped", str(skipped))

    suites = ET.Element("testsuites")
    suites.append(suite)
    ET.ElementTree(suites).write(
        os.path.join(output_dir, base), encoding="utf-8", xml_declaration=True
    )
    for path in rank_files:
        os.replace(path, os.path.join(per_rank_dir, os.path.basename(path)))
    print(
        f"Merged {len(rank_files)}/{expected_ranks} rank report(s) -> {base} "
        f"(rc={rc}, tests={tests}, failures={failures}, errors={errors}, "
        f"skipped={skipped}{', CRASH-INJECTED' if (crashed and not already_red) else ''})"
    )
EOF

# Merge all individual CSVs into one unified report
python - <<EOF
import os
import pandas as pd

output_dir = "output"
csv_files = [f for f in os.listdir(output_dir) if f.startswith("test_report_") and f.endswith(".csv")]

dfs = []
for file in csv_files:
    path = os.path.join(output_dir, file)
    df = pd.read_csv(path)
    df["source_file"] = file  # Optional: track which file the results came from
    dfs.append(df)

if dfs:
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df.to_csv(os.path.join(output_dir, "unified_test_report.csv"), index=False)
    print("Unified test report saved to output/unified_test_report.csv")
else:
    print("No test report CSV files found to merge.")
EOF

exit $ANY_FAIL
