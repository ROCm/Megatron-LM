#!/bin/bash

set -u -o pipefail

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "Number of GPUs: $NUM_GPUS"

# Keep RCCL/NCCL chatter out of the logs; only warnings and errors.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

OUT_DIR=output
mkdir -p "$OUT_DIR"

# Per-file logs are sorted into these subfolders by the file's overall outcome:
#   failed  -> any errors or failures (or a hard crash / unparseable result)
#   passed  -> only passed and/or skipped tests
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR/passed" "$LOG_DIR/failed"

# Sum failures + errors across the given JUnit XML file(s)/globs. This is the
# authoritative signal for "did this file actually have failed or error tests":
# deselected-only, skipped-only, "no tests ran", and all-passed runs report 0,
# while a synthetic crash report contributes an error. Unparseable XML counts as
# a failure so a broken/partial report is never silently treated as a pass.
count_junit_failures() {
    python - "$@" <<'EOF'
import sys, glob
from xml.etree import ElementTree as ET

total = 0
for pattern in sys.argv[1:]:
    for path in glob.glob(pattern):
        try:
            root = ET.parse(path).getroot()
        except Exception:
            total += 1
            continue
        suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
        for s in suites:
            total += int(s.get("failures", 0) or 0) + int(s.get("errors", 0) or 0)
print(total)
EOF
}

# Per-test-case timeout (pytest-timeout). Applies to each individual test, not the
# whole file, so a file with many tests is never killed just for its total runtime.
# A single hung test (e.g. RCCL/NCCL deadlock) is stopped after this many seconds.
PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-3600}       # 1 hour per test

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

# Synthesize a JUnit report for a file whose run died hard (signal/timeout) or
# whose XML is missing/incomplete, so the test reporter shows it as a failure
# instead of silently dropping it (which makes a crashed run look all-green).
write_crash_xml() {
    local src_file="$1"
    local name="$2"
    local rc="$3"
    local log="$4"
    local reason="${5:-}"
    local crash_xml="$OUT_DIR/junit_report_${name}_crash.xml"

    SRC_FILE="$src_file" CRASH_NAME="$name" CRASH_RC="$rc" \
    CRASH_LOG="$log" CRASH_XML="$crash_xml" CRASH_REASON="$reason" python - <<'EOF'
import os
from xml.sax.saxutils import escape, quoteattr

src_file = os.environ["SRC_FILE"]
name = os.environ["CRASH_NAME"]
rc = int(os.environ["CRASH_RC"])
log = os.environ["CRASH_LOG"]
crash_xml = os.environ["CRASH_XML"]
reason = os.environ.get("CRASH_REASON", "").strip()

# Human-readable label for the exit code (an explicit reason takes precedence).
labels = {
    124: "TIMEOUT (SIGTERM after wall-clock cap)",
    137: "TIMEOUT/KILLED (SIGKILL, code 137)",
    134: "ABORTED (SIGABRT, code 134)",
    139: "SEGFAULT (SIGSEGV, code 139)",
    143: "TERMINATED (SIGTERM, code 143)",
    2: "INTERRUPTED (pytest exit 2)",
    3: "INTERNAL ERROR (pytest exit 3)",
}
if reason:
    label = reason
elif rc > 128:
    label = labels.get(rc, f"KILLED by signal {rc - 128} (code {rc})")
else:
    label = labels.get(rc, f"hard exit (code {rc})")

tail = ""
try:
    with open(log, "r", errors="replace") as f:
        tail = "".join(f.readlines()[-50:])
except OSError:
    tail = "(log file unavailable)"

message = f"{src_file} did not finish cleanly: {label}"
body = f"Exit code: {rc}\nLabel: {label}\nFile: {src_file}\n\n--- last 50 log lines ---\n{tail}"

xml = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<testsuites>\n'
    f'  <testsuite name="crash::{escape(name)}" tests="1" failures="0" errors="1" skipped="0">\n'
    f'    <testcase classname={quoteattr("crash." + name)} name={quoteattr(os.path.basename(src_file) + "::process")}>\n'
    f'      <error message={quoteattr(message)} type="ProcessCrash">{escape(body)}</error>\n'
    '    </testcase>\n'
    '  </testsuite>\n'
    '</testsuites>\n'
)

with open(crash_xml, "w") as f:
    f.write(xml)

print(f"Wrote synthetic crash report: {crash_xml}")
EOF
}

# Find all test files recursively
TEST_FILES=$(find tests/unit_tests -type f -name "test_*.py")

ANY_FAIL=0

for file in $TEST_FILES; do
    # Create unique filename by replacing slashes with underscores and removing tests/unit_tests/ prefix
    # E.g., tests/unit_tests/dist_checkpointing/test_optimizer.py -> dist_checkpointing_test_optimizer
    test_name=$(echo "$file" | sed 's|tests/unit_tests/||' | sed 's|/|_|g' | sed 's|\.py$||')

    csv_file="$OUT_DIR/test_report_${test_name}.csv"
    xml_file="$OUT_DIR/junit_report_${test_name}.xml"
    log_file="$OUT_DIR/pytest_${test_name}.log"

    echo "Running test file: $file"

    # Full verbose output (per-test names + uncaptured stdout) goes to the per-file
    # log (uploaded as an artifact); the console only gets the concise status line.
    # The timeout is enforced PER TEST CASE via pytest-timeout, not per file, so a
    # large file with many tests is not killed just for having a lot of cases.
    torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
        -v -s --tb=long --showlocals \
        --timeout="$PYTEST_TIMEOUT" --timeout-method=thread \
        --reruns 2 --reruns-delay 5 \
        -m "$PYTEST_MARKERS" \
        --csv "$csv_file" \
        --junitxml "$xml_file" \
        "$file" > "$log_file" 2>&1
    rc=$?

    summary=$(grep -E "==.*(passed|failed|error|skipped).*==" "$log_file" | tail -1)

    # pytest-timeout prints a "+ Timeout +" banner (and, for the signal method,
    # a "from pytest-timeout" message) when a single test exceeds --timeout. The
    # thread method then os._exit(1)s, so detect it from the log rather than rc.
    timeout_reason=""
    if grep -qE '\+ Timeout \+|from pytest-timeout' "$log_file"; then
        timeout_reason="TIMEOUT (per-test ${PYTEST_TIMEOUT}s cap exceeded)"
    fi

    # Capture hard crashes the JUnit XML missed: the process did not exit cleanly
    # (rc != 0) and either died on a signal/timeout (rc != 1) or never wrote a
    # usable XML. rc == 0 is always a clean run (incl. deselected/no-tests) and is
    # never synthesized as a failure. Runs while $log_file is at its original path.
    crash_xml="$OUT_DIR/junit_report_${test_name}_crash.xml"
    if [[ $rc -ne 0 ]] && { [[ $rc -ne 1 ]] || [[ ! -s "$xml_file" ]]; }; then
        write_crash_xml "$file" "$test_name" "$rc" "$log_file" "$timeout_reason"
    fi

    # A file goes to failed/ only if it actually has failed/error tests (per the
    # JUnit counts, including any synthetic crash report) or a per-test timeout.
    fail_errors=$(count_junit_failures "$xml_file" "$crash_xml")
    if [[ -n "$timeout_reason" || ${fail_errors:-0} -gt 0 ]]; then
        outcome="failed"
    else
        outcome="passed"
    fi
    mv "$log_file" "$LOG_DIR/$outcome/" 2>/dev/null || true
    final_log="$LOG_DIR/$outcome/$(basename "$log_file")"

    if [[ -n "$timeout_reason" ]]; then
        echo "[TIMEOUT] $file -- a test exceeded the ${PYTEST_TIMEOUT}s per-test cap -- $final_log"
        ANY_FAIL=1
    elif [[ "$outcome" == "failed" ]]; then
        echo "[FAIL] ($rc) $file -- ${summary:-no summary} -- $final_log"
        ANY_FAIL=1
    else
        echo "[PASS] $file -- ${summary:-no summary} -- $final_log"
    fi

    # Reap any stragglers so a killed run doesn't leak GPU memory into the next file.
    pkill -9 -f "pytest" || true
    pkill -9 -f "torchrun" || true
done

if [[ $ANY_FAIL -ne 0 ]]; then
    echo "One or more test files failed."
else
    echo "All test files passed successfully."
fi

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
