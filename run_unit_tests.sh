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

# Classify a run using the pytest summary line and the process exit code.
classify_outcome() {
    local rc="$1"
    local summary="$2"
    local n_failed n_error n_passed n_skipped
    n_failed=$(echo "$summary" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
    n_error=$(echo "$summary" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | head -1)
    n_passed=$(echo "$summary" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
    n_skipped=$(echo "$summary" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' | head -1)

    if [[ ( "$rc" -ne 0 && "$rc" -ne 1 ) || ${n_error:-0} -gt 0 || ${n_failed:-0} -gt 0 ]]; then
        echo "failed"
    elif [[ ${n_passed:-0} -gt 0 || ${n_skipped:-0} -gt 0 ]]; then
        echo "passed"
    else
        echo "failed"
    fi
}

# Hard per-file wall-clock cap. A single RCCL/NCCL collective deadlock can hang
# torchrun forever, so this is the backstop that guarantees forward progress.
PER_FILE_TIMEOUT=${PER_FILE_TIMEOUT:-1200}   # 20 minutes
# Graceful in-pytest timeout (fires first, gives a traceback before the hard kill).
PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-900}        # 15 minutes

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
    local crash_xml="$OUT_DIR/junit_report_${name}_crash.xml"

    SRC_FILE="$src_file" CRASH_NAME="$name" CRASH_RC="$rc" \
    CRASH_LOG="$log" CRASH_XML="$crash_xml" python - <<'EOF'
import os
from xml.sax.saxutils import escape, quoteattr

src_file = os.environ["SRC_FILE"]
name = os.environ["CRASH_NAME"]
rc = int(os.environ["CRASH_RC"])
log = os.environ["CRASH_LOG"]
crash_xml = os.environ["CRASH_XML"]

# Human-readable label for the exit code.
labels = {
    124: "TIMEOUT (SIGTERM after wall-clock cap)",
    137: "TIMEOUT/KILLED (SIGKILL, code 137)",
    134: "ABORTED (SIGABRT, code 134)",
    139: "SEGFAULT (SIGSEGV, code 139)",
    143: "TERMINATED (SIGTERM, code 143)",
    2: "INTERRUPTED (pytest exit 2)",
    3: "INTERNAL ERROR (pytest exit 3)",
}
if rc > 128:
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

    # Full verbose output goes to the per-file log (uploaded as an artifact);
    # the console only gets the concise status line below.
    timeout --signal=SIGTERM --kill-after=60 "$PER_FILE_TIMEOUT" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
        --tb=short --capture=fd \
        --timeout="$PYTEST_TIMEOUT" --timeout-method=thread \
        --reruns 2 --reruns-delay 5 \
        -m "$PYTEST_MARKERS" \
        --csv "$csv_file" \
        --junitxml "$xml_file" \
        "$file" > "$log_file" 2>&1
    rc=$?

    summary=$(grep -E "==.*(passed|failed|error|skipped).*==" "$log_file" | tail -1)

    # Capture hard crashes the JUnit XML missed (signals/timeout, or missing/empty XML).
    # Must run while $log_file is still at its original path so the tail can be embedded.
    if [[ ( $rc -ne 0 && $rc -ne 1 ) || ! -s "$xml_file" ]]; then
        write_crash_xml "$file" "$test_name" "$rc" "$log_file"
    fi

    # Sort this file's log into output/logs/<outcome>/ by its overall result.
    outcome=$(classify_outcome "$rc" "$summary")
    mv "$log_file" "$LOG_DIR/$outcome/" 2>/dev/null || true
    final_log="$LOG_DIR/$outcome/$(basename "$log_file")"

    if [[ $rc -eq 0 ]]; then
        echo "[PASS] $file -- ${summary:-no summary} -- $final_log"
    elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
        echo "[TIMEOUT] $file (killed after ${PER_FILE_TIMEOUT}s) -- $final_log"
        ANY_FAIL=1
    else
        echo "[FAIL] ($rc) $file -- ${summary:-no summary} -- $final_log"
        ANY_FAIL=1
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
