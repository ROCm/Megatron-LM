#!/bin/bash
set -e
set -x

# Install mock for unit tests
pip install mock

# Install Go version of yq (required for ignore logic)
# Matches upstream installation in Dockerfile.ci.dev
YQ_VERSION=4.44.1
if ! command -v yq &> /dev/null || ! yq eval '.test' <<< 'test: value' &> /dev/null 2>&1; then
    echo "Installing Go version of yq..."
    wget https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_amd64 -O /usr/local/bin/yq
    chmod a+x /usr/local/bin/yq
    echo "yq installed successfully"
fi

chmod +x run_unit_tests_bucketed.sh

BUCKETS=(
    "tests/unit_tests/data/"
    "tests/unit_tests/dist_checkpointing/*.py"
    "tests/unit_tests/dist_checkpointing/models/"
    "tests/unit_tests/transformer/*.py"
    "tests/unit_tests/transformer/moe"
    "tests/unit_tests/distributed/fsdp"
    "tests/unit_tests"
)

for bucket in "${BUCKETS[@]}"; do
    echo "========================================"
    echo "Running bucket: $bucket"
    echo "========================================"
    ./run_unit_tests_bucketed.sh --bucket "$bucket" "$@"
done

echo "All buckets completed!"


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