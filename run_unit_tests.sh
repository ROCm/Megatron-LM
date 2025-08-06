#!/bin/bash

set -x
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

echo "=============================================================================="
echo "Starting main unit tests with markers: $PYTEST_MARKERS"
echo "=============================================================================="

torchrun --nproc_per_node=8 -m pytest -v -s -m "$PYTEST_MARKERS" --csv output/test_report.csv tests/unit_tests/ --dist=loadscope
echo "Main unit tests completed. Report saved to output/test_report.csv"

echo ""
echo "=============================================================================="
echo "Starting experimental unit tests."

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and experimental"
echo "Using markers: $PYTEST_MARKERS"
echo "=============================================================================="

torchrun --nproc_per_node=8 -m pytest -v -s -m "$PYTEST_MARKERS" --csv output/experimental_test_report.csv tests/unit_tests/ --dist=loadscope --experimental

echo "Experimental unit tests completed. Report saved to output/experimental_test_report.csv"

echo ""
echo "All test runs finished."