#!/bin/bash
###############################################################################
# Run DeepSeek-V3-671B proxy training across three precision configs (bf16,
# fp8/delayed, fp8/mxfp8), generate TraceLens performance reports from the
# PyTorch profiler traces, and produce comparison .xlsx files against the
# pre-existing baseline CSV directories.
#
# Pre-requisites (already on the system):
#   - examples/deepseek_v3/proxy_mi355x_deepseekv3_671B.sh  (proxy env file)
#   - examples/deepseek_v3/train_deepseekv3.sh              (Megatron launcher)
#   - TraceLens CLI tools on PATH (auto-installed from GitHub if missing):
#       TraceLens_generate_perf_report_pytorch
#       TraceLens_compare_perf_reports_pytorch
#   - Baseline CSV archive at the repo root (optional):
#       deepseek_baselines.zip
#     (auto-extracted into ${BASELINE_ROOT}/ on first run; expected to contain
#      proxy_deepseekv3_bf16_baseline/, proxy_deepseekv3_fp8_baseline/,
#      proxy_deepseekv3_mxfp8_baseline/). If absent, compare step is skipped
#      gracefully per precision.
#
# Usage:
#   cd /path/to/Megatron-LM
#   bash examples/deepseek_v3/run_deepseekv3_perf_compare.sh             # all 3
#   bash examples/deepseek_v3/run_deepseekv3_perf_compare.sh bf16        # only bf16
#   bash examples/deepseek_v3/run_deepseekv3_perf_compare.sh bf16 fp8    # subset
#   bash examples/deepseek_v3/run_deepseekv3_perf_compare.sh all         # explicit
#   PRECISIONS="fp8,mxfp8" bash examples/deepseek_v3/run_deepseekv3_perf_compare.sh
#
# Precisions (positional args or PRECISIONS env var, space- or comma-
# separated). Valid tokens: bf16 | fp8 | mxfp8 | all (default: all).
#
# Optional overrides:
#   RUNS_ROOT=...           (default: tracelens_runs/deepseekv3_<TS>)
#   PRECISIONS=...          (default: all; subset of bf16 fp8 mxfp8)
#   SKIP_TRAIN_BF16=1       (skip training bf16, reuse $RUNS_ROOT/output_bf16)
#   SKIP_TRAIN_FP8=1        (skip training fp8/delayed)
#   SKIP_TRAIN_MXFP8=1      (skip training fp8/mxfp8)
#   SKIP_COMPARE=1          (skip the compare step)
#   BASELINE_ZIP=...        (default: deepseek_baselines.zip)
#   BASELINE_ROOT=...       (default: deepseek_baselines; extraction target dir)
#   BASELINE_BF16_DIR=...   (default: ${BASELINE_ROOT}/proxy_deepseekv3_bf16_baseline)
#   BASELINE_FP8_DIR=...    (default: ${BASELINE_ROOT}/proxy_deepseekv3_fp8_baseline)
#   BASELINE_MXFP8_DIR=...  (default: ${BASELINE_ROOT}/proxy_deepseekv3_mxfp8_baseline)
#   TRACELENS_GIT_URL=...   (default: git+https://github.com/AMD-AGI/TraceLens.git)
#   SKIP_TRACELENS_INSTALL=1 (do not auto-install TraceLens even if missing)
###############################################################################

set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${CURRENT_DIR}")")"
cd "${REPO_ROOT}"

# --- precision selection ---------------------------------------------------
# Positional CLI args (if any) take precedence over the PRECISIONS env var.
# Accepts space- or comma-separated tokens from: bf16, fp8, mxfp8, all.
PRECISIONS_INPUT="${PRECISIONS:-all}"
if [ "$#" -gt 0 ]; then
    PRECISIONS_INPUT="$*"
fi
PRECISIONS_INPUT="${PRECISIONS_INPUT//,/ }"

SELECTED=()
_have_bf16=0; _have_fp8=0; _have_mxfp8=0
for _tok in ${PRECISIONS_INPUT}; do
    case "${_tok,,}" in
        all)
            [ "${_have_bf16}"  = "0" ] && SELECTED+=(bf16)  && _have_bf16=1
            [ "${_have_fp8}"   = "0" ] && SELECTED+=(fp8)   && _have_fp8=1
            [ "${_have_mxfp8}" = "0" ] && SELECTED+=(mxfp8) && _have_mxfp8=1
            ;;
        bf16)
            [ "${_have_bf16}"  = "0" ] && SELECTED+=(bf16)  && _have_bf16=1
            ;;
        fp8)
            [ "${_have_fp8}"   = "0" ] && SELECTED+=(fp8)   && _have_fp8=1
            ;;
        mxfp8)
            [ "${_have_mxfp8}" = "0" ] && SELECTED+=(mxfp8) && _have_mxfp8=1
            ;;
        *)
            echo "ERROR: unknown precision '${_tok}' (expected: bf16|fp8|mxfp8|all)" >&2
            exit 1
            ;;
    esac
done
if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo "ERROR: no precisions selected" >&2
    exit 1
fi

# --- proxy env: model size, parallelism, profiler on, etc. ---
# shellcheck disable=SC1091
source examples/deepseek_v3/proxy_mi355x_deepseekv3_671B.sh

TS=$(date +%Y%m%d_%H%M%S)
RUNS_ROOT=${RUNS_ROOT:-"tracelens_runs/deepseekv3_${TS}"}
mkdir -p "${RUNS_ROOT}"

BASELINE_ZIP=${BASELINE_ZIP:-"deepseek_baselines.zip"}
BASELINE_ROOT=${BASELINE_ROOT:-"deepseek_baselines"}
BASELINE_BF16_DIR=${BASELINE_BF16_DIR:-"${BASELINE_ROOT}/proxy_deepseekv3_bf16_baseline"}
BASELINE_FP8_DIR=${BASELINE_FP8_DIR:-"${BASELINE_ROOT}/proxy_deepseekv3_fp8_baseline"}
BASELINE_MXFP8_DIR=${BASELINE_MXFP8_DIR:-"${BASELINE_ROOT}/proxy_deepseekv3_mxfp8_baseline"}

# Extract the baseline CSV archive (idempotent) so that the comparison step
# below has access to the reference perf reports. The zip is expected at the
# repo root and contains the proxy_deepseekv3_*_baseline/ directories at top
# level. If the zip is missing, the compare step warns and continues.
extract_baselines() {
    if [ ! -f "${BASELINE_ZIP}" ]; then
        echo "WARN: baseline zip not found at ${BASELINE_ZIP}; skipping extraction" >&2
        return 0
    fi
    if [ -d "${BASELINE_BF16_DIR}" ] \
        && [ -d "${BASELINE_FP8_DIR}" ] \
        && [ -d "${BASELINE_MXFP8_DIR}" ]; then
        echo "Baselines already extracted under ${BASELINE_ROOT}/"
        return 0
    fi
    echo "Extracting ${BASELINE_ZIP} -> ${BASELINE_ROOT}/"
    mkdir -p "${BASELINE_ROOT}"
    if command -v unzip >/dev/null 2>&1; then
        unzip -oq "${BASELINE_ZIP}" -d "${BASELINE_ROOT}"
    else
        python3 -m zipfile -e "${BASELINE_ZIP}" "${BASELINE_ROOT}"
    fi
}
extract_baselines

# Ensure the TraceLens CLI tools are available; install from GitHub if not.
TRACELENS_GIT_URL=${TRACELENS_GIT_URL:-"git+https://github.com/AMD-AGI/TraceLens.git"}
ensure_tracelens() {
    if command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1 \
        && command -v TraceLens_compare_perf_reports_pytorch >/dev/null 2>&1; then
        echo "TraceLens CLI tools already installed."
        return 0
    fi
    if [ "${SKIP_TRACELENS_INSTALL:-0}" = "1" ]; then
        echo "ERROR: TraceLens CLI tools not on PATH and SKIP_TRACELENS_INSTALL=1" >&2
        return 1
    fi
    echo "Installing TraceLens from ${TRACELENS_GIT_URL} ..."
    python3 -m pip install --disable-pip-version-check --quiet "${TRACELENS_GIT_URL}"
    if ! command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1 \
        || ! command -v TraceLens_compare_perf_reports_pytorch >/dev/null 2>&1; then
        echo "ERROR: TraceLens install completed but CLI tools still not on PATH" >&2
        return 1
    fi
}
ensure_tracelens

echo "REPO_ROOT  : ${REPO_ROOT}"
echo "RUNS_ROOT  : ${RUNS_ROOT}"
echo "PRECISIONS : ${SELECTED[*]}"
echo "BASELINES  : ${BASELINE_BF16_DIR} | ${BASELINE_FP8_DIR} | ${BASELINE_MXFP8_DIR}"
echo

# Run a single training configuration.
# Args:
#   $1 = tag (bf16 | fp8 | mxfp8)
#   $2 = skip-train flag value (1 to skip)
#   rest = extra `KEY=VAL` pairs to prepend before `bash train_deepseekv3.sh`
run_train() {
    local tag="${1:-}"
    if [ -z "${tag}" ]; then
        echo "run_train: missing tag arg" >&2; return 1
    fi
    shift
    local skip="${1:-0}"
    if [ "$#" -ge 1 ]; then shift; fi
    local out_base="${RUNS_ROOT}/output_${tag}"
    local train_log="${RUNS_ROOT}/train_${tag}.log"

    if [ "${skip}" = "1" ]; then
        echo "[${tag}] SKIP_TRAIN set; reusing ${out_base}"
        return 0
    fi

    echo "============================================================"
    echo "[${tag}] training -> ${out_base}"
    echo "============================================================"
    rm -rf "${out_base}"
    mkdir -p "${out_base}"

    # train_deepseekv3.sh respects OUTPUT_BASEPATH if already set.
    # Use `env` to inject per-run vars without polluting our shell.
    env OUTPUT_BASEPATH="${out_base}" "$@" \
        bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee "${train_log}"
}

# Generate a TraceLens perf report (CSV dir) from the most-recent PyTorch
# profiler trace JSON inside ${out_base}/tensorboard/.
run_perf_report() {
    local tag="${1:-}"
    if [ -z "${tag}" ]; then
        echo "run_perf_report: missing tag arg" >&2; return 1
    fi
    local out_base="${RUNS_ROOT}/output_${tag}"
    local csv_dir="${RUNS_ROOT}/perf_${tag}"
    local trace_json

    trace_json=$(ls -t "${out_base}/tensorboard/"*.pt.trace.json 2>/dev/null | head -1 || true)
    if [ -z "${trace_json}" ]; then
        echo "[${tag}] ERROR: no .pt.trace.json found under ${out_base}/tensorboard/" >&2
        return 1
    fi

    echo "============================================================"
    echo "[${tag}] TraceLens generate -> ${csv_dir}"
    echo "  trace: ${trace_json}"
    echo "============================================================"
    rm -rf "${csv_dir}"
    TraceLens_generate_perf_report_pytorch \
        --profile_json_path "${trace_json}" \
        --output_csvs_dir "${csv_dir}" \
        --enable_kernel_summary
}

# Compare a candidate run against its baseline CSV directory.
run_compare() {
    local tag="${1:-}"
    local baseline_dir="${2:-}"
    if [ -z "${tag}" ]; then
        echo "run_compare: missing tag arg" >&2; return 1
    fi
    local csv_dir="${RUNS_ROOT}/perf_${tag}"
    local cmp_xlsx="${RUNS_ROOT}/compare_${tag}.xlsx"

    if [ -z "${baseline_dir}" ]; then
        echo "[${tag}] WARN: baseline dir not provided; skipping compare" >&2
        return 0
    fi
    if [ ! -d "${baseline_dir}" ]; then
        echo "[${tag}] WARN: baseline dir not found (${baseline_dir}); skipping compare" >&2
        return 0
    fi
    if [ ! -d "${csv_dir}" ]; then
        echo "[${tag}] WARN: candidate csv dir not found (${csv_dir}); skipping compare" >&2
        return 0
    fi

    echo "============================================================"
    echo "[${tag}] TraceLens compare -> ${cmp_xlsx}"
    echo "  baseline : ${baseline_dir}"
    echo "  candidate: ${csv_dir}"
    echo "============================================================"
    TraceLens_compare_perf_reports_pytorch \
        "${baseline_dir}" "${csv_dir}" \
        --names "baseline_${tag}" "${tag}" \
        --sheets all \
        -o "${cmp_xlsx}"
}

# --- training + perf reports -----------------------------------------------
for tag in "${SELECTED[@]}"; do
    case "${tag}" in
        bf16)
            run_train bf16  "${SKIP_TRAIN_BF16:-0}"  PR=bf16
            ;;
        fp8)
            run_train fp8   "${SKIP_TRAIN_FP8:-0}"   PR=fp8
            ;;
        mxfp8)
            run_train mxfp8 "${SKIP_TRAIN_MXFP8:-0}" PR=fp8 FP8_RECIPE=mxfp8
            ;;
    esac
    run_perf_report "${tag}"
done

# --- comparisons ------------------------------------------------------------
if [ "${SKIP_COMPARE:-0}" = "1" ]; then
    echo "SKIP_COMPARE set; done."
    exit 0
fi

for tag in "${SELECTED[@]}"; do
    case "${tag}" in
        bf16)  run_compare bf16  "${BASELINE_BF16_DIR}"  ;;
        fp8)   run_compare fp8   "${BASELINE_FP8_DIR}"   ;;
        mxfp8) run_compare mxfp8 "${BASELINE_MXFP8_DIR}" ;;
    esac
done

echo
echo "All done. Artifacts in: ${RUNS_ROOT}"
ls -la "${RUNS_ROOT}"
