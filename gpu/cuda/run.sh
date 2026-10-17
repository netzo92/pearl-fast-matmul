#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pearl-cuda.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
cd "$repo_dir"
output="${1:-$repo_dir/results/cuda-matrix-stage.json}"
mode="${2:-cuda}"
case "$mode" in
    cuda|host) ;; *) echo "Usage: bash gpu/cuda/run.sh [REPORT_JSON] [cuda|host]" >&2; exit 2 ;;
esac
python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import numpy' || { echo 'Install gpu/cuda/requirements.txt in your Python environment.' >&2; exit 1; }
if [[ "$mode" == cuda ]]; then
    bash gpu/cuda/build.sh "$work_dir/libpearl_fast_matmul.so"
fi
# Existing portable fixture format; four distinct A commitments per fixed-B job.
cargo run --locked --release -- --export-metal-cached "$work_dir/fixtures"
if [[ "$mode" == host ]]; then
    "${CXX:-c++}" -std=c++17 -O2 gpu/cuda/plan_check.cpp -o "$work_dir/plan-check"
    "$python_bin" gpu/cuda/benchmark.py "$work_dir/fixtures" --validate-only \
        --plan-checker "$work_dir/plan-check" --output "$output"
else
    "$python_bin" gpu/cuda/benchmark.py "$work_dir/fixtures" \
        --library "$work_dir/libpearl_fast_matmul.so" --output "$output"
fi
