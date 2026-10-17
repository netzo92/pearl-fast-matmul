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
profile_args=()
if [[ "${PEARL_PROFILE_STAGES:-0}" == 1 ]]; then profile_args+=(--profile-stages); fi
"$python_bin" -c 'import numpy' || { echo 'Install gpu/cuda/requirements.txt in your Python environment.' >&2; exit 1; }
if [[ "$mode" == cuda ]]; then
    bash gpu/cuda/build.sh "$work_dir/libpearl_fast_matmul.so"
fi
if [[ "$mode" == cuda ]]; then
    # Smaller tiles and dense nonzero A exercise boundaries and larger residuals.
    cargo run --locked --release -- --export-metal "$work_dir/original-fixtures"
    "$python_bin" gpu/cuda/benchmark.py "$work_dir/original-fixtures" \
        --library "$work_dir/libpearl_fast_matmul.so" --rounds 3 --warmups 1 \
        --output "$work_dir/original-report.json"
fi
# Existing portable fixture format; four distinct A commitments per fixed-B job.
cargo run --locked --release -- --export-metal-cached "$work_dir/fixtures"
if [[ "$mode" == host ]]; then
    "${CXX:-c++}" -std=c++17 -O2 gpu/cuda/plan_check.cpp -o "$work_dir/plan-check"
    "$python_bin" gpu/cuda/benchmark.py "$work_dir/fixtures" --validate-only \
        --plan-checker "$work_dir/plan-check" --output "$output"
else
    "$python_bin" gpu/cuda/benchmark.py "$work_dir/fixtures" \
        --library "$work_dir/libpearl_fast_matmul.so" "${profile_args[@]}" --output "$output"
    "$python_bin" - "$output" "$work_dir/original-report.json" <<'PYREPORT'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads(path.read_text())
original = json.loads(Path(sys.argv[2]).read_text())
report['additional_gpu_checks'] = original['checks']
report['additional_gpu_check_cases'] = original['cases']
report['additional_gpu_check_scope'] = 'Original 12 fixtures exercise m=8, n=32 and dense nonzero A; 3 timing rounds are retained but not used for primary speed claims.'
path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
PYREPORT
fi
