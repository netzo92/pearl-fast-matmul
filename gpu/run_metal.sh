#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pearl-metal.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
cd "$repo_dir"
output="${1:-$repo_dir/results/metal-m4.json}"
mode="${2:-prefix}"
export_flag=--export-metal
if [[ "$mode" == cached ]]; then
    export_flag=--export-metal-cached
elif [[ "$mode" != prefix ]]; then
    echo "Usage: bash gpu/run_metal.sh [REPORT_JSON] [prefix|cached]" >&2
    exit 2
fi
cargo run --locked --release -- "$export_flag" "$work_dir/fixtures"
swiftc -O -module-cache-path "$work_dir/swift-modules" gpu/metal_benchmark.swift -o "$work_dir/benchmark"
if [[ "$mode" == cached ]]; then
    "$work_dir/benchmark" "$work_dir/fixtures" gpu/prefix_kernels.metal "$output" --cached
else
    "$work_dir/benchmark" "$work_dir/fixtures" gpu/prefix_kernels.metal "$output"
fi
