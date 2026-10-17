#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pearl-metal.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
cd "$repo_dir"
output="${1:-$repo_dir/results/metal-m4.json}"
cargo run --locked --release -- --export-metal "$work_dir/fixtures"
swiftc -O -module-cache-path "$work_dir/swift-modules" gpu/metal_benchmark.swift -o "$work_dir/benchmark"
"$work_dir/benchmark" "$work_dir/fixtures" gpu/prefix_kernels.metal "$output"
