#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
output="${1:-$source_dir/libpearl_fast_matmul.so}"
arch="${CUDA_ARCH:-sm_90}"
command -v nvcc >/dev/null || { echo 'nvcc is required (CUDA 12.8+ devel toolkit).' >&2; exit 1; }
case "$arch" in sm_80|sm_86|sm_89|sm_90) ;; *) echo "Unsupported CUDA_ARCH: $arch" >&2; exit 2 ;; esac
mkdir -p -- "$(dirname -- "$output")"
nvcc --std=c++17 -O3 -lineinfo -arch="$arch" -Xcompiler=-fPIC \
    --shared "$source_dir/benchmark.cu" -lcublas -o "$output"
echo "Built $output ($arch)"
