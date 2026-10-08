#!/usr/bin/env bash
# CPU/plain-proof conformance only. Requires git, a stable Rust toolchain, and cc.
# Usage: ./run.sh [Pearl checkout directory] [results directory]
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
pearl_checkout="${1:-${TMPDIR:-/tmp}/pearl-fast-matmul-upstream}"
results_dir="${2:-$script_dir/results}"
commit='2f8b770cac8f8b74be05c1fc51c2baeb76ce0701'
if [[ ! -e "$pearl_checkout" ]]; then
  git init "$pearl_checkout"
  git -C "$pearl_checkout" remote add origin https://github.com/pearl-research-labs/pearl.git
  git -C "$pearl_checkout" sparse-checkout init --cone
  git -C "$pearl_checkout" sparse-checkout set zk-pow pearl-blake3 plonky2
  git -C "$pearl_checkout" fetch --depth 1 --filter=blob:none origin "$commit"
  git -C "$pearl_checkout" checkout --detach FETCH_HEAD
fi
actual="$(git -C "$pearl_checkout" rev-parse HEAD)"
if [[ "$actual" != "$commit" ]]; then
  echo "Expected pinned Pearl commit $commit, found $actual" >&2
  exit 1
fi
if [[ -n "$(git -C "$pearl_checkout" status --porcelain --untracked-files=no)" ]]; then
  echo 'Tracked Pearl source has local modifications; use a clean checkout.' >&2
  exit 1
fi
cp "$script_dir/grouping_conformance.rs" "$pearl_checkout/zk-pow/src/bin/grouping_conformance.rs"
cargo run --locked --release --manifest-path "$pearl_checkout/zk-pow/Cargo.toml" --bin grouping_conformance -- "$results_dir"
