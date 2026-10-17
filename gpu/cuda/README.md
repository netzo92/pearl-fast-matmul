# CUDA matrix-stage benchmark

Question: does exact zero-B grouping beat an INT8 cuBLAS control on NVIDIA hardware once grouping, corrections and B preparation are counted?

This is a feasibility benchmark. It does not mine, contact a pool, run inference or measure profit. It uses the same portable Rust fixtures as the Metal experiment.

## Methods

Seven paths share the same fixed-B fixtures and exact required prefix/state semantics:

1. Direct INT8 cuBLAS strided-batched chunk products and state reduction.
2. Grouped cuBLAS with per-candidate grouping and exact residual correction.
3. Direct cuBLAS with per-candidate A packing.
4. Fused direct `m16n8k16` tensor instructions, prefix accumulation and state reduction.
5. Prepared grouping followed by fused MMA16, corrections and states.
6. Grouping inside the shared-memory load followed by fused MMA16, corrections and states.
7. Inline grouping with dense width forced to 128 as a padding control.

Ratios compare the faster grouped MMA16 method (5 or 6) with the fastest direct control (1, 3 or 4). The legacy grouped method and padded control remain in every report. A new graph is prepared per B job, and grouping is paid for each A candidate. Grouped coefficients outside INT8 use exact signed residuals; no contribution is dropped.

The legacy cuBLAS methods use inner dimension 128 and materialize chunk products. The fused methods retain accumulators in registers. Grouped MMA16 rounds the active dimension to 16, usually 112 for these fixtures. All methods check every prefix in correctness mode and separately check the state-only timed path. Timings omit full-prefix writes for every method.

## Run

Requires Linux, an NVIDIA GPU, CUDA 12.8+ development toolkit with cuBLAS, a C++ compiler, Rust/Cargo, Python 3.11+ and NumPy. The default target is SM90 (H100/H200); `CUDA_ARCH=sm_80`, `sm_86` or `sm_89` can build the standalone experiment for other cards. That does not establish support for Pearl's official miner on those cards.

```sh
python3 -m venv /tmp/pearl-cuda-env
/tmp/pearl-cuda-env/bin/pip install -r gpu/cuda/requirements.txt
PYTHON=/tmp/pearl-cuda-env/bin/python bash gpu/cuda/run.sh results/cuda-matrix-stage.json
```

The runner exports all 32 cached fixtures. It checks all seven GPU paths before recording timings; a mismatch aborts. Eleven rounds rotate method order and include B preparation amortized across four candidates. Rounds replay those finite fixtures. Buffers, allocations, compilation and initial fixture uploads are outside timing.

For a shorter first check, prepare fixtures once and select a shape:

```sh
cargo run --locked --release -- --export-metal-cached /tmp/pearl-cuda-fixtures
bash gpu/cuda/build.sh /tmp/libpearl_fast_matmul.so
/tmp/pearl-cuda-env/bin/python gpu/cuda/benchmark.py /tmp/pearl-cuda-fixtures \
  --library /tmp/libpearl_fast_matmul.so --shape 128x128 --output /tmp/cuda-smoke.json
```

Host validation of the arithmetic, exported bases, residuals, prefixes and states needs no CUDA:

```sh
PYTHON=/tmp/pearl-cuda-env/bin/python bash gpu/cuda/run.sh /tmp/cuda-host-check.json host
```

For the smaller original 12-fixture set, use `--export-metal` instead of `--export-metal-cached`, then invoke `benchmark.py --validate-only`. Host validation also compiles the same C++ planning code used by CUDA and checks it against Python and Rust when `--plan-checker` is supplied.

## Recorded H100 result

The retained vector-load implementation takes about **50 µs direct versus 153–155 µs grouped** at 2048 × 512, including B preparation amortized across four candidates. Without B preparation, it takes about 46 µs direct versus 101 µs grouped. All seven methods pass 44 fixtures, including small boundary tiles and dense nonzero A.

[Latest results and iterations](../../notes/017-fused-mma16-h100.md) · [Raw MMA16 samples](../../results/cuda-mma16-h100.json) · [First CUDA experiment](../../notes/016-cuda-feasibility.md).

For GCP serial collection, the startup script emits compressed chunks with a SHA256 of the original JSON. Recover them with:

```sh
python3 gpu/cuda/collect_serial.py serial.log report.json
```

## Interpretation

Reports retain every timing sample, device/runtime information and source/fixture SHA256 hashes. `wall_us_per_candidate_including_amortized_b` is the primary measurement. CUDA events report elapsed stream span, which can include host launch/copy gaps; this is not active GPU execution time.

Excluded: input commitments, seed/noise generation, generation of new candidates, final keyed BLAKE3, winning proofs and verification, network acceptance, fees, inference and customer revenue. This is neither complete mining throughput nor a dollars/hour result. The baseline is the fastest of cuBLAS and our fused direct MMA16 kernel, **not Pearl's fused NVIDIA kernel**. A win requires a subsequent comparison with that implementation. A loss only rejects this CUDA implementation and shape, not every possible grouping implementation.

References: [cuBLAS strided-batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex), [current scope of the M4 result](../../notes/012-cached-b-distinct-candidates.md).
