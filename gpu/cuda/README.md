# CUDA matrix-stage benchmark

Question: does exact zero-B grouping beat an INT8 cuBLAS control on NVIDIA hardware once grouping, corrections and B preparation are counted?

This is a feasibility benchmark. It does not mine, contact a pool, run inference or measure profit. It uses the same portable Rust fixtures as the Metal experiment.

## Methods

1. Direct INT8 cuBLAS strided-batched GEMM, one independent product per rank-128 chunk.
2. Grouped INT8 cuBLAS with exact sparse residual correction. Grouped coefficients are clipped to signed INT8; all remaining signed values are added back using INT16 residuals and INT32 accumulation. There is no quantization loss.
3. Direct INT8 cuBLAS with A packed into the grouped chunk layout. Its per-candidate packing cost is included. Ratios use the faster direct control.

All methods get fixed-B reuse, the same candidates, identical chunk-product storage and identical fused prefix accumulation/tile-state reduction. Correctness mode writes every prefix and compares it with the Rust direct reference. It checks the state-only timed path separately. Timing omits full-prefix writes for every method.

The dense inner dimension is **128 on both paths**. The active grouped dimensions often round up to 128 on tensor cores. This implementation tests whether any wall-time benefit remains despite that padding and the residual correction; it does not assume the Metal speedup transfers.

## Run

Requires Linux, an NVIDIA GPU, CUDA 12.8+ development toolkit with cuBLAS, a C++ compiler, Rust/Cargo, Python 3.11+ and NumPy. The default target is SM90 (H100/H200); `CUDA_ARCH=sm_80`, `sm_86` or `sm_89` can build the standalone experiment for other cards. That does not establish support for Pearl's official miner on those cards.

```sh
python3 -m venv /tmp/pearl-cuda-env
/tmp/pearl-cuda-env/bin/pip install -r gpu/cuda/requirements.txt
PYTHON=/tmp/pearl-cuda-env/bin/python bash gpu/cuda/run.sh results/cuda-matrix-stage.json
```

The runner exports all 32 cached fixtures. It checks all three GPU paths before recording timings; a mismatch aborts. Eleven rounds rotate method order and include B preparation amortized across four candidates. Rounds replay those finite fixtures. Buffers, allocations, compilation and initial fixture uploads are outside timing.

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

On H100 80GB, the largest shape takes about **291 µs grouped versus 91 µs direct**, including B preparation amortized over four candidates. Grouping loses all eleven paired rounds in every tested case. All three paths pass the prefix/state checks. [Results and limits](../../notes/016-cuda-feasibility.md) · [Raw samples](../../results/cuda-h100.json).

For GCP serial collection, the startup script emits compressed chunks with a SHA256 of the original JSON. Recover them with:

```sh
python3 gpu/cuda/collect_serial.py serial.log report.json
```

## Interpretation

Reports retain every timing sample, device/runtime information and source/fixture SHA256 hashes. `wall_us_per_candidate_including_amortized_b` is the primary measurement. CUDA events report elapsed stream span, which can include host launch/copy gaps; this is not active GPU execution time.

Excluded: input commitments, seed/noise generation, generation of new candidates, final keyed BLAKE3, winning proofs and verification, network acceptance, fees, inference and customer revenue. This is neither complete mining throughput nor a dollars/hour result. The baseline is cuBLAS with our state reduction, **not Pearl's fused NVIDIA kernel**. A win requires a subsequent comparison with that implementation. A loss only rejects this CUDA implementation and shape, not every possible grouping implementation.

References: [cuBLAS strided-batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex), [current scope of the M4 result](../../notes/012-cached-b-distinct-candidates.md).
