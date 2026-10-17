# pearl-fast-matmul

Exact signed-noise grouping experiments for Pearl's integer proof of useful work.

**[Research site](https://netzo92.github.io/pearl-fast-matmul/) · [Latest research note](notes/018-commitment-seed-search.md) · [Raw benchmark samples](results/benchmarks.json) · [Official conformance report](conformance/report.json)**

## What works

For committed raw `B = 0`, the noise has the form `B'[s,j] = V[p_s,j] - V[q_s,j]`. Group contributions by the selected vertices. Within each connected component their coefficients sum to zero, allowing one coefficient to be eliminated. This computes exactly the same intermediate products with fewer scalar multiplications.

- **2,046 local tile tickets checked:** every prefix matrix, tile hash state, and final hash matched direct multiplication across 30 fixtures.
- **Two official CPU verifier cases passed:** independently generated noise, full prefixes, jackpot states, and hashes matched Pearl's implementation. Its actual salted plain-proof verifier accepted both fixtures. Four negative controls rejected corrupted input or a zero target.
- **15.87–17.92% fewer scalar multiplications** in the measured batch fixtures.
- **No demonstrated speed advantage against the stronger CPU baseline.** Packing the direct method's inputs removed the apparent advantage over the original direct loop.

The verifier fixtures use an intentionally easy threshold. They establish functional compatibility with the pinned CPU plain-proof implementation, not a mined block, a verified ZK certificate, or network acceptance. No advantage against a competitive GPU miner or in total mining cost is established.

## Local GPU version

Fixed-B reuse across distinct A candidates now shows a repeatable matrix-stage benefit on this Mac’s M4 GPU. At output shape **2048 × 512**, including A grouping and B preparation amortized over four candidates:

| Synthetic job key | Direct per candidate | Grouped per candidate | Throughput ratio |
| --- | ---: | ---: | ---: |
| 0 | 22.15 ms | 17.29 ms | 1.282x |
| 1 | 22.15 ms | 17.31 ms | 1.280x |

All three GPU paths match **178,257,920 reference prefix entries and 43,520 tile hash states** across 32 fixtures. Each family has four different sparse raw A inputs, fresh commitment-derived A noise and fixed raw B=0; the useful result AB is always zero. Direct receives the same B reuse, and an additional A-packing control pays to match the grouped input layout. Ratios use the faster direct control. Timing rounds replay those finite fixtures.

This is approximately **28% more matrix-stage throughput**, with no established mining advantage. Timings exclude commitments, noise generation, final state/hash computation and proof generation. The baseline is our local tiled integer prototype, rather than Pearl’s NVIDIA miner. [Methodology and limits](notes/012-cached-b-distinct-candidates.md) · [Raw samples](results/metal-cached-m4.json).

```sh
bash gpu/run_metal.sh results/metal-cached-m4.json cached
```

The [earlier uncached GPU experiment](notes/011-metal-prefix-benchmark.md) was effectively tied after preparation. Its benchmark remains available with `bash gpu/run_metal.sh`.

The earlier narrow CPU controls remain faster than CPU grouping: 2.28ms direct versus 4.61ms grouped. These CPU checks ran locally; the H100 cloud run is recorded below.

## Fused MMA16 result on H100

A custom fused direct kernel improves the local matrix-stage control from about **92 µs to 50 µs** at 2048 × 512. The grouped MMA16 path performs fewer dense tensor steps, but still takes **153–155 µs** including B setup amortized across four candidates. Without B setup, it takes about **101 µs grouped versus 46 µs direct**. There is no grouped speed advantage in these tests.

All seven methods pass **44 fixtures, 183,517,184 prefix entries and 44,804 tile states per method**, including a separate check of the state-only timed path. Scalar-load and graph-staging experiments are preserved alongside the retained vector-load implementation. The temporary VM and disk were deleted; estimated compute for the session was about $1.10, excluding other charges and final billing adjustments.

[Methods and results](notes/017-fused-mma16-h100.md) · [Raw samples](results/cuda-mma16-h100.json) · [CUDA code](gpu/cuda/README.md).

## Searching committed inputs for cheaper graphs

We tested **262,144 near-zero B commitments** across four fixed job keys, looking for noise graphs that remove more 16-wide tensor-core steps. Selecting a favorable graph removes only **0.89–1.79% more dense MMA work** than the zero-B baseline in this search. These are arithmetic counts, not new GPU timings. The local search took about two minutes and launched no cloud resources.

Selected inputs reproduce **393,216 prefix entries and 96 tile states/hashes** across fresh A commitments. Omitting the sparse nonzero-B correction fails 32 negative controls. This does not establish new official verifier or network acceptance. The result does not justify another H100 run of the current grouped kernel, which remains slower than direct in the existing measurements.

[Method and limitations](notes/018-commitment-seed-search.md) · [Summary](results/seed-search.json) · [Complete compressed trial logs](results/seed-search-raw/).

```sh
cargo run --offline --locked --release -- --seed-search /tmp/pearl-seed-search 65536
python3 experiments/analyze_seed_search.py /tmp/pearl-seed-search \
  --source-dir . --output /tmp/pearl-seed-search-summary.json
```

## First CUDA result on H100

The [CUDA port](gpu/cuda/README.md) runs correctly on H100, but **grouping is slower than the INT8 cuBLAS control**. At output shape 2048 × 512, including B preparation amortized across four candidates:

| Synthetic job key | Faster direct | Grouped | Grouped throughput / direct |
| --- | ---: | ---: | ---: |
| 0 | 90.84 µs | 290.53 µs | 0.313× |
| 1 | 91.06 µs | 291.24 µs | 0.313× |

Grouping loses all 11 paired rounds in all eight tested cases. Even excluding B preparation, the largest candidate stage takes about 230 µs grouped versus 87 µs direct. The M4 speedup does not transfer to this implementation.

All three CUDA methods match **178,257,920 prefix entries and 43,520 tile states** across 32 fixtures; the state-only timed path is also checked. Exact sparse correction preserves coefficients outside INT8. This is a matrix-stage comparison, not complete mining or a profit measurement, and cuBLAS is not Pearl's fused mining kernel. [Method and results](notes/016-cuda-feasibility.md) · [Raw H100 samples](results/cuda-h100.json).

The temporary GCP VM and its boot disk were deleted after collection. Estimated compute cost was below $1; final billing was not checked.

## CPU results

Apple M4, arm64, one thread; Rust 1.93.1; release build; `r=128`, `k=2048`, tiles `8 × 32`. Medians of seven alternating timing batches, in microseconds per whole batch:

| Matrix output | Tickets | Direct | Packed direct | Grouped, including setup |
| --- | ---: | ---: | ---: | ---: |
| 8 × 32 | 1 | 37.30 | 59.41 | 156.40 |
| 32 × 32 | 4 | 180.12 | 133.16 | 268.61 |
| 128 × 32 | 16 | 711.00 | 425.77 | 719.59 |
| 128 × 128 | 64 | 2,706.75 | 1,743.98 | 2,222.59 |
| 512 × 128 | 256 | 10,878.69 | 6,354.43 | 7,575.48 |

Grouping reuses each graph/contrast plan across row tiles and each grouped row across column tiles. Its timing includes that setup. Packed direct includes copying each chunk into contiguous storage. All paths compute separate tile states and final hashes using the same sliced int32 dot-product helper. Timings exclude commitments, seed expansion/noising, direct B materialization, and proof generation. These are CPU prototypes, not competitive GPU miners. See the [complete samples and scope](results/benchmarks.json).

## Run

Requires Rust/Cargo. The main crate has one pinned dependency, BLAKE3.

```sh
cargo run --release                    # Commitment/permutation experiment
cargo run --release -- --grouping      # Single-tile identity and benchmark
cargo run --release -- --batch-grouping # All-prefix checks, then three CPU baselines
```

Use `--offline` after dependencies are cached. Correctness assertions run before performance measurements; an assertion failure exits unsuccessfully.

Run official conformance with git, stable Rust, and a C compiler:

```sh
bash conformance/run.sh
```

The runner downloads the necessary Pearl directories at commit [`2f8b770cac8f8b74be05c1fc51c2baeb76ce0701`](https://github.com/pearl-research-labs/pearl/tree/2f8b770cac8f8b74be05c1fc51c2baeb76ce0701), adds the local experiment binary, and invokes the unmodified official verifier. Dependencies may take several minutes to build. An existing checkout and output directory can be supplied:

```sh
bash conformance/run.sh /tmp/pearl-checkout /tmp/pearl-results
```

Generated serialized plain proofs and a fresh report are written to the results directory. The checked-in report contains synthetic test headers and seeds, not private mining or wallet material.

## Repository map

- `src/`: seed/noise reproduction, single-tile grouping, and batch benchmark.
- `conformance/`: pinned official implementation integration and recorded evidence.
- `results/`: unrounded benchmark samples and environment.
- `notes/`: research history, proof, limits, and the next question.
- `experiments/`: small Python demonstrations of permutation symmetry, factorization, and demand provenance.
- `docs/`: dependency-free GitHub Pages site, published from `main:/docs`.

To preview the site locally:

```sh
python3 -m http.server 8000 --directory docs
```

Earlier notes preserve the conclusions reached at each step and link to the current result. This repository was extracted from [netzo92/math](https://github.com/netzo92/math).

## Conclusion

The strongest tested grouped H100 implementation is about 3× slower than the fastest direct control at the largest shape. We have no demonstrated NVIDIA mining advantage or higher dollars/hour.

Using [Pearl's illustrative 10% incremental-cost assumption](https://pearlresearch.ai/research/int-whitepaper), a hypothetical **complete-cost** speedup of 1.28x would leave useless mining at about 78% of the reference cost, versus 10% for adding mining to existing inference: roughly 7.8x as expensive. Our 1.28x measurement covers only the local matrix stage, so it does not establish that complete-cost ratio. The estimate gives a conditional threshold; it does not demonstrate an exploit. [Calculation and ELI5](notes/013-inference-overhead-threshold.md#conclusion-using-pearls-own-estimate).

## Chain evidence and inference overhead

- [Chain inspection](notes/014-chain-observability.md): the checked proof schema has no required customer receipt. Ten public API block responses (129036–129045) omit certificate data, so their real-inference status is unknown. [Report](results/chain-observability.json).
- [Inference overhead](notes/015-measuring-inference-overhead.md): Pearl reports 3.9% and 5.08% for specific H200 serving setups. Those figures remain unverified here. The official code has a mining-off control for a matched A/B measurement; the earlier 10% figure is an economic example.

## Next question

Can structure on both sides, or a way to compute the exact required XOR states without evaluating every prefix entry, save substantially more work than one-sided graph grouping? [The latest search](notes/018-commitment-seed-search.md) found only small additional arithmetic savings. An eventual mining claim must also pass the [complete reward-normalized cost comparison](notes/013-inference-overhead-threshold.md).
