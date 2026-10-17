# pearl-fast-matmul

Exact signed-noise grouping experiments for Pearl's integer proof of useful work.

**[Research site](https://netzo92.github.io/pearl-fast-matmul/) · [Latest research note](notes/013-inference-overhead-threshold.md) · [Raw benchmark samples](results/benchmarks.json) · [Official conformance report](conformance/report.json)**

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

The earlier narrow CPU controls remain faster than CPU grouping: 2.28ms direct versus 4.61ms grouped. No cloud resources have been created.

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

## Next question

Can complete grouped useless mining beat the **incremental cost of adding mining to useful inference**, per expected accepted reward, on the same hardware? No predetermined speedup factor is required. The current matrix-stage measurements do not measure that incremental cost. [Cost model and next measurement](notes/013-inference-overhead-threshold.md).
