# pearl-fast-matmul

Exact signed-noise grouping experiments for Pearl's integer proof of useful work.

**[Research site](https://netzo92.github.io/pearl-fast-matmul/) · [Latest research note](notes/011-metal-prefix-benchmark.md) · [Raw benchmark samples](results/benchmarks.json) · [Official conformance report](conformance/report.json)**

## What works

For committed raw `B = 0`, the noise has the form `B'[s,j] = V[p_s,j] - V[q_s,j]`. Group contributions by the selected vertices. Within each connected component their coefficients sum to zero, allowing one coefficient to be eliminated. This computes exactly the same intermediate products with fewer scalar multiplications.

- **2,046 local tile tickets checked:** every prefix matrix, tile hash state, and final hash matched direct multiplication across 30 fixtures.
- **Two official CPU verifier cases passed:** independently generated noise, full prefixes, jackpot states, and hashes matched Pearl's implementation. Its actual salted plain-proof verifier accepted both fixtures. Four negative controls rejected corrupted input or a zero target.
- **15.87–17.92% fewer scalar multiplications** in the measured batch fixtures.
- **No demonstrated speed advantage against the stronger CPU baseline.** Packing the direct method's inputs removed the apparent advantage over the original direct loop.

The verifier fixtures use an intentionally easy threshold. They establish functional compatibility with the pinned CPU plain-proof implementation, not a mined block, a verified ZK certificate, or network acceptance. No GPU or end-to-end mining advantage is established.

## Local GPU version

Both direct and grouped prefix kernels now run on this Mac’s M4 GPU. They match **5,259,264 reference prefix entries and 1,284 tile hash states** across 12 commitment-derived fixtures.

For the largest tested batch, the prepared grouped kernel took **1.21ms versus 1.51ms** for the tiled direct GPU kernel. Including grouping preparation and command overhead gave **1.75ms versus 1.78ms**: effectively tied given the sample variation. Smaller batches were slower overall. This is a local prototype comparison, with no established mining advantage. [Full methodology and limits](notes/011-metal-prefix-benchmark.md) · [Raw GPU samples](results/metal-m4.json).

```sh
bash gpu/run_metal.sh
```

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

Can grouped coefficient construction and multiplication beat packed direct computation while accounting for the wider grouped coefficients, particularly on GPU integer hardware?
