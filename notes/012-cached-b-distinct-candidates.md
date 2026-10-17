# Reusing zero-B preparation across distinct A candidates on Apple M4

## Question

Does fixed-B reuse make exact grouping consistently faster at the matrix-computation stage when each candidate has a different committed raw A and freshly derived A noise?

## Construction and controls

All raw B entries are zero. For candidate t in 0..3, raw A is zero except A[0,t]=1. Thus every useful result AB is zero. Moving the 1 changes A's commitment, its noise seed, the noised A operand, and the keyed ticket hashes. This does not reuse a previous transcript or hash.

The V3 seed chain derives B's seed from the job key and B commitment/dimension; A's commitment/dimension subsequently affects A's seed. Holding the job key and B fixed therefore permits reusing B's selection graph and contrasts across these candidates. A new job key invalidates that cache. The same derivation and exact grouping identity were checked against the pinned official CPU code in [note 009](009-batch-and-conformance.md); the new GPU cases here use the local reference.

Both direct controls get equivalent reuse: materialize and pack B once, then use it for all four A operands. One additionally packs A into the same chunk/row layout as grouped A, paying that GPU packing cost for every candidate. The comparison uses the faster direct control per case. Grouped computation builds its graph/CSR and contrasts once, but runs GPU A grouping for **every** candidate. Both paths use the existing tiled integer kernels and write every prefix matrix. There is no new arithmetic shortcut in this iteration.

Four shapes and two predetermined synthetic job keys are used, without seed selection. All four A seeds, noised operands and reference hash arrays must differ within each family. All B operands, V arrays and selection pairs must agree. The CPU exporter also checks the regenerated B and changed A seeds. A hash-array inequality is a fixture freshness check, not a measurement of independent network successes.

## Correctness and measurements

The uncached and actual cached GPU executions both match the independent direct Rust reference for every prefix and tile state. Across 32 fixtures, this covers **178,257,920 prefix entries and 43,520 ticket states**. Final BLAKE3 hashes are computed by the Rust reference, rather than the GPU host. These new fixtures have not been submitted to the official verifier or a live network.

Apple M4; macOS 26.3.1; Swift 6.2.4, `swiftc -O`; integer Metal kernels. Parameters r=128, k=2048, tiles 8 by 32. Eleven rounds per case, rotating the order of all three methods. Each timed command computes four distinct, already prepared A operands, with one command completion wait. Later timing rounds replay that finite fixture set; they do not generate additional mining candidates.

The table gives median **milliseconds per candidate**, including B preparation amortized over four candidates. Each sample is candidate-batch wall time divided by four plus its measured B preparation wall time divided by four. Ratios use the medians of those sums. Every sample is in [the raw report](../results/metal-cached-m4.json).

| Output | Job key id | Direct | A-packed direct | Grouped | Ratio vs faster direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 by 128 | 0 | 0.532 | 0.542 | 0.512 | 1.038x |
| 128 by 128 | 1 | 0.515 | 0.529 | 0.502 | 1.027x |
| 512 by 128 | 0 | 1.553 | 1.590 | 1.415 | 1.098x |
| 512 by 128 | 1 | 1.553 | 1.588 | 1.390 | 1.117x |
| 512 by 512 | 0 | 5.704 | 5.718 | 4.528 | 1.260x |
| 512 by 512 | 1 | 5.690 | 5.706 | 4.537 | 1.254x |
| 2048 by 512 | 0 | 22.152 | 22.194 | 17.285 | 1.282x |
| 2048 by 512 | 1 | 22.151 | 22.187 | 17.308 | 1.280x |

The largest cases show about 28% more matrix-stage throughput, equivalent to approximately 22% less elapsed time. Packing A does not improve the direct baseline in these cases. The grouped method wins all eleven paired rounds for both largest shapes and both keys. The smallest case has large scheduling variation, so its marginal median difference is not a useful speed claim. Two job keys do not establish behavior over the full job distribution.

Both methods include command encoding/completion and the same full-prefix output traffic. B setup includes direct GPU materialization or grouped CPU graph construction/copying plus GPU contrasts. A grouping is timed for every grouped candidate. Buffers and candidate A inputs are resident before timing. Shader compilation, Metal buffer allocation, input commitments, seed expansion, A noising, final state/hash generation and proof generation are excluded. Grouped and direct timings share these exclusions. This is not a total mining-cost measurement.

The baseline is the faster of our tiled integer Metal kernels, with chunk-packed B and with or without per-candidate A packing. There is no comparison against an optimized Metal GEMM library or Pearl's NVIDIA miner. A faster local matrix stage is not sufficient evidence of an exploitable mining advantage.

## How much more can this identity save?

These fixtures remove about 16–17% of scalar multiplications. Considering multiplication count alone gives roughly a 1.2x ratio; elapsed ratios can differ because of input layouts, loads, integer widths and other costs. Reuse and larger batches amortize overhead, but do not turn this identity into a manyfold arithmetic reduction.

Writing both zero-input noise operands as sparse-selection factorizations exposes a small middle matrix. Reassociating those factors alone does not establish a further speedup: every required prefix must still be computed, and forming the intermediate factors adds work. No faster two-sided algorithm is established here.

For larger gains, a next implementation could retain each tile's accumulators and produce its state on the GPU instead of writing and rereading complete prefix matrices. The direct control must receive the same output reduction, and any resulting advantage must be measured again. A manyfold gain is not the success criterion: a stable improvement in complete cost per expected accepted reward can be sufficient. Competing with an existing inference provider additionally requires comparison against its incremental mining cost; see [the cost model](013-inference-overhead-threshold.md).

## Reproduce

```sh
bash gpu/run_metal.sh results/metal-cached-m4.json cached
```

The default mode remains the original uncached benchmark:

```sh
bash gpu/run_metal.sh results/metal-m4.json
```

Both require a Metal-capable Mac, Cargo and Swift. Fixtures are generated in a temporary directory and removed by the runner. No cloud resources are used. Original fixture binaries are unchanged after factoring out the shared exporter, and the original GPU mode still passes its 12 fixtures.

## Result, scope, uncertainty, and next question

**Fixed-B reuse and larger batches give a repeatable matrix-stage benefit against the existing direct Metal prototype: about 1.28x at the largest shape on both tested job keys.** The raw useful result is always zero, while different A inputs produce fresh noise and reference hashes. This establishes a faster local computation path for these fixtures. It does not establish a competitive miner, network acceptance, or a break of the paper's hardness conjecture.

The main uncertainties are the excluded candidate-generation/hash/proof costs, the limited job-key sample and the strength of the baseline. Can the advantage survive the complete candidate/proof pipeline and beat the measured incremental mining cost of useful inference?
