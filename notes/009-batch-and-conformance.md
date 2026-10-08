# Shared grouping work and official CPU conformance

## Question

Can reusing graph and contrast preparation across row tiles make exact signed-noise grouping faster per ticket, and does the construction agree with Pearl's official integer verifier?

## Algorithm and model

The component identity is proved in [the grouping note](008-pearl-noise-grouping.md). For each width-r chunk, prepare its component basis and all column contrasts once. Reuse this plan for every row tile. Group each A' row once, then reuse the grouped coefficients across all column tiles. Each 8 by 32 tile retains its own cumulative prefix state and keyed jackpot hash.

Dimensions of the fully committed matrices are used when deriving salted seeds. Moving from one row tile to many therefore regenerates A's noise correctly; it does not repeat the original A rows or ticket. B's seed and grouping plan are legitimately shared by the rows of that same committed job.

Arithmetic savings remain a constant-factor observation, not an asymptotic result. Coefficients can exceed int8. Timings use scalar int32 expressions compiled with CPU vectorization and include graph/contrast setup, row grouping, tile state accumulation, and keyed hashes. The packed direct control includes packing. Both multiplication loops use the same sliced dot-product helper.

## Correctness evidence

`cargo run --release -- --batch-grouping` tests five output shapes, three deterministic job keys per shape, and zero/nonzero raw A, with raw B=0: 30 fixtures and 2,046 tile tickets. Every prefix entry, separate tile state, and final hash agrees among direct, packed direct, and grouping implementations before timing begins. These local fixture job keys are synthetic.

Separately, `bash conformance/run.sh` constructs two actual header/config-derived fixtures, official Merkle trees/openings, and salted INT V3 parameters at the pinned [Pearl revision](https://github.com/pearl-research-labs/pearl/tree/2f8b770cac8f8b74be05c1fc51c2baeb76ce0701).

For each fixture, the independent noise reproduction matches all 81,920 official noise entries. All 16 complete prefix matrices match direct computation. The grouped state and keyed hash equal official `compute_jackpot` and `compute_jackpot_hash`, and official `verify_plain_proof` accepts. A corrupted input byte and a zero-target override each fail verification. The [recorded report](../conformance/report.json) contains the exact headers, roots, seeds, states, hashes, and errors.

The threshold `0x207fffff` saturates the scaled target. Acceptance verifies functional conformance. It does not establish network-difficulty success, a ZK certificate, or live-network acceptance. The official conformance harness independently implements the same identity; it does not run the batch implementation against every official tile. No claim of official validation for all 2,046 local tickets is made.

## CPU measurements

Apple M4, arm64, single thread; rustc 1.93.1 (01f6ddf75 2026-02-11); Cargo release. Parameters r128, k2048, tile8 by32. Seven alternating batches per path; 244/61/15/4/4 repetitions respectively. [Full unrounded samples](../results/benchmarks.json) and [console output](../results/batch-cpu.txt) are retained.

| Output shape | Tickets | Direct, microseconds | Packed direct | Grouped including plan |
| --- | ---: | ---: | ---: | ---: |
| 8 by 32 | 1 | 37.30 | 59.41 | 156.40 |
| 32 by 32 | 4 | 180.12 | 133.16 | 268.61 |
| 128 by 32 | 16 | 711.00 | 425.77 | 719.59 |
| 128 by 128 | 64 | 2706.75 | 1743.98 | 2222.59 |
| 512 by 128 | 256 | 10878.69 | 6354.43 | 7575.48 |

For the largest batch, grouping costs 29.59 microseconds per ticket versus 24.82 for packed direct, approximately 19.2% longer. Multiplication counts are 15.87% lower for that fixture; this reduction does not overcome the other costs. Grouping remains slower even if its graph/contrast plan is cached in that measured case (7.11 ms versus 6.35 ms).

The initial row-sharing benchmark appeared 1.36 times faster than an unpacked direct baseline. That comparison had different memory access patterns. Adding chunk packing to direct multiplication reversed the result. The reported comparison uses the better of direct and packed direct; selecting only the weaker baseline would be misleading.

Timings exclude raw commitment hashing, seed expansion, A noising, direct B materialization, and proof generation. Packed direct starts from already-materialized B'. Both paths produce all tile states and hashes but omit full prefix copies during timing. Some raw batches contain scheduling outliers; medians and every sample are published. The CPU methods are neither GPU mining kernels nor optimized library GEMMs.

## Result and scope

**Exact grouping agrees with the pinned official CPU implementation and passes plain verification on two functional fixtures. Sharing preparation reduces overhead, but does not beat packed direct computation on this CPU.** The conjectured unfair mining advantage remains unestablished. No broader proof-of-useful-work impossibility follows from these measurements.

## Next question

Can grouping's coefficient construction and multiplication be implemented faster than packed direct computation while retaining the same exact intermediate products and accounting for arithmetic width on GPU hardware?
