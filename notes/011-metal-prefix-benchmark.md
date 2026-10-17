# Exact direct and grouped prefix computation on Apple M4

## Question

Can the grouping identity reduce GPU time on this Mac after accounting for grouping preparation and a tiled direct baseline?

## Implementation

`gpu/prefix_kernels.metal` implements both paths as integer Metal kernels. A 256-thread group owns an 8 by 32 output tile. Cooperative contiguous loads stage each rank-128 chunk in shared memory; each thread maintains an int32 output accumulator across all chunks. Both paths write every cumulative prefix matrix. Direct B is packed by chunk, giving both paths comparable memory layouts.

The grouped path constructs connected-component bases and signed-incidence lists on the CPU. The GPU gathers grouped A coefficients into int16 storage, constructs int8 V contrasts, and multiplies with the actual reduced inner dimension. Loading shared memory uses fixed strides to avoid repeated division by the variable dimension. Integer bounds are unchanged from the earlier proof; no floating-point rounding is used.

## Correctness

Rust exports commitment-derived noise fixtures and independent direct-reference prefixes and tile hash states. The Swift host independently reconstructs the graph and incidence lists and checks them against the exported basis. Both GPU paths match all **5,259,264 prefix entries** and **1,284 tile hash states** across 12 fixtures: three output shapes, two deterministic job keys, and zero/nonzero raw A, with raw B=0. Parameters are r=128, k=2048, tile 8 by 32.

The fixture header/job keys are synthetic. This is agreement with the local Rust reference. Earlier official CPU plain-proof checks remain a separate result. Final keyed hashes are not computed or timed on the GPU; exact state equality implies they would match under the same key.

## Measurements

Apple M4, macOS 26.3.1, Apple Swift 6.2.4 with `swiftc -O`, runtime Metal compilation. Seven alternating samples; kernel-only GPU samples average ten dispatches per command. Timing fixtures use job id0 and A=0; correctness covers both input variants. [Raw samples and scope](../results/metal-m4.json).

| Output | Tickets | Direct job wall | Grouped job wall | Direct GPU kernel | Grouped GPU kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 by 32 | 1 | 657.38 us | 822.29 us | 209.48 us | 257.09 us |
| 128 by 128 | 64 | 975.79 us | 1312.17 us | 629.59 us | 523.17 us |
| 512 by 128 | 256 | 1784.17 us | 1750.25 us | 1506.80 us | 1209.79 us |

At the largest shape, the prepared grouped prefix kernel is about 1.25 times as fast as this tiled direct kernel. Including preparation and dispatch/waiting produces effectively tied job times: 1.78ms versus 1.75ms. The roughly 2% median difference is small compared with the sample variation and is **not a demonstrated robust total-time advantage**. Smaller cases are slower overall. Raw samples show GPU frequency/host scheduling variation; this is an initial prototype measurement.

Both job paths start from A', V and signed B selections. Direct includes GPU B materialization into chunk-packed storage. Grouping includes fresh CPU graph/incidence construction and copying, GPU A grouping, and GPU contrasts. Job wall time includes command encoding and waiting. Metal buffer allocation, shader compilation, fixture loading, raw commitments, A noising, final hashing, and proof generation are excluded. Host graph array allocations are included. Kernel-only timings exclude all preparation. Full-prefix storage is additional traffic beyond a miner and is identical on both paths.

This is a comparison of two local tiled integer prototypes, not a comparison against an optimized Metal library or Pearl's NVIDIA miner.

## Reproduce

```sh
bash gpu/run_metal.sh
```

Requires a Metal-capable Mac, Rust/Cargo, and Swift with the Metal framework. The script creates temporary binary fixtures, compiles the host, checks each fixture before timing it, saves `results/metal-m4.json`, and removes its temporary build files. No cloud resources are used.

## Result and next question

**The GPU implementation works and matches the reference. Kernel arithmetic improves at larger sizes, while preparation currently removes a clear overall advantage.** No competitive miner or network advantage is established.

Can caching B-dependent graph/contrast preparation across changing A inputs preserve a total-time advantage after adding commitment, A-noise and final-hash costs? The direct control must receive equivalent B reuse.
