# Narrow integer kernels and local GPU access

The full Metal implementation and measurements are now in [note011](011-metal-prefix-benchmark.md).

## Question

Does narrower integer storage make the exact grouping construction cheaper than equally narrowed direct multiplication?

## Result

No CPU advantage in this iteration. All prefixes, separate ticket states, and hashes agree for 18 fixtures comprising 1,926 tickets. At output shape 512 by 128, r128,k2048, the median batch times were 2.279ms for packed direct int8, 3.290ms for packed direct int16, 6.165ms for packed direct int32, 5.081ms for grouped int16, and 4.606ms for clipped-int8 grouping with exact sparse corrections. Seven alternating batches of four calls on Apple M4, single thread, Rust 1.93.1 release. Setup and packing are included; commitments,noising,input storage conversions,and proofs are excluded. Full samples are in `results/narrow-cpu.txt`. These narrower controls supersede any interpretation of the prior int32 baseline as the fastest local method.

`cargo run --release -- --narrow-grouping` reproduces the comparison. `cargo run --release -- --width-stats` reproduces a separate 16-fixture coefficient study, preserving all exact prefixes/states/hashes. Clipping grouped coefficients to int8 plus a sparse residual is an identity, with no rounding loss. For raw A=B=0,0.4133% of grouped coefficients exceeded int8 in that study; for nonzero A,6.7112% exceeded it. Raw results: `results/coefficient-widths.txt`.

## GPU mapping constraint

At the pinned Pearl revision, the Hopper INT8 WGMMA path uses inner-dimension32 instructions. The observed grouped dimensions 100–115 all pad to 128. This erases dense instruction-count savings for independently computed rank128 chunks. Other GPU instruction paths require their own implementations and comparisons. Source: [pinned kernel traits](https://github.com/pearl-research-labs/pearl/blob/2f8b770cac8f8b74be05c1fc51c2baeb76ce0701/miner/pearl-gemm/csrc/gemm/kernel_traits.hpp).

## Local Metal smoke test

This machine has a 10-core Apple M4 GPU with Metal support. `gpu/metal_smoke.swift` executed a signed-int8 to int32 prefix calculation on that GPU and matched all 4,096 prefix entries against a CPU reference. These are synthetic operands; this checks GPU execution and integer exactness, not Pearl noise, verifier conformance, grouping speed, or mining advantage.

```sh
swiftc -O gpu/metal_smoke.swift -o /tmp/pearl-metal-smoke
/tmp/pearl-metal-smoke
```

No cloud resources were created. A local Metal implementation can test the algorithm without GPU rental. Apple GPU results do not predict performance against Pearl's NVIDIA-specific Hopper kernel.

## Scope and uncertainty

The exact grouping identity and earlier official plain-proof conformance remain valid. Narrowing improves both candidates and direct controls, with direct int8 still faster. No GPU grouping implementation or competitive miner has been demonstrated.

## Next question

Can a Metal implementation of direct and grouped integer prefix computation achieve a reproducible advantage on the same M4 GPU after including grouping and dispatch costs?
