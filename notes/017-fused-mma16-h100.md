# Fused 16-wide tensor-core experiment on H100

## Question

Does a fused MMA16 kernel preserve the grouped arithmetic saving and beat the fastest direct control after grouping, exact correction and B preparation?

## Implementation and controls

NVIDIA documents signed INT8 `mma.sync.m16n8k16` on SM80 and later. The SM90 build emits actual `IMMA.16816.S8.S8` instructions, verified from cuobjdump output in [the compile check](https://github.com/netzo92/pearl-fast-matmul/actions/runs/39016018186). [PTX fragment layout and instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

At the largest tested shape, both job keys fit each grouped chunk into 112 terms: seven 16-wide steps instead of eight. This removes 12.5% of the dense MMA steps before correction work. Another fixture family has one chunk requiring eight steps. Fewer dense steps do not imply the same reduction in wall time.

A block computes 16 × 128 outputs using four warps, retains accumulators across every rank-128 prefix and writes the two 8 × 32 ticket states per warp. Correctness mode additionally captures every prefix. The grouped paths retain exact signed residuals for coefficients outside INT8. Shared row pitch is 144 bytes; the retained implementation copies input tiles with aligned 16-byte loads.

Seven methods run in rotating order:

1. Direct INT8 cuBLAS with state reduction.
2. Previous grouped cuBLAS plus residual correction.
3. Direct cuBLAS with per-candidate A packing.
4. Fused direct MMA16, with the same prefix/state work as grouped MMA16.
5. Group A in a separate GPU pass, then fused grouped MMA16, correction and states.
6. Group A inside the fused kernel, then MMA16, correction and states.
7. The same inline grouping with the dense inner dimension forced to 128.

Primary ratios use the faster of methods 1, 3 and 4. Methods 5 and 6 are both reported; method 5 wins their comparison in every case. The older grouped method and padded control remain in the raw report. Setup is amortized over four distinct prepared candidates, and timing rounds replay those finite fixtures.

## Correctness

Every method passed **44 fixtures: 183,517,184 prefix entries and 44,804 tile states per method**. The state-only timed path was separately checked for every candidate. The original 12-fixture set covers 8-row and 32-column boundary tiles and dense nonzero A with larger coefficient corrections. The primary timing set is the existing 32 cached fixtures. The additional original-set three-round timings are preserved but are not used for speed claims.

These are exact local GPU checks. No final keyed BLAKE3, winning proof, customer inference or network acceptance is tested.

## Measured result

H100 80GB HBM3, CUDA runtime 12.9, cuBLAS 12.9.0. The retained vector-load implementation was measured at commit `f86e3aa89e79a7d7b0aa90e0193dbfbace58c10d`. [Raw samples](../results/cuda-mma16-h100.json), SHA256 `e7289e7f5da9514f987e1686c10ad5fb7781e4fab25059272ccdd0820ebbf66b`.

Largest shape, 2048 × 512. Median microseconds per candidate including B setup amortized across four candidates:

| Method | Job key 0 | Job key 1 |
| --- | ---: | ---: |
| direct_cublas | 91.55 | 91.66 |
| grouped_cublas_residual | 280.89 | 283.07 |
| packed_direct_cublas | 105.26 | 105.51 |
| direct_mma16_fused | 50.08 | 49.79 |
| grouped_mma16_prepared | 152.54 | 155.05 |
| grouped_mma16_inline | 240.51 | 243.61 |
| grouped_mma16_inline_padded128 | 239.19 | 241.06 |

The direct fused kernel improves this matrix-stage control from about **92 µs to 50 µs**, around 1.8× throughput. The strongest grouped version takes **153–155 µs**, approximately 3.0–3.1 times longer than that direct control. Grouping loses every paired round in all eight primary cases. Direct cuBLAS remains faster than our fused direct kernel at the smaller shapes.

Removing B setup does not reverse the result: largest-shape candidate execution is **100.7–101.0 µs grouped versus 46.1–46.2 µs fused direct**, around 2.2 times longer. Larger B reuse alone therefore cannot make these recorded grouped candidate timings beat the direct candidate timings. The report also contains explicitly modeled reuse counts of 64 and 1024; those are recombinations of the four-candidate measurements, not executions of that many fresh candidates.

## Iterations retained

- [First fused draft](../results/cuda-mma16-h100-initial.json), commit `75c51d76960a30d26744b5e7dcc162896bcaffa8`: correct, but scalar byte copies made even the direct control slow, about 292 µs at the largest shape. This motivated vectorizing both controls rather than treating the loss as an algorithmic limit.
- [Vector loads](../results/cuda-mma16-h100.json), commit `f86e3aa89e79a7d7b0aa90e0193dbfbace58c10d`: direct control about 50 µs; grouped prepared 153–155 µs. This is the retained version.
- [Shared graph staging and trimmed B loads](../results/cuda-mma16-h100-staged.json), commit `b79b29d0c0d0a9dbc6f5504f2ce2c1f092f30fe2`: all correctness checks pass, but the grouped prepared path takes 159–160 µs and the inline path 242–244 µs. It does not improve the relevant comparison. The source is retained in git history and the default returns to the faster vector-load version.

The inline trimmed and padded controls are close; the measured time difference changes across these implementations. Dense instruction-count savings alone do not explain the wall-time result. No profiler breakdown was collected, so this experiment does not assign the remaining runtime to particular instructions or prove a hardware lower bound.

## Cloud cost and scope

All three implementations ran on the same temporary H100 Spot VM, with a 30-minute automatic deletion limit. It and its boot disk were deleted after collection. Listed-rate compute was approximately $1.10 for the session; this excludes boot disk/network charges and is not a final billing statement. [Runtime and cleanup record](../results/gcp-mma16-run.json).

Both grouped and direct methods omit commitments, noise generation, generation of fresh candidates, final hashes, winning proofs and network submission. The comparator is the fastest measured local direct control, not Pearl's fused miner. The direct optimization can also be used for useful inputs; its speedup is not evidence of an advantage reserved for useless work. No inference revenue or mining payout was measured.

## Result, uncertainty and next question

**The 16-wide arithmetic saving is real and the fused kernels are correct, but the tested grouped implementations still lose to the strongest direct control.** We have a faster direct H100 matrix-stage implementation than the earlier cuBLAS control, not a demonstrated grouped-mining advantage. This does not rule out a different mapping or algorithm, and it does not establish that useful mining is economically optimal.

Can a different way to form and consume the grouped coefficients reduce complete grouped candidate time below the measured direct control, including all required prefixes and states?
