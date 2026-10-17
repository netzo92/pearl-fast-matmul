# CUDA feasibility benchmark

## Question

Can the exact zero-B grouping construction beat an INT8 cuBLAS control on NVIDIA hardware after including grouping, residual correction and B preparation?

## Implementation

[The CUDA benchmark](../gpu/cuda/README.md) adds three methods: direct INT8 cuBLAS, grouped INT8 cuBLAS with exact sparse correction, and direct INT8 cuBLAS with per-candidate A packing. The ratio uses the faster direct control. All methods receive the same fixed-B reuse and run identical prefix accumulation and 8-by-32 state reduction.

Grouping can produce coefficients outside signed INT8. The implementation clips the dense part to INT8 and stores every remaining signed residual in INT16, then adds its exact contribution using INT32 arithmetic. It does not drop or approximate residuals. Planning, contrast formation and uploads are included in B preparation. A grouping runs for every candidate.

The active grouped dimensions often pad back to 128 on tensor cores. This first implementation therefore uses dense inner dimension 128 in every method. It asks whether a wall-time benefit survives that limitation rather than inferring a CUDA advantage from the Metal result.

Correctness runs write every prefix matrix and compare it with the Rust direct reference. The timed path omits full-prefix writes and returns only the 16-word tile states; it receives a separate correctness check. Raw per-round wall times and CUDA event spans are retained. Event spans can contain host launch/copy gaps and are not active GPU time.

## Checks completed

- [CUDA 12.8.1 SM90 compilation and Linux host conformance passed](https://github.com/netzo92/pearl-fast-matmul/actions/runs/39014008998) at commit `cffec470081a6b384c2c4b690ef79cf981a30e5a`. Compiling a GPU binary requires no GPU rental and is not a GPU execution test.
- The original 12-fixture set passes independent host checks for **5,259,264 prefix entries and 1,284 tile states**. The same C++ planner used by CUDA matches both the independent Python planner and exported Rust bases. [Report](../results/cuda-host-conformance.json).
- The complete 32-fixture fixed-B set passes host checks for **178,257,920 prefix entries and 43,520 tile states**, including **176,288 nonzero coefficient residuals**. [Report](../results/cuda-cached-host-conformance.json).
- Changing one reference prefix entry causes the host check to reject the fixture.

The larger local validation was rerun with NumPy 2.4.2, which provides a Python 3.14 wheel. The initial NumPy 2.2.6 build produced a testing-library failure on that interpreter. The dependency now pins 2.4.2; this does not change CUDA arithmetic.

## H100 execution and result

The benchmark ran on **NVIDIA H100 80GB HBM3**, compute capability 9.0, CUDA runtime 12.9, cuBLAS 12.9.0, in a GCP `a3-highgpu-1g` Spot VM. Source revision: `7b367270bda76c52e52d20143546ed85208e5b41`. The JSON was created at 2026-10-17 07:40:20 UTC. [Raw report](../results/cuda-h100.json), SHA256 `93d8f2765a516439543f9b7ec69360e7d09d8d48f1ba8cbed7c972076c9abf6d`.

All three methods passed GPU comparisons for every prefix and tile state across **32 fixtures, 178,257,920 prefix entries and 43,520 tile states per method**. The state-only timed path was checked separately for each method and candidate. These are local GPU correctness checks, not network-accepted proofs.

Median wall microseconds per candidate, including B preparation amortized across four distinct A candidates:

| Output | Job key | Faster direct | Grouped | Grouped throughput / direct |
| --- | ---: | ---: | ---: | ---: |
| 128 × 128 | 0 | 16.67 µs | 79.21 µs | 0.211× |
| 128 × 128 | 1 | 16.98 µs | 80.00 µs | 0.212× |
| 512 × 128 | 0 | 18.30 µs | 90.12 µs | 0.203× |
| 512 × 128 | 1 | 18.39 µs | 92.01 µs | 0.200× |
| 512 × 512 | 0 | 26.03 µs | 113.26 µs | 0.230× |
| 512 × 512 | 1 | 26.05 µs | 116.74 µs | 0.223× |
| 2048 × 512 | 0 | 90.84 µs | 290.53 µs | 0.313× |
| 2048 × 512 | 1 | 91.06 µs | 291.24 µs | 0.313× |

**Grouping loses all 11 paired rounds in all eight cases.** At the largest shape it takes about **291 µs versus 91 µs**, approximately **3.2 times longer**. The unpacked direct cuBLAS control is faster than the packed direct control in every case.

This is not solely B planning overhead. At 2048 × 512, median candidate execution excluding B preparation is about **229–230 µs grouped versus 87 µs direct**. Median B preparation is about 243–247 µs grouped versus 15.6 µs direct, amortized over four candidates. The candidate-only result remains about 2.6 times slower even if B preparation is ignored. Medians of separate components need not sum to the median total.

Both dense methods still execute inner dimension 128; the grouped path additionally runs A grouping and residual correction. These structural costs are present in the implementation. No profiler breakdown was collected to assign a measured fraction of runtime to each cause. The M4 1.28× advantage does not transfer to this H100 implementation.

## Cloud run and collection

Global quota for one GPU and regional quota for one Spot H100 were requested and approved. The temporary VM had a 30-minute maximum runtime, termination action DELETE and an automatically deleted boot disk. No inference API, pool or blockchain was contacted. The benchmark completed about 90 seconds after boot; the VM was deleted after collecting the report. Runtime was a few minutes, giving an estimated compute cost below $1 at the listed ~$6.31/hour Spot rate; this is not a final billing statement.

The first serial transfer attempted one long base64 line, exceeding the guest metadata runner's line limit after the benchmark had saved its report. The complete JSON was retrieved over SCP instead, and its SHA256 matched the checksum emitted before the transfer failure. The startup script now emits compressed 4,000-character chunks; [the collector](../gpu/cuda/collect_serial.py) verifies completeness and SHA256, tolerates duplicate serial-log lines and rejects truncated transfers. Chunk recovery was checked locally. No second GPU run was needed for that transport fix.

## Scope

This is a matrix-stage feasibility benchmark. Its direct baseline is cuBLAS with our reduction, not Pearl's fused mining kernel. Both methods materialize chunk products outside the fused reduction. Input commitments, noise generation, fresh candidate generation, final keyed BLAKE3, winning proofs, network acceptance and useful inference are excluded. Timing rounds replay a finite fixture set and do not measure a stream of new mining candidates.

A win would justify comparing against the actual Pearl kernel and integrating omitted costs. A loss would reject this implementation and these shapes; it would not establish that all grouping implementations must lose. Neither outcome by itself measures customer revenue or net dollars per hour.

## Result, uncertainty and next question

**CUDA correctness is established for these fixtures, but this implementation is slower than the direct cuBLAS control.** It supplies no evidence of higher mining earnings, lower complete mining cost or a defeat of Pearl's economic argument. It also does not prove that every possible grouped CUDA kernel must lose.

Can a fused implementation retain exact grouped arithmetic and states while avoiding enough grouping, correction and intermediate-buffer traffic to beat the direct control?
