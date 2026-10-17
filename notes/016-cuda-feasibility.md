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

## GPU execution

No NVIDIA execution or CUDA timing result has been recorded. Existing accessible GCP projects with Compute Engine enabled have global GPU quota zero. Requests for a global allowance of one GPU and one Spot H100 in `us-central1` were submitted for the short benchmark. Quota requests create neither VMs nor reservations. Both requests were approved to one GPU. The run remains subject to available capacity; no VM has been created.

The prepared [GCP startup script](../gpu/cuda/gcp-startup.sh) requires an exact source revision, uses a CUDA-ready image, compiles the benchmark and generates its fixtures, then emits the report and its SHA256 into the serial log. It performs no pool or blockchain submission. Any launch must use an enforced 30-minute run limit with instance termination action DELETE and an automatically deleted boot disk. The script's benchmark phase also has a 15-minute timeout. This bounds runtime rather than guaranteeing a particular bill.

## Scope

This is a matrix-stage feasibility benchmark. Its direct baseline is cuBLAS with our reduction, not Pearl's fused mining kernel. Both methods materialize chunk products outside the fused reduction. Input commitments, noise generation, fresh candidate generation, final keyed BLAKE3, winning proofs, network acceptance and useful inference are excluded. Timing rounds replay a finite fixture set and do not measure a stream of new mining candidates.

A win would justify comparing against the actual Pearl kernel and integrating omitted costs. A loss would reject this implementation and these shapes; it would not establish that all grouping implementations must lose. Neither outcome by itself measures customer revenue or net dollars per hour.

## Result, uncertainty and next question

**The CUDA port compiles, and its host arithmetic/planning checks pass. NVIDIA correctness and performance are unmeasured.** The one-GPU GCP quota requests have been approved; GPU execution is the next step. The uncertainty remains whether grouping can save time against NVIDIA's optimized integer matrix hardware after paying its extra costs.

Does this compiled implementation produce exact states on H100, and is it faster than the stronger direct cuBLAS control when all included preparation is counted?
