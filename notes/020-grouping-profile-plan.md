# Grouping and correction profiling: prepared, awaiting approval

## Question

Can a warp-based grouping pass and a different graph anchor reduce complete matrix-stage time enough to beat the fastest local direct control on H100?

**GPU execution is not authorized yet.** The user explicitly requested approval before starting this run. Compile-only CI and host arithmetic validation are preparation. No new GPU timings, GPU correctness results, official proof checks or cloud charges are established by this note.

## Changes prepared

The seven earlier methods remain. Three additional valid candidate methods isolate the changes:

| Method | Grouping pass | Anchor |
| --- | --- | --- |
| Existing prepared MMA16 | One 128-thread block per row | Smallest vertex in each component |
| Warp MMA16 | One warp per row, four rows per block | Same original anchor |
| Warp + degree MMA16 | One warp per row, four rows per block | Highest-degree vertex in each component |
| Original pass + degree MMA16 | Original block per row | Highest-degree vertex |

The new grouping pass uses warp ballots and population counts to assign residual slots, replacing block-wide synchronization and shared atomic counters. Each warp still writes all 128 coefficient slots and preserves every exact residual.

The degree anchor is determined only by the B selection graph. Its contrasts can therefore be reused across fresh A candidates just like the original contrasts. Eliminating a high-degree coefficient may remove more grouping additions and more overflow corrections. The active dimension and padded tensor width do not change. Ties use the smallest vertex index; no A-dependent or per-row anchor is selected.

## Measurements prepared

The primary benchmark keeps the existing complete matrix-stage comparison: eleven rotating rounds, four distinct preloaded A inputs per fixed B job, B preparation amortized across those four inputs, and the fastest direct control chosen from cuBLAS, packed cuBLAS and fused direct MMA16. The three new methods must pass every prefix and state check before timing. Both the prefix-capturing and state-only paths are checked.

An optional diagnostic uses CUDA graph replay to reduce host gaps between kernel launches. It records:

1. Complete direct and grouped candidate batches.
2. Grouping alone for the four prepared inputs.
3. The fused grouped multiplication/correction/state kernel alone, repeating prepared candidate zero.
4. The same fused kernel with residual correction deliberately omitted.

The fourth diagnostic is **not a valid mining algorithm** and is excluded from the method list and every valid-method performance ratio. Its states are checked against an independently computed clipped-only reference. Complete graph batches must match the last candidate's exact Rust states; fused-only valid diagnostics must match candidate zero's exact states.

These controls do not form an additive breakdown of kernel runtime. Removing corrections may change compiler register allocation or occupancy. Group-only and fused-only controls also have different buffer reuse from complete batches. Each graph has four candidate slots; complete grouped slots contain two kernel nodes each. Capture, instantiation, graph warmup and B preparation are outside the diagnostic timer. Replays use finite prepared data, not a stream of fresh mining candidates.

## Preparation checks

Host validation independently constructs both anchor plans in Python, compares them with the actual C++ planning code, and checks exact grouped-plus-residual arithmetic against the existing Rust prefixes. It also retains the original exported vertex/anchor comparison. Planned CUDA compilation uses the existing CPU-only CI job; compiling CUDA code does not run it on a GPU.

The evidence files accompanying this branch describe host checks only. GPU warp compaction, graph capture, performance and final states still need execution on NVIDIA hardware after approval.

## Proposed cloud run, only after explicit approval

- Project: `tracegrove-research-20260911`; zone: `us-central1-a`.
- One `a3-highgpu-1g` H100 Spot VM, using the same CUDA 12.9 image as the previous test.
- GCE-enforced maximum runtime: **600 seconds**, termination action **DELETE**.
- One 60 GiB auto-deleted boot disk, no service account or API scopes.
- Exact reviewed source revision, with `PEARL_PROFILE_STAGES=1`.
- Startup prepares dependencies; the benchmark subprocess is limited to 420 seconds. The 600-second VM limit also covers startup.
- Retrieve the report and verify its SHA256, then delete the VM immediately. Confirm the boot disk is absent. Preserve failures or preemption without automatically launching another VM.

The published Iowa rate checked for this plan is **$6.307365068 per hour**, about **$1.05 compute for the full ten-minute limit**, plus storage/network charges and billing adjustments. Spot rates can change; recheck before launch. This is a duration limit and a cost estimate, not a billing-account spending cap. [Google's Spot price table](https://cloud.google.com/spot-vms/pricing?hl=en).

## What a result would establish

A valid, repeatable full-path matrix-stage speedup would justify testing fresh candidate generation and Pearl's official fused kernel next. A stage-only or incorrect-diagnostic win would not establish a valid speedup. A loss would identify the remaining cost in this implementation, without proving that every cheap-input algorithm is impossible.

The experiment omits commitments, seed/noise generation, fresh candidate production, final keyed hashes, winning proofs, network submission and useful inference. It cannot establish higher mining profit or refute Pearl's economic claims on its own.

## Run after approval

```sh
PEARL_PROFILE_STAGES=1 PYTHON=/path/to/cuda-env/bin/python \
  bash gpu/cuda/run.sh results/cuda-grouping-profile-h100.json
```

## Result, uncertainty, next question

**Preparation only; no GPU run has started.** The runtime effect of warp grouping and degree anchors is unknown. NVIDIA execution and user approval remain required.

After approval, which of grouping, fused multiplication/state work, and residual correction limits complete candidate performance in the matched H100 comparison?
