# Exact grouping of Pearl's signed noise selections

Historical research note. See [the current batch and official-verifier results](009-batch-and-conformance.md) for the latest status.

## Question

For committed raw `B=0`, can grouping the signed selections in Pearl's integer noise reduce the work of computing every tile prefix exactly?

## Construction and proof

Use the ordinary commitment-derived noise; no seed search or cancellation is required. Within one width-r chunk, write the noised B operand as

```text
B'[s,j] = V[p_s,j] - V[q_s,j].
G[i,t] = sum_{s:p_s=t} A'[i,s] - sum_{s:q_s=t} A'[i,s].
Delta C[i,j] = sum_t G[i,t] V[t,j].
```

Build a graph on r vertices with one edge `(p_s,q_s)` per inner position. For each connected component T, the grouped coefficients sum to zero: every edge contributes its value once positively and once negatively. Choose an anchor a_T. Then exactly

```text
Delta C[i,j] = sum_T sum_{t in T except a_T}
                  G[i,t] (V[t,j] - V[a_T,j]).
```

If the graph has c components, including isolated vertices, this uses d=r-c multiplications per output entry instead of r. This identity holds for arbitrary A', so raw A need not be zero. It preserves every chunk product and therefore every cumulative prefix, XOR/rotation state, and keyed hash. It does not require the commitment or hash key to remain fixed across trials.

The noise structure follows [Pearl's noise implementation](https://github.com/pearl-research-labs/pearl/blob/master/zk-pow/src/circuit/pearl_noise.rs). Seed derivation follows [the V3 seed implementation](https://github.com/pearl-research-labs/pearl/blob/master/zk-pow/src/api/seed.rs). These upstream master links are mutable. The local reproduction checks the full 32-byte published seed vectors; it has not been checked against an official full noise/transcript vector or certificate verifier.

## Computational model

Count scalar integer multiplications and additions separately from graph traversal and memory operations. Per chunk, direct multiplication performs h*w*r multiplies and h*w*r accumulator additions. Grouping performs h*w*d multiplies and accumulator additions, plus 2*h*r grouping additions and w*d contrast subtractions. Graph construction adds near-linear union-find work. This is a concrete arithmetic-count reduction, not an established asymptotic improvement or lower mining cost.

The experiment uses r=128, k=2048, and an 8 by 32 tile. At these sizes the exact integer computations fit signed int32, including grouped coefficients and partial accumulators. Grouped coefficients can exceed int8, which matters when comparing against GPU integer tensor cores.

## Reproduce

```sh
cargo run --offline --release -- --grouping
```

Remove `--offline` if the pinned BLAKE3 dependency is not cached. Implementation: `src/grouped_noise.rs`.

## Results

Measured on Apple M4, arm64, Rust 1.93.1 (01f6ddf75 2026-02-11), Cargo release profile, single CPU thread.

- All prefix matrices and final keyed hashes agreed in 32 commitment-derived fixtures: 16 deterministic job keys, each with raw A zero and nonzero signed-int7 A. B is zero throughout. Because B's seed is independent of A, these cover 16 distinct B selection graphs, each with two A inputs.
- Two additional algebraic boundary cases passed: repeated identical edges and a connected chain. Those forced graphs are not claimed to be obtainable cheaply from valid commitments.
- Direct scalar multiplications across the 32 fixtures: 16,777,216.
- Grouped scalar multiplications: 13,943,808, a **16.89% reduction**.
- Adds plus multiplies: 33,554,432 direct versus 30,679,168 grouped, an 8.57% reduction before graph and memory costs.
- Median CPU time per tile over seven alternating batches of 100 calls: **40.37 microseconds direct versus 88.60 microseconds grouped**. Grouping was approximately **2.19 times slower**.

Both kernels materialize every prefix. The direct baseline starts with materialized A' and B'; grouping starts with A', V, and the selection pairs. Timings include graph construction, grouping, contrasts, allocations, and prefix copying, but exclude commitment hashing, noise generation, final hashing, and proof construction. They are not end-to-end mining measurements. The arithmetic totals cover all fixtures; timing uses the first all-zero-input fixture.

An initial indexed direct loop took about 245 microseconds and made grouping appear 2.7 times faster. Slicing the input rows before the direct dot product enabled compiler optimization and reversed that result. The improved baseline is the one retained in the experiment. Neither implementation is a competitive GPU mining baseline.

The original permutation experiment also still passes, including the strengthened full seed-vector assertions.

## Result and scope

**Proved algebraic identity; measured multiplication reduction; no measured speed advantage against the improved CPU baseline.** This is an exact alternative way to compute the local transcript reproduction for committed zero B. It is not a verified network certificate, a jackpot-winning example, a proof that the dimensions satisfy every certificate constraint, or a break of Pearl's hardness assumption. Job keys here are fixed test inputs, not derived from a live header and mining configuration.

An exploitation claim still requires agreement with the official verifier and higher expected accepted tickets per unit of total resource cost than the competitive miner, including commitments, seed expansion, proof costs, and difficulty normalization.

## Next question

Can graph analysis and V contrasts be reused across enough row tiles to make this exact grouping algorithm faster per ticket, after including its wider coefficient arithmetic and comparing with the deployed GPU kernel?
