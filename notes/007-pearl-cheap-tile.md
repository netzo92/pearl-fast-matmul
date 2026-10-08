# Can a miner compute Pearl's winning-tile hash without the tile GEMM?

Historical research note. See [the current batch and official-verifier results](009-batch-and-conformance.md) for the latest status.

## Question

For a miner-chosen cheap input, especially `A=B=0`, is there an algorithm that computes the **exact** hash state of one valid tile with materially fewer operations per lottery ticket than the prescribed tiled matrix multiplication? A positive answer, with a cost advantage that survives input commitments and proof generation, would directly challenge the security claim of Pearl's deployed integer protocol.

Sources: [Pearl INT whitepaper, sections 4.3–4.6](https://pearlresearch.ai/research/int-whitepaper) for commitment-derived noise and the hash rule; [Komargodski–Weinstein, section 6.5](https://arxiv.org/html/2504.09971v4) for the related transcript-hardness assumption. This note addresses the integer protocol's tile-opening rule, which differs from the paper's full-transcript presentation.

## Exact attack target

Pearl derives noise seeds from hashes of `A`, `B`, the mining configuration, and chain state. Set `A=B=0` before deriving the seeds. Write the resulting noised operands as `U P` and `Q V`, with `U` of shape `m x r`, `P` of shape `r x k`, `Q` of shape `k x r`, and `V` of shape `r x n`. In the whitepaper's noise generation, each column of `P` and each row of `Q` has two signed nonzero entries.

For one `h x w` hash tile, after chunk `l` of width `r`, define

```text
S_l = sum_{s < (l+1)r} P[:,s] Q[s,:]       (r x r)
C_l = U_tile S_l V_tile                   (h x w)
X_l = XOR_{i,j} int32(C_l[i,j])
M[l mod 16] = rotl32(M[l mod 16],13) XOR X_l
```

The last two lines are the quantity that feeds the keyed BLAKE3 threshold test. A successful attack must compute the same `M` that the official verifier reconstructs. Merely computing `AB=0`, a final tile, or a different hash does not suffice.

**Research problem:** exploit the sparse `P,Q` and the XOR reduction to compute all `X_l` in `o(h w k)` word operations per candidate tile, or show a concrete GPU speedup against the honest kernel at allowed parameters. Count commitment hashing, seed expansion, candidate trials, and proof generation. Compare accepted tickets per second at equal resources, not time for one chosen tile against time for an entire GEMM.

## Concrete synthetic one-tile miner

The [integer whitepaper's Algorithm 4](https://pearlresearch.ai/research/int-whitepaper) declares an individual output tile opened when its final hash meets the target; the verifier's proof reconstructs that tile's input strips. Therefore a miner can choose synthetic `A=B=0`, commit to them, derive the bound noise, compute the accumulator and hash state for one valid tile, and submit a proof if that tile wins. The rest of `A'B'` and `AB` need not be materialized for that proof. With explicit hashing of the full inputs, one trial costs `O((m+n)k + h w k)` elementary operations rather than the `Theta(m n k)` operations of a full classical GEMM, plus the winning proof cost. This is a valid *partial-computation strategy* under the whitepaper's tile-opening rule, assuming all concrete certificate checks are met; it does not require finding a special noise seed.

This observation alone does not show an unfair PoW advantage. An honest full GEMM computes about `(m/h)(n/w)` tile tickets, and current consensus code scales the target with `h*w*k` (subject to rank penalty). The relevant comparison is expected accepted tickets per second after commitment and proof costs. A proof that one synthetic tile is cheap relative to an entire GEMM is insufficient to show that per-ticket mining is cheaper. It does establish that acceptance of one tile does not enforce delivery of the full useful product.

## Shifting zeros and ones: a genuine transcript symmetry

Let `P` permute inner coordinates **within each width-`r` chunk**. For already-noised operands `A',B'`, replace them by `A'P, P^{-1}B'`. Every chunk product, every prefix accumulator, and therefore every XOR hash-state word is unchanged. This follows from `(A'P)_chunk (P^{-1}B')_chunk = A'_chunk B'_chunk` for each chunk. Run `python3 -m experiments.pearl_permutation_symmetry` for a tiny exact example. Permutations across chunk boundaries preserve the final product but generally change the hashed prefix sequence.

This symmetry would let a miner reuse one expensive transcript while trying many input encodings **if the noise and hash key stayed fixed**. In the deployed protocol, the commitments hash the actual operand bytes and the noise seeds depend on those hashes. Moving zeros or ones changes the commitments, regenerates the noise, and changes the keyed final hash. To turn the symmetry into an attack, one would need a cheap way to generate new committed inputs whose *new* noised operands are `A'P,P^{-1}B'`, or a way to keep the binding hash unchanged. Neither is established here; the latter would challenge the commitment's collision resistance.

## First candidate and its limit

The factorization gives a cheap recurrence for `S_l`: each scalar position in `k` adds at most four signed entries, so maintaining all `S_l` takes `O(k)` scalar updates. This alone does **not** give a cheap `M`. Reconstructing `C_l` by `(U_tile S_l) V_tile` costs `O(h r^2 + h w r)` operations per chunk, or `O(k h r + k h w)` overall. Direct tiled multiplication costs `Theta(k h w)`. The apparent low-rank shortcut therefore fails to establish an asymptotic advantage for normal tile sizes. The missing step is an exact way to get the XORs without materializing every `C_l` entry.

Run `python3 -m experiments.pearl_tile_factorization` for a tiny integer example that checks the factorization and XOR states against direct multiplication. It is not a protocol implementation or a performance benchmark.

## Adaptive cancellation in the isolated seed-only encoding

There is a concrete adaptive-input shortcut in Algorithm 6.4 **if its seed-only Encode is used in isolation**. Its `Encode` parses the public `sigma` into low-rank `E,F` and sets `A'=A+E`, `B'=B+F`. After seeing `sigma`, choose `A=-E` and `B=0`. Then `A'=0`, so every partial-sum tile and the entire transcript are zero. Constructing `A` costs `O(n^2 r)` field operations; streaming the explicit zero transcript costs `Theta(n^3/r)` words. For growing `r=o(n)`, both are `o(n^3)`, whereas the classical honest transcript computation is `Theta(n^3)`. This is a conditional attack on that isolated seed-only, adaptive-input interpretation. The paper's Section 2 overview instead derives the factors as a random-oracle output of `(sigma,A,B)`, which blocks direct cancellation; thus this observation is not a break of the paper's intended full protocol. It does not beat the paper's best asymptotic rectangular-multiplication baseline when that baseline itself reaches the transcript output-size bound. The paper's transcript-unpredictability statement fixes `A,B` before sampling `sigma`; a miner choosing its inputs after seeing `sigma` reverses those quantifiers.

The deployed integer protocol adds the missing input binding: `s_A,s_B` depend on keyed hashes of `A,B`. Thus `A=-E(s_A)` becomes a hash fixed-point search, and the construction above does not transfer. The current [consensus sanity checks](https://github.com/pearl-research-labs/pearl/blob/master/zk-pow/src/api/sanity_checks.rs) also require rank at least 128 after the rank-penalty rule and scale the winning threshold with `h*w*k` (with a rank penalty). Choosing the shortest permitted `k` alone therefore does not improve tickets per nominal multiply-add.

## Commitment-bound V3 experiment

`cargo run --release` (run with `--offline` if `blake3` is cached) reproduces Pearl's published V3 seed test vector, its keyed BLAKE3 commitment chain, sparse noise generation, `int32` tile accumulation, XOR/rotation state, and keyed jackpot hash. It uses a fixed 32-byte job key, `r=128`, `k=2048`, and a `4 x 8` tile. The source follows Pearl's `zk-pow/src/api/{seed.rs,proof_utils.rs}` and `zk-pow/src/circuit/pearl_noise.rs`; it is a narrow CPU reproduction, **not** a generated or verified network certificate.

The raw inputs have a single matching `1` in inner position 0. Moving both `1`s to another position in the same width-128 chunk preserves their useful product. If we permute the *already-noised* operands with the same permutation, the tile state remains exactly equal, as algebra predicts. But recommitting the shifted raw inputs changes both commitments and both V3 noise seeds. The recomputed tile state and keyed jackpot hash differ. In a sweep of 16 within-chunk shifts, none repeated the original state or hash. This is a finite check of the proposed shift attack, not a proof that no other structured input or larger search can work.

## Result

**Partial result.** The isolated seed-only encoding has the adaptive cancellation shortcut above; the paper's overview and the deployed protocol bind the noise to the input. For the deployed commitment-bound protocol, we have an explicit mathematical target and exact factorization, but no faster hash-state algorithm or valid mining exploit. The simple cancellation idea `A=-E` is invalid for the deployed protocol because the noise seed depends on a commitment to `A`; it asks for a hash fixed point. The paper's abstract Algorithm 6.4 does not include this commitment binding, so its security model must be discussed separately.

## Follow-up experiment

[Signed-noise grouping](008-pearl-noise-grouping.md) gives an exact component-based identity for raw B=0. It saved 16.89% of scalar multiplications in the tested fixtures but was slower than the improved direct CPU baseline. No mining advantage is established.

## Next question

Can the XOR of all signed 32-bit entries of `U_tile S_l V_tile` be computed for every prefix `l` without evaluating all `h w` dot products per prefix, while preserving exact wraparound and the specified rotation/XOR update?
