# Factoring both sparse noise maps

## Question

Can using structure on both sides of the product remove substantially more dense tensor work than one-sided grouping, while keeping the transformed coefficients small?

## Exact construction

For raw `A=B=0`, the noised operands are `U P` and `Q V^T`. The two sparse incidence maps share the inner coordinate. For one width-128 chunk:

```text
T = P_chunk Q_chunk
Delta C = U T V^T
```

Each inner position adds four signed entries to the 128-by-128 integer core T. Factor `T=X Y` exactly and compute `(U X)(Y V^T)`. Accumulate these chunk products to obtain every required prefix.

This is different from reconstructing the whole cumulative core at every prefix in [note 007](007-pearl-cheap-tile.md). We factor the sparse chunk increments and retain their output accumulator. Both transformations and any wider-integer corrections still cost work.

We generate 32 commitment-derived fixtures: shapes 128×128, 512×512, 2048×512 and 2048×2048; two fixed synthetic job keys per shape; and four distinct A inputs per job. B stays zero. A is zero or has one `+1` at inner position 0, 1 or 2 of row zero. Raw AB is zero throughout. A's commitment and both of its noise factors are regenerated. Nonzero A adds a small exact correction to the affected chunk and all following prefixes.

## Factorization and coefficient growth

The prototype repeatedly removes a rank-one outer product using a `+1` or `-1` pivot, preferring pivots with low predicted fill. Every step uses integers. Any remainder is represented directly by its nonzero rows or columns, so factorization always remains exact.

The first version allowed updated core coefficients up to 127. Although the factors themselves fit INT8, multiplying them by U and V produced transformed operands with absolute values as large as 12,691 and 14,833. About 2.28% of left and 2.21% of right coefficients exceeded INT8.

The retained version first tries a much tighter update cap, then increases it only if needed to reach the minimum number of 16-wide steps established below. Of 512 chunks, 510 use cap 2 and two use cap 3. Original core entries can exceed the update cap; the actual retained factors have maximum absolute coefficient **3**.

| Variant | Left coefficients outside INT8 | Right coefficients outside INT8 | Maximum factor coefficient |
| --- | ---: | ---: | ---: |
| Initial cap 127 | 2.280% | 2.209% | 127 |
| Retained adaptive caps | 0.594% | 0.537% | 3 |

The tighter version retains the same padded width as the first version. It reduces correction demand; it does not eliminate it. Timings and instruction counts for these corrections have not been measured on a GPU.

## How far can this factorization reduce the width?

We compute rank modulo the prime 65521, whose primality the script checks. A nonzero minor modulo a prime is also nonzero over the integers. It therefore gives a lower bound on rational rank. Our exact integer factorization supplies an upper bound: `rank(T) <= factor_width`.

For **all 512 chunks**, these lower and upper bounds round to the same multiple of 16. Thus the retained construction already uses the fewest possible 16-wide steps **among exact linear core factorizations of these tested matrices**. The exact unpadded rank can remain between the two bounds. This does not establish a lower bound for other inputs, other multiplication algorithms, or computing only the required XOR states.

| Quantity across the 32 complete fixtures | Result |
| --- | --- |
| Ordinary direct inner sum | 2048 in every fixture |
| Retained two-sided padded inner sum | 1776 twice; 1792 in 29 cases; 1808 once |
| Better one-sided grouping chosen separately per chunk | Same count in 31 cases; 16 more terms in one case |
| Minimum padded inner sum for independently reconstructing every cumulative prefix core | At least 2032 in every fixture |

The both-sided construction saves only one additional 16-wide step in one of 32 cases relative to choosing the better one-sided direction for each chunk. That comparison grants the one-sided route free choice of direction; it is an arithmetic control, not an implemented mixed GPU kernel. It shows that simply finding a better core rank factorization cannot produce a large additional dense-work saving on these fixtures.

The cumulative-prefix alternative has even less room: each fixture's rank lower bounds require at least 2032 inner terms, versus 2048 direct. It also needs transformations and an accumulator update. The XOR target would need a different shortcut to avoid that work.

## Exact prefixes, states, and failed shortcuts

An independent Rust path reconstructs the commitment-bound noise and checks two 8×32 output tiles per fixture, one including the nonzero A row and one outside it. It checks **262,144 prefix values, 64 full tile states and final keyed hashes** against ordinary direct multiplication. The retained factorization passes every comparison.

The check also exercises these failure cases:

- Omitting the raw-A correction fails all 24 nonzero-A controls.
- Using only clipped INT8 operands changes the chunk product in all 1,024 sampled tile/chunk combinations. Exact residual corrections restore it.
- Replacing the XOR of the accumulated prefix with the previous prefix XOR XORed with the chunk XOR fails in all 960 checked noninitial prefixes. Integer carries prevent that naive distributive shortcut. This does not rule out a more sophisticated exact XOR algorithm.

These are local arithmetic checks. We did not run a new official plain-proof verifier check, construct a ZK certificate, submit network work, or benchmark a useful inference service.

## Cost and scope

The retained Python/NumPy screening pass took about 14 seconds on this Mac, including rank checks and full transformed-operand statistics. The pivot/factor phases alone took about 6 seconds across 512 cores. These are prototype CPU wall times, not a lower bound on planning cost or GPU performance.

The core contains P, which changes with every fresh A commitment. Holding B fixed does not permit reusing the complete factorization across new A candidates. A working miner must include that planning cost, both transformations, all residual corrections, raw-input corrections and the remaining mining pipeline. We have not demonstrated that complete path is faster than direct.

No cloud resources were launched for this iteration. Initial and retained measurements, exact factor files, noise factors, seeds, hashes and source snapshots are preserved in the raw archive.

## Reproduce

Install NumPy 2.4.2, matching `gpu/cuda/requirements.txt`, then run:

```sh
cargo run --offline --locked --release -- --export-bilateral /tmp/pearl-bilateral
python3 experiments/bilateral_core.py /tmp/pearl-bilateral \
  --output /tmp/pearl-bilateral-screen.json
cargo run --offline --locked --release -- --verify-bilateral /tmp/pearl-bilateral
```

[Full report](../results/bilateral-core.json) · [Compressed fixtures and iteration history](../results/bilateral-core-fixtures.tar.gz) · [Exact Rust checks](../src/bilateral.rs) · [Factorization and rank bounds](../experiments/bilateral_core.py).

## Result, uncertainty, next question

**We have an exact both-sided construction with small factor coefficients, and a concrete limit on its dense-width saving for these fixtures. No new H100 speedup is demonstrated.** Other inputs or an algorithm that avoids evaluating each output entry remain open. Existing grouping kernels also have implementation overhead that a rank calculation does not measure.

Can profiling and a different grouping/anchor implementation turn the existing arithmetic saving into a runtime gain, without relying on a larger rank reduction?
