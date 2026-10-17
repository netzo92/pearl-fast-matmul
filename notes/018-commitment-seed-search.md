# Searching committed near-zero inputs for cheaper noise graphs

## Question

Can a miner select a mostly zero B commitment whose noise graph removes enough additional tensor-core work to rescue the grouped algorithm?

## Construction

Keep `m=2048`, `n=512`, `k=2048`, `r=128` and the same four fixed synthetic job keys used by the batch fixture generator. Keys 0 and 1 match the largest-shape H100 fixture families. Neither the job key nor the dimensions are optimized during the search.

For each key, test 65,536 distinct B inputs. B is stored transposed as 1,048,576 bytes. Trial `t`, starting at zero, puts a single `+1` at flat position `(t * 1000003) mod 1048576`; every other entry is zero. The odd multiplier gives unique positions over the tested range. Each full B byte string is keyed-BLAKE3 committed before deriving its V3 B noise seed. We do not select an arbitrary seed and pretend a matching commitment exists.

For an entry `B[j,s]=1`, the exact chunk update is:

```text
A'[i,s] in output column j
```

That contribution must remain in all subsequent prefixes. It costs `m` extra additions per fresh A candidate, with no extra scalar multiplication for the value `+1`. Constructing the grouped coefficients and their INT8 residual corrections still has its existing costs.

For each width-128 chunk, the graph grouping dimension is `d=128-components`. The primary search objective is `sum(16*ceil(d/16))` across all 16 chunks, because the H100 kernel uses 16-wide integer MMA steps. Raw dimension sum is the tie breaker. A secondary selection minimizes the raw sum first. Chunks from different commitments are never combined: each selected candidate belongs to one complete committed B and its bound noise.

## Results

Apple M4, single-thread Rust release build. **262,144 distinct committed candidates** across four job keys. The earlier 8,192-per-key pilot is a prefix of this search, not additional independent samples.

| Job key | Zero-B raw dimension sum | Best raw dimension sum | Zero-B MMA16 inner sum | Best MMA16 inner sum | Extra dense MMA work removed | Search phase wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1696 | 1652 | 1792 | 1760 | 1.79% | 29.54 s |
| 1 | 1700 | 1644 | 1792 | 1776 | 0.89% | 30.10 s |
| 2 | 1692 | 1653 | 1792 | 1760 | 1.79% | 30.57 s |
| 3 | 1690 | 1651 | 1792 | 1760 | 1.79% | 30.42 s |

The raw and MMA16 columns can select different candidates. For example, job 0's MMA16 winner has raw dimension sum 1671, while its raw-count winner has sum 1652. The full report records both selections and all checkpoint histories.

Only six candidates across the complete search reach the best padded sum of 1760, and none go below it. The best case removes two more 16-wide steps across the 16 chunks: 110 steps instead of the zero-B baseline's 112. Ordinary direct multiplication uses 128 steps. Thus the best selected grouped construction removes 14.06% of dense MMA work versus direct, compared with 12.5% for zero B. These are instruction-work counts before grouping, correction, memory traffic and search costs; they are not measured GPU speedups.

All search phases took about 121 seconds, and total screening wall time was 121.71 seconds, excluding compilation and subsequent analysis/compression. Full commitment hashing accounts for approximately 92% of the timed search phases. We rehashed each complete 1 MiB B input, totaling 256 GiB. Incremental Merkle-tree commitment updates could reduce that cost; these times are not a lower bound on seed search cost. CSV formatting/writes and correctness checks are outside the phase timers and inside the total screening wall time. JSON fields named `cpu_seconds` contain elapsed wall time for the corresponding single-thread CPU phase, not process CPU counters.

## Exactness checks

For each job key, check the zero-B baseline, the primary winner and the secondary winner. For each of these 12 selections:

- Create four fresh full-size A commitments and their bound A noise seeds, keeping selected B fixed.
- Choose A's lone `+1` at a different inner coordinate from B's `+1`, so raw `AB=0` for every fixture.
- Compare an 8-by-32 output tile containing the B nonce and another tile outside it.
- Compare every accumulated prefix, all 16 state words, and the final keyed hash against ordinary direct multiplication.

Totals: **393,216 exact prefix entries, 96 tile states and hashes, and 32 negative controls** in which omitting the nonzero-B correction produces an incorrect prefix. The optimized search generator also agrees with the existing sparse-pair generator for each checked seed. The Python analysis rechecks every CSV score, unique position/root/seed, selected minimum and search checkpoint. Neither a new official plain-proof check nor a ZK/network acceptance check was run for these selected inputs. The older official verifier results remain separate.

## Does this justify another H100 run?

Not for the current grouped implementation. In [note 017](017-fused-mma16-h100.md), its candidate stage took about 101 microseconds versus 46 microseconds for direct, even before B preparation. This search removes at most 1.79% more dense MMA work than that grouped baseline. It does not remove A grouping or INT8 residual correction, and adds a sparse raw-B correction. There is no measured mechanism here for closing the observed time gap. This is an engineering decision about this experiment, not a lower bound on all structured-input algorithms.

Search must also pay for itself before its job key changes. If an implementation eventually measures a per-candidate GPU saving `delta`, a simplified resource-price comparison is:

```text
reuse_count > search_seconds * CPU_price_per_hour
              / (delta_GPU_seconds * GPU_price_per_hour)
```

Include additional B preparation and other extra costs as well. We have not measured `delta` for these selected inputs, so there is no demonstrated break-even reuse count or profit advantage. Direct must receive the same B reuse and freedom to choose its implementation.

## Reproduce and evidence

```sh
cargo run --offline --locked --release -- --seed-search /tmp/pearl-seed-search 65536
python3 experiments/analyze_seed_search.py /tmp/pearl-seed-search \
  --source-dir . --output /tmp/pearl-seed-search-summary.json
```

[Summary and distributions](../results/seed-search.json) · [Complete compressed trial logs and selected-case checks](../results/seed-search-raw/) · [Rust search and exact checks](../src/seed_search.rs) · [Independent log analysis](../experiments/analyze_seed_search.py).

## Result, scope, uncertainty, next question

**Selecting near-zero B inputs gives a small additional arithmetic saving, not a demonstrated fast H100 algorithm.** This bounded search covers one nonce construction, four synthetic job keys and fixed dimensions. It does not establish a global optimum, an impossibility result, or miner profitability. No cloud resources were launched.

Can an algorithm exploit structure on both sides of the product, or compute the exact required XOR states without reconstructing every dense prefix, to obtain a substantially larger saving than one-sided graph grouping? The simple low-rank reconstruction in [note 007](007-pearl-cheap-tile.md) already failed its operation-count comparison; a new method must beat that limitation.
