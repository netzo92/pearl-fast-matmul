# The threshold for competing with useful inference

## Question

What measured advantage would make useless grouped matrix multiplication competitive with mining added to an existing inference service?

## The relevant comparisons

A manyfold speedup is not a requirement by itself. The criterion is lower resource cost per expected accepted reward, measured on comparable hardware and with equivalent difficulty and eligibility rules.

Let W be the complete resource cost of a dedicated useless-work mining run, and Qw its expected accepted reward. Include commitments, noising, matrix computation, states, hashes, winning-proof costs amortized by win probability, and actual resource use.

Let U be the cost of serving a fixed useful workload with mining enabled, and Qh its expected mining reward. Let I be the cost of serving equivalent useful output without mining. Use matched output quality, request volume and service constraints. Then:

```text
Raw resource efficiency:       W / Qw < U / Qh
Incremental mining efficiency: W / Qw < (U - I) / Qh
```

The second inequality is the target for competing with a provider whose inference workload would already have run. Under equal expected rewards it becomes W < U - I. It compares the entire useless-mining cost with the extra cost of adding mining to inference.

The first inequality tests mining throughput per total resource use. Passing it does not imply passing the second. Likewise, a measured percentage reduction in one matrix kernel is not directly comparable with a percentage overhead on a complete inference workload: the denominators and included work differ.

Elapsed time can stand in for resource cost only when it tracks the hardware capacity actually consumed. If kernels overlap, use sustained capacity, GPU time, energy, or another explicit resource metric; do not equate request latency with cost. Include any extra hardware and constraints needed to sustain useful output. If inference-plus-mining uses a different quantization scheme, match quality and account for that change instead of attributing all differences to mining.

## What Pearl claims

Section 6.1 of the [INT whitepaper](https://pearlresearch.ai/research/int-whitepaper) models useful-work providers as paying incremental mining overhead and providers without useful work as paying their full compute cost. It uses a 10% overhead cost as an illustrative assumption, rather than establishing a universal measured rate. The document also explicitly allows participation without useful work. A demonstration of useless mining alone therefore does not refute that economic model.

Separately, the [paper's section 2](https://arxiv.org/html/2504.09971v4) acknowledges that an attacker can skip useful-output recovery. Its security conjecture concerns obtaining an additional advantage in transcript computation. Measuring only the cost of skipping output recovery would not be a new refutation of that conjecture.

## What our measurements establish

[Note 012](012-cached-b-distinct-candidates.md) measures a matrix stage on Apple M4: approximately 17.3ms grouped versus 22.15ms direct at the largest tested shape, after the preparation costs specified there. That is a local saving of about 4.85ms, or 22% of the direct stage's time. It excludes commitments, noise generation, final state/hash generation and proof generation, and does not run a useful inference workload.

Consequently, we have measured neither W nor U - I. The 28% throughput ratio cannot establish the incremental-cost inequality above. Even common omitted work can dilute the raw throughput ratio; shared work between inference and mining must additionally be credited in the incremental comparison.

A small stable advantage in the **complete, reward-normalized comparison** would be enough to establish the corresponding practical advantage. Its magnitude does not need to be fixed in advance. Competitive NVIDIA kernels and real useful workloads remain necessary before making a claim about deployed miners.

## Measurement to run next

Measure three complete paths for a fixed useful workload: inference alone, inference with Pearl mining, and dedicated grouped useless mining. Match useful output and normalize expected rewards under the same protocol configuration. Preserve both total and incremental resource costs. Inspect the actual useful-output recovery and inference integration instead of assuming a separate extra matrix multiplication or using the whitepaper's illustrative overhead as a measurement.

## Result, scope, uncertainty, and next question

**The research target is lower cost per expected accepted reward, including comparison with the incremental cost of mining during inference. No predetermined speedup factor is required.** This is a cost-accounting criterion, not a new performance or network-acceptance result. Our current kernel measurements do not settle it.

Can complete grouped useless mining beat the measured incremental mining cost of an equivalent, correctly served inference workload on the same hardware?
