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

## Conclusion using Pearl's own estimate

We can use the published estimate for a **conditional calculation**. The 10% figure comes from [the INT whitepaper's economic example in section 6.1](https://pearlresearch.ai/research/int-whitepaper); it is not a measured overhead for our fixtures. The theoretical paper's asymptotic overhead claim does not supply a universal numerical percentage.

Normalize costs to the same expected mining reward and let C denote the reference full dedicated-mining cost. Adopt Pearl's illustrative incremental-cost parameter tau=0.10. If a useless-work algorithm achieves a speedup s in **complete resource cost**, its cost is C/s. Then:

| Quantity | Normalized cost |
| --- | ---: |
| Reference dedicated mining | C |
| Inference mining, extra | 0.10 C |
| Useless mining, if s=1.28 | 0.78125 C |

```text
Cost ratio = (C / 1.28) / (0.10 C) = 7.8125
To be cheaper: C / s < tau C, hence s > 1 / tau.
At tau=0.10: s > 10, or complete cost below 10% of C.
```

**Under that assumption, a 1.28x improvement would not beat the provider's incremental mining cost.** It would remain about 7.8x as expensive per expected reward. This is our inference from the stated model, not a result reported by Pearl. Our measured 1.28x concerns only a local matrix stage; treating it as a full-cost improvement here is an explicitly hypothetical extrapolation. It does not estimate actual H100 performance or complete mining cost.

The threshold depends on the actual incremental cost. For illustration, using the same normalized model:

| Assumed incremental cost / C | Complete-cost speedup needed to be cheaper | Hypothetical useless / incremental cost at s=1.28 |
| --- | ---: | ---: |
| 10% | Greater than 10x | 7.81x |
| 50% | Greater than 2x | 1.56x |
| 80% | Greater than 1.25x | 0.98x |

A modest improvement can therefore be sufficient when the measured threshold is close enough. The source estimate gives us a model to test; it does not establish that its assumptions hold. The defensible present conclusion is that exact useless grouping can speed up this local matrix stage, while neither a complete mining advantage nor a defeat of the illustrative 10% economic model has been demonstrated. A valid useless proof also does not, by itself, establish that a useful customer was served.

### ELI5

Imagine two people earning the same number of raffle tickets. One already runs a machine for a paying customer; adding ticket collection costs an extra $10. The other runs a machine only for tickets, normally costing $100.

Our shortcut makes part of the second machine faster. **If** the entire $100 bill fell by the same proportion, it would become about $78. That is an improvement, but $78 is still more than $10. To win on cost, we must get the whole bill below $10, or show that the first person's extra bill is actually higher. These dollars are an illustration, not measured operating costs.

Getting a valid ticket does not prove the machine did a job anyone wanted. Getting a ticket more cheaply is a separate question.

## Scope, uncertainty, and next question

**The criterion is lower complete resource cost per expected accepted reward.** Using Pearl's 10% example supplies a conditional threshold, not a measured outcome. Our matrix timings and easy-target plain-proof checks do not establish a live-network exploit or disprove the theoretical hardness conjecture. The unresolved quantities are complete dedicated-mining cost, actual incremental inference cost, and comparable expected rewards on competitive hardware.

Can complete grouped useless mining beat the measured incremental mining cost of an equivalent, correctly served inference workload on the same hardware?
