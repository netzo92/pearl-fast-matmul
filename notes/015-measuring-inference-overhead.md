# Measuring the inference overhead of mining

## Question

What is the extra resource cost of enabling Pearl mining on an otherwise matched inference workload?

## Published figures and what they establish

Pearl's [blog](https://pearlresearch.ai/research/blog/proof-of-useful-work), checked on 2026-10-17, reports end-to-end overhead of **5.08% for Llama 70B on four H200s with DP=4**, and **3.9% for DeepSeek V3.2 on eight H200s with DP=8 plus expert parallelism**. These are vendor-reported results, not measurements reproduced in this repository. Paired raw logs sufficient to independently reconstruct those exact percentages were not found in the cited blog or the inspected miner READMEs. The overhead metric's precise denominator must also be confirmed before converting it to another metric.

The whitepaper's 10% figure used in [note 013](013-inference-overhead-threshold.md) is a separate illustrative economic assumption. It should not be presented as measured universal inference overhead.

The [INT whitepaper's table 1](https://pearlresearch.ai/research/int-whitepaper) compares Meta BF16 and Pearl-certified execution on four H200s. Its PP=4 times are 441.100 seconds and 391.457 seconds. Their ratio gives an apparent time change of approximately -11.25%. TP=4 times of 510 and 507.789 seconds give approximately -0.43%. These comparisons change the model representation as well as the serving path; they do not isolate the cost of enabling mining. The BF16 DP=4 row is out of memory and supplies no finite baseline time.

## Calculation for a matched workload

Let T0 be elapsed time without mining and T1 elapsed time with mining, for the same fixed work on the same GPU allocation:

```text
time overhead = T1 / T0 - 1
extra GPU-seconds = number_of_GPUs * (T1 - T0)
```

For example, 100 seconds becoming 105 seconds is 5% time overhead. If measured using throughput R0 and R1 for equivalent fixed work, the corresponding time overhead is R0/R1 - 1. A 5% drop in throughput is therefore about 5.26% more time, not exactly 5%.

This calculation assumes matched work and sustained conditions. A request-latency increase need not equal additional GPU resource consumption if work overlaps. For the mining economics comparison, also record energy or occupied capacity, additional CPU/prover costs, and expected difficulty-normalized rewards. The relevant quantity is extra cost per expected accepted reward, not only a percentage of inference latency.

## A control exists in the official code

Inspected source revision: [`6f996e3bb7d9a74ad6345d24a19c18ab782585fa`](https://github.com/pearl-research-labs/pearl/tree/6f996e3bb7d9a74ad6345d24a19c18ab782585fa).

The official [execution tests](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/miner/vllm-miner/tests/test_vllm_execution.py) use `MINER_NO_MINING=true` and `false` before initializing vLLM. In a 7-bit mining layer, [PearlKernel](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/miner/vllm-miner/src/vllm_miner/vllm_kernels.py) retains the quantization path and uses vanilla GEMM when the global no-mining setting is true. Its [scheme selection](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/miner/vllm-miner/src/vllm_miner/vllm_config.py) depends on the checkpoint's bit-width configuration.

This gives a concrete starting control: same 7-bit checkpoint and configuration, toggle mining, and check outputs and execution traces. Changing to an 8-bit or BF16 checkpoint would answer a different comparison. `MINER_NO_GATEWAY` and skipping block submission should not be substituted for disabling mining; omitting the network operation alone does not remove kernel work.

## Measurement to run

Use the same pinned checkpoint, prompts, tokenization, request schedule, fixed output-token budget, GPU count/type, parallelism, vLLM build and kernel settings. Warm up both modes, alternate repeated runs, synchronize the timed work, and retain every raw sample. Check output quality and that the intended mining kernels actually execute on eligible layers. Report prefill, decode and overall serving performance separately when their mix changes the result.

Track commitments, noising, hashing, work refresh and asynchronous processing. Account for winning-proof computation by measured cost and expected win frequency; triggering proof generation artificially often answers a different workload. Do not submit test blocks to a live network merely to measure overhead.

The published miner [requirements](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/miner/README.md) specify NVIDIA CUDA and currently supported SM90 H100/H200 testing. The M4 matrix benchmark cannot reproduce these vLLM results. No NVIDIA VM or cloud spending was started for this investigation.

## Result, scope, uncertainty, and next question

**Inference overhead is measurable with a matched A/B run, and the official code provides a useful mining-off control.** The currently located reported values are 3.9% and 5.08%; neither has been independently verified here. Source inspection and arithmetic are complete, but they are not an executed inference benchmark. The chain sample in [note 014](014-chain-observability.md) supplies no serving-time measurements.

What are the paired times, resource costs and eligible mining rewards for the same checkpoint with `MINER_NO_MINING` toggled on one supported NVIDIA setup?
