# Can a chain scan identify real inference?

## Question

Do public Pearl block records establish that a customer requested and received inference?

## Source inspection

Source revision: [`6f996e3bb7d9a74ad6345d24a19c18ab782585fa`](https://github.com/pearl-research-labs/pearl/tree/6f996e3bb7d9a74ad6345d24a19c18ab782585fa), inspected on 2026-10-17. This is a schema audit of that revision, not a claim that all historical blocks used it.

The [public proof parameters](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/zk-pow/src/api/proof.rs) contain mining configuration, matrix commitments, a jackpot hash, dimensions and tile indices, with extra routing data for MoE. The raw matrix strips are private witnesses. There is no required customer-signed request, model identifier, prompt or inference-result receipt in this structure. A valid multiplication proof alone therefore cannot establish external demand or delivery of an inference result. An optional external receipt would require a separate check.

The [certificate format](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/node/wire/certificate.go) carries public proof parameters and ZK proof bytes. The [getblock implementation](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/node/rpcserver.go) can return serialized blocks at verbosity zero. A node with raw-block access would permit inspecting those certificate fields.

## Live API sample

The public Blockbook host is published in Pearl's [wallet source](https://github.com/pearl-research-labs/pearl/blob/6f996e3bb7d9a74ad6345d24a19c18ab782585fa/apps/apps/pearl-desktop-wallet/src/main/clients/blockbook-client.ts). Ten consecutive mainnet blocks, heights **129036 through 129045**, were read from `https://blockbook.pearlresearch.ai/api/v2/block/HEIGHT`. Adjacent hash links matched within this indexer's view. That is a consistency check, not independent chain or proof verification.

The responses contain block metadata and transactions, but none exposes certificate public data or proof bytes. This sample therefore cannot be used for a matrix-commitment comparison. It also cannot establish a count of real inference jobs. The [report](../results/chain-observability.json) deliberately records that count as `null`, rather than zero. [Compressed raw API responses](../results/chain-sample-129036-129045.json.gz) are preserved with source URLs and response hashes in the report. Confirmation counts will change in later API responses.

No full node was synced, transactions submitted, raw ZK proofs verified, or private inference logs examined. Transaction counts and ordinary payments are not evidence of inference requests.

## What a deeper scan could test

With raw certificates, the public matrix hashes and job parameters could be compared with commitments to known candidate inputs, such as all-zero matrices. This requires reproducing the exact keyed commitment and version-specific serialization. It has **not** been run on these blocks. Even a confirmed zero matrix would establish an input property, not by itself prove that no customer wanted that computation.

Positive attribution to an actual request would need authenticated evidence connecting the input commitments to a model execution and a customer request or result receipt. Publicly identifying a model's weights would not establish that anyone requested its output.

## Result, scope, uncertainty, and next question

**The checked proof schema does not require inference provenance; the ten-block public API sample cannot classify real inference.** This does not show that inference was absent. More blocks from the same API would not recover certificate bytes it omits.

Can raw certificates be obtained from a read-only node RPC and checked against known candidate matrix commitments?
