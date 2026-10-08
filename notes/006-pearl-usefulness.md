# Can a Pearl certificate prove economic usefulness?

Historical research note. See [the current batch and official-verifier results](009-batch-and-conformance.md) for the latest status.

## Question

Pearl's paper lets a miner choose arbitrary input matrices and defines a proof whose verifier receives the seed and a certificate containing those matrices and a multiplication transcript. Can that certificate alone establish that an external customer requested the product? This is a question about *demand provenance*, separate from whether the transcript was costly to compute.

The formal target is Definition 5.1 and Algorithm 6.2 of Komargodski and Weinstein, *Proofs of Useful Work from Arbitrary Matrix Multiplication*, arXiv:2504.09971v4, sections 5 and 6: https://arxiv.org/html/2504.09971v4 . Pearl's public explanation makes the stronger assertion that every coin was mined through matrix multiplications serving real inference requests: https://pearlresearch.ai/research/blog/proof-of-useful-work .

## Claim and proof

Let `D` be a bit indicating whether an external customer wanted the product `AB`. Suppose the verifier's entire view is `V = (sigma, A, B, z)`, where `z` is the transcript and any certificate fields are functions of that view and public randomness. No authenticated request or independent customer evidence is supplied.

Construct two worlds with the same `sigma`, `A`, `B`, and miner randomness. In world 1, a customer requested `AB`, so `D=1`. In world 0, the miner chose the identical matrices without a customer, so `D=0`. Run the same honest `Solve` algorithm in both. The transcript and certificate are identical, hence every verifier gives the same answer. More generally, if the distribution of `V` is the same in both worlds, every statistical test has distinguishing advantage zero. In the deterministic coupling above, this follows pointwise and needs no cryptographic assumption.

Thus **a certificate derived only from these inputs cannot certify external demand**. An accepted certificate may show that a matrix multiplication was performed, conditional on the paper's hardness conjecture, but acceptance does not imply that its result was used by an inference customer. An authenticated request or other outside evidence would change the verifier's view and would need a separate analysis.

## Reproducible tiny example

Run `python3 -m experiments.pearl_usefulness` from the repository root. It computes the same finite-field transcript twice, changing only the external demand label. Both transcript digests match. The program illustrates the coupling; the argument above proves it for arbitrary dimensions and any verifier with the stated view.

This is a toy model over `F_7`, not a Pearl network certificate or a protocol compatibility test. It makes no elapsed-time or cryptographic-security claim.

## Result

**Proved within the stated verifier-view model.** This refutes the stronger claim that an accepted certificate by itself establishes that the multiplication served a real inference request. It does **not** refute the paper's conditional proof-of-work result: Definition 5.1 asks whether arbitrary chosen matrix products can generate certificates efficiently and whether cheating saves work. The economic value of a particular input is not verified there.

## Next question

Can one find an algorithm that computes every `r x r` partial-sum tile of two random rank-`r`, `n x n` matrices in `o(n^(omega_r+1)/r)` field operations for a parameter range where the transcript's `Theta(n^3/r)` output size does not already rule this out? That would directly challenge Assumption 6.4 and needs an explicit algorithm, cost model, and correctness proof. No such algorithm is established here.
