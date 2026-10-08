"""Tiny illustration of the demand-provenance indistinguishability argument."""

from hashlib import sha256


Q = 7
A = ((1, 2), (3, 4))
B = ((2, 0), (1, 5))


def transcript(a, b):
    """Classical partial-sum transcript for 2x2 matrices over F_7."""
    out = []
    for k in range(2):
        for i in range(2):
            for j in range(2):
                previous = out[(k - 1) * 4 + i * 2 + j] if k else 0
                out.append((previous + a[i][k] * b[k][j]) % Q)
    return tuple(out)


def visible_certificate(demand_label):
    """The label is external information absent from the certificate."""
    assert demand_label in ("customer", "miner")
    return (A, B, transcript(A, B))


def main():
    customer = visible_certificate("customer")
    miner = visible_certificate("miner")
    assert customer == miner
    digest = sha256(repr(customer).encode()).hexdigest()
    print("F_7, n=2, block size=1")
    print("transcript:", customer[2])
    print("same certificate in both worlds:", customer == miner)
    print("certificate digest:", digest)


if __name__ == "__main__":
    main()
