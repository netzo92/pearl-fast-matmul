"""Show inner-coordinate permutation symmetry and its commitment limit."""

from hashlib import sha256

A = ((1, 0, 2, 0), (0, 1, 0, 3), (2, 1, 1, 0), (0, 2, 3, 1))
B = ((1, 2, 0, 1), (3, 0, 1, 0), (2, 1, 2, 0), (0, 1, 3, 2))
R = 2
PERM = (1, 0, 3, 2)  # Swaps coordinates within each width-R chunk.


def prefixes(a, b):
    n = len(a)
    return tuple(
        tuple(tuple(sum(a[i][s] * b[s][j] for s in range(limit))
                    for j in range(n)) for i in range(n))
        for limit in range(R, n + 1, R)
    )


def permute_inner(a, b):
    new_a = tuple(tuple(row[PERM[s]] for s in range(len(PERM))) for row in a)
    new_b = tuple(b[PERM[s]] for s in range(len(PERM)))
    return new_a, new_b


def main():
    shifted_a, shifted_b = permute_inner(A, B)
    assert (shifted_a, shifted_b) != (A, B)
    assert prefixes(A, B) == prefixes(shifted_a, shifted_b)
    hash_a = sha256(repr((A, B)).encode()).hexdigest()
    hash_shifted = sha256(repr((shifted_a, shifted_b)).encode()).hexdigest()
    assert hash_a != hash_shifted
    print("same chunk-prefix matrices:", True)
    print("same input commitment:", False)
    print("original digest:", hash_a)
    print("shifted digest:", hash_shifted)
    print("A toy digest illustrates rebinding; it is not Pearl's commitment hash.")


if __name__ == "__main__":
    main()
