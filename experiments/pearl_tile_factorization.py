"""Check direct and factored partial tile states on a tiny integer example."""

H, W, R, K = 2, 2, 2, 4
U = ((1, 2), (3, -1))
P = ((1, 0, -1, 1), (-1, 1, 1, 0))
Q = ((1, -1), (0, 1), (-1, 1), (1, 0))
V = ((2, -1), (1, 3))
MASK = (1 << 32) - 1


def rotl32(x, amount):
    return ((x << amount) | (x >> (32 - amount))) & MASK


def direct_prefix(limit):
    return tuple(tuple(sum(sum(U[i][a] * P[a][s] for a in range(R)) *
                               sum(Q[s][b] * V[b][j] for b in range(R))
                               for s in range(limit)) for j in range(W)) for i in range(H))


def factored_prefix(limit):
    state = [[sum(P[a][s] * Q[s][b] for s in range(limit))
              for b in range(R)] for a in range(R)]
    return tuple(tuple(sum(U[i][a] * state[a][b] * V[b][j]
                           for a in range(R) for b in range(R))
                       for j in range(W)) for i in range(H))


def xor32(tile):
    value = 0
    for row in tile:
        for cell in row:
            value ^= cell & MASK
    return value


def main():
    digest = [0] * 16
    for chunk, limit in enumerate(range(R, K + 1, R)):
        direct = direct_prefix(limit)
        factored = factored_prefix(limit)
        assert direct == factored
        x = xor32(direct)
        digest[chunk % 16] = rotl32(digest[chunk % 16], 13) ^ x
        print(f"prefix={limit} tile={direct} XOR={x:#010x}")
    print("matching factorization and hash state:", True)
    print("hash state:", tuple(digest))


if __name__ == "__main__":
    main()
