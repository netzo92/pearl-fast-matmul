import json
import hashlib
import math
from pathlib import Path
from time import perf_counter
import numpy as np

assert all(65521 % d for d in range(2, math.isqrt(65521)+1)), 'rank modulus must be prime'


def graph_dimension(edges):
    parents = list(range(128))
    def root(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v
    for p, q in edges:
        p, q = root(int(p)), root(int(q))
        parents[max(p,q)] = min(p,q)
    return 128-len({root(v) for v in range(128)})


def core(p, q):
    t = np.zeros((128, 128), dtype=np.int64)
    for a, b, sign in [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]:
        np.add.at(t, (p[:, a], q[:, b]), sign)
    return t


def modular_rank(t, prime=65521):
    a = t.copy() % prime
    row = 0
    for col in range(a.shape[1]):
        indices = np.flatnonzero(a[row:, col])
        if len(indices) == 0:
            continue
        pivot = row + indices[0]
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, prime) % prime
        below = a[row+1:, col].copy()
        a[row+1:] = (a[row+1:] - below[:, None] * a[row]) % prime
        row += 1
        if row == a.shape[0]: break
    return row


def factor(t, cap=127):
    w = t.copy()
    xs, ys = [], []
    for _ in range(128):
        units = np.argwhere(np.abs(w) == 1)
        if not len(units): break
        nr = np.count_nonzero(w, axis=1)
        nc = np.count_nonzero(w, axis=0)
        costs = (nr[units[:, 0]] - 1) * (nc[units[:, 1]] - 1)
        best = None
        for ix in np.argsort(costs, kind='stable'):
            i, j = units[ix]
            x = w[:, j] * w[i, j]
            y = w[i].copy()
            updated = w - x[:, None] * y
            if np.max(np.abs(updated)) <= cap:
                best = (x, y, updated)
                break
        if best is None: break
        x, y, w = best
        xs.append(x)
        ys.append(y)
    pivots = len(xs)
    cols = np.flatnonzero(np.any(w, axis=0))
    rows = np.flatnonzero(np.any(w, axis=1))
    if len(cols) <= len(rows):
        for j in cols:
            x = w[:, j].copy()
            y = np.zeros(128, dtype=np.int64); y[j] = 1
            xs.append(x); ys.append(y)
    else:
        for i in rows:
            x = np.zeros(128, dtype=np.int64); x[i] = 1
            xs.append(x); ys.append(w[i].copy())
    x = np.array(xs, dtype=np.int64).T
    y = np.array(ys, dtype=np.int64)
    assert np.array_equal(x @ y, t)
    return x, y, pivots


def main():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument('directory', type=Path)
    a.add_argument('--limit', type=int, default=32)
    a.add_argument('--caps', default='2,3,4,8,16,32,64,127')
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    caps = [int(c) for c in args.caps.split(',')]
    assert caps and all(2 <= c <= 127 for c in caps)
    manifest = json.loads((args.directory/'manifest.json').read_text())
    cases = []
    wall = perf_counter()
    for case in manifest['cases'][:args.limit]:
        directory = args.directory/case['name']
        pq = np.fromfile(directory/'pairs.u32', dtype='<u4').reshape(2, 2048, 2)
        u = np.fromfile(directory/'u.i8', dtype='i1').reshape(case['m'],128).astype(np.int64)
        v = np.fromfile(directory/'v.i8', dtype='i1').reshape(case['n'],128).astype(np.int64)
        # Full modular column rank makes the core's modular rank a lower bound
        # on the corresponding noise-only output product's rational rank.
        u_rank = modular_rank(u[:128])
        v_rank = modular_rank(v[:128])
        assert u_rank == v_rank == 128
        chunks = []
        cumulative = np.zeros((128,128),dtype=np.int64)
        for c in range(16):
            t = core(pq[0, c*128:(c+1)*128], pq[1, c*128:(c+1)*128])
            cumulative += t
            rank = modular_rank(t)
            started = perf_counter()
            tried = []
            for cap in caps:
                x, y, pivots = factor(t, cap)
                tried.append(cap)
                if (x.shape[1]+15)//16 == (rank+15)//16:
                    break
            factor_seconds = perf_counter() - started
            prefix_rank = modular_rank(cumulative)
            d = x.shape[1]
            left = u @ x
            right = v @ y.T
            x.astype('<i4').tofile(directory/f'chunk-{c}-x.i32')
            y.astype('<i4').tofile(directory/f'chunk-{c}-y.i32')
            chunks.append({'chunk': c, 'rank_lower_bound_mod65521': rank, 'factor_width': d,
                'selected_update_cap': cap, 'tried_update_caps': tried,
                'one_sided_a_width': graph_dimension(pq[0,c*128:(c+1)*128]),
                'one_sided_b_width': graph_dimension(pq[1,c*128:(c+1)*128]),
                'rank_proven_exact': rank == d, 'unit_pivots': pivots,
                'factor_seconds': factor_seconds,
                'prefix_rank_lower_bound_mod65521': prefix_rank,
                'left_count': int(left.size), 'right_count': int(right.size),
                'left_outside_i8': int(np.count_nonzero((left < -128) | (left > 127))),
                'right_outside_i8': int(np.count_nonzero((right < -128) | (right > 127))),
                'left_max_abs': int(np.max(np.abs(left))), 'right_max_abs': int(np.max(np.abs(right))),
                'x_nonzeros': int(np.count_nonzero(x)), 'y_nonzeros': int(np.count_nonzero(y)),
                'max_factor_abs': int(max(np.max(np.abs(x)), np.max(np.abs(y)))),
                'max_core_abs': int(np.max(np.abs(t))), 'core_nonzeros': int(np.count_nonzero(t))})
        np.array([c['factor_width'] for c in chunks], dtype='<u4').tofile(directory/'widths.u32')
        case = {**case, 'chunks': chunks,
                'u_modular_rank': u_rank, 'v_modular_rank': v_rank,
                'rank_sum': sum(c['rank_lower_bound_mod65521'] for c in chunks),
                'factor_width_sum': sum(c['factor_width'] for c in chunks),
                'factor_padded16_sum': sum((c['factor_width']+15)//16*16 for c in chunks),
                'rank_lower_bound_padded16_sum': sum((c['rank_lower_bound_mod65521']+15)//16*16 for c in chunks),
                'prefix_rank_lower_bound_padded16_sum': sum((c['prefix_rank_lower_bound_mod65521']+15)//16*16 for c in chunks),
                'one_sided_a_padded16_sum': sum((c['one_sided_a_width']+15)//16*16 for c in chunks),
                'one_sided_b_padded16_sum': sum((c['one_sided_b_width']+15)//16*16 for c in chunks),
                'best_one_sided_per_chunk_padded16_sum': sum((min(c['one_sided_a_width'],c['one_sided_b_width'])+15)//16*16 for c in chunks),
                'full_transform_scalar_multiplications': sum(case['m']*c['x_nonzeros'] + case['n']*c['y_nonzeros'] for c in chunks)}
        cases.append(case)
        print(case['name'], 'rank', case['rank_sum'], 'width', case['factor_width_sum'],
            'padded', case['factor_padded16_sum'], 'max factor', max(c['max_factor_abs'] for c in chunks), flush=True)
    args.output.write_text(json.dumps({'scope':'Exact local integer factorization and arithmetic screening. No GPU benchmark or new official/network proof acceptance.',
        'factor_update_caps':caps,'cases': cases, 'wall_seconds': perf_counter()-wall,
        'numpy_version':np.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__': main()
