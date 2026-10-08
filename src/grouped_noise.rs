//! Exact signed-selection grouping for committed B=0; no full certificate claim.
use super::{
    flatten, hash, jackpot_hash, keyed, label, seeds_from_roots, sparse_pairs, uniform_rows, Key,
    K, R,
};
use std::{hint::black_box, time::Instant};

const H: usize = 8;
const W: usize = 32;
type Matrix = Vec<Vec<i32>>;
type Trace = Vec<Vec<i32>>;

struct Fixture {
    a: Matrix,
    bt: Matrix,
    v: Vec<Vec<i8>>,
    pairs: Vec<(usize, usize)>,
    seed_a: Key,
}

fn fixture(id: u32, nonzero_a: bool) -> Fixture {
    let mut raw_a = vec![vec![0i8; K]; H];
    if nonzero_a {
        for (i, row) in raw_a.iter_mut().enumerate() {
            for (s, value) in row.iter_mut().enumerate() {
                *value = ((i * 13 + s * 7 + id as usize) % 128) as i8 - 64;
            }
        }
    }
    let raw_bt = vec![vec![0i8; K]; W];
    let job_key = hash(format!("pearl-grouping-case-{id}").as_bytes());
    let (sb, sa) = seeds_from_roots(
        &job_key,
        &keyed(&job_key, &flatten(&raw_a)),
        &keyed(&job_key, &flatten(&raw_bt)),
        H as u32,
        W as u32,
    );
    let ua = uniform_rows(&sa, &label(b"A_tensor"), H);
    let pa = sparse_pairs(&sa, &label(b"A_tensor"));
    let v = uniform_rows(&sb, &label(b"B_tensor"), W);
    let pairs = sparse_pairs(&sb, &label(b"B_tensor"));
    let a = (0..H)
        .map(|i| {
            (0..K)
                .map(|s| raw_a[i][s] as i32 + ua[i][pa[s].0] as i32 - ua[i][pa[s].1] as i32)
                .collect()
        })
        .collect();
    let bt = materialize_b(&v, &pairs);
    Fixture {
        a,
        bt,
        v,
        pairs,
        seed_a: sa,
    }
}

fn materialize_b(v: &[Vec<i8>], pairs: &[(usize, usize)]) -> Matrix {
    v.iter()
        .map(|row| {
            pairs
                .iter()
                .map(|&(p, q)| row[p] as i32 - row[q] as i32)
                .collect()
        })
        .collect()
}

fn direct(a: &Matrix, bt: &Matrix) -> Trace {
    let mut acc = vec![0i32; H * W];
    let mut trace = Vec::with_capacity(K / R);
    for start in (0..K).step_by(R) {
        for i in 0..H {
            for j in 0..W {
                let mut sum = acc[i * W + j];
                // Slice bounds once so LLVM can vectorize the same contiguous
                // dot-product reduction used by the grouped kernel.
                for (&x, &y) in a[i][start..start + R].iter().zip(&bt[j][start..start + R]) {
                    sum += x * y;
                }
                acc[i * W + j] = sum;
            }
        }
        trace.push(acc.clone());
    }
    trace
}

fn root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

// Each edge p->q contributes +a_s at p and -a_s at q. Thus grouped
// coefficients sum to zero independently within every connected component.
pub(super) fn basis(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut parent: Vec<_> = (0..R).collect();
    for &(p, q) in pairs {
        let a = root(&mut parent, p);
        let b = root(&mut parent, q);
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    (0..R)
        .filter_map(|t| {
            let anchor = root(&mut parent, t);
            (t != anchor).then_some((t, anchor))
        })
        .collect()
}

fn grouped(a: &Matrix, v: &[Vec<i8>], pairs: &[(usize, usize)]) -> Trace {
    let mut acc = vec![0i32; H * W];
    let mut trace = Vec::with_capacity(K / R);
    for start in (0..K).step_by(R) {
        let chunk = &pairs[start..start + R];
        let active = basis(chunk);
        let d = active.len();
        let mut compressed = vec![0i32; H * d];
        for i in 0..H {
            let mut g = [0i32; R];
            for (offset, &(p, q)) in chunk.iter().enumerate() {
                let value = a[i][start + offset];
                g[p] += value;
                g[q] -= value;
            }
            for (t, &(vertex, _)) in active.iter().enumerate() {
                compressed[i * d + t] = g[vertex];
            }
        }
        let mut contrasts = vec![0i32; W * d];
        for j in 0..W {
            for (t, &(vertex, anchor)) in active.iter().enumerate() {
                contrasts[j * d + t] = v[j][vertex] as i32 - v[j][anchor] as i32;
            }
        }
        for i in 0..H {
            for j in 0..W {
                let mut sum = acc[i * W + j];
                for t in 0..d {
                    sum += compressed[i * d + t] * contrasts[j * d + t];
                }
                acc[i * W + j] = sum;
            }
        }
        trace.push(acc.clone());
    }
    trace
}

fn state(trace: &Trace) -> [u32; 16] {
    let mut out = [0u32; 16];
    for (chunk, values) in trace.iter().enumerate() {
        let x = values.iter().fold(0u32, |a, &b| a ^ b as u32);
        out[chunk % 16] = out[chunk % 16].rotate_left(13) ^ x;
    }
    out
}

fn check(f: &Fixture) {
    let expected = direct(&f.a, &f.bt);
    let actual = grouped(&f.a, &f.v, &f.pairs);
    assert_eq!(actual, expected, "every prefix matrix must match");
    assert_eq!(
        jackpot_hash(&state(&actual), &f.seed_a),
        jackpot_hash(&state(&expected), &f.seed_a)
    );
}

fn micros_per_call(mut f: impl FnMut() -> Trace) -> f64 {
    let now = Instant::now();
    for _ in 0..100 {
        black_box(f());
    }
    now.elapsed().as_secs_f64() * 1e6 / 100.0
}

pub fn run() {
    let mut d_total = 0usize;
    for id in 0..16 {
        for nonzero_a in [false, true] {
            let f = fixture(id, nonzero_a);
            check(&f);
            d_total += f.pairs.chunks(R).map(|c| basis(c).len()).sum::<usize>();
        }
    }
    // Algebraic boundary cases only: these graphs are not claimed to be
    // cheaply reachable from a committed seed.
    let mut edge = fixture(0, false);
    edge.pairs = vec![(0, 1); K];
    edge.bt = materialize_b(&edge.v, &edge.pairs);
    check(&edge);
    edge.pairs = (0..K)
        .map(|s| {
            let p = s % (R - 1);
            (p, p + 1)
        })
        .collect();
    edge.bt = materialize_b(&edge.v, &edge.pairs);
    check(&edge);
    let direct_mult = 32 * H * W * K;
    let grouped_mult = H * W * d_total;
    let grouping_adds = 32 * 2 * H * K;
    let contrast_subs = W * d_total;
    let grouped_adds = grouped_mult + grouping_adds + contrast_subs;
    println!("full V3 seed vectors: pass");
    println!("r={R}, k={K}, tile={H}x{W}; B=0, A=zero or signed int7");
    println!(
        "exact prefix/state/hash checks: 32 seed-derived cases and 2 algebraic boundaries pass"
    );
    println!("direct scalar multiplications: {direct_mult}");
    println!("grouped scalar multiplications: {grouped_mult}");
    println!(
        "multiplication reduction: {:.2}%",
        100.0 * (1.0 - grouped_mult as f64 / direct_mult as f64)
    );
    println!("direct scalar adds plus multiplies: {}", 2 * direct_mult);
    println!(
        "grouped scalar adds plus multiplies: {}",
        grouped_mult + grouped_adds
    );
    println!("counts exclude graph traversal, memory, noising, commitments and proof generation");
    let f = fixture(0, false);
    let mut direct_us = Vec::new();
    let mut grouped_us = Vec::new();
    black_box(direct(&f.a, &f.bt));
    black_box(grouped(&f.a, &f.v, &f.pairs));
    for round in 0..7 {
        if round % 2 == 0 {
            direct_us.push(micros_per_call(|| {
                direct(black_box(&f.a), black_box(&f.bt))
            }));
            grouped_us.push(micros_per_call(|| {
                grouped(black_box(&f.a), black_box(&f.v), black_box(&f.pairs))
            }));
        } else {
            grouped_us.push(micros_per_call(|| {
                grouped(black_box(&f.a), black_box(&f.v), black_box(&f.pairs))
            }));
            direct_us.push(micros_per_call(|| {
                direct(black_box(&f.a), black_box(&f.bt))
            }));
        }
    }
    println!("direct CPU microseconds, 7 batches of 100: {direct_us:?}");
    println!("grouped CPU microseconds, 7 batches of 100: {grouped_us:?}");
    direct_us.sort_by(f64::total_cmp);
    grouped_us.sort_by(f64::total_cmp);
    println!(
        "median direct/grouped kernel speedup: {:.3}x",
        direct_us[3] / grouped_us[3]
    );
    println!("CPU microbenchmark excludes commitment/noise/proof costs; includes graph analysis and materialized prefixes");
}
