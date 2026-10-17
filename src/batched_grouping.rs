//! Reuse graph/contrast plans across row tiles and grouped rows across column tiles.
use super::{
    flatten, grouped_noise::basis, hash, jackpot_hash, keyed, label, seeds_from_roots,
    sparse_pairs, uniform_rows, Key, K, R,
};
use std::{hint::black_box, time::Instant};
#[path = "metal_fixtures.rs"]
mod metal_fixtures;
pub fn export_metal(dir: &std::path::Path) -> std::io::Result<()> {
    metal_fixtures::export(dir)
}
#[path = "narrow_kernel.rs"]
mod narrow_kernel;
#[path = "width_stats.rs"]
mod width_stats;
pub fn run_width_stats() {
    width_stats::run();
}
pub fn run_narrow() {
    narrow_kernel::run();
}

const H: usize = 8;
const W: usize = 32;
struct Fixture {
    m: usize,
    n: usize,
    a: Vec<i32>,
    bt: Vec<i32>,
    v: Vec<Vec<i8>>,
    pairs: Vec<(usize, usize)>,
    seed: Key,
}
struct ChunkPlan {
    active: Vec<(usize, usize)>,
    contrasts: Vec<i32>,
}
#[derive(Debug, PartialEq)]
struct Output {
    prefixes: Vec<Vec<i32>>,
    states: Vec<[u32; 16]>,
    hashes: Vec<Key>,
}

fn fixture(m: usize, n: usize, id: u32, nonzero_a: bool) -> Fixture {
    let a: Vec<Vec<i8>> = (0..m)
        .map(|i| {
            (0..K)
                .map(|s| {
                    if nonzero_a {
                        ((i * 13 + s * 7 + id as usize) % 128) as i8 - 64
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();
    let b = vec![vec![0i8; K]; n];
    let key = hash(format!("pearl-batch-case-{id}").as_bytes());
    let (sb, sa) = seeds_from_roots(
        &key,
        &keyed(&key, &flatten(&a)),
        &keyed(&key, &flatten(&b)),
        m as u32,
        n as u32,
    );
    let u = uniform_rows(&sa, &label(b"A_tensor"), m);
    let pa = sparse_pairs(&sa, &label(b"A_tensor"));
    let v = uniform_rows(&sb, &label(b"B_tensor"), n);
    let pairs = sparse_pairs(&sb, &label(b"B_tensor"));
    let mut aa = Vec::with_capacity(m * K);
    for i in 0..m {
        for s in 0..K {
            aa.push(a[i][s] as i32 + u[i][pa[s].0] as i32 - u[i][pa[s].1] as i32);
        }
    }
    let bt = v
        .iter()
        .flat_map(|row| pairs.iter().map(|&(p, q)| row[p] as i32 - row[q] as i32))
        .collect();
    Fixture {
        m,
        n,
        a: aa,
        bt,
        v,
        pairs,
        seed: sa,
    }
}

#[inline]
fn dot(a: &[i32], b: &[i32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn advance(acc: &[i32], states: &mut [[u32; 16]], m: usize, n: usize, chunk: usize) {
    for ti in 0..m / H {
        for tj in 0..n / W {
            let mut x = 0u32;
            for i in ti * H..(ti + 1) * H {
                for &value in &acc[i * n + tj * W..i * n + (tj + 1) * W] {
                    x ^= value as u32;
                }
            }
            let word = &mut states[ti * (n / W) + tj][chunk % 16];
            *word = word.rotate_left(13) ^ x;
        }
    }
}

fn finish(prefixes: Vec<Vec<i32>>, states: Vec<[u32; 16]>, seed: &Key) -> Output {
    let hashes = states.iter().map(|s| jackpot_hash(s, seed)).collect();
    Output {
        prefixes,
        states,
        hashes,
    }
}

fn direct<const CAPTURE: bool, const PACKED: bool>(f: &Fixture) -> Output {
    let mut acc = vec![0i32; f.m * f.n];
    let mut states = vec![[0u32; 16]; (f.m / H) * (f.n / W)];
    let mut prefixes = Vec::new();
    for chunk in 0..K / R {
        let start = chunk * R;
        let packed: Vec<i32> = if PACKED {
            f.bt.chunks_exact(K)
                .flat_map(|row| row[start..start + R].iter().copied())
                .collect()
        } else {
            Vec::new()
        };
        for i in 0..f.m {
            let a = &f.a[i * K + start..i * K + start + R];
            for j in 0..f.n {
                let b = if PACKED {
                    &packed[j * R..(j + 1) * R]
                } else {
                    &f.bt[j * K + start..j * K + start + R]
                };
                acc[i * f.n + j] += dot(a, b);
            }
        }
        advance(&acc, &mut states, f.m, f.n, chunk);
        if CAPTURE {
            prefixes.push(acc.clone());
        }
    }
    finish(prefixes, states, &f.seed)
}

fn prepare(f: &Fixture) -> Vec<ChunkPlan> {
    f.pairs
        .chunks(R)
        .map(|edges| {
            let active = basis(edges);
            let contrasts =
                f.v.iter()
                    .flat_map(|row| active.iter().map(|&(t, a)| row[t] as i32 - row[a] as i32))
                    .collect();
            ChunkPlan { active, contrasts }
        })
        .collect()
}

fn grouped<const CAPTURE: bool>(f: &Fixture, plans: &[ChunkPlan]) -> Output {
    let mut acc = vec![0i32; f.m * f.n];
    let mut states = vec![[0u32; 16]; (f.m / H) * (f.n / W)];
    let mut prefixes = Vec::new();
    for (chunk, plan) in plans.iter().enumerate() {
        let start = chunk * R;
        let d = plan.active.len();
        // A row is grouped once and reused across all column tiles.
        let mut compressed = vec![0i32; d];
        for i in 0..f.m {
            let mut g = [0i32; R];
            for (s, &(p, q)) in f.pairs[start..start + R].iter().enumerate() {
                let a = f.a[i * K + start + s];
                g[p] += a;
                g[q] -= a;
            }
            for (t, &(vertex, _)) in plan.active.iter().enumerate() {
                compressed[t] = g[vertex];
            }
            for j in 0..f.n {
                acc[i * f.n + j] += dot(&compressed, &plan.contrasts[j * d..(j + 1) * d]);
            }
        }
        advance(&acc, &mut states, f.m, f.n, chunk);
        if CAPTURE {
            prefixes.push(acc.clone());
        }
    }
    finish(prefixes, states, &f.seed)
}

fn timing<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e6 / reps as f64
}
fn median(v: &[f64]) -> f64 {
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}

pub fn run() {
    let shapes = [(8, 32), (32, 32), (128, 32), (128, 128), (512, 128)];
    let mut tickets = 0;
    for &(m, n) in &shapes {
        for id in 0..3 {
            for nonzero in [false, true] {
                let f = fixture(m, n, id, nonzero);
                let plans = prepare(&f);
                assert_eq!(direct::<true,false>(&f),grouped::<true>(&f,&plans),
            "all prefixes, separate tile states and hashes: m={m},n={n},id={id},nonzero={nonzero}");
                assert_eq!(
                    direct::<true, false>(&f),
                    direct::<true, true>(&f),
                    "packed direct must match"
                );
                tickets += (m / H) * (n / W);
            }
        }
    }
    println!("correctness: 30 commitment-derived fixtures, {tickets} separate tickets, all prefixes/states/hashes match");
    println!("benchmark: CPU, i32 sliced dot on both paths; 7 alternating batches; times include all tile hashes");
    println!("grouped total includes one plan per committed batch; cached time excludes plan, never excludes A grouping");
    for (m, n) in shapes {
        let f = fixture(m, n, 0, false);
        let plan = prepare(&f);
        let d_sum: usize = plan.iter().map(|p| p.active.len()).sum();
        let reps = (128_000_000 / (m * n * K)).clamp(4, 256);
        let mut tb = Vec::new();
        let mut td = Vec::new();
        let mut tg = Vec::new();
        let mut tc = Vec::new();
        let mut tp = Vec::new();
        black_box(direct::<false, false>(&f));
        black_box(direct::<false, true>(&f));
        black_box(grouped::<false>(&f, &plan));
        for round in 0..7 {
            for offset in 0..4 {
                match (round + offset) % 4 {
                    0 => td.push(timing(reps, || direct::<false, false>(black_box(&f)))),
                    1 => tg.push(timing(reps, || {
                        let p = prepare(black_box(&f));
                        grouped::<false>(black_box(&f), &p)
                    })),
                    2 => tb.push(timing(reps, || direct::<false, true>(black_box(&f)))),
                    _ => tc.push(timing(reps, || {
                        grouped::<false>(black_box(&f), black_box(&plan))
                    })),
                }
            }
            tp.push(timing(reps, || prepare(black_box(&f))));
        }
        let tickets = (m / H) * (n / W);
        println!(
            "m={m} n={n} tickets={tickets} reps={reps} mul_saved_pct={:.3}",
            100.0 * (1.0 - d_sum as f64 / K as f64)
        );
        println!("packed_direct_us={tb:?}\ndirect_us={td:?}\ngrouped_total_us={tg:?}\ngrouped_cached_us={tc:?}\nplan_us={tp:?}");
        println!("medians_us: direct={:.3} packed_direct={:.3} grouped_total={:.3} grouped_cached={:.3} plan={:.3}; speedup_vs_best_direct={:.3}x; us_per_ticket={:.3}/{:.3}",
            median(&td),median(&tb),median(&tg),median(&tc),median(&tp),median(&td).min(median(&tb))/median(&tg),median(&td).min(median(&tb))/tickets as f64,median(&tg)/tickets as f64);
    }
    println!("excluded: raw commitments, seed expansion/A noising, direct B materialization, proof generation; no GPU or official verifier claim");
}
