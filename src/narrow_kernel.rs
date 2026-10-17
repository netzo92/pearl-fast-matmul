//! Narrow storage and SIMD-friendly padding, with equally narrow direct controls.
use super::{advance, basis, direct, finish, fixture, median, timing, Fixture, Output, H, K, R, W};
use std::hint::black_box;

struct Plan {
    active: Vec<(usize, usize)>,
    stride: usize,
    contrasts: Vec<i16>,
}

fn narrow<T: TryFrom<i32>>(values: &[i32]) -> Vec<T>
where
    T::Error: std::fmt::Debug,
{
    values
        .iter()
        .map(|&x| T::try_from(x).expect("coefficient must fit exactly"))
        .collect()
}

#[inline]
fn dot<T: Copy + Into<i32>>(a: &[T], b: &[T]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x.into() * y.into()).sum()
}

fn direct_narrow<T: Copy + Into<i32>, const CAPTURE: bool>(
    f: &Fixture,
    a: &[T],
    bt: &[T],
) -> Output {
    let mut acc = vec![0i32; f.m * f.n];
    let mut states = vec![[0; 16]; f.m / H * f.n / W];
    let mut prefixes = Vec::new();
    for chunk in 0..K / R {
        let start = chunk * R;
        let packed: Vec<T> = bt
            .chunks_exact(K)
            .flat_map(|row| row[start..start + R].iter().copied())
            .collect();
        for i in 0..f.m {
            let row = &a[i * K + start..i * K + start + R];
            for j in 0..f.n {
                acc[i * f.n + j] += dot(row, &packed[j * R..(j + 1) * R]);
            }
        }
        advance(&acc, &mut states, f.m, f.n, chunk);
        if CAPTURE {
            prefixes.push(acc.clone());
        }
    }
    finish(prefixes, states, &f.seed)
}

fn prepare(f: &Fixture) -> Vec<Plan> {
    f.pairs
        .chunks(R)
        .map(|edges| {
            let active = basis(edges);
            let stride = (active.len() + 7) / 8 * 8;
            let mut contrasts = vec![0i16; f.n * stride];
            for (j, row) in f.v.iter().enumerate() {
                for (t, &(vertex, anchor)) in active.iter().enumerate() {
                    contrasts[j * stride + t] = row[vertex] as i16 - row[anchor] as i16;
                }
            }
            Plan {
                active,
                stride,
                contrasts,
            }
        })
        .collect()
}

fn grouped<const CAPTURE: bool>(f: &Fixture, a: &[i16], plans: &[Plan]) -> Output {
    let mut acc = vec![0i32; f.m * f.n];
    let mut states = vec![[0; 16]; f.m / H * f.n / W];
    let mut prefixes = Vec::new();
    for (chunk, plan) in plans.iter().enumerate() {
        let start = chunk * R;
        // Bounds: |A'|<=127, so every |G|<=128*127=16256 fits i16.
        let mut compressed = vec![0i16; plan.stride];
        for i in 0..f.m {
            let mut g = [0i16; R];
            for (s, &(p, q)) in f.pairs[start..start + R].iter().enumerate() {
                let value = a[i * K + start + s];
                g[p] += value;
                g[q] -= value;
            }
            for (t, &(vertex, _)) in plan.active.iter().enumerate() {
                compressed[t] = g[vertex];
            }
            for j in 0..f.n {
                acc[i * f.n + j] += dot(
                    &compressed,
                    &plan.contrasts[j * plan.stride..(j + 1) * plan.stride],
                );
            }
        }
        advance(&acc, &mut states, f.m, f.n, chunk);
        if CAPTURE {
            prefixes.push(acc.clone());
        }
    }
    finish(prefixes, states, &f.seed)
}

struct BytePlan {
    active: Vec<(usize, usize)>,
    stride: usize,
    contrasts: Vec<i8>,
}

fn prepare_bytes(f: &Fixture) -> Vec<BytePlan> {
    f.pairs
        .chunks(R)
        .map(|edges| {
            let active = basis(edges);
            let stride = active.len().div_ceil(8) * 8;
            let mut contrasts = vec![0i8; f.n * stride];
            for (j, row) in f.v.iter().enumerate() {
                for (t, &(vertex, anchor)) in active.iter().enumerate() {
                    contrasts[j * stride + t] = row[vertex] - row[anchor];
                }
            }
            BytePlan {
                active,
                stride,
                contrasts,
            }
        })
        .collect()
}

fn grouped_bytes<const CAPTURE: bool>(f: &Fixture, a: &[i16], plans: &[BytePlan]) -> Output {
    let mut acc = vec![0i32; f.m * f.n];
    let mut states = vec![[0; 16]; f.m / H * f.n / W];
    let mut prefixes = Vec::new();
    for (chunk, plan) in plans.iter().enumerate() {
        let start = chunk * R;
        let mut compressed = vec![0i8; plan.stride];
        let mut corrections = Vec::with_capacity(R);
        for i in 0..f.m {
            let mut g = [0i16; R];
            for (s, &(p, q)) in f.pairs[start..start + R].iter().enumerate() {
                let value = a[i * K + start + s];
                g[p] += value;
                g[q] -= value;
            }
            corrections.clear();
            for (t, &(vertex, _)) in plan.active.iter().enumerate() {
                let exact = g[vertex];
                let clipped = exact.clamp(-128, 127) as i8;
                compressed[t] = clipped;
                if exact != clipped as i16 {
                    corrections.push((t, exact - clipped as i16));
                }
            }
            for j in 0..f.n {
                let row = &plan.contrasts[j * plan.stride..(j + 1) * plan.stride];
                let mut value = dot(&compressed, row);
                for &(t, residual) in &corrections {
                    value += residual as i32 * row[t] as i32;
                }
                acc[i * f.n + j] += value;
            }
        }
        advance(&acc, &mut states, f.m, f.n, chunk);
        if CAPTURE {
            prefixes.push(acc.clone());
        }
    }
    finish(prefixes, states, &f.seed)
}

pub(super) fn run() {
    let shapes = [(8, 32), (128, 128), (512, 128)];
    let mut checked = 0;
    for &(m, n) in &shapes {
        for id in 0..3 {
            for nonzero in [false, true] {
                let f = fixture(m, n, id, nonzero);
                let a8 = narrow::<i8>(&f.a);
                let b8 = narrow::<i8>(&f.bt);
                let a16 = narrow::<i16>(&f.a);
                let b16 = narrow::<i16>(&f.bt);
                let expected = direct::<true, true>(&f);
                assert_eq!(
                    expected,
                    direct_narrow::<i8, true>(&f, &a8, &b8),
                    "packed int8 prefixes/state/hash"
                );
                assert_eq!(
                    expected,
                    direct_narrow::<i16, true>(&f, &a16, &b16),
                    "packed int16 prefixes/state/hash"
                );
                assert_eq!(
                    expected,
                    grouped::<true>(&f, &a16, &prepare(&f)),
                    "grouped int16 prefixes/state/hash"
                );
                assert_eq!(
                    expected,
                    grouped_bytes::<true>(&f, &a16, &prepare_bytes(&f)),
                    "clipped int8 and residual prefixes/state/hash"
                );
                checked += m / H * n / W;
            }
        }
    }
    println!("narrow correctness: 18 fixtures, {checked} tickets, all prefixes/states/hashes match int32 reference");
    for (m, n) in shapes {
        let f = fixture(m, n, 0, false);
        let a8 = narrow::<i8>(&f.a);
        let b8 = narrow::<i8>(&f.bt);
        let a16 = narrow::<i16>(&f.a);
        let b16 = narrow::<i16>(&f.bt);
        let plan = prepare(&f);
        let padded: usize = plan.iter().map(|p| p.stride).sum();
        let actual: usize = plan.iter().map(|p| p.active.len()).sum();
        let reps = (128_000_000 / (m * n * K)).clamp(4, 256);
        let mut t8 = Vec::new();
        let mut t16 = Vec::new();
        let mut t32 = Vec::new();
        let mut tg = Vec::new();
        let mut tc = Vec::new();
        black_box(direct_narrow::<i8, false>(&f, &a8, &b8));
        black_box(direct_narrow::<i16, false>(&f, &a16, &b16));
        black_box(direct::<false, true>(&f));
        black_box(grouped::<false>(&f, &a16, &plan));
        black_box(grouped_bytes::<false>(&f, &a16, &prepare_bytes(&f)));
        for round in 0..7 {
            for offset in 0..5 {
                match (round + offset) % 5 {
                    0 => t8.push(timing(reps, || {
                        direct_narrow::<i8, false>(black_box(&f), black_box(&a8), black_box(&b8))
                    })),
                    1 => t16.push(timing(reps, || {
                        direct_narrow::<i16, false>(black_box(&f), black_box(&a16), black_box(&b16))
                    })),
                    2 => t32.push(timing(reps, || direct::<false, true>(black_box(&f)))),
                    3 => tc.push(timing(reps, || {
                        grouped_bytes::<false>(
                            black_box(&f),
                            black_box(&a16),
                            &prepare_bytes(black_box(&f)),
                        )
                    })),
                    _ => tg.push(timing(reps, || {
                        grouped::<false>(black_box(&f), black_box(&a16), &prepare(black_box(&f)))
                    })),
                }
            }
        }
        println!("m={m} n={n} tickets={} reps={reps} true_inner={actual} padded_inner={padded} direct_inner={K}",m/H*n/W);
        println!("grouped_clipped_i8_us={tc:?}");
        println!("packed_i8_us={t8:?}\npacked_i16_us={t16:?}\npacked_i32_us={t32:?}\ngrouped_i16_us={tg:?}");
        println!(
            "medians_us: clipped_i8={:.3} i8={:.3} i16={:.3} i32={:.3} grouped={:.3}; speedup_vs_best_direct={:.3}x",
            median(&tc),
            median(&t8),
            median(&t16),
            median(&t32),
            median(&tg),
            median(&t8).min(median(&t16)).min(median(&t32)) / median(&tg)
        );
    }
    println!("includes direct packing and group setup; excludes noising/commitments/proofs; conversions of prepared inputs excluded equally; CPU only");
}
