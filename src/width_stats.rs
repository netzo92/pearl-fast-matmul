//! Exact signed-byte clipping plus sparse-residual arithmetic exploration.
//! Fixture seeds, raw matrices, prefix extraction, and direct reference come
//! from the parent batching experiment. No official-verifier or GPU claim.
use super::{advance, direct, finish, fixture, prepare, H, K, R, W};

#[derive(Default)]
struct WidthStats {
    count: usize,
    outside: usize,
    outside_residual: usize,
    extra_i8_terms: usize,
    max_g: i32,
    min_g: i32,
    max_residual: i32,
    min_residual: i32,
    basis_total: usize,
    dmin: usize,
    dmax: usize,
    pad16: usize,
    pad32: usize,
    max_row_residuals: usize,
}
pub(super) fn run() {
    let (m, n) = (128, 32);
    for nonzero in [false, true] {
        let mut st = WidthStats {
            dmin: R,
            ..Default::default()
        };
        for id in 0..8 {
            let f = fixture(m, n, id, nonzero);
            let plans = prepare(&f);
            let mut acc = vec![0i32; m * n];
            let mut states = vec![[0u32; 16]; (m / H) * (n / W)];
            let mut prefixes = vec![];
            for (chunk, plan) in plans.iter().enumerate() {
                let d = plan.active.len();
                st.basis_total += d;
                st.dmin = st.dmin.min(d);
                st.dmax = st.dmax.max(d);
                st.pad16 += d.div_ceil(16) * 16;
                st.pad32 += d.div_ceil(32) * 32;
                for i in 0..m {
                    let mut g = [0i32; R];
                    for (s, &(p, q)) in f.pairs[chunk * R..(chunk + 1) * R].iter().enumerate() {
                        let a = f.a[i * K + chunk * R + s];
                        g[p] += a;
                        g[q] -= a;
                    }
                    let mut row_residuals = 0;
                    for (t, &(vertex, _anchor)) in plan.active.iter().enumerate() {
                        let x = g[vertex];
                        let small = x.clamp(-128, 127) as i8;
                        let residual = x - i32::from(small);
                        assert_eq!(i32::from(small) + residual, x);
                        st.count += 1;
                        st.min_g = st.min_g.min(x);
                        st.max_g = st.max_g.max(x);
                        if residual != 0 {
                            st.outside += 1;
                            row_residuals += 1;
                        }
                        if !(-128..=127).contains(&residual) {
                            st.outside_residual += 1;
                        }
                        st.min_residual = st.min_residual.min(residual);
                        st.max_residual = st.max_residual.max(residual);
                        let mut remaining = residual;
                        while remaining != 0 {
                            let part = remaining.clamp(-128, 127) as i8;
                            remaining -= i32::from(part);
                            st.extra_i8_terms += 1;
                        }
                        for j in 0..n {
                            let c = plan.contrasts[j * d + t];
                            assert!((-63..=63).contains(&c));
                            acc[i * n + j] += i32::from(small) * c;
                            if residual != 0 {
                                acc[i * n + j] += residual * c;
                            }
                        }
                    }
                    st.max_row_residuals = st.max_row_residuals.max(row_residuals);
                }
                advance(&acc, &mut states, m, n, chunk);
                prefixes.push(acc.clone());
            }
            let candidate = finish(prefixes, states, &f.seed);
            assert_eq!(
                direct::<true, true>(&f),
                candidate,
                "fixture id={id}, nonzero={nonzero}"
            );
        }
        let direct = 8 * m * n * K;
        let grouped = st.count * n;
        let corrected = (st.count + st.outside) * n;
        let bytesplit = (st.count + st.extra_i8_terms) * n;
        println!("nonzero_a={nonzero} fixtures=8 m={m} n={n} r={R} k={K}: every prefix/state/hash matches direct packed");
        println!("coefficients={} outside_i8={} outside_pct={:.6} g_range={}..{} residual_range={}..{} residual_outside_i8={} total_extra_i8_terms={} max_row_residuals={}",st.count,st.outside,100.0*st.outside as f64/st.count as f64,st.min_g,st.max_g,st.min_residual,st.max_residual,st.outside_residual,st.extra_i8_terms,st.max_row_residuals);
        println!(
            "d_range={}..{} basis_total={} average_d={:.6} average_pad16={:.6} average_pad32={:.6}",
            st.dmin,
            st.dmax,
            st.basis_total,
            st.basis_total as f64 / 128.0,
            st.pad16 as f64 / 128.0,
            st.pad32 as f64 / 128.0
        );
        println!("scalar_mults_direct={direct} grouped_wide={grouped} clipped_plus_wide_residual={corrected} clipped_plus_byte_residual={bytesplit}; saved_wide_residual_pct={:.6} saved_byte_residual_pct={:.6}",100.0*(1.0-corrected as f64/direct as f64),100.0*(1.0-bytesplit as f64/direct as f64));
        println!(
            "padded32_primary_plus_wide_residual_mults={} (direct={direct})",
            st.pad32 * m * n + st.outside * n
        );
    }
}
