//! Portable, exact fixtures for the local Metal implementation.
//! This child module accesses the existing batch fixture and direct reference.
use super::{basis, direct, fixture, fixture_with_raw_a, prepare, Fixture, H, K, R, W};
use std::{fs, hint::black_box, io, path::Path, time::Instant};

fn write_i8(path: &Path, values: &[i32]) -> io::Result<()> {
    let bytes: Vec<u8> = values
        .iter()
        .map(|&x| i8::try_from(x).expect("noised fixture must fit signed int8") as u8)
        .collect();
    fs::write(path, bytes)
}
fn write_u32(path: &Path, values: impl IntoIterator<Item = u32>) -> io::Result<()> {
    fs::write(
        path,
        values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn export(dir: &Path) -> io::Result<()> {
    let mut cases = Vec::new();
    for (m, n) in [(8, 32), (128, 128), (512, 128)] {
        for id in 0..2 {
            for nonzero_a in [false, true] {
                cases.push((
                    fixture(m, n, id, nonzero_a),
                    id,
                    nonzero_a,
                    format!("m{m}-n{n}-id{id}-a{}", usize::from(nonzero_a)),
                ));
            }
        }
    }
    export_cases(dir, cases)
}

/// Same synthetic job key and zero B; four sparse raw A commitments per shape.
/// A's seed is regenerated for each candidate. No graph/seed selection search.
pub fn export_cached(dir: &Path) -> io::Result<()> {
    let mut cases = Vec::new();
    for (m, n) in [(128, 128), (512, 128), (512, 512), (2048, 512)] {
        for id in 0..2 {
            let mut previous = None;
            for candidate in 0..4 {
                let mut raw_a = vec![vec![0i8; K]; m];
                raw_a[0][candidate] = 1;
                let f = fixture_with_raw_a(m, n, id, raw_a);
                if let Some((ref bt, ref pairs, ref seed)) = previous {
                    assert_eq!(&f.bt, bt, "B must be unchanged across A commitments");
                    assert_eq!(&f.pairs, pairs);
                    assert_ne!(&f.seed, seed, "A seed must change");
                }
                previous = Some((f.bt.clone(), f.pairs.clone(), f.seed));
                cases.push((
                    f,
                    id,
                    true,
                    format!("m{m}-n{n}-id{id}-candidate{candidate}"),
                ));
            }
        }
    }
    export_cases(dir, cases)
}

fn export_cases(dir: &Path, cases: Vec<(Fixture, u32, bool, String)>) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut entries = Vec::new();
    let mut tickets = 0usize;
    let mut prefix_entries = 0usize;
    for (f, id, nonzero_a, name) in cases {
        let (m, n) = (f.m, f.n);
        // All expected products/states/hashes come from the direct method.
        let expected = direct::<true, false>(&f);
        let plans = prepare(&f);
        let case_dir = dir.join(&name);
        fs::create_dir_all(&case_dir)?;
        write_i8(&case_dir.join("a.i8"), &f.a)?;
        write_i8(&case_dir.join("bt.i8"), &f.bt)?;
        fs::write(
            case_dir.join("v.i8"),
            f.v.iter().flatten().map(|&v| v as u8).collect::<Vec<_>>(),
        )?;
        write_u32(
            &case_dir.join("pairs.u32"),
            f.pairs.iter().flat_map(|&(p, q)| [p as u32, q as u32]),
        )?;
        let mut vertices = vec![0u32; K];
        let mut anchors = vec![0u32; K];
        for (chunk, plan) in plans.iter().enumerate() {
            assert!(plan.active.len() <= R);
            for (t, &(v, a)) in plan.active.iter().enumerate() {
                vertices[chunk * R + t] = v as u32;
                anchors[chunk * R + t] = a as u32;
            }
        }
        write_u32(&case_dir.join("vertices.u32"), vertices)?;
        write_u32(&case_dir.join("anchors.u32"), anchors)?;
        write_u32(
            &case_dir.join("d.u32"),
            plans.iter().map(|p| p.active.len() as u32),
        )?;
        write_u32(
            &case_dir.join("prefixes.i32"),
            expected.prefixes.iter().flatten().map(|&x| x as u32),
        )?;
        write_u32(
            &case_dir.join("states.u32"),
            expected.states.iter().flatten().copied(),
        )?;
        fs::write(
            case_dir.join("hashes.u8"),
            expected
                .hashes
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )?;

        // Basis planning only. GPU grouping/contrasts remain in the Metal measurement.
        // Keep seven raw samples in the manifest so setup accounting is auditable.
        let mut graph_samples = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..64 {
                black_box(
                    f.pairs
                        .chunks_exact(R)
                        .map(|edges| basis(black_box(edges)))
                        .collect::<Vec<_>>(),
                );
            }
            graph_samples.push(start.elapsed().as_nanos() as f64 / 64.0);
        }
        let mut sorted = graph_samples.clone();
        sorted.sort_by(f64::total_cmp);
        let graph_cpu_ns = sorted[sorted.len() / 2];
        let count = (m / H) * (n / W);
        let sample_json = graph_samples
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
                    "{{\"name\":\"{name}\",\"m\":{m},\"n\":{n},\"k\":{K},\"r\":{R},\"h\":{H},\"w\":{W},\"id\":{id},\"nonzero_a\":{nonzero_a},\"tickets\":{count},\"seed_a\":\"{}\",\"graph_cpu_ns\":{graph_cpu_ns},\"graph_cpu_ns_samples\":[{sample_json}],\"a\":\"{name}/a.i8\",\"bt\":\"{name}/bt.i8\",\"v\":\"{name}/v.i8\",\"pairs\":\"{name}/pairs.u32\",\"vertices\":\"{name}/vertices.u32\",\"anchors\":\"{name}/anchors.u32\",\"d\":\"{name}/d.u32\",\"prefixes\":\"{name}/prefixes.i32\",\"states\":\"{name}/states.u32\",\"hashes\":\"{name}/hashes.u8\"}}", hex(&f.seed)
                ));
        tickets += count;
        prefix_entries += (K / R) * m * n;
        println!(
            "exported {name}: {count} tickets, {} prefix entries",
            (K / R) * m * n
        );
    }
    let manifest = format!("{{\"version\":1,\"k\":{K},\"r\":{R},\"h\":{H},\"w\":{W},\"byte_order\":\"little\",\"fixtures\":[{}]}}\n", entries.join(",\n"));
    fs::write(dir.join("manifest.json"), manifest)?;
    println!("export complete: {} fixtures, {tickets} tickets, {prefix_entries} exact prefix entries; {}", entries.len(), dir.join("manifest.json").display());
    Ok(())
}
