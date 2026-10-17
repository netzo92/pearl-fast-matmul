//! Portable, exact fixtures for the local Metal implementation.
//! This child module accesses the existing batch fixture and direct reference.
use super::{basis, direct, fixture, prepare, H, K, R, W};
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
    fs::create_dir_all(dir)?;
    let mut entries = Vec::new();
    let mut tickets = 0usize;
    let mut prefix_entries = 0usize;
    for (m, n) in [(8, 32), (128, 128), (512, 128)] {
        for id in 0..2 {
            for nonzero_a in [false, true] {
                let f = fixture(m, n, id, nonzero_a);
                // All expected products/states/hashes come from the direct method.
                let expected = direct::<true, false>(&f);
                let plans = prepare(&f);
                let name = format!("m{m}-n{n}-id{id}-a{}", usize::from(nonzero_a));
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
        }
    }
    let manifest = format!("{{\"version\":1,\"k\":{K},\"r\":{R},\"h\":{H},\"w\":{W},\"byte_order\":\"little\",\"fixtures\":[{}]}}\n", entries.join(",\n"));
    fs::write(dir.join("manifest.json"), manifest)?;
    println!("export complete: {} fixtures, {tickets} tickets, {prefix_entries} exact prefix entries; {}", entries.len(), dir.join("manifest.json").display());
    Ok(())
}
