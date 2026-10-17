//! Export commitment-derived factors for an exact both-sided integer study.
use super::{hash, jackpot_hash, keyed, label, seeds_from_roots, sparse_pairs, uniform_rows, K, R};
use std::{fs, io::Write, path::Path};

fn hex(x: &[u8]) -> String {
    x.iter().map(|b| format!("{b:02x}")).collect()
}

fn read_i32(path: &Path) -> Vec<i64> {
    let bytes = fs::read(path).expect("factor file");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()) as i64)
        .collect()
}

pub fn verify(out: &Path) -> std::io::Result<()> {
    let mut records = Vec::new();
    let mut total_entries = 0usize;
    let mut total_states = 0usize;
    let mut raw_controls = 0usize;
    let mut clip_controls = 0usize;
    let mut xor_controls = 0usize;
    for (m, n) in [(128usize, 128usize), (512, 512), (2048, 512), (2048, 2048)] {
        for id in 0..2 {
            let job = hash(format!("pearl-batch-case-{id}").as_bytes());
            let root_b = keyed(&job, &vec![0u8; n * K]);
            for candidate in 0..4usize {
                let mut raw_a = vec![0u8; m * K];
                let raw_position = if candidate == 0 {
                    None
                } else {
                    Some(candidate - 1)
                };
                if let Some(s) = raw_position {
                    raw_a[s] = 1;
                }
                let (sb, sa) =
                    seeds_from_roots(&job, &keyed(&job, &raw_a), &root_b, m as u32, n as u32);
                let p = sparse_pairs(&sa, &label(b"A_tensor"));
                let q = sparse_pairs(&sb, &label(b"B_tensor"));
                let u = uniform_rows(&sa, &label(b"A_tensor"), m);
                let v = uniform_rows(&sb, &label(b"B_tensor"), n);
                let name = format!("m{m}-n{n}-id{id}-candidate{candidate}");
                let dir = out.join(&name);
                for (file, rows) in [("u.i8", &u), ("v.i8", &v)] {
                    let expected: Vec<u8> = rows.iter().flatten().map(|&b| b as u8).collect();
                    assert_eq!(fs::read(dir.join(file))?, expected, "bound uniform factors");
                }
                let widths = read_i32(&dir.join("widths.u32"));
                assert_eq!(widths.len(), K / R);
                let mut tile_records = Vec::new();
                for (row_start, col_start) in [(0, 0), (m - 8, n - 32)] {
                    let mut direct = vec![0i64; 8 * 32];
                    let mut factored = direct.clone();
                    let mut ds = [0u32; 16];
                    let mut fs_state = ds;
                    let mut prefix_hash = blake3::Hasher::new();
                    for c in 0..K / R {
                        let d = widths[c] as usize;
                        let x = read_i32(&dir.join(format!("chunk-{c}-x.i32")));
                        let y = read_i32(&dir.join(format!("chunk-{c}-y.i32")));
                        assert_eq!(x.len(), R * d);
                        assert_eq!(y.len(), d * R);
                        let mut left = vec![0i64; 8 * d];
                        let mut right = vec![0i64; 32 * d];
                        for i in 0..8 {
                            for t in 0..d {
                                for a in 0..R {
                                    left[i * d + t] += u[row_start + i][a] as i64 * x[a * d + t];
                                }
                            }
                        }
                        for j in 0..32 {
                            for t in 0..d {
                                for b in 0..R {
                                    right[j * d + t] += y[t * R + b] * v[col_start + j][b] as i64;
                                }
                            }
                        }
                        let old_xor = direct.iter().fold(0u32, |a, &b| a ^ b as u32);
                        let mut delta_xor = 0u32;
                        let mut without_raw = factored.clone();
                        let mut any_clip_bad = false;
                        for i in 0..8 {
                            for j in 0..32 {
                                let mut delta = 0i64;
                                for s in c * R..(c + 1) * R {
                                    let av = u[row_start + i][p[s].0] as i64
                                        - u[row_start + i][p[s].1] as i64
                                        + raw_a[(row_start + i) * K + s] as i64;
                                    let bv = v[col_start + j][q[s].0] as i64
                                        - v[col_start + j][q[s].1] as i64;
                                    delta += av * bv;
                                }
                                let mut fd = 0i64;
                                let mut exact_correction = 0i64;
                                let mut low_only = 0i64;
                                for t in 0..d {
                                    let l = left[i * d + t];
                                    let r = right[j * d + t];
                                    let ll = l.clamp(-128, 127);
                                    let rr = r.clamp(-128, 127);
                                    fd += l * r;
                                    low_only += ll * rr;
                                    exact_correction += ll * rr + (l - ll) * r + ll * (r - rr);
                                }
                                assert_eq!(fd, exact_correction, "both exact residual corrections");
                                any_clip_bad |= low_only != fd;
                                without_raw[i * 32 + j] += fd;
                                if row_start + i == 0 {
                                    if let Some(s) = raw_position {
                                        if s / R == c {
                                            fd += v[col_start + j][q[s].0] as i64
                                                - v[col_start + j][q[s].1] as i64;
                                        }
                                    }
                                }
                                direct[i * 32 + j] += delta;
                                factored[i * 32 + j] += fd;
                                assert!(i32::try_from(direct[i * 32 + j]).is_ok());
                                delta_xor ^= delta as u32;
                                total_entries += 1;
                            }
                        }
                        assert_eq!(
                            factored, direct,
                            "every factored prefix including raw A correction"
                        );
                        if row_start == 0 && c == 0 && raw_position.is_some() {
                            assert_ne!(
                                without_raw, direct,
                                "negative control: omit raw A correction"
                            );
                            raw_controls += 1;
                        }
                        if any_clip_bad {
                            clip_controls += 1;
                        }
                        let dx = direct.iter().fold(0u32, |a, &b| a ^ b as u32);
                        let fx = factored.iter().fold(0u32, |a, &b| a ^ b as u32);
                        // XOR is not distributive through the integer prefix sum.
                        if old_xor ^ delta_xor != dx {
                            xor_controls += 1;
                        }
                        ds[c % 16] = ds[c % 16].rotate_left(13) ^ dx;
                        fs_state[c % 16] = fs_state[c % 16].rotate_left(13) ^ fx;
                        for &value in &direct {
                            prefix_hash.update(&(value as i32).to_le_bytes());
                        }
                    }
                    assert_eq!(ds, fs_state);
                    let digest = jackpot_hash(&ds, &sa);
                    assert_eq!(digest, jackpot_hash(&fs_state, &sa));
                    total_states += 1;
                    tile_records.push(format!(r#"{{"row_start":{},"column_start":{},"state":{:?},"hash":"{}","prefix_digest":"{}"}}"#,
                        row_start,col_start,ds,hex(&digest),prefix_hash.finalize().to_hex()));
                }
                records.push(format!(
                    r#"{{"name":"{}","seed_a":"{}","seed_b":"{}","tiles":[{}]}}"#,
                    name,
                    hex(&sa),
                    hex(&sb),
                    tile_records.join(",")
                ));
            }
        }
    }
    assert_eq!(total_entries, 32 * 2 * (K / R) * 8 * 32);
    assert_eq!(raw_controls, 24);
    assert!(clip_controls > 0 && xor_controls > 0);
    let report = format!(
        r#"{{"scope":"Independent local direct-prefix and hash checks. Not a new official verifier or network check.","prefix_entries_checked":{},"states_and_hashes_checked":{},"raw_a_correction_negative_controls":{},"prefixes_with_clip_only_mismatch":{},"prefixes_with_xor_linearity_mismatch":{},"cases":[{}]}}"#,
        total_entries,
        total_states,
        raw_controls,
        clip_controls,
        xor_controls,
        records.join(",")
    );
    fs::write(out.join("exact-checks.json"), report)?;
    eprintln!("Both-sided exact checks: {total_entries} prefix entries, {total_states} states/hashes, {raw_controls} raw-correction controls, {clip_controls} clip-only mismatches, {xor_controls} XOR-linearity mismatches");
    Ok(())
}

pub fn export(out: &Path) -> std::io::Result<()> {
    fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    for (m, n) in [(128usize, 128usize), (512, 512), (2048, 512), (2048, 2048)] {
        for id in 0..2 {
            let job = hash(format!("pearl-batch-case-{id}").as_bytes());
            let raw_b = vec![0u8; n * K];
            let root_b = keyed(&job, &raw_b);
            for candidate in 0..4usize {
                let mut raw_a = vec![0u8; m * K];
                // Four distinct commitments: zero A, then one +1 at 0,1,2.
                let raw_position = if candidate == 0 {
                    None
                } else {
                    Some(candidate - 1)
                };
                if let Some(s) = raw_position {
                    raw_a[s] = 1;
                }
                let root_a = keyed(&job, &raw_a);
                let (sb, sa) = seeds_from_roots(&job, &root_a, &root_b, m as u32, n as u32);
                let p = sparse_pairs(&sa, &label(b"A_tensor"));
                let q = sparse_pairs(&sb, &label(b"B_tensor"));
                let u = uniform_rows(&sa, &label(b"A_tensor"), m);
                let v = uniform_rows(&sb, &label(b"B_tensor"), n);
                let name = format!("m{m}-n{n}-id{id}-candidate{candidate}");
                let dir = out.join(&name);
                fs::create_dir_all(&dir)?;
                for (file, rows) in [("u.i8", &u), ("v.i8", &v)] {
                    let bytes: Vec<u8> = rows.iter().flatten().map(|&v| v as u8).collect();
                    fs::write(dir.join(file), bytes)?;
                }
                let mut f = fs::File::create(dir.join("pairs.u32"))?;
                for pairs in [&p, &q] {
                    for &(a, b) in pairs {
                        f.write_all(&(a as u32).to_le_bytes())?;
                        f.write_all(&(b as u32).to_le_bytes())?;
                    }
                }
                cases.push(format!(r#"{{"name":"{}","m":{},"n":{},"id":{},"candidate":{},"raw_a_position":{},"job_key":"{}","root_a":"{}","root_b":"{}","seed_a":"{}","seed_b":"{}"}}"#,
                    name,m,n,id,candidate,raw_position.map_or("null".into(),|s|s.to_string()),hex(&job),hex(&root_a),hex(&root_b),hex(&sa),hex(&sb)));
            }
        }
    }
    fs::write(
        out.join("manifest.json"),
        format!(
            r#"{{"schema":1,"k":{},"r":{},"cases":[{}]}}"#,
            K,
            R,
            cases.join(",")
        ),
    )?;
    eprintln!(
        "Exported {} both-sided fixtures to {}",
        cases.len(),
        out.display()
    );
    Ok(())
}
