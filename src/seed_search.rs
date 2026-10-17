//! Commitment-bound near-zero B search. Arithmetic screening, not GPU timing.
use super::{
    grouped_noise::basis, hash, jackpot_hash, keyed, label, random_hash, seeds_from_roots,
    sparse_pairs, uniform_rows, Key, K, R,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const M: usize = 2048;
const N: usize = 512;
const JOBS: usize = 4;
const CHUNKS: usize = K / R;
const TILE_H: usize = 8;
const TILE_W: usize = 32;

fn hex(x: &[u8]) -> String {
    x.iter().map(|b| format!("{b:02x}")).collect()
}

// Exactly the existing sparse_pairs generator, sharing each digest's 8 words.
fn pairs_fast(seed: &Key) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(K);
    for block in 0..K / 8 {
        for bytes in random_hash(block, &label(b"B_tensor"), seed, 1).chunks_exact(4) {
            let word = u32::from_le_bytes(bytes.try_into().unwrap());
            let p = word as usize & (R - 1);
            let hi = (((R - 1) as u64 * word as u64) >> 32) as usize;
            out.push((p, p ^ (1 + hi)));
        }
    }
    out
}

#[derive(Clone)]
struct Candidate {
    trial: usize,
    position: Option<usize>,
    root: Key,
    seed: Key,
    dims: Vec<usize>,
    scalar: usize,
    padded: usize,
}
impl Candidate {
    fn json(&self) -> String {
        let position = self.position.map_or("null".into(), |p| p.to_string());
        format!(
            r#"{{"trial":{},"flat_b_position":{},"raw_b_value":{},"root_b":"{}","seed_b":"{}","chunk_dimensions":{:?},"scalar_inner_sum":{},"mma16_inner_sum":{}}}"#,
            self.trial,
            position,
            usize::from(self.position.is_some()),
            hex(&self.root),
            hex(&self.seed),
            self.dims,
            self.scalar,
            self.padded
        )
    }
}
fn score(
    trial: usize,
    position: Option<usize>,
    root: Key,
    seed: Key,
    pairs: &[(usize, usize)],
) -> Candidate {
    let dims: Vec<_> = pairs.chunks(R).map(|p| basis(p).len()).collect();
    Candidate {
        trial,
        position,
        root,
        seed,
        scalar: dims.iter().sum(),
        padded: dims.iter().map(|d| d.div_ceil(16) * 16).sum(),
        dims,
    }
}

// Check full 16-prefix traces for four fresh A commitments on two tile locations:
// one contains the B nonce, one does not. A and B are committed at full size.
fn check(job: &Key, selected: &Candidate, out: &Path, name: &str) -> std::io::Result<String> {
    let mut raw_b = vec![0u8; N * K];
    if let Some(p) = selected.position {
        raw_b[p] = 1;
    }
    assert_eq!(keyed(job, &raw_b), selected.root);
    let q = pairs_fast(&selected.seed);
    assert_eq!(q, sparse_pairs(&selected.seed, &label(b"B_tensor")));
    let b_col = selected.position.unwrap_or(0) / K;
    let starts = [
        b_col / TILE_W * TILE_W,
        ((b_col / TILE_W + 1) % (N / TILE_W)) * TILE_W,
    ];
    let v_all = uniform_rows(&selected.seed, &label(b"B_tensor"), N);
    let mut records = Vec::new();
    let mut previous_seed = None;
    let mut checks = 0usize;
    let mut correction_hits = 0usize;
    for nonce in 0..4 {
        let mut raw_a = vec![0u8; M * K];
        // Keep raw A*B zero while making all four commitments distinct.
        let avoid = selected.position.unwrap_or(0) % K;
        let a_s = (avoid + 1 + nonce) % K;
        raw_a[a_s] = 1;
        let root_a = keyed(job, &raw_a);
        let (sb, sa) = seeds_from_roots(job, &root_a, &selected.root, M as u32, N as u32);
        assert_eq!(sb, selected.seed);
        assert_ne!(previous_seed, Some(sa));
        previous_seed = Some(sa);
        let u = uniform_rows(&sa, &label(b"A_tensor"), TILE_H);
        let p = sparse_pairs(&sa, &label(b"A_tensor"));
        let aa: Vec<Vec<i32>> = (0..TILE_H)
            .map(|i| {
                (0..K)
                    .map(|s| raw_a[i * K + s] as i32 + u[i][p[s].0] as i32 - u[i][p[s].1] as i32)
                    .collect()
            })
            .collect();
        let mut hashes = Vec::new();
        for &col_start in &starts {
            let mut direct_acc = vec![0i32; TILE_H * TILE_W];
            let mut grouped_acc = direct_acc.clone();
            let mut ds = [0u32; 16];
            let mut gs = ds;
            let mut prefix_hasher = blake3::Hasher::new();
            let mut correction_omitted = false;
            for c in 0..CHUNKS {
                let start = c * R;
                let edges = &q[start..start + R];
                let active = basis(edges);
                for i in 0..TILE_H {
                    let mut g = [0i32; R];
                    for (s, &(a, b)) in edges.iter().enumerate() {
                        g[a] += aa[i][start + s];
                        g[b] -= aa[i][start + s];
                    }
                    for j in 0..TILE_W {
                        let v = &v_all[col_start + j];
                        for s in start..start + R {
                            let bv = raw_b[(col_start + j) * K + s] as i32 + v[q[s].0] as i32
                                - v[q[s].1] as i32;
                            direct_acc[i * TILE_W + j] += aa[i][s] * bv;
                        }
                        for &(t, a) in &active {
                            grouped_acc[i * TILE_W + j] += g[t] * (v[t] as i32 - v[a] as i32);
                        }
                    }
                    if let Some(pos) = selected.position {
                        let col = pos / K;
                        let s = pos % K;
                        if col >= col_start && col < col_start + TILE_W && s / R == c {
                            let delta = aa[i][s];
                            // Negative control: omitting a nonzero correction differs.
                            correction_omitted |= delta != 0;
                            grouped_acc[i * TILE_W + col - col_start] += delta;
                        }
                    }
                }
                assert_eq!(
                    grouped_acc, direct_acc,
                    "every exact prefix including raw-B correction"
                );
                if let Some(pos) = selected.position {
                    let col = pos / K;
                    let s = pos % K;
                    if col >= col_start && col < col_start + TILE_W && s / R == c {
                        let mut missing_correction = grouped_acc.clone();
                        for i in 0..TILE_H {
                            missing_correction[i * TILE_W + col - col_start] -= aa[i][s];
                        }
                        assert_ne!(
                            missing_correction, direct_acc,
                            "negative control: omitting raw B must fail"
                        );
                    }
                }
                let dx = direct_acc.iter().fold(0u32, |x, &v| x ^ v as u32);
                let gx = grouped_acc.iter().fold(0u32, |x, &v| x ^ v as u32);
                ds[c % 16] = ds[c % 16].rotate_left(13) ^ dx;
                gs[c % 16] = gs[c % 16].rotate_left(13) ^ gx;
                for &x in &direct_acc {
                    prefix_hasher.update(&x.to_le_bytes());
                }
                checks += direct_acc.len();
            }
            assert_eq!(gs, ds);
            let dh = jackpot_hash(&ds, &sa);
            assert_eq!(dh, jackpot_hash(&gs, &sa));
            if selected.position.is_some() && col_start == starts[0] {
                assert!(correction_omitted, "nonzero-B correction must be exercised");
                correction_hits += 1;
            }
            hashes.push(format!(
                r#"{{"column_start":{},"prefix_digest":"{}","state":{:?},"hash":"{}"}}"#,
                col_start,
                prefix_hasher.finalize().to_hex(),
                ds,
                hex(&dh)
            ));
        }
        records.push(format!(r#"{{"candidate":{},"a_nonzero_flat_position":{},"root_a":"{}","seed_a":"{}","tiles":[{}]}}"#,
            nonce, a_s, hex(&root_a), hex(&sa), hashes.join(",")));
    }
    let text = format!(
        r#"{{"job_key":"{}","m":{},"n":{},"k":{},"r":{},"h":{},"w":{},"selected":{},"prefix_entries_checked":{},"tile_states_and_hashes_checked":8,"correction_negative_controls":{},"raw_product_zero":true,"cases":[{}]}}"#,
        hex(job),
        M,
        N,
        K,
        R,
        TILE_H,
        TILE_W,
        selected.json(),
        checks,
        correction_hits,
        records.join(",")
    );
    fs::write(out.join(format!("{name}.json")), &text)?;
    Ok(format!(
        r#"{{"file":"{name}.json","blake3":"{}","prefix_entries_checked":{},"tile_states_and_hashes_checked":8,"correction_negative_controls":{}}}"#,
        blake3::hash(text.as_bytes()).to_hex(),
        checks,
        correction_hits
    ))
}

pub fn run(out: &Path, trials: usize) -> std::io::Result<()> {
    assert!(
        (1..=N * K).contains(&trials),
        "one-entry candidates must have unique positions"
    );
    fs::create_dir_all(out)?;
    let wall = Instant::now();
    let mut jobs = Vec::new();
    for job_id in 0..JOBS {
        let job = hash(format!("pearl-batch-case-{job_id}").as_bytes());
        let mut raw_b = vec![0u8; N * K];
        let zero_root = keyed(&job, &raw_b);
        let (zero_seed, _) = seeds_from_roots(&job, &[0; 32], &zero_root, M as u32, N as u32);
        let zero_pairs = pairs_fast(&zero_seed);
        assert_eq!(zero_pairs, sparse_pairs(&zero_seed, &label(b"B_tensor")));
        let baseline = score(0, None, zero_root, zero_seed, &zero_pairs);
        let mut best_padded: Option<Candidate> = None;
        let mut best_scalar: Option<Candidate> = None;
        let mut commitment_s = 0.0;
        let mut seeds_s = 0.0;
        let mut graphs_s = 0.0;
        let mut history = Vec::new();
        let mut csv = BufWriter::new(File::create(out.join(format!("job-{job_id}-trials.csv")))?);
        writeln!(
            csv,
            "trial,flat_b_position,root_b,seed_b,scalar_inner_sum,mma16_inner_sum,chunk_dimensions"
        )?;
        for trial in 0..trials {
            // Odd multiplier permutes all 2^20 positions, without duplicate B inputs.
            let position = trial * 1_000_003 % (N * K);
            raw_b[position] = 1;
            let t = Instant::now();
            let root = keyed(&job, &raw_b);
            commitment_s += t.elapsed().as_secs_f64();
            raw_b[position] = 0;
            let t = Instant::now();
            let (seed, _) = seeds_from_roots(&job, &[0; 32], &root, M as u32, N as u32);
            let pairs = pairs_fast(&seed);
            seeds_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let candidate = score(trial, Some(position), root, seed, &pairs);
            graphs_s += t.elapsed().as_secs_f64();
            let dimensions = candidate
                .dims
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(";");
            writeln!(
                csv,
                "{trial},{position},{},{},{},{},{dimensions}",
                hex(&root),
                hex(&seed),
                candidate.scalar,
                candidate.padded
            )?;
            if best_padded.as_ref().is_none_or(|b| {
                (candidate.padded, candidate.scalar, trial) < (b.padded, b.scalar, b.trial)
            }) {
                best_padded = Some(candidate.clone());
            }
            if best_scalar.as_ref().is_none_or(|b| {
                (candidate.scalar, candidate.padded, trial) < (b.scalar, b.padded, b.trial)
            }) {
                best_scalar = Some(candidate);
            }
            if [
                1, 16, 256, 1024, 4096, 8192, 32768, 65536, 262144, 1048576, trials,
            ]
            .contains(&(trial + 1))
            {
                let b = best_padded.as_ref().unwrap();
                let s = best_scalar.as_ref().unwrap();
                history.push(format!(
                    r#"{{"trials":{},"best_padded":{},"best_scalar":{},"search_cpu_seconds":{}}}"#,
                    trial + 1,
                    b.json(),
                    s.json(),
                    commitment_s + seeds_s + graphs_s
                ));
                eprintln!("job {job_id}, {} candidates: MMA16 {} -> {}; raw inner {} -> {}; search CPU {:.3}s",
                    trial + 1, baseline.padded, b.padded, baseline.scalar, s.scalar, commitment_s + seeds_s + graphs_s);
            }
        }
        csv.flush()?;
        let b = best_padded.unwrap();
        let s = best_scalar.unwrap();
        let baseline_check = check(
            &job,
            &baseline,
            out,
            &format!("job-{job_id}-baseline-check"),
        )?;
        let padded_check = check(&job, &b, out, &format!("job-{job_id}-padded-check"))?;
        let scalar_check = check(&job, &s, out, &format!("job-{job_id}-scalar-check"))?;
        jobs.push(format!(r#"{{"job_id":{},"job_key":"{}","trials":{},"baseline":{},"best_padded":{},"best_scalar":{},"commitment_cpu_seconds":{},"seed_and_pairs_cpu_seconds":{},"graph_cpu_seconds":{},"search_cpu_seconds":{},"history":[{}],"checks":[{},{},{}]}}"#,
            job_id, hex(&job), trials, baseline.json(), b.json(), s.json(), commitment_s, seeds_s, graphs_s,
            commitment_s + seeds_s + graphs_s, history.join(","), baseline_check, padded_check, scalar_check));
    }
    let text = format!(
        r#"{{"schema":1,"scope":"Local commitment-bound arithmetic screening. No new GPU timings, official verifier acceptance or network submission.","m":{},"n":{},"k":{},"r":{},"jobs":[{}],"wall_seconds":{}}}"#,
        M,
        N,
        K,
        R,
        jobs.join(","),
        wall.elapsed().as_secs_f64()
    );
    fs::write(out.join("screen.json"), text)?;
    eprintln!("Report: {}", out.join("screen.json").display());
    Ok(())
}
