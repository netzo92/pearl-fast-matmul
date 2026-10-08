//! Local CPU conformance experiment. This file is not part of upstream Pearl.
use anyhow::Result;
use pearl_blake3::{MerkleTree, pad_to_chunk_boundary};
use serde_json::json;
use std::{env, fs, path::Path};
use zk_pow::{
    api::{
        proof::{
            IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern,
            PublicProofParams, SeedDerivation,
        },
        proof_utils::{CompiledPublicParams, compute_jackpot_hash},
        verify::verify_plain_proof,
    },
    circuit::{chip::compute_jackpot, pearl_noise::compute_noise},
    ffi::plain_proof::{MatrixMerkleProof, PlainProof},
};

const M: usize = 8;
const N: usize = 32;
const K: usize = 2048;
const R: usize = 128;
const UPSTREAM: &str = "2f8b770cac8f8b74be05c1fc51c2baeb76ce0701";
type Matrix = Vec<Vec<i32>>;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}
fn keyed(data: &[u8], key: &[u8; 32]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}
fn label(side: u8) -> [u8; 32] {
    let mut l = [0; 32];
    l[..8].copy_from_slice(if side == b'A' {
        b"A_tensor"
    } else {
        b"B_tensor"
    });
    l
}
fn random_hash(index: usize, side: u8, seed: &[u8; 32], slot: usize) -> [u8; 32] {
    let mut msg = [0; 64];
    msg[slot * 4..slot * 4 + 4].copy_from_slice(&((index + 1) as i32).to_le_bytes());
    msg[32..].copy_from_slice(&label(side));
    keyed(&msg, seed)
}
fn uniform(rows: usize, side: u8, seed: &[u8; 32]) -> Matrix {
    (0..rows)
        .map(|i| {
            let mut row = Vec::with_capacity(R);
            for block in i * R / 32..(i + 1) * R / 32 {
                row.extend(random_hash(block, side, seed, 0).map(|b| (b & 63) as i32 - 32));
            }
            row
        })
        .collect()
}
fn pairs(side: u8, seed: &[u8; 32]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(K);
    for block in 0..K / 8 {
        let digest = random_hash(block, side, seed, 1);
        for bytes in digest.chunks_exact(4) {
            let word = u32::from_le_bytes(bytes.try_into().unwrap());
            let p = word as usize & (R - 1);
            let q = p ^ (1 + (((R - 1) as u64 * word as u64) >> 32) as usize);
            out.push((p, q));
        }
    }
    out
}
fn root(parent: &mut [usize], mut p: usize) -> usize {
    while parent[p] != p {
        parent[p] = parent[parent[p]];
        p = parent[p];
    }
    p
}
fn grouped(a: &Matrix, v: &Matrix, pairs: &[(usize, usize)]) -> (Vec<Vec<i32>>, usize) {
    let mut acc = vec![0; M * N];
    let mut trace = Vec::new();
    let mut total_d = 0;
    for start in (0..K).step_by(R) {
        let edges = &pairs[start..start + R];
        let mut parent: Vec<usize> = (0..R).collect();
        for &(p, q) in edges {
            let x = root(&mut parent, p);
            let y = root(&mut parent, q);
            parent[x.max(y)] = x.min(y);
        }
        let basis: Vec<(usize, usize)> = (0..R)
            .filter_map(|t| {
                let anchor = root(&mut parent, t);
                (t != anchor).then_some((t, anchor))
            })
            .collect();
        total_d += basis.len();
        for i in 0..M {
            let mut g = [0i32; R];
            for (s, &(p, q)) in edges.iter().enumerate() {
                g[p] += a[i][start + s];
                g[q] -= a[i][start + s];
            }
            for j in 0..N {
                for &(t, anchor) in &basis {
                    acc[i * N + j] += g[t] * (v[j][t] - v[j][anchor]);
                }
            }
        }
        trace.push(acc.clone());
    }
    (trace, total_d)
}
fn direct_prefixes(a: &Matrix, bt: &Matrix) -> Vec<Vec<i32>> {
    let mut acc = vec![0; M * N];
    let mut trace = Vec::new();
    for start in (0..K).step_by(R) {
        for i in 0..M {
            for j in 0..N {
                for s in start..start + R {
                    acc[i * N + j] += a[i][s] * bt[j][s];
                }
            }
        }
        trace.push(acc.clone());
    }
    trace
}
fn state(prefixes: &[Vec<i32>]) -> [u32; 16] {
    let mut state = [0u32; 16];
    for (chunk, prefix) in prefixes.iter().enumerate() {
        let x = prefix.iter().fold(0u32, |x, &v| x ^ v as u32);
        state[chunk % 16] = state[chunk % 16].rotate_left(13) ^ x;
    }
    state
}
fn make_merkle(raw: &[Vec<i8>], key: [u8; 32]) -> MatrixMerkleProof {
    let rows: Vec<usize> = (0..raw.len()).collect();
    let bytes: Vec<u8> = raw.iter().flatten().map(|&x| x as u8).collect();
    let tree = MerkleTree::new(&pad_to_chunk_boundary(&bytes), key);
    let leaves = MerkleTree::compute_leaf_indices_from_rows(&rows, (raw.len(), K));
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaves),
        row_indices: rows,
    }
}
fn run_case(nonzero_a: bool, out: &Path) -> Result<serde_json::Value> {
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [0x11; 32],
        merkle_root: [0x22; 32],
        timestamp: 1_800_000_000 + u32::from(nonzero_a),
        nbits: 0x207fffff,
    };
    let config = MiningConfiguration {
        common_dim: K as u32,
        rank: R as u16,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&(0..M as u32).collect::<Vec<_>>())?,
        cols_pattern: PeriodicPattern::from_list(&(0..N as u32).collect::<Vec<_>>())?,
        moe: None,
    };
    let context = PublicProofParams::new_dummy(
        header,
        SeedDerivation::Salted,
        config,
        M as u32,
        N as u32,
        0,
        0,
    );
    let job_key = context.job_key();
    let raw_a: Vec<Vec<i8>> = (0..M)
        .map(|i| {
            (0..K)
                .map(|s| {
                    if nonzero_a {
                        ((i * 13 + s * 7) % 128) as i8 - 64
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();
    let raw_bt = vec![vec![0i8; K]; N];
    let plain = PlainProof {
        m: M,
        n: N,
        k: K,
        noise_rank: R,
        a: make_merkle(&raw_a, job_key),
        bt: make_merkle(&raw_bt, job_key),
        moe: None,
    };
    let (private, public) = plain.parse_proof(header, SeedDerivation::Salted)?;
    public.sanity_check()?;
    public.sanity_check_private_params(&private)?;
    assert_eq!(public.job_key(), job_key);
    let compiled = CompiledPublicParams::from(&public);
    let official_noise = compute_noise(&compiled);
    let (b_seed, a_seed) = compiled.commitment_hash;
    let u = uniform(M, b'A', &a_seed);
    let v = uniform(N, b'B', &b_seed);
    let p = pairs(b'A', &a_seed);
    let q = pairs(b'B', &b_seed);
    let a: Matrix = (0..M)
        .map(|i| {
            (0..K)
                .map(|s| {
                    let local_noise = u[i][p[s].0] - u[i][p[s].1];
                    assert_eq!(local_noise, official_noise.a[i][s] as i32);
                    raw_a[i][s] as i32 + local_noise
                })
                .collect()
        })
        .collect();
    let bt: Matrix = (0..N)
        .map(|j| {
            (0..K)
                .map(|s| {
                    let local_noise = v[j][q[s].0] - v[j][q[s].1];
                    assert_eq!(local_noise, official_noise.b[j][s] as i32);
                    local_noise
                })
                .collect()
        })
        .collect();
    let (prefixes, total_d) = grouped(&a, &v, &q);
    assert_eq!(
        prefixes,
        direct_prefixes(&a, &bt),
        "every full prefix matrix"
    );
    let grouped_state = state(&prefixes);
    let official_state = compute_jackpot(&compiled, &private.s_a, &private.s_b, &official_noise);
    assert_eq!(grouped_state, official_state, "official jackpot state");
    let msg: Vec<u8> = grouped_state.iter().flat_map(|x| x.to_le_bytes()).collect();
    let candidate_hash = keyed(&msg, &a_seed);
    let official_hash = compute_jackpot_hash(&official_state, compiled.a_noise_seed());
    assert_eq!(candidate_hash, official_hash, "official jackpot hash");
    verify_plain_proof(&header, &plain, None, SeedDerivation::Salted)?;
    let mut corrupt = plain.clone();
    corrupt.a.proof.leaf_data[0][0] ^= 1;
    let corrupt_error = verify_plain_proof(&header, &corrupt, None, SeedDerivation::Salted)
        .expect_err("changed raw byte must fail");
    let zero_target_error = verify_plain_proof(&header, &plain, Some(0), SeedDerivation::Salted)
        .expect_err("zero target must fail");
    let name = if nonzero_a { "nonzero_a" } else { "zero_a" };
    let proof_bytes = bincode::serialize(&plain)?;
    let proof_file = format!("plain_{name}.bin");
    fs::write(out.join(&proof_file), &proof_bytes)?;
    let prefix_bytes: Vec<u8> = prefixes
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    Ok(json!({
        "case": name, "header": {"version": header.version, "prev_block": hex(&header.prev_block), "merkle_root": hex(&header.merkle_root), "timestamp": header.timestamp, "nbits": "0x207fffff"},
        "job_key": hex(&job_key), "root_a": hex(&plain.a.proof.root), "root_b": hex(&plain.bt.proof.root),
        "a_seed": hex(&a_seed), "b_seed": hex(&b_seed), "jackpot_state": grouped_state,
        "jackpot_hash": hex(&candidate_hash), "prefix_digest": hex(blake3::hash(&prefix_bytes).as_bytes()),
        "noise_entries_compared": (M + N) * K,
        "full_prefix_matrices_compared": K / R, "prefix_entries_compared": M * N * K / R,
        "direct_scalar_multiplications": M * N * K, "grouped_scalar_multiplications": M * N * total_d,
        "official_state_equal": true, "official_hash_equal": true, "official_plain_verifier_accepted": true,
        "corrupted_raw_byte_rejected": true, "corrupt_error": corrupt_error.to_string(),
        "zero_target_rejected": true, "zero_target_error": zero_target_error.to_string(),
        "plain_proof_file": proof_file, "plain_proof_bytes": proof_bytes.len(),
    }))
}
fn main() -> Result<()> {
    let dest = env::args()
        .nth(1)
        .unwrap_or_else(|| "/private/tmp/pearl-grouping-conformance".to_string());
    let out = Path::new(&dest);
    fs::create_dir_all(out)?;
    let cases = [run_case(false, out)?, run_case(true, out)?];
    let report = json!({
        "upstream_commit": UPSTREAM, "protocol": "INT V3 salted seed derivation", "m": M, "n": N, "h": M, "w": N, "r": R, "k": K,
        "scope": "CPU correctness and plain proof acceptance. Easy threshold saturates adjusted target. No ZK certificate, network acceptance, or speed claim.",
        "cases": cases,
    });
    let text = serde_json::to_string_pretty(&report)?;
    fs::write(out.join("report.json"), &text)?;
    println!("{text}");
    Ok(())
}
