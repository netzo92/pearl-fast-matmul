//! Narrow V3 seed/noise/tile-state reproduction from Pearl's public Rust sources.
//! This does not construct or verify a network certificate.
mod batched_grouping;
mod grouped_noise;

const R: usize = 128;
const K: usize = 2048;
const M: usize = 8;
const N: usize = 8;
const H: usize = 4;
const W: usize = 8;

type Key = [u8; 32];

fn hash(input: &[u8]) -> Key {
    *blake3::hash(input).as_bytes()
}
fn keyed(key: &Key, input: &[u8]) -> Key {
    *blake3::keyed_hash(key, input).as_bytes()
}
fn bind_root(root: &Key, dim: u32, label: &[u8]) -> Key {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(root);
    msg[32..36].copy_from_slice(&dim.to_le_bytes());
    keyed(&hash(label), &msg)
}
fn seeds_from_roots(job_key: &Key, raw_a: &Key, raw_b: &Key, m: u32, n: u32) -> (Key, Key) {
    let bound_a = bind_root(raw_a, m, b"pearl/cert-v3/noise-seed/A");
    let bound_b = bind_root(raw_b, n, b"pearl/cert-v3/noise-seed/B");
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(job_key);
    msg[32..].copy_from_slice(&bound_b);
    let seed_b = hash(&msg);
    msg[..32].copy_from_slice(&seed_b);
    msg[32..].copy_from_slice(&bound_a);
    let seed_a = hash(&msg);
    (seed_b, seed_a)
}
fn pad_1024(mut bytes: Vec<u8>) -> Vec<u8> {
    let rem = bytes.len() % 1024;
    if rem != 0 {
        bytes.resize(bytes.len() + 1024 - rem, 0);
    }
    bytes
}
fn flatten(x: &[Vec<i8>]) -> Vec<u8> {
    pad_1024(x.iter().flatten().map(|&b| b as u8).collect())
}
fn seeds(job_key: &Key, a: &[Vec<i8>], bt: &[Vec<i8>]) -> (Key, Key, Key, Key) {
    let ha = keyed(job_key, &flatten(a));
    let hb = keyed(job_key, &flatten(bt));
    let (sb, sa) = seeds_from_roots(job_key, &ha, &hb, M as u32, N as u32);
    (ha, hb, sb, sa)
}
fn label(s: &[u8; 8]) -> Key {
    let mut out = [0; 32];
    out[..8].copy_from_slice(s);
    out
}
fn random_hash(index: usize, seed_label: &Key, key: &Key, prepend_slot: usize) -> Key {
    let mut msg = [0u8; 64];
    msg[prepend_slot * 4..prepend_slot * 4 + 4]
        .copy_from_slice(&((index + 1) as i32).to_le_bytes());
    msg[32..].copy_from_slice(seed_label);
    keyed(key, &msg)
}
fn uniform_rows(key: &Key, seed_label: &Key, count: usize) -> Vec<Vec<i8>> {
    (0..count)
        .map(|row| {
            (0..R)
                .map(|col| {
                    let idx = row * R + col;
                    let block = random_hash(idx / 32, seed_label, key, 0);
                    ((block[idx % 32] & 63) as i8) - 32
                })
                .collect()
        })
        .collect()
}
fn sparse_pairs(key: &Key, seed_label: &Key) -> Vec<(usize, usize)> {
    (0..K)
        .map(|s| {
            let block = random_hash(s / 8, seed_label, key, 1);
            let start = (s % 8) * 4;
            let word = u32::from_le_bytes(block[start..start + 4].try_into().unwrap());
            let first = (word & (R as u32 - 1)) as usize;
            let hi = (((R as u32 - 1) as u64 * word as u64) >> 32) as usize;
            (first, first ^ (1 + hi))
        })
        .collect()
}
fn noised(
    a: &[Vec<i8>],
    bt: &[Vec<i8>],
    seed_a: &Key,
    seed_b: &Key,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let la = label(b"A_tensor");
    let lb = label(b"B_tensor");
    let ua = uniform_rows(seed_a, &la, M);
    let vb = uniform_rows(seed_b, &lb, N);
    let pa = sparse_pairs(seed_a, &la);
    let pb = sparse_pairs(seed_b, &lb);
    let aa = (0..M)
        .map(|i| {
            (0..K)
                .map(|s| a[i][s] as i32 + ua[i][pa[s].0] as i32 - ua[i][pa[s].1] as i32)
                .collect()
        })
        .collect();
    let bb = (0..N)
        .map(|j| {
            (0..K)
                .map(|s| bt[j][s] as i32 + vb[j][pb[s].0] as i32 - vb[j][pb[s].1] as i32)
                .collect()
        })
        .collect();
    (aa, bb)
}
fn tile_state(aa: &[Vec<i32>], bb: &[Vec<i32>]) -> [u32; 16] {
    let mut acc = [[0i32; W]; H];
    let mut state = [0u32; 16];
    for chunk in 0..K / R {
        let mut x = 0u32;
        for i in 0..H {
            for j in 0..W {
                for s in chunk * R..(chunk + 1) * R {
                    acc[i][j] += aa[i][s] * bb[j][s];
                }
                x ^= acc[i][j] as u32;
            }
        }
        state[chunk % 16] = state[chunk % 16].rotate_left(13) ^ x;
    }
    state
}
fn jackpot_hash(state: &[u32; 16], seed_a: &Key) -> Key {
    let mut msg = [0u8; 64];
    for (i, x) in state.iter().enumerate() {
        msg[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    keyed(seed_a, &msg)
}
fn hex_short(key: &Key) -> String {
    key[..8].iter().map(|x| format!("{x:02x}")).collect()
}
fn main() {
    // Pearl's pinned V3 seed vector checks the salting and seed chain independently.
    let (pinned_b, pinned_a) = seeds_from_roots(&[0x11; 32], &[0xaa; 32], &[0xbb; 32], 192, 320);
    assert_eq!(
        pinned_b,
        [
            0x60, 0xed, 0x9b, 0x73, 0xc5, 0xa9, 0x59, 0x9b, 0x20, 0x0b, 0x6c, 0xd5, 0x63, 0xe7,
            0xf0, 0xd5, 0xd9, 0xa6, 0x7d, 0x24, 0x02, 0xd8, 0x5f, 0xd4, 0xef, 0x96, 0x6c, 0x58,
            0x00, 0x80, 0xd0, 0xe5
        ]
    );
    assert_eq!(
        pinned_a,
        [
            0x30, 0x17, 0x84, 0x16, 0x80, 0x05, 0xec, 0x83, 0x3a, 0xb0, 0xaa, 0x60, 0x00, 0x6f,
            0x7f, 0xe7, 0xfa, 0xaa, 0x95, 0x30, 0x7d, 0x8c, 0x1f, 0xc6, 0x81, 0x9b, 0x2f, 0xfd,
            0xd7, 0x17, 0xec, 0xcf
        ]
    );
    if std::env::args().any(|arg| arg == "--batch-grouping") {
        batched_grouping::run();
        return;
    }
    if std::env::args().any(|arg| arg == "--grouping") {
        grouped_noise::run();
        return;
    }
    let mut a = vec![vec![0i8; K]; M];
    let mut bt = vec![vec![0i8; K]; N];
    a[0][0] = 1;
    bt[0][0] = 1;
    let job_key = [0x11u8; 32];
    let (ha, hb, sb, sa) = seeds(&job_key, &a, &bt);
    let (aa, bb) = noised(&a, &bt, &sa, &sb);
    let state = tile_state(&aa, &bb);
    let jackpot = jackpot_hash(&state, &sa);
    let mut shifted_a = a.clone();
    let mut shifted_bt = bt.clone();
    for row in &mut shifted_a {
        row.swap(0, 1);
    }
    for col in &mut shifted_bt {
        col.swap(0, 1);
    }
    assert_eq!(
        a[0][0] as i32 * bt[0][0] as i32 + a[0][1] as i32 * bt[0][1] as i32,
        shifted_a[0][0] as i32 * shifted_bt[0][0] as i32
            + shifted_a[0][1] as i32 * shifted_bt[0][1] as i32
    );
    let (ha2, hb2, sb2, sa2) = seeds(&job_key, &shifted_a, &shifted_bt);
    let (aa2, bb2) = noised(&shifted_a, &shifted_bt, &sa2, &sb2);
    let state2 = tile_state(&aa2, &bb2);
    let jackpot2 = jackpot_hash(&state2, &sa2);
    let mut control_a = aa.clone();
    let mut control_b = bb.clone();
    for row in &mut control_a {
        row.swap(0, 1);
    }
    for col in &mut control_b {
        col.swap(0, 1);
    }
    assert_eq!(state, tile_state(&control_a, &control_b));
    assert_ne!(ha, ha2);
    assert_ne!(hb, hb2);
    assert_ne!(sa, sa2);
    assert_ne!(sb, sb2);
    println!("Pearl V3 pinned seed vector: pass");
    println!("fixed-noise inner permutation preserves state: true");
    println!(
        "raw A commitment: {} -> {}",
        hex_short(&ha),
        hex_short(&ha2)
    );
    println!(
        "raw B commitment: {} -> {}",
        hex_short(&hb),
        hex_short(&hb2)
    );
    println!("A noise seed: {} -> {}", hex_short(&sa), hex_short(&sa2));
    println!("B noise seed: {} -> {}", hex_short(&sb), hex_short(&sb2));
    println!("recommitted tile state equal: {}", state == state2);
    println!(
        "recommitted jackpot hash: {} -> {}",
        hex_short(&jackpot),
        hex_short(&jackpot2)
    );
    println!("recommitted jackpot hash equal: {}", jackpot == jackpot2);
    let mut repeated_states = 0usize;
    let mut repeated_hashes = 0usize;
    for dest in 1..=16 {
        let mut trial_a = a.clone();
        let mut trial_bt = bt.clone();
        for row in &mut trial_a {
            row.swap(0, dest);
        }
        for col in &mut trial_bt {
            col.swap(0, dest);
        }
        let (_, _, trial_sb, trial_sa) = seeds(&job_key, &trial_a, &trial_bt);
        let (trial_noised_a, trial_noised_b) = noised(&trial_a, &trial_bt, &trial_sa, &trial_sb);
        let trial_state = tile_state(&trial_noised_a, &trial_noised_b);
        repeated_states += usize::from(trial_state == state);
        repeated_hashes += usize::from(jackpot_hash(&trial_state, &trial_sa) == jackpot);
    }
    println!("within-chunk shifts tested: 16");
    println!("original tile-state repeats: {repeated_states}");
    println!("original jackpot-hash repeats: {repeated_hashes}");
}
