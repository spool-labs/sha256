//! First digest word of short messages: `hash_short_heads` vs full digests
//!
//! A proof-of-work or prefix search hashes `prefix || counter` (one block) and
//! only compares the leading bits, so it never needs the other 28 digest bytes.
//! `hash_short_heads` skips their serialisation where the active backend has a
//! dedicated kernel (AVX2); elsewhere it is `hash_many_prefixed` plus a
//! truncation, so the two rows should match. Run with
//! `cargo bench --bench short_heads`.

use {
    std::time::Instant,
    tape_sha256::{hash_many_prefixed, hash_short_heads},
};

const PREFIX: &[u8] = b"tape-pow:0123456789abcdef:";
const BATCH: usize = 64;

/// Times `f`, reporting the best of several rounds; see merkle_nodes
fn bench<F: FnMut()>(name: &str, hashes: usize, mut f: F) -> f64 {
    const ROUNDS: usize = 15;
    const ITERS: usize = 2000;
    for _ in 0..100 {
        f();
    }
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            f();
        }
        let s = t0.elapsed().as_secs_f64() / ITERS as f64;
        best = best.min(s);
        worst = worst.max(s);
    }
    let spread = 100.0 * (worst - best) / best;
    println!(
        "{name:<32}{:>9.2} us{:>12.2} Mhash/s   (spread {spread:>4.1}%)",
        best * 1e6,
        hashes as f64 / best / 1e6
    );
    best
}

fn main() {
    // Decimal counters of 10 or 11 digits after a 26-byte prefix: one block each.
    let counters: Vec<String> = (0..BATCH as u64)
        .map(|i| (1_000_000_007u64 * (i + 1)).to_string())
        .collect();
    let bodies: Vec<&[u8]> = counters.iter().map(|c| c.as_bytes()).collect();
    let lens = bodies.iter().map(|b| PREFIX.len() + b.len());
    let (min, max) = (lens.clone().min().unwrap(), lens.max().unwrap());
    assert!(max <= 55, "every message must fit one block");
    println!("{BATCH} messages of {min}-{max} bytes, one block each");
    println!("active backend: {}", tape_sha256::backend());
    println!();

    let mut full = vec![[0u8; 32]; BATCH];
    let mut heads = vec![0u32; BATCH];
    let t_full = bench("hash_many_prefixed (full digest)", BATCH, || {
        hash_many_prefixed(PREFIX, &bodies, &mut full);
        std::hint::black_box(&full);
    });
    let t_heads = bench("hash_short_heads", BATCH, || {
        hash_short_heads(PREFIX, &bodies, &mut heads);
        std::hint::black_box(&heads);
    });

    for (h, d) in heads.iter().zip(&full) {
        assert_eq!(*h, u32::from_be_bytes([d[0], d[1], d[2], d[3]]));
    }
    println!();
    println!("heads vs full digest: {:.2}x", t_full / t_heads);
}
