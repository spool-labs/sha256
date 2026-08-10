//! Shred verification: Merkle proof work vs the signature check beside it
//!
//! the Solana validator's shred verification does three
//! things per received shred: rebuild the Merkle root from the shred's leaf
//! and proof, look up `(signature, pubkey, root)` in an LRU cache, and verify
//! the signature only on a miss.
//!
//! All 64 shreds of an erasure batch share one signature and one root, so the
//! cache means roughly one signature check per batch. The Merkle rebuild runs
//! *before* the cache lookup, so it is paid on every shred. This measures
//! which of the two actually dominates, and what batching the Merkle side
//! would buy.
//!
//! Each rebuild is `get_proof_size(64)` = 6 sequential joins, and the 64
//! shreds are independent, so a level of the walk is 64 independent hashes.
//!
//! Run with `cargo bench --bench shred_verify`.

use {
    ed25519_dalek::{Signer, SigningKey, Verifier},
    sha2::{Digest, Sha256},
    std::time::Instant,
    tape_sha256::hash_many_prefixed,
};

const PREFIX_NODE: &[u8] = b"\x01SOLANA_MERKLE_SHREDS_NODE";
const PROOF_ENTRY: usize = 20;
const SHREDS: usize = 64;
const PROOF_DEPTH: usize = 6; // get_proof_size(64)

fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
    const ROUNDS: usize = 15;
    const ITERS: usize = 50;
    for _ in 0..10 {
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
    println!("{name:<38}{:>9.2} us   (spread {spread:>4.1}%)", best * 1e6);
    best
}

fn join(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut d = Sha256::new();
    d.update(PREFIX_NODE);
    d.update(&a[..PROOF_ENTRY]);
    d.update(&b[..PROOF_ENTRY]);
    d.finalize().into()
}

fn main() {
    let leaves: Vec<[u8; 32]> = (0..SHREDS)
        .map(|i| {
            let mut h = [0u8; 32];
            h[0] = i as u8;
            h
        })
        .collect();
    // Each shred carries PROOF_DEPTH sibling entries.
    let proofs: Vec<Vec<[u8; 32]>> = (0..SHREDS)
        .map(|i| {
            (0..PROOF_DEPTH)
                .map(|j| {
                    let mut h = [0u8; 32];
                    h[0] = (i * 7 + j) as u8;
                    h
                })
                .collect()
        })
        .collect();

    println!("one erasure batch: {SHREDS} shreds, proof depth {PROOF_DEPTH}");
    println!("active backend: {}", tape_sha256::backend());
    println!();

    // What validators do now: per shred, walk the proof serially.
    let serial = bench("merkle rebuild, serial (validator)", || {
        let mut roots = [[0u8; 32]; SHREDS];
        for (i, leaf) in leaves.iter().enumerate() {
            let mut node = *leaf;
            for entry in &proofs[i] {
                node = join(&node, entry);
            }
            roots[i] = node;
        }
        std::hint::black_box(&roots);
    });

    // Same work, loop order swapped: one level at a time across all shreds, so
    // each level is 64 independent hashes filling the SIMD lanes.
    let mut nodes = [[0u8; 32]; SHREDS];
    let mut bodies = vec![[0u8; 2 * PROOF_ENTRY]; SHREDS];
    let batched = bench("merkle rebuild, batched by level", || {
        nodes.copy_from_slice(&leaves);
        for level in 0..PROOF_DEPTH {
            for (b, (n, p)) in bodies.iter_mut().zip(nodes.iter().zip(&proofs)) {
                b[..PROOF_ENTRY].copy_from_slice(&n[..PROOF_ENTRY]);
                b[PROOF_ENTRY..].copy_from_slice(&p[level][..PROOF_ENTRY]);
            }
            let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
            hash_many_prefixed(PREFIX_NODE, &refs, &mut nodes);
        }
        std::hint::black_box(&nodes);
    });

    // Correctness: swapping the loop order must not change the roots.
    {
        let mut want = [[0u8; 32]; SHREDS];
        for (i, leaf) in leaves.iter().enumerate() {
            let mut node = *leaf;
            for entry in &proofs[i] {
                node = join(&node, entry);
            }
            want[i] = node;
        }
        assert_eq!(nodes, want, "level-batched roots must equal serial walk");
        println!("(verified: batched roots == serial walk for all {SHREDS})");
    }

    // The signature check the cache lets them do roughly once per batch.
    let sk = SigningKey::from_bytes(&[7u8; 32]);
    let vk = sk.verifying_key();
    let msg = [0x5au8; 32];
    let sig = sk.sign(&msg);
    let one_sig = bench("ed25519 verify (1 per batch)", || {
        std::hint::black_box(vk.verify(&msg, &sig).is_ok());
    });

    // Same batched work split across two caller-owned threads. Run under
    // `taskset -c a,b` with a and b siblings of one physical core to see
    // whether the merkle rebuild has the same spare-SMT headroom the leaf
    // kernel does. Throughput measure: no per-iteration sync.
    let half = SHREDS / 2;
    let smt = {
        const ROUNDS: usize = 15;
        const ITERS: usize = 200;
        let mut best = f64::INFINITY;
        for r in 0..=ROUNDS {
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for part in 0..2 {
                    let leaves = &leaves;
                    let proofs = &proofs;
                    sc.spawn(move || {
                        let lo = part * half;
                        let mut nodes = vec![[0u8; 32]; half];
                        let mut bodies = vec![[0u8; 2 * PROOF_ENTRY]; half];
                        for _ in 0..ITERS {
                            nodes.copy_from_slice(&leaves[lo..lo + half]);
                            for level in 0..PROOF_DEPTH {
                                for (b, (n, p)) in bodies
                                    .iter_mut()
                                    .zip(nodes.iter().zip(&proofs[lo..lo + half]))
                                {
                                    b[..PROOF_ENTRY].copy_from_slice(&n[..PROOF_ENTRY]);
                                    b[PROOF_ENTRY..].copy_from_slice(&p[level][..PROOF_ENTRY]);
                                }
                                let refs: Vec<&[u8]> =
                                    bodies.iter().map(|b| b.as_slice()).collect();
                                hash_many_prefixed(PREFIX_NODE, &refs, &mut nodes);
                            }
                            std::hint::black_box(&nodes);
                        }
                    });
                }
            });
            if r == 0 {
                continue;
            }
            best = best.min(t0.elapsed().as_secs_f64() / ITERS as f64);
        }
        println!(
            "merkle rebuild, SMT pair (2 threads)  {:>9.2} us",
            best * 1e6
        );
        best
    };

    println!();
    println!(
        "SMT pair vs single thread (batched):  {:.2}x",
        batched / smt
    );
    println!(
        "merkle is {:.1}x the cost of the batch's signature check",
        serial / one_sig
    );
    println!(
        "batching the merkle side: {:.2}x  ({:.1} -> {:.1} us per batch)",
        serial / batched,
        serial * 1e6,
        batched * 1e6
    );
    println!(
        "per-batch verify cost: {:.1} us now -> {:.1} us batched",
        (serial + one_sig) * 1e6,
        (batched + one_sig) * 1e6
    );
}
