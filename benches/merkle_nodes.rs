//! Merkle interior nodes: the validator's serial join vs batching them
//!
//! Validators hash shred-tree leaves in a batch (that is what this crate is
//! integrated for) but joins interior nodes one at a time:
//!
//!   join_nodes(a, b) = hashv(&[MERKLE_HASH_PREFIX_NODE, a[..20], b[..20]])
//!
//! (ledger/src/shred/merkle_tree.rs). Proof entries are 20 bytes, so every
//! interior hash is 26 + 20 + 20 = 66 bytes, two blocks, and identical in
//! shape across the whole tree. A level of the tree is a set of independent
//! equal-length messages, which is exactly what multi-buffer wants.
//!
//! Under Alpenglow this matters more, not less: the block id becomes a
//! "double merkle root", a second tree built over the per-FEC-set roots.
//!
//! Run with `cargo bench --bench merkle_nodes`.

use {
    sha2::{Digest, Sha256},
    std::time::Instant,
    tape_sha256::{hash_many_prefixed, hash_pairs},
};

const PREFIX_NODE: &[u8] = b"\x01SOLANA_MERKLE_SHREDS_NODE";
const PROOF_ENTRY: usize = 20;

/// One 32:32 erasure batch is 64 shreds, so 63 interior joins per FEC set.
const LEAVES: usize = 64;
/// FEC sets in a decent-sized block, so the whole block's interior work.
const FEC_SETS: usize = 32;

/// Times `f`, reporting the best of several rounds.
///
/// Each round runs ITERS calls inside one timed window. A single ~80 us call
/// per round left the batched rows swinging 45-110% while the slower serial
/// row sat at 4%, which is scheduler jitter on a short window rather than any
/// property of the code being measured.
fn bench<F: FnMut()>(name: &str, hashes: usize, mut f: F) -> f64 {
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
    println!(
        "{name:<32}{:>9.1} us{:>12.2} Mhash/s   (spread {spread:>4.1}%)",
        best * 1e6,
        hashes as f64 / best / 1e6
    );
    best
}

/// the validator's serial node join, against the independent sha2 crate.
fn join_serial(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut d = Sha256::new();
    d.update(PREFIX_NODE);
    d.update(&a[..PROOF_ENTRY]);
    d.update(&b[..PROOF_ENTRY]);
    d.finalize().into()
}

fn main() {
    // One block's worth of interior nodes: each FEC set tree has LEAVES-1
    // interior joins, and a full binary level halves each time.
    let joins_per_set = LEAVES - 1;
    let total = joins_per_set * FEC_SETS;

    let nodes: Vec<[u8; 32]> = (0..2 * (LEAVES - 1) * FEC_SETS)
        .map(|i| {
            let mut h = [0u8; 32];
            for (j, b) in h.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(31).wrapping_add(j as u8);
            }
            h
        })
        .collect();

    assert_eq!(nodes.len(), 2 * total, "need two inputs per join");
    assert_eq!(
        nodes.chunks_exact(2).count(),
        total,
        "pair count must equal total"
    );
    println!("{FEC_SETS} FEC sets x {joins_per_set} interior joins = {total} node hashes/block");
    println!("each is 26 + 20 + 20 = 66 bytes (2 blocks), uniform");
    println!("active backend: {}", tape_sha256::backend());
    println!();

    let mut sout = vec![[0u8; 32]; total];
    let serial = bench("validator join (serial)", total, || {
        for (o, p) in sout.iter_mut().zip(nodes.chunks_exact(2)) {
            *o = join_serial(&p[0], &p[1]);
        }
        std::hint::black_box(&sout);
    });

    // Batched: materialise each 40-byte body, then one prefixed batch call.
    // The concat is charged to us, since without a pair-taking API an
    // integration would have to do exactly this.
    // Buffers hoisted out of the timed region; a real integration reuses them.
    // The 40-byte concat stays inside, since without a pair-taking API the
    // caller has to do it every time.
    let mut bodies = vec![[0u8; 2 * PROOF_ENTRY]; total];
    let mut out = vec![[0u8; 32]; total];
    // Row 1: everything the caller must do per batch -- concat both halves and
    // build the slice table -- then hash.
    let batched = bench("prefixed + concat (per call)", total, || {
        for (b, p) in bodies.iter_mut().zip(nodes.chunks_exact(2)) {
            b[..PROOF_ENTRY].copy_from_slice(&p[0][..PROOF_ENTRY]);
            b[PROOF_ENTRY..].copy_from_slice(&p[1][..PROOF_ENTRY]);
        }
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        hash_many_prefixed(PREFIX_NODE, &refs, &mut out);
        std::hint::black_box(&out);
    });

    // Row 2: the same call with concat and slice table hoisted out entirely.
    // Compared against hash_pairs this isolates the API path, since neither
    // does any per-call setup.
    for (b, p) in bodies.iter_mut().zip(nodes.chunks_exact(2)) {
        b[..PROOF_ENTRY].copy_from_slice(&p[0][..PROOF_ENTRY]);
        b[PROOF_ENTRY..].copy_from_slice(&p[1][..PROOF_ENTRY]);
    }
    let prebuilt: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
    let prehashed = bench("prefixed, all setup hoisted", total, || {
        hash_many_prefixed(PREFIX_NODE, &prebuilt, &mut out);
        std::hint::black_box(&out);
    });

    // Correctness: the batched path must equal the serial join.
    {
        let pairs: Vec<_> = nodes.chunks_exact(2).take(total).collect();
        let bodies: Vec<[u8; 2 * PROOF_ENTRY]> = pairs
            .iter()
            .map(|p| {
                let mut b = [0u8; 2 * PROOF_ENTRY];
                b[..PROOF_ENTRY].copy_from_slice(&p[0][..PROOF_ENTRY]);
                b[PROOF_ENTRY..].copy_from_slice(&p[1][..PROOF_ENTRY]);
                b
            })
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; refs.len()];
        hash_many_prefixed(PREFIX_NODE, &refs, &mut got);
        for (i, p) in pairs.iter().enumerate() {
            assert_eq!(got[i], join_serial(&p[0], &p[1]), "node {i}");
        }
        println!("\n(verified: batched output == serial join for all {total})");
    }

    // Same work, but batched per FEC set (63 at a time) rather than per block
    // (2016). Real integrations build one tree at a time, so this is the batch
    // size that actually applies.
    let mut sbodies = vec![[0u8; 2 * PROOF_ENTRY]; joins_per_set];
    let mut sout2 = vec![[0u8; 32]; joins_per_set];
    let per_set = bench("tape, batched per FEC set (63)", total, || {
        for set in 0..FEC_SETS {
            let base = set * joins_per_set * 2;
            for (b, p) in sbodies
                .iter_mut()
                .zip(nodes[base..].chunks_exact(2).take(joins_per_set))
            {
                b[..PROOF_ENTRY].copy_from_slice(&p[0][..PROOF_ENTRY]);
                b[PROOF_ENTRY..].copy_from_slice(&p[1][..PROOF_ENTRY]);
            }
            let refs: Vec<&[u8]> = sbodies.iter().map(|b| b.as_slice()).collect();
            hash_many_prefixed(PREFIX_NODE, &refs, &mut sout2);
            std::hint::black_box(&sout2);
        }
    });

    // The point of hash_pairs: the two 20-byte halves are read in place, so
    // there is no concat and no bodies buffer at all.
    let lefts: Vec<&[u8]> = nodes
        .chunks_exact(2)
        .map(|p| &p[0][..PROOF_ENTRY])
        .collect();
    let rights: Vec<&[u8]> = nodes
        .chunks_exact(2)
        .map(|p| &p[1][..PROOF_ENTRY])
        .collect();
    let mut pout = vec![[0u8; 32]; total];
    let paired = bench("tape hash_pairs (no concat)", total, || {
        hash_pairs(PREFIX_NODE, &lefts, &rights, &mut pout);
        std::hint::black_box(&pout);
    });
    for (i, p) in nodes.chunks_exact(2).enumerate() {
        assert_eq!(pout[i], join_serial(&p[0], &p[1]), "hash_pairs node {i}");
    }

    // Same batched work split across two caller-owned threads, each handling
    // different FEC sets. Within one tree the levels are sequential, but FEC
    // sets are independent, so this is the split a real integration has.
    // Run under `taskset -c a,b` with a and b siblings of one physical core.
    let per_thread = FEC_SETS / 2;
    let smt = {
        const ROUNDS: usize = 15;
        const ITERS: usize = 200;
        let mut best = f64::INFINITY;
        for r in 0..=ROUNDS {
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for part in 0..2 {
                    let nodes = &nodes;
                    sc.spawn(move || {
                        let lo = part * per_thread * joins_per_set * 2;
                        let n = per_thread * joins_per_set;
                        let mut b2 = vec![[0u8; 2 * PROOF_ENTRY]; n];
                        let mut o2 = vec![[0u8; 32]; n];
                        for _ in 0..ITERS {
                            for (b, p) in b2.iter_mut().zip(nodes[lo..].chunks_exact(2)) {
                                b[..PROOF_ENTRY].copy_from_slice(&p[0][..PROOF_ENTRY]);
                                b[PROOF_ENTRY..].copy_from_slice(&p[1][..PROOF_ENTRY]);
                            }
                            let refs: Vec<&[u8]> = b2.iter().map(|b| b.as_slice()).collect();
                            hash_many_prefixed(PREFIX_NODE, &refs, &mut o2);
                            std::hint::black_box(&o2);
                        }
                    });
                }
            });
            if r == 0 {
                continue;
            }
            best = best.min(t0.elapsed().as_secs_f64() / ITERS as f64);
        }
        println!("tape, SMT pair (2 threads)      {:>9.1} us", best * 1e6);
        best
    };

    println!();
    println!(
        "SMT pair vs single thread (batched):       {:.2}x",
        batched / smt
    );
    println!(
        "hash_pairs speedup:                        {:.2}x",
        serial / paired
    );
    println!(
        "caller-side setup cost (concat + slice table): {:.1}%",
        100.0 * (batched - prehashed) / prehashed
    );
    println!(
        "hash_pairs vs hoisted-prefixed (pure API):     {:+.1}%",
        100.0 * (paired - prehashed) / prehashed
    );
    println!(
        "interior-node speedup, whole block (2016): {:.2}x",
        serial / batched
    );
    println!(
        "interior-node speedup, per FEC set (63):   {:.2}x",
        serial / per_set
    );
    println!(
        "per FEC set: serial {:.2} us -> batched {:.2} us",
        serial * 1e6 / FEC_SETS as f64,
        per_set * 1e6 / FEC_SETS as f64
    );
}
